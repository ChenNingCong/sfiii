"""Fault-tolerant DIAMBRA env layer.

Principle (per design): a genuine engine hang SHOULD block the gRPC call — so we
put a watchdog *above* gRPC. If a step/reset exceeds its timeout (hang) or raises
(crash/desync), we DETACH that env (kill+respawn its engine on the same port) and
ATTACH a fresh client, then hand back a clean terminal transition so the caller's
replay buffer just drops the corrupted partial episode.

  RobustDiambraEnv      — single env that self-heals on hang/crash.
  FaultTolerantVecEnv   — threaded vec wrapper; one stuck env never blocks the rest.

Off-policy (DQN) is the natural consumer: a dropped episode costs nothing; there is
no synchronized on-policy rollout to corrupt.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import diambra.arena
from engine_manager import EngineManager


class EnvRecovered(Exception):
    """Internal signal: the env was detached+reattached during this call."""


class RobustDiambraEnv:
    def __init__(self, mgr: EngineManager, rank: int, settings_fn, game_id="sfiii3n",
                 wrappers_fn=None, step_timeout=10.0, reset_timeout=40.0,
                 make_timeout=60.0, max_retries=3, verbose=True, with_stats=True):
        self.mgr = mgr
        self.rank = rank
        self.settings_fn = settings_fn
        self.wrappers_fn = wrappers_fn
        self.game_id = game_id
        self.with_stats = with_stats
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.make_timeout = make_timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.n_recoveries = 0
        self.env = None
        self._make()

    def _log(self, msg):
        if self.verbose:
            print(f"[robust rank={self.rank}] {msg}", flush=True)

    # ── attach / detach ───────────────────────────────────────────────────────
    def _make(self):
        if not self.mgr.is_healthy(self.rank):
            self._log("engine unhealthy at make -> restart")
            self.mgr.restart(self.rank)
        # point diambra at our engine pool (rank -> base_port+rank). Without this,
        # make() falls back to localhost:50051 and channel_ready blocks for the full
        # grpc_timeout — the classic silent hang.
        os.environ["DIAMBRA_ENVS"] = self.mgr.addresses()
        from sfiii_env import build_env   # stats-enabled make() equivalent
        t = time.time()
        wr = self.wrappers_fn() if self.wrappers_fn else None
        self.env = build_env(self.settings_fn(), wr, rank=self.rank,
                             game_id=self.game_id, with_stats=self.with_stats)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._log(f"attached to {self.mgr.addresses().split()[self.rank]} in {time.time()-t:.1f}s")

    def _detach_attach(self):
        """KILL the engine first (this frees any gRPC call still blocked in a
        watchdog thread, and avoids close() hanging on a frozen engine), then
        respawn + re-make a fresh client. We deliberately do NOT call
        self.env.close() — its channel points at a dead/frozen engine and Close()
        would block forever."""
        self.n_recoveries += 1
        self._log(f"DETACH+ATTACH (recovery #{self.n_recoveries})")
        self.env = None                                  # drop ref; thread will error out
        self.mgr.restart(self.rank, ready_timeout=self.make_timeout)
        self._make()

    def _watch(self, fn, timeout, *a, **kw):
        """Run a blocking gRPC call under a watchdog daemon thread. On timeout the
        thread is abandoned (it unblocks & dies when _detach_attach kills the
        engine), so we never reuse a stuck worker."""
        box = {}
        def run():
            try:
                box["v"] = fn(*a, **kw)
            except BaseException as e:   # noqa: BLE001 — propagate gRPC errors
                box["e"] = e
        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise TimeoutError(f"call exceeded {timeout}s (engine hang)")
        if "e" in box:
            raise box["e"]
        return box["v"]

    # ── gym API (self-healing) ────────────────────────────────────────────────
    def reset(self, **kw):
        for attempt in range(self.max_retries):
            try:
                return self._watch(self.env.reset, self.reset_timeout, **kw)
            except Exception as e:
                self._log(f"reset failed ({type(e).__name__}); attempt {attempt+1}/{self.max_retries}")
                self._detach_attach()
        raise RuntimeError(f"rank {self.rank}: reset failed after {self.max_retries} recoveries")

    def step(self, action):
        try:
            return self._watch(self.env.step, self.step_timeout, action)
        except Exception as e:
            self._log(f"step hang/err ({type(e).__name__}) -> recover, end episode")
            self._detach_attach()
            obs, info = self._watch(self.env.reset, self.reset_timeout)
            info = dict(info or {}); info["corrupted"] = True
            # terminal=True so the buffer drops the partial episode cleanly
            return obs, 0.0, True, False, info

    def close(self):
        if self.env is not None:
            try:
                self._watch(self.env.close, 5.0)   # don't hang on a frozen engine
            except Exception:
                pass


class FaultTolerantVecEnv:
    """Synchronous vec env over RobustDiambraEnv; envs stepped concurrently with
    threads (gRPC releases the GIL). A hung env recovers in place without blocking
    the others — its slot just returns a terminal/corrupted transition that step()."""
    def __init__(self, mgr, settings_fn, game_id="sfiii3n", wrappers_fn=None, **kw):
        self.mgr = mgr
        self.num_envs = mgr.num_envs
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs)
        # build in parallel (each RobustDiambraEnv self-heals if its make races)
        self.envs = list(self._pool.map(
            lambda i: RobustDiambraEnv(mgr, i, settings_fn, game_id, wrappers_fn, **kw),
            range(self.num_envs)))
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        return list(self._pool.map(lambda e: e.reset(), self.envs))

    def step(self, actions):
        outs = list(self._pool.map(lambda ea: ea[0].step(ea[1]), zip(self.envs, actions)))
        return outs

    def recoveries(self):
        return {i: e.n_recoveries for i, e in enumerate(self.envs)}

    def close(self):
        for e in self.envs:
            e.close()
        self._pool.shutdown(wait=False)
