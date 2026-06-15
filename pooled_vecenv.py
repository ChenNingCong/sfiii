"""PooledVecEnv — fault-tolerant, hot-spare vec env for the slow DIAMBRA engine.

The problem: reset() is sync-blocking (~3s) and a normal vec env stalls the whole
batch on every episode end. The fix (your design): keep a pool of P = N + spares
engines. N serve the active slots; the extras are pre-reset "hot spares". When a
slot's episode ends we DON'T reset inline — we swap in a warm spare instantly and
hand the finished engine to a background recycler (async reset → back to the spare
pool). step() therefore never blocks on reset as long as spares are available.

Each engine is a RobustDiambraEnv (self-heals on hang/crash via EngineManager) and
carries the StatsWrapper, so info['episode_stats'] surfaces on episode end.

Clean gym-VecEnv-style interface:
    obs            = venv.reset()                       # batched
    obs, rew, done, infos = venv.step(actions)          # batched; autoreset via spares
SB3-style autoreset: on done, `obs[i]` is the NEW episode's first frame and the
real terminal frame is in infos[i]['terminal_observation'].
"""
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from engine_manager import EngineManager
from robust_env import RobustDiambraEnv


def _stack_obs(obs_list):
    """Batch a list of obs (Box array or Dict-of-arrays) into vec form."""
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in first}
    return np.stack(obs_list)


class PooledVecEnv:
    def __init__(self, num_envs, settings_fn, wrappers_fn=None, num_spares=4,
                 base_port=51000, prefix="engine", game_id="sfiii3n",
                 recycler_threads=None, verbose=True, **robust_kw):
        assert num_spares >= 0
        self.num_envs = num_envs
        self.num_spares = num_spares
        self.total = num_envs + num_spares
        self.verbose = verbose
        self.mgr = EngineManager(num_envs=self.total, base_port=base_port, prefix=prefix)
        t = time.time()
        self.mgr.start()
        os.environ["DIAMBRA_ENVS"] = self.mgr.addresses()
        self._launch_t = time.time() - t

        # build all P engines in parallel (containment fix => no race)
        self._build_exec = ThreadPoolExecutor(max_workers=self.total)
        t = time.time()
        self.engines = list(self._build_exec.map(
            lambda r: RobustDiambraEnv(mgr=self.mgr, rank=r, settings_fn=settings_fn,
                                       wrappers_fn=wrappers_fn, game_id=game_id,
                                       verbose=False, **robust_kw),
            range(self.total)))
        self._make_t = time.time() - t
        self.action_space = self.engines[0].action_space
        self.observation_space = self.engines[0].observation_space

        # spare machinery
        self._spare_q: "queue.Queue" = queue.Queue()
        self._spare_obs: dict = {}
        self._recycle_q: "queue.Queue" = queue.Queue()
        self.active = [None] * num_envs
        self._step_exec = ThreadPoolExecutor(max_workers=num_envs)
        self._recycle_exec = ThreadPoolExecutor(
            max_workers=recycler_threads or max(1, num_spares))
        self._n_swaps = 0
        self._n_stalls = 0          # times we had to reset inline (pool exhausted)
        self._closed = False

    def _log(self, m):
        if self.verbose:
            print(f"[pool] {m}", flush=True)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def reset(self):
        outs = list(self._build_exec.map(lambda e: (e, e.reset()), self.engines))
        obs_active = [None] * self.num_envs
        for i in range(self.num_envs):
            e, (o, _info) = outs[i]
            self.active[i] = e
            obs_active[i] = o
        for j in range(self.num_envs, self.total):
            e, (o, _info) = outs[j]
            self._spare_obs[e] = o
            self._spare_q.put(e)
        self._log(f"reset: {self.num_envs} active + {self._spare_q.qsize()} spares "
                  f"(launch {self._launch_t:.1f}s, make {self._make_t:.1f}s)")
        return _stack_obs(obs_active)

    def _recycle(self, e):
        """Background: reset a finished engine and return it to the spare pool."""
        try:
            o, _info = e.reset()
            self._spare_obs[e] = o
            self._spare_q.put(e)
        except Exception as ex:        # RobustDiambraEnv already tried to self-heal
            self._log(f"recycle failed permanently: {ex!r}; engine dropped")

    def step(self, actions):
        def one(i):
            e = self.active[i]
            o, r, term, trunc, info = e.step(actions[i])
            if term or trunc:
                info = dict(info)
                info["terminal_observation"] = o
                try:
                    spare = self._spare_q.get_nowait()        # hot swap, no reset wait
                    o = self._spare_obs.pop(spare)
                    self.active[i] = spare
                    self._recycle_exec.submit(self._recycle, e)
                    self._n_swaps += 1
                except queue.Empty:
                    o, _ = e.reset()                          # fallback: inline (blocks)
                    self._n_stalls += 1
            return o, r, bool(term or trunc), info

        res = list(self._step_exec.map(one, range(self.num_envs)))
        obs = _stack_obs([x[0] for x in res])
        rew = np.array([x[1] for x in res], dtype=np.float32)
        done = np.array([x[2] for x in res], dtype=bool)
        infos = [x[3] for x in res]
        return obs, rew, done, infos

    # ── introspection ─────────────────────────────────────────────────────────
    def stats(self):
        return dict(spares_ready=self._spare_q.qsize(), swaps=self._n_swaps,
                    inline_stalls=self._n_stalls,
                    recoveries={i: e.n_recoveries for i, e in enumerate(self.engines)})

    def close(self):
        if self._closed:
            return
        self._closed = True
        for e in self.engines:
            try: e.close()
            except Exception: pass
        self._step_exec.shutdown(wait=False)
        self._recycle_exec.shutdown(wait=False)
        self._build_exec.shutdown(wait=False)
        self.mgr.stop_all()
