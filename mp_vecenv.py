"""MPVecEnv — the production vec env for sfiii.

Combines everything proven out:
  * MULTIPROCESS (one worker process per slot) -> near-linear SPS scaling
    (threaded was GIL-bound at ~2.3k SPS; multiprocess hit ~5.8k at N=16).
  * PARALLEL LAUNCH via EngineManager (--contain fix) -> all engines up in ~1s.
  * FAULT TOLERANT: each worker runs a RobustDiambraEnv that detaches+reattaches
    its engine on hang/crash (observable; logged).
  * HOT-SPARE per worker: an extra pre-reset engine per worker. On episode end the
    worker swaps to the warm spare INSTANTLY and resets the finished engine in a
    background thread -> step() doesn't stall the batch on the ~3s reset.
  * STATS: each engine carries the StatsWrapper, so info['episode_stats'] surfaces.

Clean SB3-style VecEnv interface:
    obs              = venv.reset()
    obs, rew, done, infos = venv.step(actions)     # autoreset; terminal in info
Engine layout: worker i owns active rank=i and (if spares) spare rank=num_envs+i,
so total engines = num_envs * (1 + spares_per_worker).
"""
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from engine_manager import EngineManager
from robust_env import RobustDiambraEnv


# ── shared-memory obs transfer (avoid pickling 64-256KB frames through pipes) ──
def _obs_layout(space):
    """Ordered [(key, shape, np.dtype)] for a gym Dict obs space."""
    return [(k, tuple(space.spaces[k].shape), np.dtype(space.spaces[k].dtype))
            for k in sorted(space.spaces.keys())]


def _layout_nbytes(layout):
    return sum(int(np.prod(sh)) * dt.itemsize for _, sh, dt in layout)


class _ShmViews:
    """Numpy views into a SharedMemory buffer, one per obs key (single env)."""
    def __init__(self, layout, buf):
        self.views = {}
        off = 0
        for k, sh, dt in layout:
            self.views[k] = np.ndarray(sh, dtype=dt, buffer=buf, offset=off)
            off += int(np.prod(sh)) * dt.itemsize

    def write(self, obs):
        for k, v in self.views.items():
            # ravel both sides: handles 0-d (scalar) views and any reset/step
            # shape quirks as long as the element count matches the declared layout.
            v.reshape(-1)[:] = np.asarray(obs[k]).reshape(-1)

    def read(self):
        return {k: v.copy() for k, v in self.views.items()}   # copy: shm is reused next step


# ── settings reconstruction inside workers (dicts are picklable; objects/fns less so)
def _build_settings(d):
    from diambra.arena import EnvironmentSettings, SpaceTypes, load_settings_flat_dict
    d = dict(d)
    if isinstance(d.get("action_space"), str):
        d["action_space"] = (SpaceTypes.DISCRETE if d["action_space"] == "discrete"
                             else SpaceTypes.MULTI_DISCRETE)
    if isinstance(d.get("frame_shape"), list):
        d["frame_shape"] = tuple(d["frame_shape"])
    return load_settings_flat_dict(EnvironmentSettings, d)


def _build_wrappers(d):
    from diambra.arena import WrappersSettings, load_settings_flat_dict
    if not d:
        return WrappersSettings()
    return load_settings_flat_dict(WrappersSettings, dict(d))


def _stack(obs_list):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in first}
    return np.stack(obs_list)


# ── worker process ────────────────────────────────────────────────────────────
def _worker(conn, rank, cfg):
    try:
        mgr = EngineManager(num_envs=cfg["total"], base_port=cfg["base_port"],
                            sif=cfg["sif"], prefix=cfg["prefix"])   # config only; parent started engines
        # per-env settings override (e.g. a different continue_game per env to
        # shape the stage-coverage distribution). Keyed by worker index `rank`;
        # the worker's active env AND its spares share the same override.
        ov = (cfg.get("per_env_settings") or {})
        override = ov.get(rank, {}) if isinstance(ov, dict) else (ov[rank] if rank < len(ov) else {})
        sfn = lambda: _build_settings({**cfg["settings"], **(override or {})})
        wfn = (lambda: _build_wrappers(cfg["wrappers"])) if cfg["wrappers"] else None
        rk = dict(step_timeout=cfg["step_timeout"], reset_timeout=cfg["reset_timeout"],
                  make_timeout=cfg["make_timeout"], game_id=cfg["game_id"], verbose=cfg["verbose"])
        active = RobustDiambraEnv(mgr=mgr, rank=rank, settings_fn=sfn, wrappers_fn=wfn, **rk)

        spares_q: "queue.Queue" = queue.Queue()
        spare_obs = {}
        n_spare = cfg["spares_per_worker"]
        for s in range(n_spare):
            srank = cfg["num_envs"] * (s + 1) + rank
            sp = RobustDiambraEnv(mgr=mgr, rank=srank, settings_fn=sfn, wrappers_fn=wfn, **rk)
            o, _ = sp.reset()
            spare_obs[sp] = o
            spares_q.put(sp)
        recyc = ThreadPoolExecutor(max_workers=max(1, n_spare))

        layout = _obs_layout(active.observation_space)
        conn.send(("ready", (active.observation_space, active.action_space, layout)))
        # parent allocates the shm and sends its name; we attach + build views.
        tag, shm_name = conn.recv()
        assert tag == "shm"
        shm = shared_memory.SharedMemory(name=shm_name)   # parent owns/unlinks it
        views = _ShmViews(layout, shm.buf)            # obs goes here, NOT through the pipe

        # optional: pin THIS worker process to its NUMA node's cores so the worker and
        # its engine (pinned to the same cores) share a CCD -> no cross-NUMA gRPC.
        wcores = (cfg.get("worker_cores") or {}).get(rank)
        if wcores:
            try: os.sched_setaffinity(0, set(wcores))
            except Exception: pass

        def recycle(env):
            o, _ = env.reset()
            spare_obs[env] = o
            spares_q.put(env)

        profile = bool(os.environ.get("MPVEC_PROFILE"))
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                _t0 = time.time()
                o, r, term, trunc, info = active.step(data)
                if profile:
                    info = dict(info); info["_wstep"] = time.time() - _t0
                done = bool(term or trunc)
                if done:
                    info = dict(info)
                    info["terminal_observation"] = o   # terminal obs still via pipe (rare)
                    try:
                        sp = spares_q.get_nowait()
                        o = spare_obs.pop(sp)
                        recyc.submit(recycle, active)
                        active = sp
                        info["hot_swap"] = True
                    except queue.Empty:
                        o, _ = active.reset()
                        info["hot_swap"] = False
                views.write(o)                         # obs -> shared memory
                conn.send((r, done, info))             # pipe carries only reward/done/info
            elif cmd == "reset":
                o, info = active.reset(seed=data)
                views.write(o)
                conn.send(("reset_done", info))
            elif cmd == "close":
                try: active.close()
                except Exception: pass
                try: shm.close()
                except Exception: pass
                conn.close()
                return
    except BaseException as e:
        import traceback
        tb = f"{e!r}\n{traceback.format_exc()}"
        try:
            with open(f"/tmp/mpvec_worker_{rank}.err", "w") as f:
                f.write(tb)
        except Exception: pass
        try: conn.send(("error", tb))
        except Exception: pass
        conn.close()


# ── NUMA topology (for optional pinning) ──────────────────────────────────────
def _numa_nodes():
    """[(node_id, [cores])...] from /sys; falls back to one node of all cpus."""
    import glob, re
    nodes = []
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        nid = int(re.search(r"node(\d+)", d).group(1))
        try:
            with open(os.path.join(d, "cpulist")) as f:
                spec = f.read().strip()
        except Exception:
            continue
        cores = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-"); cores += list(range(int(a), int(b) + 1))
            elif part:
                cores.append(int(part))
        if cores:
            nodes.append((nid, cores))
    if not nodes:
        nodes = [(0, sorted(os.sched_getaffinity(0)))]
    return nodes


# ── parent vec env ──────────────────────────────────────────────────────────
class MPVecEnv:
    def __init__(self, num_envs, settings, wrappers=None, spares_per_worker=1,
                 base_port=51000, prefix="engine", sif=None, game_id="sfiii3n",
                 step_timeout=10.0, reset_timeout=60.0, make_timeout=60.0,
                 ready_timeout=120.0, verbose_workers=False, per_env_settings=None,
                 numa_pin=False, numa_offset=0):
        # per_env_settings: optional list[dict] (len num_envs) or {rank: dict} of
        # per-env setting overrides merged onto `settings` (e.g. continue_game).
        from engine_manager import DEFAULT_SIF
        self.num_envs = num_envs
        self.spares_per_worker = spares_per_worker
        self.total = num_envs * (1 + spares_per_worker)
        # optional NUMA pinning: map worker rank i -> node ((i+offset) % n_nodes), and
        # pin its engines (active rank i + spares num_envs*s+i) to that node's cores.
        cpu_pin = None; self._worker_cores = {}
        if numa_pin:
            nodes = _numa_nodes(); nn = len(nodes)
            def _node_for_worker(w): return nodes[(w + numa_offset) % nn][1]
            for w in range(num_envs):
                self._worker_cores[w] = _node_for_worker(w)
            def cpu_pin(engine_idx, _nf=_node_for_worker):
                cores = _nf(engine_idx % num_envs)
                return f"{min(cores)}-{max(cores)}"
        # 1) parallel launch ALL engines (active + spares) from the parent
        self.mgr = EngineManager(num_envs=self.total, base_port=base_port,
                                 sif=sif or DEFAULT_SIF, prefix=prefix, cpu_pin=cpu_pin)
        t = time.time()
        self.mgr.start(ready_timeout=ready_timeout)
        self.launch_t = time.time() - t

        cfg = dict(total=self.total, num_envs=num_envs, base_port=base_port,
                   sif=sif or DEFAULT_SIF, prefix=prefix, settings=settings,
                   per_env_settings=per_env_settings,
                   wrappers=wrappers or {}, game_id=game_id, step_timeout=step_timeout,
                   reset_timeout=reset_timeout, make_timeout=make_timeout,
                   spares_per_worker=spares_per_worker, verbose=verbose_workers,
                   worker_cores=self._worker_cores)

        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        t = time.time()
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, i, cfg), daemon=True)
            p.start()
            self._conns.append(parent)
            self._procs.append(p)
        # wait for all workers ready, then hand each a shared-memory block for obs
        spaces = []
        self._shms = []; self._views = []
        for i, c in enumerate(self._conns):
            tag, payload = c.recv()
            if tag == "error":
                self.close()
                raise RuntimeError(f"worker {i} failed to start: {payload}")
            obs_space, act_space, layout = payload
            spaces.append((obs_space, act_space))
            shm = shared_memory.SharedMemory(create=True, size=max(1, _layout_nbytes(layout)))
            self._shms.append(shm)
            self._views.append(_ShmViews(layout, shm.buf))
            c.send(("shm", shm.name))
        self.make_t = time.time() - t
        self.observation_space, self.action_space = spaces[0]
        self._conn_index = {c: i for i, c in enumerate(self._conns)}   # for wait_ready

    def _read_obs(self):
        return _stack([v.read() for v in self._views])     # from shared memory, no pickle

    def reset(self):
        for c in self._conns:
            c.send(("reset", None))
        for c in self._conns:
            c.recv()                                       # ("reset_done", info); obs is in shm
        return self._read_obs()

    def step_async(self, actions):
        for c, a in zip(self._conns, actions):
            c.send(("step", a))

    def step_wait(self):
        res = [c.recv() for c in self._conns]              # (reward, done, info) only
        obs = self._read_obs()                             # obs from shm
        rew = np.array([x[0] for x in res], dtype=np.float32)
        done = np.array([x[1] for x in res], dtype=bool)
        infos = [x[2] for x in res]
        return obs, rew, done, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    # ── per-worker async primitives (hide per-engine stalls) ──────────────────
    # The policy is frozen during a rollout, so each env may step at its own pace
    # with ZERO off-policy error. Service whichever workers are ready instead of
    # blocking the whole batch on the one mid round/stage transition (~2.4s stall).
    def send_action(self, i, action):
        self._conns[i].send(("step", action))

    def wait_ready(self, timeout=None):
        """Return indices of workers whose step result is ready to read."""
        from multiprocessing.connection import wait as _wait
        ready = _wait(self._conns, timeout=timeout)
        return [self._conn_index[c] for c in ready]

    def recv_one(self, i):
        """Read one worker's result; obs comes from its shm view."""
        r, done, info = self._conns[i].recv()
        return self._views[i].read(), r, done, info

    def close(self):
        for c in self._conns:
            try: c.send(("close", None))
            except Exception: pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        for shm in getattr(self, "_shms", []):
            try: shm.close(); shm.unlink()
            except Exception: pass
        self.mgr.stop_all()


# ── double-buffered (async) vec env ───────────────────────────────────────────
class AsyncMPVecEnv:
    """Correct double-buffered sampling (Sample-Factory style). Split `num_envs` into
    TWO HALVES (num_envs//2 each) that step one phase out of sync: while the learner
    runs the policy on one half (and that half steps), the OTHER half is already
    stepping. The per-step learner window thus overlaps env stepping -> kills the
    per-step starvation.

    DROP-IN engine count: two MPVecEnv(num_envs//2) groups => num_envs*(1+spares)
    engines total, exactly like MPVecEnv(num_envs). Effective width = num_envs.

    Interface: reset() -> half-wide obs (group 0); step(half_actions) applies them to
    the half whose obs were last returned and returns the OTHER half's in-flight
    result (half-wide). `last_group` tells which half the returned batch belongs to.
    The returned (obs,rew,done,info) is the result of THAT half's PREVIOUS action.
    Each env keeps a continuous trajectory -> STANDARD GAE; the ONLY care needed is
    that bootstrap (next_value/next_done) is assembled per half (different phases).
    """
    def __init__(self, num_envs, settings, wrappers=None, spares_per_worker=1,
                 base_port=51000, prefix="env", game_id="sfiii3n", **kw):
        assert num_envs % 2 == 0, "num_envs must be even for double buffering"
        self.num_envs = num_envs
        self.half = num_envs // 2
        span = self.half * (1 + spares_per_worker)
        self.g = [MPVecEnv(self.half, settings, wrappers=wrappers,
                           spares_per_worker=spares_per_worker,
                           base_port=base_port + k * span, prefix=f"{prefix}{k}",
                           game_id=game_id, **kw) for k in (0, 1)]
        self.launch_t = sum(x.launch_t for x in self.g)
        self.make_t = max(x.make_t for x in self.g)
        self.observation_space = self.g[0].observation_space
        self.action_space = self.g[0].action_space
        self._inflight = [False, False]
        self._reset_obs = [None, None]
        self.last_group = 0

    def reset(self):
        for k in (0, 1):
            self._reset_obs[k] = self.g[k].reset()
        self._inflight = [False, False]
        self.last_group = 0
        return self._reset_obs[0]            # half-0 obs (half-wide)

    def step(self, half_actions):
        cur = self.last_group
        self.g[cur].step_async(half_actions); self._inflight[cur] = True
        other = 1 - cur
        if self._inflight[other]:
            res = self.g[other].step_wait(); self._inflight[other] = False
        else:                                # first touch of `other`: hand back its reset obs
            res = (self._reset_obs[other],
                   np.zeros(self.half, np.float32),
                   np.zeros(self.half, bool),
                   [{} for _ in range(self.half)])
        self.last_group = other
        return res

    def close(self):
        for x in self.g:
            x.close()
