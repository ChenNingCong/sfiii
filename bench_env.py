"""Smoke-test + SPS benchmark for the DIAMBRA sfiii env.

Spawns N engines via EngineManager (patched no-auth image), creates N gym envs
in-process (round-robin stepping), runs random actions, reports steps/sec.

Usage:
    python bench_env.py --num-envs 1  --steps 300
    python bench_env.py --num-envs 8  --steps 2000 --parallel-make
"""
import argparse
import os
import time

import numpy as np

from engine_manager import EngineManager


def build_settings():
    from diambra.arena import EnvironmentSettings, SpaceTypes
    s = EnvironmentSettings()
    s.game_id = "sfiii3n"
    s.characters = "Ryu"
    s.difficulty = 6
    s.action_space = SpaceTypes.DISCRETE
    s.step_ratio = 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=51000)
    ap.add_argument("--steps", type=int, default=300, help="total steps across all envs")
    ap.add_argument("--make-mode", choices=["serial", "parallel"], default="serial",
                    help="how to construct the gym envs (test the spawn race)")
    ap.add_argument("--prefix", type=str, default="engine",
                    help="apptainer instance name prefix (use distinct prefixes to run "
                         "isolated benchmarks concurrently)")
    opt = ap.parse_args()

    import diambra.arena

    mgr = EngineManager(num_envs=opt.num_envs, base_port=opt.base_port, prefix=opt.prefix)
    print(f"[spawn] starting {opt.num_envs} engines...")
    t0 = time.time()
    mgr.start()
    os.environ["DIAMBRA_ENVS"] = mgr.addresses()
    print(f"[spawn] engines ready in {time.time()-t0:.1f}s  DIAMBRA_ENVS={mgr.addresses()}")
    print(f"[health] {mgr.health_report()}")

    settings = build_settings()
    envs = []
    try:
        # ── make envs ────────────────────────────────────────────────────────
        tmk = time.time()
        if opt.make_mode == "serial":
            for i in range(opt.num_envs):
                te = time.time()
                envs.append(diambra.arena.make("sfiii3n", settings, rank=i))
                print(f"[make] env {i} created in {time.time()-te:.1f}s")
        else:
            import threading
            results = [None] * opt.num_envs
            errs = [None] * opt.num_envs
            def _mk(i):
                try:
                    results[i] = diambra.arena.make("sfiii3n", build_settings(), rank=i)
                except Exception as e:
                    errs[i] = e
            ths = [threading.Thread(target=_mk, args=(i,)) for i in range(opt.num_envs)]
            [t.start() for t in ths]; [t.join() for t in ths]
            for i, e in enumerate(errs):
                if e: print(f"[make] env {i} ERROR: {e!r}")
            envs = [r for r in results if r is not None]
        print(f"[make] {len(envs)}/{opt.num_envs} envs created in {time.time()-tmk:.1f}s "
              f"({opt.make_mode})")

        if not envs:
            print("[FAIL] no envs created"); return

        # ── reset ────────────────────────────────────────────────────────────
        trs = time.time()
        obs_list = []
        for i, e in enumerate(envs):
            o, _ = e.reset(seed=1000 + i)
            obs_list.append(o)
        print(f"[reset] {len(envs)} envs reset in {time.time()-trs:.1f}s")

        # ── step loop (CONCURRENT via threads; gRPC releases GIL) + SPS ───────
        from concurrent.futures import ThreadPoolExecutor
        rng = np.random.default_rng(0)
        per_env = max(1, opt.steps // len(envs))
        nstep = 0; ndone = 0
        pool = ThreadPoolExecutor(max_workers=len(envs))

        def _one(e):
            a = e.action_space.sample()
            obs, rew, term, trunc, info = e.step(a)
            if term or trunc:
                e.reset(seed=int(rng.integers(0, 1_000_000)))
                return 1
            return 0

        tstep = time.time()
        for _ in range(per_env):
            for d in pool.map(_one, envs):   # barrier per vec-step, like SubprocVecEnv
                nstep += 1; ndone += d
        pool.shutdown()
        dt = time.time() - tstep
        print(f"\n===== RESULT =====")
        print(f"envs={len(envs)} steps={nstep} episodes_done={ndone} time={dt:.1f}s")
        print(f"SPS (aggregate) = {nstep/dt:,.1f}   per-env = {nstep/dt/len(envs):,.1f}")
    finally:
        for e in envs:
            try: e.close()
            except Exception: pass
        mgr.stop_all()
        print("[cleanup] engines stopped")


if __name__ == "__main__":
    main()
