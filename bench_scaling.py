"""SPS scaling benchmark: sweep N envs, report launch/make/reset time + aggregate
SPS and scaling efficiency. Uses the real 128x128x1 frame (matches training).

    python bench_scaling.py --ns 1,2,4,8,12,16 --secs 8
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from engine_manager import EngineManager


def settings_fn():
    from diambra.arena import EnvironmentSettings, SpaceTypes
    s = EnvironmentSettings()
    s.game_id = "sfiii3n"; s.characters = "Ryu"; s.difficulty = 6
    s.action_space = SpaceTypes.DISCRETE; s.step_ratio = 1
    s.frame_shape = (128, 128, 1)      # match training config
    return s


def run_n(n, base_port, secs):
    import diambra.arena
    mgr = EngineManager(num_envs=n, base_port=base_port, prefix=f"sc{n}")
    t = time.time(); mgr.start(); launch_t = time.time() - t
    os.environ["DIAMBRA_ENVS"] = mgr.addresses()

    pool = ThreadPoolExecutor(max_workers=n)
    try:
        t = time.time()
        envs = list(pool.map(lambda i: diambra.arena.make("sfiii3n", settings_fn(), rank=i), range(n)))
        make_t = time.time() - t

        t = time.time()
        list(pool.map(lambda ei: ei[1].reset(seed=1000 + ei[0]), enumerate(envs)))
        reset_t = time.time() - t

        rng = np.random.default_rng(0)
        def one(e):
            o, r, term, trunc, info = e.step(e.action_space.sample())
            if term or trunc:
                e.reset(seed=int(rng.integers(0, 1_000_000)))
            return 1
        # timed loop
        nstep = 0
        t0 = time.time()
        while time.time() - t0 < secs:
            for _ in pool.map(one, envs):
                nstep += 1
        dt = time.time() - t0
        sps = nstep / dt
        for e in envs:
            try: e.close()
            except Exception: pass
        return dict(n=n, launch=launch_t, make=make_t, reset=reset_t,
                    sps=sps, per_env=sps / n)
    finally:
        pool.shutdown(wait=False)
        mgr.stop_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="1,2,4,8,12,16")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--base-port", type=int, default=57000)
    opt = ap.parse_args()
    ns = [int(x) for x in opt.ns.split(",")]

    rows = []
    for k, n in enumerate(ns):
        print(f"\n===== N={n} =====", flush=True)
        r = run_n(n, opt.base_port + k * 100, opt.secs)
        rows.append(r)
        print(f"  launch={r['launch']:.1f}s make={r['make']:.1f}s reset={r['reset']:.1f}s "
              f"SPS={r['sps']:,.0f} per_env={r['per_env']:,.0f}", flush=True)
        time.sleep(2)

    base = rows[0]["per_env"]
    print("\n================ SCALING SUMMARY ================")
    print(f"{'N':>3} {'launch':>7} {'make':>6} {'reset':>6} {'SPS':>9} {'per-env':>8} {'efficiency':>10}")
    for r in rows:
        eff = r["per_env"] / base
        print(f"{r['n']:>3} {r['launch']:>6.1f}s {r['make']:>5.1f}s {r['reset']:>5.1f}s "
              f"{r['sps']:>9,.0f} {r['per_env']:>8,.0f} {eff:>9.0%}")
    print("efficiency = per-env SPS relative to N=1 (100% = perfect linear scaling)")


if __name__ == "__main__":
    main()
