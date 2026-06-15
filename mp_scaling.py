"""Multiprocess scaling check: each env stepped in its OWN process (no shared GIL).
Compares against the threaded vec env to decide the production architecture."""
import argparse, os, time
import multiprocessing as mp
from engine_manager import EngineManager


def worker(rank, addresses, secs, q):
    os.environ["DIAMBRA_ENVS"] = addresses
    import diambra.arena as A
    from diambra.arena import EnvironmentSettings, SpaceTypes
    s = EnvironmentSettings(); s.game_id="sfiii3n"; s.characters="Ryu"; s.difficulty=6
    s.action_space=SpaceTypes.DISCRETE; s.step_ratio=1; s.frame_shape=(128,128,1); s.grpc_timeout=60
    e = A.make("sfiii3n", s, rank=rank)
    e.reset(seed=rank)
    n = 0; t0 = time.time()
    while time.time() - t0 < secs:
        o, r, term, trunc, info = e.step(e.action_space.sample())
        n += 1
        if term or trunc: e.reset()
    q.put(n / (time.time() - t0))
    e.close()


def run_n(n, base_port, secs):
    mgr = EngineManager(num_envs=n, base_port=base_port, prefix=f"mp{n}")
    t=time.time(); mgr.start(); launch=time.time()-t
    addr = mgr.addresses()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, addr, secs, q)) for i in range(n)]
    [p.start() for p in ps]
    spss = [q.get() for _ in range(n)]
    [p.join() for p in ps]
    mgr.stop_all()
    agg = sum(spss)
    return dict(n=n, launch=launch, sps=agg, per_env=agg/n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="1,4,8,12")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--base-port", type=int, default=60000)
    opt = ap.parse_args()
    ns = [int(x) for x in opt.ns.split(",")]
    rows=[]
    for k,n in enumerate(ns):
        r = run_n(n, opt.base_port + k*100, opt.secs); rows.append(r)
        print(f"N={n}: launch={r['launch']:.1f}s SPS={r['sps']:,.0f} per_env={r['per_env']:,.0f}", flush=True)
        time.sleep(2)
    base = rows[0]["per_env"]
    print("\n==== MULTIPROCESS SCALING ====")
    print(f"{'N':>3} {'SPS':>9} {'per-env':>8} {'efficiency':>10}")
    for r in rows:
        print(f"{r['n']:>3} {r['sps']:>9,.0f} {r['per_env']:>8,.0f} {r['per_env']/base:>9.0%}")
