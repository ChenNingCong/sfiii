"""Surface the REAL exception behind diambra's masked 'failed to connect' on
concurrent make, and compare thread vs process parallelism."""
import argparse, os, sys, time, traceback
from engine_manager import EngineManager

# unmask diambra's bare-except in env_init / connect
import diambra.arena.engine.interface as IF
_orig_init = IF.DiambraEngine.env_init
def _loud_init(self, pb):
    try:
        return self.client.EnvInit(pb)
    except Exception as e:
        code = getattr(e, "code", lambda: "?")()
        det = getattr(e, "details", lambda: "?")()
        print(f"  [REAL env_init pid={os.getpid()}] code={code} details={det!r}", flush=True)
        raise
IF.DiambraEngine.env_init = _loud_init

import diambra.arena
def settings():
    from diambra.arena import EnvironmentSettings, SpaceTypes
    s = EnvironmentSettings(); s.game_id="sfiii3n"; s.characters="Ryu"
    s.difficulty=6; s.action_space=SpaceTypes.DISCRETE; s.step_ratio=1
    return s

def mk(i):
    try:
        e = diambra.arena.make("sfiii3n", settings(), rank=i)
        return (i, "OK")
    except Exception as e:
        return (i, f"{type(e).__name__}: {str(e)[:60]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--mode", choices=["thread","process","stagger"], default="thread")
    ap.add_argument("--base-port", type=int, default=54000)
    ap.add_argument("--stagger", type=float, default=2.0)
    opt = ap.parse_args()
    mgr = EngineManager(num_envs=opt.num_envs, base_port=opt.base_port, prefix="diag")
    mgr.start(); os.environ["DIAMBRA_ENVS"]=mgr.addresses()
    print(f"engines ready: {mgr.health_report()}", flush=True)
    t0=time.time()
    if opt.mode=="thread":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=opt.num_envs) as p:
            res=list(p.map(mk, range(opt.num_envs)))
    elif opt.mode=="stagger":
        from concurrent.futures import ThreadPoolExecutor
        def mk_s(i): time.sleep(i*opt.stagger); return mk(i)
        with ThreadPoolExecutor(max_workers=opt.num_envs) as p:
            res=list(p.map(mk_s, range(opt.num_envs)))
    else:  # process
        import multiprocessing as MP
        ctx=MP.get_context("spawn")
        with ctx.Pool(opt.num_envs) as p:
            res=p.map(mk, range(opt.num_envs))
    print(f"\nmode={opt.mode} time={time.time()-t0:.1f}s")
    for i,st in res: print(f"  env {i}: {st}")
    print(f"OK={sum(1 for _,s in res if s=='OK')}/{opt.num_envs}")
    mgr.stop_all()
