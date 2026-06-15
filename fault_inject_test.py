"""Fault-injection test for RobustDiambraEnv.

Spawns N engines, steps them, then injects a fault into engine #1:
  - HANG: SIGSTOP the diambraEngineServer (port stays open, no responses) -> the
          step watchdog must fire and the env must detach+reattach.
  - CRASH: kill the server process -> gRPC raises -> env recovers.
Verifies the faulted env recovers and the others keep stepping throughout.
"""
import argparse
import subprocess
import time

import numpy as np

from engine_manager import EngineManager
from robust_env import RobustDiambraEnv


def server_pids(port):
    out = subprocess.run(["pgrep", "-f", f"diambraEngineServer.*{port}"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def signal_engine(port, sig):
    for pid in server_pids(port):
        subprocess.run(["kill", sig, str(pid)])


def build_settings():
    from diambra.arena import EnvironmentSettings, SpaceTypes
    s = EnvironmentSettings()
    s.game_id = "sfiii3n"; s.characters = "Ryu"; s.difficulty = 6
    s.action_space = SpaceTypes.DISCRETE; s.step_ratio = 1
    s.grpc_timeout = 30   # fail fast on a bad connect instead of blocking 600s
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=52000)
    ap.add_argument("--fault", choices=["hang", "crash"], default="hang")
    ap.add_argument("--victim", type=int, default=1)
    opt = ap.parse_args()

    mgr = EngineManager(num_envs=opt.num_envs, base_port=opt.base_port)
    mgr.start()
    print(f"[setup] {opt.num_envs} engines up: {mgr.health_report()}", flush=True)

    envs = [RobustDiambraEnv(mgr, i, build_settings, step_timeout=8.0) for i in range(opt.num_envs)]
    for i, e in enumerate(envs):
        e.reset(seed=100 + i)
    print("[setup] all envs reset; stepping...", flush=True)

    victim, vport = opt.victim, mgr.port(opt.victim)
    rng = np.random.default_rng(0)

    def vec_step(n, tag):
        ok = np.zeros(opt.num_envs, dtype=int)
        for _ in range(n):
            for i, e in enumerate(envs):
                o, r, term, trunc, info = e.step(e.action_space.sample())
                ok[i] += 1
                if term or trunc:
                    e.reset(seed=int(rng.integers(0, 1e6)))
        print(f"[{tag}] steps_per_env={ok.tolist()} recoveries={[e.n_recoveries for e in envs]}", flush=True)

    vec_step(20, "warmup")

    print(f"\n>>> INJECT {opt.fault.upper()} on env {victim} (port {vport}, pids {server_pids(vport)})", flush=True)
    if opt.fault == "hang":
        signal_engine(vport, "-STOP")   # freeze: port open, no responses
    else:
        signal_engine(vport, "-KILL")   # crash

    t0 = time.time()
    vec_step(20, "during-fault")   # victim must time out, detach+reattach, continue
    print(f">>> recovery window took {time.time()-t0:.1f}s", flush=True)

    vec_step(20, "after-recovery")

    print("\n===== VERDICT =====", flush=True)
    rec = envs[victim].n_recoveries
    others = [envs[i].n_recoveries for i in range(opt.num_envs) if i != victim]
    print(f"victim env {victim}: recoveries={rec}  (expected >=1)")
    print(f"other envs recoveries={others}  (expected all 0)")
    healthy = all(mgr.is_healthy(i) for i in range(opt.num_envs))
    print(f"all engines healthy at end: {healthy}")
    ok = rec >= 1 and all(o == 0 for o in others) and healthy
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")

    for e in envs:
        e.close()
    mgr.stop_all()


if __name__ == "__main__":
    main()
