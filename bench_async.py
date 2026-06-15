"""Prove the double-buffer win: sync MPVecEnv vs AsyncMPVecEnv at the SAME effective
num_envs, with a simulated per-step learner latency L (the inference+update window
that starves the envs in the sync loop). Async overlaps L with the other group's
stepping -> higher SPS.

    python bench_async.py --num-envs 8 --latency 0.008 --steps 400
"""
import argparse, time
import numpy as np
from mp_vecenv import MPVecEnv, AsyncMPVecEnv

SETTINGS = dict(game_id="sfiii3n", step_ratio=1, frame_shape=[128, 128, 1],
                continue_game=0.0, action_space="multi_discrete", characters="Ryu",
                difficulty=6, outfits=2)
WRAPPERS = dict(normalize_reward=True, normalization_factor=0.2, stack_frames=4, dilation=6,
                add_last_action=True, stack_actions=6, scale=True, exclude_image_scaling=True,
                role_relative=True, flatten=True,
                filter_keys=["action", "own_health", "opp_health", "own_side", "opp_side",
                             "opp_character", "stage", "timer"])


def run(env, n, steps, latency, act):
    obs = env.reset()
    nstep = 0
    t0 = time.time()
    for _ in range(steps):
        time.sleep(latency)            # simulate learner compute (inference+buffer+update window)
        out = env.step(act(n))
        nstep += n
    dt = time.time() - t0
    env.close()
    return nstep / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--latency", type=float, default=0.008)   # 8ms simulated learner/step
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--base-port", type=int, default=60000)
    opt = ap.parse_args()
    N = opt.num_envs
    act = lambda n: np.stack([np.array([np.random.randint(9), np.random.randint(10)]) for _ in range(n)])

    print(f"=== SYNC  MPVecEnv(N={N}) [{N} engines] latency={opt.latency*1000:.0f}ms/step ===", flush=True)
    s = run(MPVecEnv(N, SETTINGS, wrappers=WRAPPERS, spares_per_worker=1,
                     base_port=opt.base_port, prefix="syncb"), N, opt.steps, opt.latency, act)
    time.sleep(2)
    print(f"=== ASYNC AsyncMPVecEnv(N={N}) [2x={2*N} engines, double-buffered] ===", flush=True)
    a = run(AsyncMPVecEnv(N, SETTINGS, wrappers=WRAPPERS, spares_per_worker=1,
                          base_port=opt.base_port + 200, prefix="asyncb"), N, opt.steps, opt.latency, act)

    print("\n================ DOUBLE-BUFFER RESULT ================")
    print(f"  sync : {s:6.0f} steps/s")
    print(f"  async: {a:6.0f} steps/s   ({a/s:.2f}x)")
    print(f"  (ceiling if learner fully hidden ~ {N/ (max(0.0001,(s and N/s - opt.latency))) :.0f}/s env-bound)")


if __name__ == "__main__":
    main()
