"""Max-throughput benchmark with the EXACT cl-run settings (config_cleanrl.yaml).
Measures sync MPVecEnv and double-buffered AsyncMPVecEnv, random actions, no learner.

    python bench_best.py --mode mpvec --n 16 --spares 1 --secs 15
    python bench_best.py --mode async --n 16 --spares 1 --secs 15
"""
import argparse, time
import numpy as np
from mp_vecenv import MPVecEnv, AsyncMPVecEnv

# mirror config/config_cleanrl.yaml exactly
SETTINGS = dict(game_id="sfiii3n", step_ratio=1, frame_shape=[128, 128, 1],
                continue_game=0.0, action_space="multi_discrete",
                characters="Ryu", difficulty=6, outfits=2)
WRAPPERS = dict(normalize_reward=True, normalization_factor=0.2, stack_frames=4,
                dilation=6, add_last_action=True, stack_actions=6, scale=True,
                exclude_image_scaling=True, role_relative=True, flatten=True,
                filter_keys=["action", "own_health", "opp_health", "own_side",
                             "opp_side", "opp_character", "stage", "timer"])


def bench_sync(n, spares, secs, base_port, prefix):
    venv = MPVecEnv(num_envs=n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=spares, base_port=base_port, prefix=prefix)
    print(f"  launch {venv.launch_t:.1f}s make {venv.make_t:.1f}s", flush=True)
    venv.reset()
    A = [venv.action_space.sample() for _ in range(n)]
    for _ in range(20): venv.step([venv.action_space.sample() for _ in range(n)])
    t0 = time.time(); steps = 0
    while time.time() - t0 < secs:
        venv.step(A); steps += 1
    dt = time.time() - t0
    venv.close()
    return steps * n / dt, steps / dt, n


def bench_async(n, spares, secs, base_port, prefix):
    venv = AsyncMPVecEnv(num_envs=n, settings=SETTINGS, wrappers=WRAPPERS,
                         spares_per_worker=spares, base_port=base_port, prefix=prefix)
    print(f"  launch {venv.launch_t:.1f}s make {venv.make_t:.1f}s half={venv.half}", flush=True)
    venv.reset()
    half = venv.half
    A = [venv.action_space.sample() for _ in range(half)]
    for _ in range(20): venv.step([venv.action_space.sample() for _ in range(half)])
    t0 = time.time(); calls = 0
    while time.time() - t0 < secs:
        venv.step(A); calls += 1       # each call advances ONE half by one step
    dt = time.time() - t0
    venv.close()
    return calls * half / dt, calls / dt, n   # env-steps/s, half-batches/s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mpvec", "async"], default="mpvec")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--base-port", type=int, default=55000)
    ap.add_argument("--prefix", type=str, default="bb")
    opt = ap.parse_args()
    fn = bench_sync if opt.mode == "mpvec" else bench_async
    sps, bps, n = fn(opt.n, opt.spares, opt.secs, opt.base_port, opt.prefix)
    print(f"RESULT mode={opt.mode} N={n} spares={opt.spares}  "
          f"{sps:.0f} env-steps/s  {bps:.1f} batches/s  {1000/bps:.1f} ms/batch", flush=True)
    print("DONE", flush=True)
