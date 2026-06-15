"""SPS benchmark for MPVecEnv (shared-memory obs). Random actions, no learner."""
import argparse, time
import numpy as np
from mp_vecenv import MPVecEnv

SETTINGS = dict(game_id="sfiii3n", step_ratio=1, frame_shape=[128, 128, 0],
                action_space="multi_discrete", continue_game=0.0, difficulty=6)
WRAPPERS = dict(no_op_max=0, stack_frames=4, dilation=1, add_last_action=True,
                stack_actions=12, scale=True, role_relative=True, flatten=True, filter_keys=[])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--base-port", type=int, default=55000)
    ap.add_argument("--prefix", type=str, default="shmb")
    opt = ap.parse_args()

    venv = MPVecEnv(num_envs=opt.n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=opt.spares, base_port=opt.base_port, prefix=opt.prefix)
    print(f"launch_t {venv.launch_t:.1f}s make_t {venv.make_t:.1f}s", flush=True)
    venv.reset()
    acts = [venv.action_space.sample() for _ in range(opt.n)]
    # warmup
    for _ in range(20):
        venv.step([venv.action_space.sample() for _ in range(opt.n)])
    t0 = time.time(); steps = 0
    while time.time() - t0 < opt.secs:
        venv.step(acts)
        steps += 1
    dt = time.time() - t0
    sps = steps * opt.n / dt
    print(f"N={opt.n} spares={opt.spares}  {steps} batched steps in {dt:.1f}s  "
          f"=> {sps:.0f} env-steps/s  ({steps/dt:.1f} batched/s)", flush=True)
    venv.close()
    print("DONE", flush=True)
