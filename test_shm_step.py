"""Minimal 1-worker shm test: reset then step, capture full worker traceback."""
import os, sys, time
import numpy as np
from mp_vecenv import MPVecEnv

SETTINGS = dict(
    game_id="sfiii3n",
    step_ratio=1,
    frame_shape=[128, 128, 0],
    action_space="multi_discrete",
    continue_game=0.0,
    difficulty=6,
)
WRAPPERS = dict(
    no_op_max=0,
    stack_frames=4,
    dilation=1,
    add_last_action=True,
    stack_actions=12,
    scale=True,
    role_relative=True,
    flatten=True,
    filter_keys=[],
)

if __name__ == "__main__":
    venv = MPVecEnv(num_envs=1, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=0, base_port=54000, prefix="shmt")
    print("launch_t", venv.launch_t, "make_t", venv.make_t, flush=True)
    obs = venv.reset()
    print("reset ok; keys", list(obs.keys()), flush=True)
    for k, v in obs.items():
        print("  ", k, v.shape, v.dtype, "nonzero", bool(np.any(v)), flush=True)
    print("worker alive?", [p.is_alive() for p in venv._procs],
          "exitcodes", [p.exitcode for p in venv._procs], flush=True)
    time.sleep(2)
    print("after 2s worker alive?", [p.is_alive() for p in venv._procs],
          "exitcodes", [p.exitcode for p in venv._procs], flush=True)
    acts = [venv.action_space.sample()]
    print("stepping with", acts, flush=True)
    obs, rew, done, infos = venv.step(acts)
    print("step ok; rew", rew, "done", done, flush=True)
    print("info0 keys", list(infos[0].keys()), flush=True)
    for i in range(20):
        obs, rew, done, infos = venv.step([venv.action_space.sample()])
    print("20 more steps ok", flush=True)
    venv.close()
    print("DONE", flush=True)
