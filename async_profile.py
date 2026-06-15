"""Measure ASYNC batched-inference collection vs lockstep, with the trained policy.

Lockstep: send all N actions, wait for ALL -> one engine's ~2.4s round/stage stall
freezes the whole batch. Async: service whichever workers are ready (batched GPU
inference over the ready set), so a stalled engine never blocks the others. Policy is
frozen during collection -> zero off-policy error.

    python async_profile.py --ckpt ... --n 12 --mode async --steps 6000
    python async_profile.py --ckpt ... --n 12 --mode lockstep --steps 6000
"""
import argparse, os, time
import numpy as np
import torch

SETTINGS = dict(game_id="sfiii3n", step_ratio=1, frame_shape=[128, 128, 1],
                continue_game=0.0, action_space="multi_discrete",
                characters="Ryu", difficulty=6, outfits=2)
WRAPPERS = dict(normalize_reward=True, normalization_factor=0.2, stack_frames=4,
                dilation=6, add_last_action=True, stack_actions=6, scale=True,
                exclude_image_scaling=True, role_relative=True, flatten=True,
                filter_keys=["action", "own_health", "opp_health", "own_side",
                             "opp_side", "opp_character", "stage", "timer"])


def to_t(obs, device):
    from ppo_sfiii import obs_to_tensor
    return obs_to_tensor(obs, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--mode", choices=["async", "lockstep", "budget"], default="async")
    ap.add_argument("--steps", type=int, default=6000)   # per-env transitions to collect
    ap.add_argument("--budget", type=int, default=36000) # mode=budget: total transitions to collect from M engines
    ap.add_argument("--base-port", type=int, default=58000)
    ap.add_argument("--prefix", type=str, default="ap")
    opt = ap.parse_args()

    os.environ["MPVEC_PROFILE"] = "1"
    from mp_vecenv import MPVecEnv
    from ppo_sfiii import Agent, obs_to_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    venv = MPVecEnv(num_envs=opt.n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=opt.spares, base_port=opt.base_port, prefix=opt.prefix)
    nvec = list(np.asarray(venv.action_space.nvec).tolist())
    obs = venv.reset()
    frame, vec = obs_to_tensor(obs, device)
    agent = Agent(frame.shape[1], vec.shape[1], nvec).to(device)
    ck = torch.load(opt.ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"] if "agent" in ck else ck); agent.eval()
    N = opt.n
    print(f"loaded; mode={opt.mode} N={N} make={venv.make_t:.1f}s", flush=True)

    def act_batch(frames, vecs):
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(frames, vecs)
        return a.cpu().numpy()

    total_env_steps = N * opt.steps
    t0 = time.time()

    if opt.mode == "lockstep":
        for s in range(opt.steps):
            a = act_batch(frame, vec)
            obs, rew, done, infos = venv.step(a)
            frame, vec = obs_to_tensor(obs, device)
    else:
        # async: service ready workers in waves; ONE batched obs_to_tensor + ONE
        # inference per wave (a stalled engine just isn't in the wave).
        counts = [0] * N
        a0 = act_batch(frame, vec)
        for i in range(N):
            venv.send_action(i, a0[i])
        while min(counts) < opt.steps:
            ready = venv.wait_ready()
            idxs = []; obs_batch = None
            raw = []
            for i in ready:
                o_i, r, d, info = venv.recv_one(i)
                counts[i] += 1
                if counts[i] < opt.steps:
                    raw.append(o_i); idxs.append(i)
            if idxs:
                # stack ready envs' obs once, single obs_to_tensor, single inference
                batch = {k: np.stack([o[k] for o in raw]) for k in raw[0]}
                fb, vb = obs_to_tensor(batch, device)
                ab = act_batch(fb, vb)
                for j, i in enumerate(idxs):
                    venv.send_action(i, ab[j])
        try:
            for i in range(N):
                if venv._conns[i].poll(0): venv.recv_one(i)
        except Exception:
            pass

    if opt.mode == "budget":
        # M free-running engines; collect a FIXED total `budget` of transitions, taking
        # them as they arrive (over-provisioned: extras cover stalling/resetting engines).
        a0 = act_batch(frame, vec)
        for i in range(N):
            venv.send_action(i, a0[i])
        collected = 0
        total_env_steps = opt.budget
        while collected < opt.budget:
            ready = venv.wait_ready()
            raw = []; idxs = []
            for i in ready:
                o_i, r, d, info = venv.recv_one(i)
                collected += 1
                raw.append(o_i); idxs.append(i)
            if idxs and collected < opt.budget:
                batch = {k: np.stack([o[k] for o in raw]) for k in raw[0]}
                fb, vb = obs_to_tensor(batch, device)
                ab = act_batch(fb, vb)
                for j, i in enumerate(idxs):
                    venv.send_action(i, ab[j])
        try:
            for i in range(N):
                if venv._conns[i].poll(0): venv.recv_one(i)
        except Exception:
            pass

    dt = time.time() - t0
    venv.close()
    print(f"\n=== {opt.mode}  N={N} ===", flush=True)
    print(f"  {total_env_steps} env-steps in {dt:.1f}s  =>  {total_env_steps/dt:.0f} env-steps/s", flush=True)
    print("DONE", flush=True)
