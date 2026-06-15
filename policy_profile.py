"""Policy-driven collect profiler — the REALISTIC reproduction. Random actions run
matches to the 99s timer (rare resets); a trained policy ends matches by KO -> many
more resets. Loads a checkpoint, runs MPVecEnv(N) with GPU inference each step (no
PPO update), and measures per-batch collect time, reset rate + clustering, _wstep.

    python policy_profile.py --ckpt results/.../model/ckpt_8001024.pt --n 48 --spares 1 --steps 12000
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


def pct(a, p): return float(np.percentile(a, p)) if len(a) else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--base-port", type=int, default=58000)
    ap.add_argument("--prefix", type=str, default="pp")
    ap.add_argument("--numa-pin", type=int, default=0)
    opt = ap.parse_args()

    os.environ["MPVEC_PROFILE"] = "1"
    from mp_vecenv import MPVecEnv
    from ppo_sfiii import Agent, obs_to_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    venv = MPVecEnv(num_envs=opt.n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=opt.spares, base_port=opt.base_port,
                    prefix=opt.prefix, numa_pin=bool(opt.numa_pin))
    print(f"launch {venv.launch_t:.1f}s make {venv.make_t:.1f}s spares={opt.spares} pin={opt.numa_pin}", flush=True)
    nvec = list(np.asarray(venv.action_space.nvec).tolist())
    obs = venv.reset()
    frame, vec = obs_to_tensor(obs, device)
    agent = Agent(frame.shape[1], vec.shape[1], nvec).to(device)
    ck = torch.load(opt.ckpt, map_location=device)
    agent.load_state_dict(ck["agent"] if "agent" in ck else ck)
    agent.eval()
    print(f"loaded {opt.ckpt}", flush=True)

    batch_ms, infer_ms = [], []
    n_done_total = 0
    cluster_hist = {}                       # dones-in-batch -> count
    slow_with_reset = 0; slow_no_reset = 0
    timeline = []
    # correlate spikes (>100ms batch) with in-game events on the SLOWEST worker
    spike_round = spike_stage = spike_game = spike_none = 0
    nonspike_round = nonspike_stage = 0
    for s in range(opt.steps):
        t0 = time.time()
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(frame, vec)
        acts = a.cpu().numpy()
        t1 = time.time()
        obs, rew, done, infos = venv.step(acts)
        t2 = time.time()
        frame, vec = obs_to_tensor(obs, device)
        infer_ms.append((t1 - t0) * 1000)
        bms = (t2 - t1) * 1000
        batch_ms.append(bms)
        nd = int(done.sum()); n_done_total += nd
        cluster_hist[nd] = cluster_hist.get(nd, 0) + 1
        ws = [float(info.get("_wstep", 0)) for info in infos]
        slow_i = int(np.argmax(ws))
        if "hot_swap" in infos[slow_i]: slow_with_reset += 1
        else: slow_no_reset += 1
        si = infos[slow_i]
        if bms > 100:                       # this batch was a spike: what happened on the straggler?
            if si.get("game_done"): spike_game += 1
            elif si.get("stage_done"): spike_stage += 1
            elif si.get("round_done"): spike_round += 1
            else: spike_none += 1
        else:
            if si.get("round_done"): nonspike_round += 1
            if si.get("stage_done"): nonspike_stage += 1
        if s % 50 == 0: timeline.append(bms)

    venv.close()
    bm = np.array(batch_ms); im = np.array(infer_ms)
    print(f"\n=== STEP TIME (n={opt.steps}, N={opt.n}, spares={opt.spares}) ===", flush=True)
    print(f"  env.step:  mean={bm.mean():.1f}ms p50={pct(bm,50):.1f} p90={pct(bm,90):.1f} "
          f"p99={pct(bm,99):.1f} max={bm.max():.0f}", flush=True)
    print(f"  inference: mean={im.mean():.1f}ms p99={pct(im,99):.1f}", flush=True)
    print(f"  env.step >100ms: {(bm>100).mean()*100:.1f}%   >500ms: {(bm>500).mean()*100:.1f}%", flush=True)
    print(f"\n=== RESETS ===", flush=True)
    print(f"  total dones: {n_done_total}  (rate {n_done_total/opt.steps/opt.n*100:.2f}% of env-steps)", flush=True)
    print(f"  episode len ~{opt.steps*opt.n/max(1,n_done_total):.0f} steps/episode", flush=True)
    print(f"  reset clustering (dones-in-a-batch -> #batches): " +
          " ".join(f"{k}:{v}" for k, v in sorted(cluster_hist.items()) if k > 0), flush=True)
    print(f"  slowest-worker WAS a reset: {slow_with_reset}   was a plain step: {slow_no_reset}", flush=True)
    print(f"\n=== SPIKE (>100ms batch) ATTRIBUTION on straggler ===", flush=True)
    print(f"  game_done:{spike_game}  stage_done:{spike_stage}  round_done:{spike_round}  none/plain:{spike_none}", flush=True)
    print(f"  (non-spike batches with round_done:{nonspike_round} stage_done:{nonspike_stage})", flush=True)
    tot_reset_cost = bm[bm > 100].sum() / 1000
    print(f"  wall in >100ms batches: {tot_reset_cost:.1f}s of {bm.sum()/1000:.1f}s total "
          f"({tot_reset_cost/(bm.sum()/1000)*100:.0f}%)", flush=True)
    print(f"\n=== TIMELINE env.step ms (per 50 steps) ===", flush=True)
    print("  " + " ".join(f"{int(b)}" for b in timeline), flush=True)
    print("DONE", flush=True)
