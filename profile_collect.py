"""Profile WHERE collection time goes. Runs MPVecEnv(N) = one double-buffer half,
random actions, MPVEC_PROFILE on (per-worker step time in info['_wstep']).

For each batched step records: batch wall time, per-worker step times (=> straggler),
and whether the straggler step was a reset (hot_swap False / absent). Reports:
  * batch-latency distribution (p50/p90/p99/max) and a timeline (per 200-step bucket)
  * per-worker mean/p99 step time  -> persistent slow workers = placement/NUMA effect
  * straggler attribution: reset-driven vs slow-normal-step (straggler)

    python profile_collect.py --n 24 --steps 3000            # no pinning
    NUMA_PIN=1 python profile_collect.py --n 24 --steps 3000  # pin worker+engine per NUMA node
"""
import argparse, os, time
import numpy as np

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
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--base-port", type=int, default=57000)
    ap.add_argument("--prefix", type=str, default="prof")
    opt = ap.parse_args()

    os.environ["MPVEC_PROFILE"] = "1"
    from mp_vecenv import MPVecEnv

    venv = MPVecEnv(num_envs=opt.n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=opt.spares, base_port=opt.base_port, prefix=opt.prefix,
                    numa_pin=bool(int(os.environ.get("NUMA_PIN", "0"))))
    print(f"launch {venv.launch_t:.1f}s make {venv.make_t:.1f}s pin={os.environ.get('NUMA_PIN','0')}", flush=True)
    venv.reset()
    A = [venv.action_space.sample() for _ in range(opt.n)]

    batch_ms = []                       # wall time per batched step
    wstep_ms = [[] for _ in range(opt.n)]   # per-worker step time
    n_reset_batches = 0                 # batches containing >=1 reset
    reset_straggler = 0                 # batches whose straggler worker was resetting
    norm_straggler = 0                  # batches whose straggler was a plain slow step
    timeline = []                       # (step, batch_ms)

    for s in range(opt.steps):
        t0 = time.time()
        obs, rew, done, infos = venv.step(A)
        bms = (time.time() - t0) * 1000
        batch_ms.append(bms)
        ws = [float(info.get("_wstep", 0)) * 1000 for info in infos]
        for i, w in enumerate(ws):
            wstep_ms[i].append(w)
        slow_i = int(np.argmax(ws))
        is_reset = ("hot_swap" in infos[slow_i])   # hot_swap key present only on a done/reset step
        any_reset = any("hot_swap" in info for info in infos)
        if any_reset: n_reset_batches += 1
        if is_reset: reset_straggler += 1
        else: norm_straggler += 1
        if s % 50 == 0: timeline.append((s, bms))

    venv.close()

    bm = np.array(batch_ms)
    print(f"\n=== BATCH LATENCY (n={opt.steps}) ===", flush=True)
    print(f"  mean={bm.mean():.1f}ms  p50={pct(bm,50):.1f}  p90={pct(bm,90):.1f} "
          f"p99={pct(bm,99):.1f}  max={bm.max():.1f}ms", flush=True)
    print(f"  >50ms: {(bm>50).mean()*100:.1f}%   >100ms: {(bm>100).mean()*100:.1f}%   "
          f">500ms: {(bm>500).mean()*100:.1f}%", flush=True)

    allw = np.concatenate([np.array(w) for w in wstep_ms])
    print(f"\n=== PER-WORKER STEP TIME (all {len(allw)} samples) ===", flush=True)
    print(f"  mean={allw.mean():.1f}ms  p50={pct(allw,50):.1f}  p90={pct(allw,90):.1f} "
          f"p99={pct(allw,99):.1f}  max={allw.max():.1f}ms", flush=True)

    print(f"\n=== STRAGGLER ATTRIBUTION ===", flush=True)
    print(f"  batches with a reset: {n_reset_batches} ({n_reset_batches/opt.steps*100:.1f}%)", flush=True)
    print(f"  straggler WAS the reset: {reset_straggler}   straggler was a slow normal step: {norm_straggler}", flush=True)

    print(f"\n=== PER-WORKER mean / p99 (ms) -> persistent slow = placement/NUMA ===", flush=True)
    means = [(i, np.mean(w), pct(w, 99)) for i, w in enumerate(wstep_ms)]
    for i, m, p in sorted(means, key=lambda x: -x[1]):
        print(f"  w{i:02d}: mean={m:5.1f}  p99={p:6.1f}", flush=True)

    print(f"\n=== TIMELINE (batch ms per 50-step sample) ===", flush=True)
    print("  " + " ".join(f"{int(b)}" for _, b in timeline), flush=True)
    print("DONE", flush=True)
