"""Validate MPVecEnv: linear scaling + stats + hot-spare swaps + clean interface."""
import argparse, time
import numpy as np
from mp_vecenv import MPVecEnv

SETTINGS = dict(game_id="sfiii3n", step_ratio=1, frame_shape=[128, 128, 1],
                continue_game=0.0, action_space="discrete", characters="Ryu",
                difficulty=6, outfits=2)
WRAPPERS = dict(normalize_reward=True, stack_frames=4, dilation=6, add_last_action=True,
                stack_actions=6, scale=True, exclude_image_scaling=True, role_relative=True,
                flatten=True, filter_keys=["action", "own_health", "opp_health", "own_side",
                "opp_side", "opp_character", "stage", "timer"])


def run(n, base_port, secs, spares):
    venv = MPVecEnv(num_envs=n, settings=SETTINGS, wrappers=WRAPPERS,
                    spares_per_worker=spares, base_port=base_port, prefix=f"mv{n}",
                    step_timeout=10.0)
    obs = venv.reset()
    okeys = list(obs.keys()) if isinstance(obs, dict) else f"Box{obs.shape}"
    rng = np.random.default_rng(0)
    nstep = 0; episodes = []; hot = 0; inline = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        actions = [int(rng.integers(0, venv.action_space.n)) for _ in range(n)]
        obs, rew, done, infos = venv.step(actions)
        nstep += n
        for info in infos:
            if "episode_stats" in info:
                episodes.append(info["episode_stats"])
            if info.get("hot_swap") is True: hot += 1
            elif info.get("hot_swap") is False: inline += 1
    dt = time.time() - t0
    sps = nstep / dt
    venv.close()
    return dict(n=n, launch=venv.launch_t, make=venv.make_t, sps=sps, per_env=sps / n,
                obs=okeys, episodes=episodes, hot=hot, inline=inline)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="4")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--spares", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=61000)
    opt = ap.parse_args()
    ns = [int(x) for x in opt.ns.split(",")]
    rows = []
    for k, n in enumerate(ns):
        print(f"\n===== MPVecEnv N={n} spares/worker={opt.spares} =====", flush=True)
        r = run(n, opt.base_port + k * 200, opt.secs, opt.spares)
        rows.append(r)
        print(f"  launch={r['launch']:.1f}s make={r['make']:.1f}s SPS={r['sps']:,.0f} "
              f"per_env={r['per_env']:,.0f} obs_keys={r['obs']}", flush=True)
        print(f"  hot_swaps={r['hot']} inline_resets={r['inline']}", flush=True)
        if r["episodes"]:
            e = r["episodes"][0]
            print(f"  episodes_finished={len(r['episodes'])}  e.g. max_stage={e['max_stage']} "
                  f"stage_progress={e['stage_progress']:.2f} rounds={e['rounds']} "
                  f"won/lost={e['rounds_won']}/{e['rounds_lost']} dmg_ratio={e['dmg_ratio']:.2f} "
                  f"win={e['win']} len={e['length']}", flush=True)
        time.sleep(2)
    if len(rows) > 1:
        base = rows[0]["per_env"]
        print("\n==== MPVecEnv SCALING ====")
        print(f"{'N':>3} {'launch':>7} {'make':>6} {'SPS':>9} {'per-env':>8} {'eff':>6}")
        for r in rows:
            print(f"{r['n']:>3} {r['launch']:>6.1f}s {r['make']:>5.1f}s {r['sps']:>9,.0f} "
                  f"{r['per_env']:>8,.0f} {r['per_env']/base:>5.0%}")
