"""Render a COLORFUL video of the cleanrl (ppo_sfiii) policy COMPLETING THE GAME BY
RETRYING (continue_game on). Two seed-synced envs: gray (training obs -> policy),
color (native RGB -> frames). continue_game<0 => deterministic retries so both envs
stay in lockstep through continues.

    python eval_video_cleanrl.py --model results/<run>/model/ckpt_XXXX.pt --out clear.mp4 \
        --cfgFile config/config_async.yaml --continue-game -50 --episodes 1 --max-steps 40000
"""
import argparse, os
import numpy as np
import torch
import yaml

from engine_manager import EngineManager
from sfiii_env import build_env
from video_saver import StreamingVideoRender
from ppo_sfiii import Agent, obs_to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfgFile", default="config/config_async.yaml")
    ap.add_argument("--out", default="clear.mp4")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--continue-game", type=float, default=-50.0)  # retries until cleared
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--frame-skip", type=int, default=1)   # save 1 of every N frames (game plays slow)
    ap.add_argument("--base-port", type=int, default=58000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true")
    opt = ap.parse_args()

    import diambra.arena as A
    from diambra.arena import EnvironmentSettings, WrappersSettings, SpaceTypes, load_settings_flat_dict

    params = yaml.safe_load(open(opt.cfgFile))
    sdict = dict(params["settings"])
    sdict["continue_game"] = float(opt.continue_game)        # RETRY mode
    if isinstance(sdict.get("action_space"), str):
        sdict["action_space"] = (SpaceTypes.DISCRETE if sdict["action_space"] == "discrete"
                                 else SpaceTypes.MULTI_DISCRETE)
    if isinstance(sdict.get("frame_shape"), list):
        sdict["frame_shape"] = tuple(sdict["frame_shape"])

    mgr = EngineManager(num_envs=2, base_port=opt.base_port, prefix="vid")
    mgr.start()
    os.environ["DIAMBRA_ENVS"] = mgr.addresses()

    gray = build_env(load_settings_flat_dict(EnvironmentSettings, dict(sdict, grpc_timeout=60)),
                     load_settings_flat_dict(WrappersSettings, dict(params["wrappers_settings"])),
                     rank=0, game_id=sdict["game_id"], with_stats=True)
    cdict = dict(sdict); cdict.pop("frame_shape", None); cdict["grpc_timeout"] = 60
    color = A.make(sdict["game_id"], load_settings_flat_dict(EnvironmentSettings, cdict),
                   rank=1, render_mode="rgb_array")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nvec = list(np.asarray(gray.action_space.nvec).tolist())
    o0, _ = gray.reset(seed=1000)
    f0, v0 = obs_to_tensor({k: np.asarray(o0[k])[None] for k in o0}, device)
    agent = Agent(f0.shape[1], v0.shape[1], nvec).to(device)
    ck = torch.load(opt.model, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"] if "agent" in ck else ck); agent.eval()
    print(f"loaded {opt.model}; continue_game={opt.continue_game}", flush=True)

    vid = StreamingVideoRender(opt.out, fps=opt.fps); started = False
    for ep in range(opt.episodes):
        seed = 1000 + ep
        gobs, _ = gray.reset(seed=seed); color.reset(seed=seed)
        frame = color.render()
        if not started: vid.start(frame.shape); started = True
        vid.step(frame)
        done = False; steps = 0; info = {}
        while not done and steps < opt.max_steps:
            f, v = obs_to_tensor({k: np.asarray(gobs[k])[None] for k in gobs}, device)
            with torch.no_grad():
                if opt.deterministic:
                    feat = agent.features(f, v); logits = agent.actor(agent.pi(feat))
                    a = torch.cat([s.argmax(-1, keepdim=True) for s in torch.split(logits, nvec, -1)], -1)
                else:
                    a, _, _, _ = agent.get_action_and_value(f, v)
            action = a[0].cpu().numpy()
            gobs, r, term, trunc, info = gray.step(action)
            color.step(list(np.asarray(action).flatten()))   # step every frame (sync)
            if steps % opt.frame_skip == 0:                   # but only SAVE 1 of every N
                vid.step(color.render())
            steps += 1
            done = term or trunc
        st = info.get("episode_stats", {})
        print(f"episode {ep}: steps={steps} max_stage={st.get('max_stage')} "
              f"clean_max_stage={st.get('clean_max_stage')} game_cleared={st.get('game_cleared')}", flush=True)

    vid.stop(); gray.close(); color.close(); mgr.stop_all()
    print(f"saved {opt.out}", flush=True)


if __name__ == "__main__":
    main()
