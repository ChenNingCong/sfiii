"""Render COLORFUL gameplay videos of a trained policy.

The long-standing headache: the policy is trained on 128x128x1 GRAYSCALE frames, so
the env it acts in produces grayscale — useless for a nice video. Solution: run TWO
seed-synchronized envs off the SAME action stream (the diambra emulator is
deterministic given seed+actions):
  * gray env  — exact training config (128x128x1 + wrappers)  -> feeds the policy
  * color env — native RGB frame (224x384x3), raw            -> frames for the video
Both reset with the same seed and stepped with the same action, so the color env
mirrors the game the policy actually plays. Frames -> mp4 via StreamingVideoRender.

    python eval_video.py --model results/<run>/model/<ckpt>.zip --out fight.mp4 --episodes 2
"""
import argparse
import os

import numpy as np
import yaml

from engine_manager import EngineManager
from sfiii_env import build_env
from video_saver import StreamingVideoRender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfgFile", default="config/config_replicate.yaml")
    ap.add_argument("--out", default="fight.mp4")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-port", type=int, default=58000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true")
    opt = ap.parse_args()

    from stable_baselines3 import PPO
    from diambra.arena import EnvironmentSettings, WrappersSettings, SpaceTypes, load_settings_flat_dict

    params = yaml.safe_load(open(opt.cfgFile))
    sdict = dict(params["settings"])
    if isinstance(sdict.get("action_space"), str):
        sdict["action_space"] = (SpaceTypes.DISCRETE if sdict["action_space"] == "discrete"
                                 else SpaceTypes.MULTI_DISCRETE)
    if isinstance(sdict.get("frame_shape"), list):
        sdict["frame_shape"] = tuple(sdict["frame_shape"])

    mgr = EngineManager(num_envs=2, base_port=opt.base_port, prefix="vid")
    mgr.start()
    os.environ["DIAMBRA_ENVS"] = mgr.addresses()

    # gray env = exact training obs (for the policy)
    def gray_settings():
        return load_settings_flat_dict(EnvironmentSettings, dict(sdict, grpc_timeout=60))
    def gray_wrappers():
        return load_settings_flat_dict(WrappersSettings, dict(params["wrappers_settings"]))
    gray = build_env(gray_settings(), gray_wrappers(), rank=0,
                     game_id=sdict["game_id"], with_stats=True)

    # color env = native RGB frame (raw, no frame downscaling) for the video
    import diambra.arena as A
    cdict = dict(sdict); cdict.pop("frame_shape", None); cdict["grpc_timeout"] = 60
    color_settings = load_settings_flat_dict(EnvironmentSettings, cdict)
    color = A.make(sdict["game_id"], color_settings, rank=1, render_mode="rgb_array")

    model = PPO.load(opt.model, device="cpu")
    vid = StreamingVideoRender(opt.out, fps=opt.fps)

    started = False
    for ep in range(opt.episodes):
        seed = 1000 + ep
        gobs, _ = gray.reset(seed=seed)
        cobs, _ = color.reset(seed=seed)
        frame = color.render()
        if not started:
            vid.start(frame.shape); started = True
        vid.step(frame)
        done = False; ret = 0.0; steps = 0
        while not done:
            action, _ = model.predict(gobs, deterministic=opt.deterministic)
            gobs, r, term, trunc, info = gray.step(action)
            cobs, _, ct, ctr, _ = color.step(list(np.asarray(action).flatten()))
            vid.step(color.render())
            ret += float(r); steps += 1
            done = term or trunc
        st = info.get("episode_stats", {})
        print(f"episode {ep}: steps={steps} return={ret:.2f} "
              f"max_stage={st.get('max_stage')} win={st.get('win')} "
              f"rounds={st.get('rounds_won')}/{st.get('rounds')}", flush=True)

    vid.stop()
    gray.close(); color.close(); mgr.stop_all()
    print(f"saved {opt.out}")


if __name__ == "__main__":
    main()
