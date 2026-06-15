"""Train PPO on sfiii using the fault-tolerant multiprocess MPVecEnv stack.

Replicates the best 2026 run (gamma 0.99, ent 1e-4, target_kl 0.01, adaptive KL-LR)
with the adaptive-LR blowup FIX, on engines that don't crash — so it can finally
train to completion. Logs the rich episode stats (max_stage, win_rate, dmg_ratio,
stage_progress, ...) to wandb.

    python train.py --cfgFile config/config_replicate.yaml
"""
import argparse
import collections
import datetime
import json
import os
import random

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule

from sb3_mpvec import SB3MPVecEnv
from adaptive_kl_lr import AdaptiveKLLRCallback


class WandbStatsCallback(BaseCallback):
    """Aggregate info['episode_stats'] (from StatsWrapper) and log to wandb:
      - rolling means of every stat (stats/<k>)
      - PROGRESS DISTRIBUTION: per-stage reach-rate (fraction of recent episodes that
        reached stage >= k) + best stage/round ever + histograms of max_stage/rounds.
        These make it obvious whether the agent is pushing into LATE stages of the
        arcade ladder (the real signal of getting better)."""
    def __init__(self, window=200, total_stages=10, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.total_stages = total_stages
        self.buf = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self.max_stage_hist = collections.deque(maxlen=window)
        self.rounds_hist = collections.deque(maxlen=window)
        self.n_ep = 0
        self.n_recover = 0
        self.best_stage = 0
        self.best_rounds = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            es = info.get("episode_stats")
            if es is not None:
                self.n_ep += 1
                for k, v in es.items():
                    if isinstance(v, (int, float, bool)):
                        self.buf[k].append(float(v))
                ms = float(es.get("max_stage", 1)); self.max_stage_hist.append(ms)
                self.best_stage = max(self.best_stage, ms)
                rr = float(es.get("rounds", 0)); self.rounds_hist.append(rr)
                self.best_rounds = max(self.best_rounds, rr)
            if info.get("corrupted"):
                self.n_recover += 1
        return True

    def _on_rollout_end(self) -> None:
        for k, dq in self.buf.items():
            if dq:
                self.logger.record(f"stats/{k}", float(np.mean(dq)))
        self.logger.record("stats/episodes_total", self.n_ep)
        self.logger.record("stats/fault_recoveries", self.n_recover)
        self.logger.record("progress/best_stage_ever", self.best_stage)
        self.logger.record("progress/best_rounds_ever", self.best_rounds)
        # per-stage reach rate: fraction of recent episodes reaching stage >= k
        if self.max_stage_hist:
            ms = np.array(self.max_stage_hist)
            for k in range(1, self.total_stages + 1):
                self.logger.record(f"stage_reach/ge_{k}", float((ms >= k).mean()))
        # histograms (distribution view in wandb)
        try:
            import wandb
            if wandb.run is not None and self.max_stage_hist:
                wandb.log({"dist/max_stage": wandb.Histogram(list(self.max_stage_hist)),
                           "dist/rounds_per_episode": wandb.Histogram(list(self.rounds_hist))},
                          step=self.num_timesteps)
        except Exception:
            pass


def main(cfg_file):
    with open(cfg_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    print("Config:", json.dumps(params, indent=2, default=str))

    num_envs = params["num_envs"]
    spares = params.get("spares_per_worker", 1)
    base_port = params.get("base_port", 51000)
    pp = params["ppo_settings"]

    seed = random.randint(0, 999999)
    name = datetime.datetime.now().strftime("%I:%M%p-%b-%d-%Y") + f"_{seed}_" + \
           os.path.basename(cfg_file) + "_" + params["folders"]["model_name"]

    import wandb
    wandb.init(project="ppo-new-200M", name=name, config=params,
               sync_tensorboard=True, save_code=True)
    tb = os.path.join(params["folders"]["parent_dir"], name, "tb")
    model_dir = os.path.join(params["folders"]["parent_dir"], name, "model")
    os.makedirs(model_dir, exist_ok=True)

    # optional per-env continue_game schedule (list len num_envs) -> shapes stage
    # coverage (envs with more continues persist at late stages instead of resetting).
    sched = params.get("continue_game_schedule")
    per_env = [{"continue_game": float(c)} for c in sched] if sched else None
    if per_env:
        print(f"[env] per-env continue_game schedule: {sched}")

    # ── fault-tolerant multiprocess env (the whole point) ─────────────────────
    env = SB3MPVecEnv(num_envs=num_envs, settings=params["settings"],
                      wrappers=params["wrappers_settings"], spares_per_worker=spares,
                      base_port=base_port, prefix=params.get("prefix", "train"),
                      game_id=params["settings"]["game_id"],
                      step_timeout=10.0, reset_timeout=60.0, per_env_settings=per_env)
    print(f"[env] {num_envs} workers + {spares}/worker spares ready: "
          f"launch={env.launch_t:.1f}s make={env.make_t:.1f}s")

    lr = linear_schedule(pp["learning_rate"][0], pp["learning_rate"][1])
    clip = linear_schedule(pp["clip_range"][0], pp["clip_range"][1])
    target_kl = pp.get("target_kl")
    finetune = bool(pp.get("finetune", False))    # phase-2: pure LR decay (e.g. ->0), no adaptive controller
    ckpt = pp.get("model_checkpoint", "0")

    if ckpt and ckpt != "0":
        print(f"RESUMING from checkpoint: {ckpt}")
        model = PPO.load(ckpt, env=env, gamma=pp["gamma"], ent_coef=pp["ent_coef"],
                         batch_size=pp["batch_size"], n_epochs=pp["n_epochs"], n_steps=pp["n_steps"],
                         learning_rate=lr, clip_range=clip, clip_range_vf=clip, target_kl=target_kl,
                         tensorboard_log=tb)
        model.lr_schedule = lr        # apply the (finetune) schedule, e.g. linear -> 0
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, gamma=pp["gamma"],
                    ent_coef=pp["ent_coef"], batch_size=pp["batch_size"], n_epochs=pp["n_epochs"],
                    n_steps=pp["n_steps"], learning_rate=lr, clip_range=clip, clip_range_vf=clip,
                    target_kl=target_kl, policy_kwargs=params["policy_kwargs"], tensorboard_log=tb)
    print(model.policy)

    # periodic checkpointing (env-steps): save_freq is per-env, so divide by num_envs
    from stable_baselines3.common.callbacks import CheckpointCallback
    save_every = int(pp.get("autosave_freq", 1_000_000))
    callbacks = [WandbStatsCallback(),
                 CheckpointCallback(save_freq=max(1, save_every // num_envs),
                                    save_path=model_dir, name_prefix="ckpt")]
    if target_kl is not None and not finetune:
        callbacks.append(AdaptiveKLLRCallback(
            target_kl=float(target_kl),
            lr_floor=float(pp.get("adaptive_lr_floor", 1e-5)),
            lr_cap_early=float(pp.get("adaptive_lr_cap_early", 1e-2)),
            lr_cap_late=float(pp.get("adaptive_lr_cap_late", 8e-4)),
            timestep_threshold=int(pp.get("adaptive_lr_timestep_threshold", 2_000_000)),
            early_stop_decay=float(pp.get("adaptive_lr_early_stop_decay", 1.2)),
            max_multiplier=float(pp.get("adaptive_lr_max_multiplier", 1.0)),
            verbose=1))
        print(f"AdaptiveKLLR enabled (target_kl={target_kl}, "
              f"max_multiplier={pp.get('adaptive_lr_max_multiplier', 1.0)})")
    elif finetune:
        print(f"FINETUNE mode: pure LR decay {pp['learning_rate'][0]}->{pp['learning_rate'][1]}, "
              f"adaptive controller OFF")

    try:
        model.learn(total_timesteps=pp["time_steps"], callback=CallbackList(callbacks),
                    progress_bar=False)
        model.save(os.path.join(model_dir, str(pp["time_steps"])))
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgFile", type=str, required=True)
    opt = ap.parse_args()
    main(opt.cfgFile)
