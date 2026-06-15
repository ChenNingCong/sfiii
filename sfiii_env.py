"""Clean env construction for sfiii + a comprehensive episode-stats wrapper.

`build_env(settings, wrappers_settings, rank)` stacks:
    DiambraGym1P(settings)            # raw dict obs (ram states: stage/health/wins/...)
      -> StatsWrapper                 # reads raw obs, accumulates episode stats (innermost)
      -> diambra env_wrapping(...)    # role_relative / flatten / frame-stack / filter

StatsWrapper passes obs/reward through unchanged and only augments info, so it is
safe to place innermost while the configured diambra wrappers still produce the
training observation. On episode end it emits info['episode_stats'] (and flat
info['stats/<k>'] keys) for wandb.

SFIII (3rd Strike) terminology, per diambra game states:
  - round : best-of-3 bout within a stage      (info['round_done'])
  - stage : opponent # on the arcade ladder    (obs['stage'], info['stage_done'])
  - episode: full playthrough until game over   (info['episode_done'])
  ('turn' is a card-game term from lc; it has no sfiii analogue — round is the unit.)
"""
import os
import logging

import numpy as np
import gymnasium as gym

# arcade ladder length for 3rd Strike (used for the stage-progress ratio).
SFIII_TOTAL_STAGES = 10


def _scalar(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, (list, tuple, np.ndarray)):
        a = np.asarray(x).flatten()
        return float(a[0]) if a.size else default
    return float(x)


class StatsWrapper(gym.Wrapper):
    """Accumulate rich per-episode statistics from the RAW dict obs + info.
    Place INNERMOST (before role_relative/flatten). Cheap; computes everything."""

    def __init__(self, env, total_stages: int = SFIII_TOTAL_STAGES):
        super().__init__(env)
        self.total_stages = total_stages
        self._new_episode()

    def _new_episode(self):
        self.ep_steps = 0
        self.ep_return = 0.0
        self.max_stage = 1
        self.start_stage = 1
        self.n_rounds = 0
        self.rounds_won = 0
        self.rounds_lost = 0
        self.n_stages_cleared = 0
        self.dmg_dealt = 0.0
        self.dmg_taken = 0.0
        self._prev_h1 = None
        self._prev_h2 = None
        self._prev_stage = 1
        self._cur = {}
        # continue_game-INVARIANT skill metrics (single-life / per-attempt)
        self.stage_attempts = 0          # stages entered (won or lost)
        self.stage_wins = 0              # stages won
        self._lost_once = False
        self._first_loss_stage = None    # stage at first loss = clean-run max stage
        self._clean_stages_cleared = 0   # stages cleared before the first loss

    @staticmethod
    def _stage(obs):
        return int(_scalar(obs.get("stage", 1), 1))

    def _observe(self, obs, reset_step=False):
        st = self._stage(obs)
        self.max_stage = max(self.max_stage, st)
        p1 = obs.get("P1", {}) or {}
        p2 = obs.get("P2", {}) or {}
        h1 = _scalar(p1.get("health"))
        h2 = _scalar(p2.get("health"))
        # skip the round_done step: health resets to 160/160 there, so the delta is
        # a reset artifact, not real damage.
        if self._prev_h1 is not None and not reset_step:
            d_taken = self._prev_h1 - h1
            d_dealt = self._prev_h2 - h2
            if d_taken > 0:
                self.dmg_taken += d_taken
            if d_dealt > 0:
                self.dmg_dealt += d_dealt
        self._prev_h1, self._prev_h2 = h1, h2
        self._prev_stage = st            # pre-transition stage (read at next step's transition check)
        self._cur = dict(
            stage=st, timer=_scalar(obs.get("timer")),
            p1_health=h1, p2_health=h2,
            p1_wins=int(_scalar(p1.get("wins"))),
            p2_wins=int(_scalar(p2.get("wins"))),
            p1_side=int(_scalar(p1.get("side"))),
            opp_char=int(_scalar(p2.get("character", -1), -1)),
            p1_super=_scalar(p1.get("super_bar")),
            p1_stun=_scalar(p1.get("stun_bar")),
        )

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._new_episode()
        self._observe(obs)
        self.start_stage = self._stage(obs)
        self.max_stage = self.start_stage
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.ep_steps += 1
        self.ep_return += float(np.sum(rew))

        if info.get("round_done"):
            self.n_rounds += 1
            # CRITICAL: diambra resets health (->160/160) and the per-stage `wins`
            # counter AT the round_done step, so the current obs is useless for
            # attribution. self._prev_h1/_prev_h2 still hold the PREVIOUS step's
            # health here (because _observe() runs AFTER this block) — that's the
            # pre-KO state, where the loser is ~0. Use it to pick the winner.
            ph1, ph2 = self._prev_h1, self._prev_h2
            if ph1 is None or ph1 == ph2:
                self.rounds_lost += 1               # timeout/double-KO -> not a win
            elif ph1 > ph2:
                self.rounds_won += 1
            else:
                self.rounds_lost += 1
        # stage outcome (continue_game-invariant): stage_done = won this stage;
        # game_done WITHOUT stage_done = lost this stage (a continue/retry, or game over).
        # On full-ladder completion both fire -> stage_done wins (a win, not a loss).
        if info.get("stage_done"):
            self.n_stages_cleared += 1
            self.stage_wins += 1
            self.stage_attempts += 1
            if not self._lost_once:
                self._clean_stages_cleared += 1
        elif info.get("game_done"):
            self.stage_attempts += 1
            if not self._lost_once:           # first loss => clean-run (single-life) ends here
                self._lost_once = True
                self._first_loss_stage = self._prev_stage

        self._observe(obs, reset_step=bool(info.get("round_done")))

        if term or trunc or info.get("episode_done"):
            stats = self.summary()
            info["episode_stats"] = stats
            for k, v in stats.items():           # flat keys for easy wandb logging
                info[f"stats/{k}"] = v
        return obs, rew, term, trunc, info

    def summary(self) -> dict:
        c = self._cur
        # "win" = cleared the WHOLE arcade ladder (beat the final stage). `stage`
        # only advances on a stage win, so reaching the last stage == winning the
        # game. (The old p1_wins>p2_wins test read the reset wins counter -> bogus.)
        game_cleared = self.max_stage >= self.total_stages
        # clean-run (single-life) max stage = where the FIRST loss happened. This is
        # invariant to continue_game and directly comparable to a continue=0 baseline.
        clean_max_stage = self._first_loss_stage if self._lost_once else self.max_stage
        return {
            # --- continue_game-INVARIANT (monitor these under continue) ---
            "clean_max_stage": clean_max_stage,
            "clean_stages_cleared": self._clean_stages_cleared,
            "stage_win_rate": self.stage_wins / max(1, self.stage_attempts),
            "round_win_rate": self.rounds_won / max(1, self.n_rounds),
            "dmg_ratio": self.dmg_dealt / max(1.0, self.dmg_taken),
            # --- distorted by continue_game (don't compare across continue settings) ---
            "length": self.ep_steps,
            "return": self.ep_return,
            "max_stage": self.max_stage,
            "start_stage": self.start_stage,
            "stage_progress": self.max_stage / max(1, self.total_stages),
            "stages_cleared": self.n_stages_cleared,
            "stage_attempts": self.stage_attempts,
            "rounds": self.n_rounds,
            "rounds_won": self.rounds_won,
            "rounds_lost": self.rounds_lost,
            "dmg_dealt": self.dmg_dealt,
            "dmg_taken": self.dmg_taken,
            "final_own_health": self._prev_h1,
            "final_opp_health": self._prev_h2,
            "final_timer": c.get("timer", 0),
            "game_cleared": bool(game_cleared),   # true full-ladder win (~0 until very strong)
            "opponent_char": c.get("opp_char", -1),
        }


class EpisodeReturnWrapper(gym.Wrapper):
    """OUTERMOST wrapper: emit SB3's info['episode']={'r','l','t'} so PPO logs
    rollout/ep_rew_mean + rollout/ep_len_mean. Placed AFTER normalize_reward, so 'r'
    is the (normalized) return the policy is actually trained on — same scale as the
    historical best run's rollout/ep_rew_mean (~21)."""
    def __init__(self, env):
        super().__init__(env)
        self._ret = 0.0
        self._len = 0

    def reset(self, **kw):
        self._ret = 0.0
        self._len = 0
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._ret += float(np.sum(rew))
        self._len += 1
        if term or trunc:
            info = dict(info)
            info["episode"] = {"r": self._ret, "l": self._len, "t": 0.0}
        return obs, rew, term, trunc, info


def build_env(settings, wrappers_settings=None, rank: int = 0,
              game_id: str = "sfiii3n", render_mode=None, with_stats: bool = True,
              log_level=logging.WARNING):
    """make()-equivalent that inserts StatsWrapper innermost. Reads DIAMBRA_ENVS
    (rank -> address) exactly like diambra.arena.make."""
    from diambra.arena.arena_gym import DiambraGym1P
    from diambra.arena.wrappers.arena_wrappers import env_wrapping
    from diambra.arena import WrappersSettings

    settings.game_id = game_id
    settings.render_mode = render_mode
    addrs = os.getenv("DIAMBRA_ENVS", "").split() or ["localhost:50051"]
    if len(addrs) < rank + 1:
        raise Exception(f"rank {rank} >= #engines {len(addrs)} in DIAMBRA_ENVS")
    settings.env_address = addrs[rank]
    settings.rank = rank

    env = DiambraGym1P(settings)
    if with_stats:
        env = StatsWrapper(env)
    if wrappers_settings is None:
        wrappers_settings = WrappersSettings()
    wrappers_settings.sanity_check()
    env = env_wrapping(env, wrappers_settings)
    env = EpisodeReturnWrapper(env)   # outermost: emits info['episode'] for SB3 rollout/ep_rew_mean
    return env
