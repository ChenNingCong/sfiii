"""SB3 VecEnv adapter for MPVecEnv.

Wraps the fault-tolerant, hot-spare, multiprocess MPVecEnv as a
stable_baselines3 VecEnv so it drops straight into PPO / DQN / QR-DQN. Learning
is unchanged ("result the same") — this only adapts the interface; episode logging
flows through info['episode_stats'] (see WandbStatsCallback).

    from sb3_mpvec import SB3MPVecEnv
    env = SB3MPVecEnv(num_envs=8, settings=SETTINGS, wrappers=WRAPPERS, spares_per_worker=1)
    model = PPO("MultiInputPolicy", env, ...)        # or DQN / QRDQN
"""
from typing import Optional

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mp_vecenv import MPVecEnv


class SB3MPVecEnv(VecEnv):
    def __init__(self, **kwargs):
        self._mp = MPVecEnv(**kwargs)
        self.launch_t = self._mp.launch_t
        self.make_t = self._mp.make_t
        super().__init__(self._mp.num_envs, self._mp.observation_space, self._mp.action_space)
        self.render_mode = None
        self._actions = None

    # ── core ──────────────────────────────────────────────────────────────────
    def reset(self):
        return self._mp.reset()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        self._mp.step_async(list(self._actions))
        return self._mp.step_wait()

    def close(self):
        self._mp.close()

    # ── SB3 VecEnv plumbing (minimal; training doesn't need remote attr access) ─
    def get_attr(self, attr_name: str, indices=None):
        idx = self._idx(indices)
        if attr_name in ("observation_space", "action_space", "render_mode"):
            val = getattr(self, attr_name, None)
            return [val for _ in idx]
        return [None for _ in idx]

    def set_attr(self, attr_name, value, indices=None):
        return None

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]

    def get_images(self):
        return [None for _ in range(self.num_envs)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
