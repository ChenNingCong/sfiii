import numpy as np
import gymnasium as gym


class RoundTerminatingWrapper(gym.Wrapper):
    """
    Wrapper that terminates the environment after each round instead of each episode.

    A game consists of multiple stages, and each stage consists of multiple rounds.
    This wrapper allows the agent to learn to complete each round independently
    by terminating when round_done=True.

    IMPORTANT:
    - This wrapper is applied INSIDE VecEnv (each subprocess has its own wrapped env)
    - When round_done, returns terminated=True and lets VecEnv call reset()
    - Accumulates reward during round, returns 0 at each step, returns total at round end
    """

    def __init__(self, env):
        super().__init__(env)
        self._current_round_reward = 0.0
        self._round_rewards = []  # Collect rewards for each round in a stage

    def step(self, action):
        # Step the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accumulate reward for current round (return 0 at each step)
        self._current_round_reward += reward

        # Terminate when round is done
        round_done = info.get("round_done", False)

        if round_done:
            # Collect the round reward
            self._round_rewards.append(self._current_round_reward)

            # Create info with round statistics
            info["round_reward"] = self._current_round_reward
            info["stage_round_rewards"] = self._round_rewards.copy()
            info["avg_round_reward"] = (
                float(np.mean(self._round_rewards)) if self._round_rewards else 0.0
            )

            # Get the reward to return (accumulated for this round)
            round_reward = self._current_round_reward
            self._current_round_reward = 0.0

            # If stage is also done, clear the round rewards list
            if info.get("stage_done", False):
                self._round_rewards = []

            # Return with terminated=True - VecEnv will call reset() on this env slot
            return obs, round_reward, True, False, info
        else:
            # Return 0 reward during round, only give reward at round end
            return obs, 0.0, False, False, info

    def reset(self, seed=None, options=None):
        # Reset the underlying environment (called by VecEnv)
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset accumulators
        self._current_round_reward = 0.0
        self._round_rewards = []

        return obs, info
