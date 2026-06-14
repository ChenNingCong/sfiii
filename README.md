def _make_sb3_env(rank, seed):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                     episode_recording_settings, render_mode, rank=rank)

            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            return env
        set_random_seed(env_settings.seed)
        return _init

This defines a function `_make_sb3_env` that creates an environment for Stable Baselines 3 (SB3) training. The function takes in a `rank` and a `seed` as arguments.
The problem is that the underlying diambra.arena.make function creates a environment that only teminates after the whole game (a game is a sequence of stages, which consists of multiple rounds). (it's similar to the episodic wrapper for atari game)
But I want to create a wrapper that terminates the environment after each round, so that the agent can learn to complete each round independently.
In addition, I want to create a wrapper tha collects the reward of each round (so I can calculate the average reward per round for each stage)
