import os
import yaml
import json
import argparse
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import EnvironmentSettings, WrappersSettings
from make_sb3_env import make_sb3_env
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
import wandb
from entropy_decay import EntropyDecayCallback
from adaptive_kl_lr import AdaptiveKLLRCallback
# diambra run -s 8 python stable_baselines3/training.py --cfgFile $PWD/stable_baselines3/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml
import datetime

# create an environment wrapper that injects the current stage and round to the info
# env is of type stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv
# So we create a wrapper around the DummyVecEnv that modifies the info returned by step() to include the current stage and round
from diambra.arena.wrappers.arena_wrappers import gym
class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(InfoWrapper, self).__init__(env)
        self.stage = 0
        self.round = 0

    def step(self, action):
        *args, info = self.env.step(action)
        if info["stage_done"]:
            self.stage += 1
        if info["round_done"]:
            self.round += 1
        info["round"] = self.round
        info["stage"] = self.stage
        return *args, info

    def reset(self, **kwargs):
        self.stage = 0
        self.round = 0
        observation = self.env.reset(**kwargs)
        return observation
# Limit Round wrapper - reset the environment after a certain number of rounds or stages
class LimitRoundWrapper(gym.Wrapper):
    def __init__(self, env, max_rounds=10, max_stages=5):
        super(LimitRoundWrapper, self).__init__(env)
        self.max_rounds = max_rounds
        self.max_stages = max_stages
    def step(self, action):
        *args, info = self.env.step(action)
        if info["round"] >= self.max_rounds or info["stage"] >= self.max_stages:
            info["env_done"] = True
            # reset the environment on the next step
            # truncated = False since we want to distinguish between truncation and termination, and here it's a termination
            obs, info = self.env.reset()
            info["round"] = 0
            info["stage"] = 0
            return obs, 0, True, False, info
        return *args, info
def insert_wrapper(env, WrapperType, *args, **kwargs):
    # if it's a vectorized environment, we need to wrap the inner env
    if hasattr(env, "envs"):
        for i in range(len(env.envs)):
            env.envs[i] = insert_wrapper(env.envs[i], WrapperType, *args, **kwargs)
        return env
    elif isinstance(env, WrapperType):
        # already wrapped
        return env
    elif isinstance(env, gym.Wrapper):
        env.env = insert_wrapper(env.env, WrapperType, *args, **kwargs)
        return env
    else:
        return WrapperType(env, *args, **kwargs)


import os
import time
import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings, RecordingSettings

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import time
import os
from filelock import FileLock, Timeout
from typing import Callable, Any
# --- Configuration ---
# All processes must agree on this path.
LOCK_FILE_PATH = "lock_barrier.lock"
TIMEOUT_SECONDS = 1000 # The maximum time a process will wait for the lock

def serial_wrapper(f : Callable[[], Any], lock_file_path: str, remote_lock: bool = False):
    if remote_lock:
        os.remove(lock_file_path) if os.path.exists(lock_file_path) else None
    def wrapper(*args, **kwargs):
        lock = FileLock(LOCK_FILE_PATH, timeout=TIMEOUT_SECONDS)
        try:
            print(f"Process {args, kwargs}: Attempting to acquire lock...")
            with lock:
                print(f"Process {args, kwargs}: ✅ Lock ACQUIRED. Executing serial task.")
                result = f(*args, **kwargs)
                print(f"Process {args, kwargs}: Serial task COMPLETE. Releasing lock.")
            print(f"Process {args, kwargs}: Lock RELEASED. Continuing execution.")
        except Timeout:
            # Handle the case where the lock couldn't be acquired within the timeout
            print(f"Process {args, kwargs}: ❌ Failed to acquire lock within {TIMEOUT_SECONDS} seconds.")
        print(f"Process {args, kwargs}: Finished execution.")
        return result
    return wrapper

# Make Stable Baselines3 Env function
def make_sb3_env(game_id: str, env_settings: EnvironmentSettings=EnvironmentSettings(),
                 wrappers_settings: WrappersSettings=WrappersSettings(),
                 episode_recording_settings: RecordingSettings=RecordingSettings(),
                 render_mode: str="rgb_array", seed: int=None, start_index: int=0,
                 allow_early_resets: bool=True, start_method: str=None, no_vec: bool=False,
                 use_subprocess: bool=True, log_dir_base: str="/tmp/DIAMBRALog/"):
    """
    Create a wrapped, monitored VecEnv.
    :param game_id: (str) the game environment ID
    :param env_settings: (EnvironmentSettings) parameters for DIAMBRA Arena environment
    :param wrappers_settings: (WrappersSettings) parameters for environment wrapping function
    :param episode_recording_settings: (RecordingSettings) parameters for environment recording wrapping function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses. See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv`
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """

    env_addresses = os.getenv("DIAMBRA_ENVS", "").split()
    if len(env_addresses) == 0:
        raise Exception("ERROR: Running script without DIAMBRA CLI.")

    num_envs = len(env_addresses)

    def _make_sb3_env(rank, seed):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                     episode_recording_settings, render_mode, rank=rank)
            env = insert_wrapper(env, LimitRoundWrapper, max_rounds=1, max_stages=1)
            env = insert_wrapper(env, InfoWrapper)
            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            return env
        set_random_seed(env_settings.seed)
        return _init

    # If not wanting vectorized envs
    if no_vec and num_envs == 1:
        env = _make_sb3_env(0, seed)()
    else:
        # When using one environment, no need to start subprocesses
        if num_envs == 1 or not use_subprocess:
            env = DummyVecEnv([_make_sb3_env(i + start_index, seed) for i in range(num_envs)])
        else:
            env = SubprocVecEnv([serial_wrapper(_make_sb3_env(i + start_index, seed), lock_file_path=LOCK_FILE_PATH, remote_lock=False) for i in range(num_envs)],
                                start_method=start_method)

    return env, num_envs

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class VarianceDiagnosticCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(VarianceDiagnosticCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Access the rollout buffer from the model
        # 'returns' are the discounted reward sums (the targets)
        # 'values' are the critic's predictions
        returns = self.model.rollout_buffer.returns.flatten()
        values = self.model.rollout_buffer.values.flatten()

        # Calculate raw statistics
        return_var = np.var(returns)
        return_mean = np.mean(returns)
        value_mean = np.mean(values)
        
        # Calculate the residual variance (what the critic fails to explain)
        residual_var = np.var(returns - values)

        # Log to the SB3 internal logger (Tensorboard/CSV/Stdout)
        self.logger.record("diagnostics/return_var", return_var)
        self.logger.record("diagnostics/return_mean", return_mean)
        self.logger.record("diagnostics/value_mean", value_mean)
        self.logger.record("diagnostics/residual_var", residual_var)

def main(cfg_file):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()
    port = os.environ["BASE_PORT"]
    if not port.isnumeric():
        raise Exception("ERROR: BASE_PORT environment variable must be set to a number.")
    base_port = int(port)
    os.environ["DIAMBRA_ENVS"] = " ".join([f"localhost:{base_port + i}" for i in range(params["num_envs"])])

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings,use_subprocess=True)
    
    print("Activated {} environment(s)".format(num_envs))

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]
    ent_coef = ppo_settings["ent_coef"]
    if "target_kl" in ppo_settings:
        target_kl = ppo_settings["target_kl"]
    else:
        target_kl = None

    import random
    # Returns an integer (seconds) - generally preferred for storage
    random_seed = random.randint(0, 999999)

    base_time_path = datetime.datetime.now().strftime("%I:%M%p-on-%B-%d-%Y") + "_" + str(random_seed) + "_" + os.path.split(cfg_file)[-1]

    # init wandb as soon as possible
    run = wandb.init(
        project="ppo-new-200M",
        name=base_time_path + "_" + params["folders"]["model_name"],
        config=params,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    tensor_board_folder = os.path.join(params["folders"]["parent_dir"], base_time_path, params["settings"]["game_id"],
                                    params["folders"]["model_name"], "tb")
    
    if model_checkpoint == "0":
        # use time to distinguish different runs
        model_folder = os.path.join(params["folders"]["parent_dir"], base_time_path, params["settings"]["game_id"], params["folders"]["model_name"], "model")
        os.makedirs(model_folder, exist_ok=False)
        # Initialize the agent
        agent = PPO("MultiInputPolicy", env, verbose=1, ent_coef=ent_coef,
                    gamma=gamma, batch_size=batch_size, target_kl=target_kl,
                    n_epochs=n_epochs, n_steps=n_steps,
                    learning_rate=learning_rate, clip_range=clip_range,
                    clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder)
    else:
        # Load the trained agent
        agent = PPO.load(model_checkpoint, env=env, ent_coef=ent_coef, target_kl=target_kl,
                         gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                         clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                         tensorboard_log=tensor_board_folder)


    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                  save_path=model_folder, filename_prefix=model_checkpoint + "_")

    # Train the agent
    time_steps = ppo_settings["time_steps"]

    wandb_callback = WandbCallback(
        gradient_save_freq=100_000,
        verbose=2,
    )
    callbacks = [wandb_callback, auto_save_callback, VarianceDiagnosticCallback(1)]

    # Set up entropy coefficient decay if needed
    def is_number(obj):
        return isinstance(obj, (int, float, complex))

    if not is_number(ent_coef):
        ent_callback = EntropyDecayCallback(total_timesteps=time_steps, initial_ent_coef=ent_coef[0], final_ent_coef=ent_coef[1])
        callbacks.append(ent_callback)
    else:
        print("Ent coeff is a number, no decay")

    if target_kl is not None:
        adaptive_kl_lr_callback = AdaptiveKLLRCallback(
            target_kl=float(target_kl),
            lr_floor=float(ppo_settings.get("adaptive_lr_floor", 1e-5)),
            lr_cap_early=float(ppo_settings.get("adaptive_lr_cap_early", 1e-2)),
            lr_cap_late=float(ppo_settings.get("adaptive_lr_cap_late", 8e-4)),
            timestep_threshold=int(ppo_settings.get("adaptive_lr_timestep_threshold", 2_000_000)),
            early_stop_decay=float(ppo_settings.get("adaptive_lr_early_stop_decay", 1.2)),
            verbose=1,
        )
        callbacks.append(adaptive_kl_lr_callback)
        print(f"AdaptiveKLLRCallback enabled with target_kl={target_kl}")

    callback_list = CallbackList(callbacks)
    agent.learn(total_timesteps=time_steps, callback=callback_list, progress_bar=True)

    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint) + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)