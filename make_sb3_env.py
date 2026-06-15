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
            # NOTE: the serial_wrapper FileLock that used to wrap each env creation
            # is REMOVED. The parallel-spawn race it guarded was caused by all engines
            # sharing the host $HOME//tmp (apptainer auto-mounts them); the fix is to
            # spawn engines with --contain + per-engine private workdir/home (see
            # engine_manager.py). With contained engines, concurrent make() is safe.
            # (The fault-tolerant production path is MPVecEnv / train.py; this legacy
            #  diambra-SubprocVecEnv path is kept only for reference.)
            env = SubprocVecEnv([_make_sb3_env(i + start_index, seed) for i in range(num_envs)],
                                start_method=start_method)

    return env, num_envs