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
    callbacks = [wandb_callback, auto_save_callback]

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
            target_kl=target_kl,
            lr_floor=ppo_settings.get("adaptive_lr_floor", 1e-5),
            lr_cap_early=ppo_settings.get("adaptive_lr_cap_early", 1e-2),
            lr_cap_late=ppo_settings.get("adaptive_lr_cap_late", 8e-4),
            timestep_threshold=ppo_settings.get("adaptive_lr_timestep_threshold", 8_000_000),
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