"""Profile where the PPO training loop spends time (is obs-pickling/IPC the cost?).
Breaks a vec-step into worker env.step vs IPC/pickle, plus inference and the
gradient-update phase."""
import os, time
import numpy as np

os.environ["MPVEC_PROFILE"] = "1"

if __name__ == "__main__":
    import yaml
    from sb3_mpvec import SB3MPVecEnv

    P = yaml.safe_load(open("config/config_replicate.yaml"))
    N = 8
    env = SB3MPVecEnv(num_envs=N, settings=P["settings"], wrappers=P["wrappers_settings"],
                      spares_per_worker=0, base_port=64000, prefix="prof",
                      game_id=P["settings"]["game_id"], step_timeout=20.0)
    print(f"[prof] env ready (launch {env.launch_t:.1f}s make {env.make_t:.1f}s)", flush=True)

    from stable_baselines3 import PPO
    from diambra.arena.stable_baselines3.sb3_utils import linear_schedule
    pp = P["ppo_settings"]
    model = PPO("MultiInputPolicy", env, gamma=pp["gamma"], n_steps=pp["n_steps"],
                batch_size=pp["batch_size"], n_epochs=pp["n_epochs"],
                learning_rate=linear_schedule(*pp["learning_rate"]), verbose=0)

    obs = env.reset()
    WARM, M = 20, 150

    # (A) pure vec-step: env.step + IPC, NO inference
    for _ in range(WARM):
        env.step(np.array([env.action_space.sample() for _ in range(N)]))
    tot, wmax = [], []
    for _ in range(M):
        a = np.array([env.action_space.sample() for _ in range(N)])
        t0 = time.time(); _o, _r, _d, infos = env.step(a); tot.append(time.time() - t0)
        wmax.append(max(i.get("_wstep", 0) for i in infos))
    tot = np.array(tot); wmax = np.array(wmax); ipc = tot - wmax
    print(f"\n[A] worker env.step (||max): {wmax.mean()*1000:6.1f} ms", flush=True)
    print(f"[A] IPC/pickle/barrier     : {ipc.mean()*1000:6.1f} ms  ({ipc.mean()/tot.mean()*100:.0f}% of vec-step)", flush=True)
    print(f"[A] total vec-step (no inf): {tot.mean()*1000:6.1f} ms -> {N/tot.mean():.0f} steps/s", flush=True)

    # (B) inference only
    for _ in range(WARM): model.predict(obs)
    t0 = time.time()
    for _ in range(M): model.predict(obs)
    t_inf = (time.time() - t0) / M
    print(f"[B] policy inference       : {t_inf*1000:6.1f} ms/step", flush=True)

    # (C) collect + update (needs _setup_learn to init logger/buffer; use the
    # SB3-wrapped env = VecTransposeImage(SB3MPVecEnv), as model.learn does)
    model._setup_learn(100000)
    cb = model._init_callback(None)
    wenv = model.get_env()
    t0 = time.time()
    model.collect_rollouts(wenv, cb, model.rollout_buffer, n_rollout_steps=pp["n_steps"])
    t_collect = time.time() - t0
    t0 = time.time(); model.train(); t_train = time.time() - t0
    it = t_collect + t_train
    print(f"\n[C] collect {pp['n_steps']} steps   : {t_collect:5.2f}s ({N*pp['n_steps']/t_collect:.0f} steps/s)", flush=True)
    print(f"[C] gradient update (train): {t_train:5.2f}s (env idle)", flush=True)
    print(f"[C] full iteration         : {it:5.2f}s -> {N*pp['n_steps']/it:.0f} steps/s effective", flush=True)
    print(f"[C] update share of iter   : {t_train/it*100:.0f}%", flush=True)
    env.close()
