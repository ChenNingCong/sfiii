"""CleanRL-style PPO for diambra sfiii — leaner than SB3 (no per-step framework
overhead), reusing our fault-tolerant multiprocess MPVecEnv.

Adapted from bejeweled/ppo_bejeweled.py (itself from CleanRL ppo.py). Differences,
kept faithful to the SB3 replication so results are comparable:
  * env: MPVecEnv (containerized diambra engines) — obs is a numpy DICT
    {frame:(N,128,128,4) uint8, + scalar ram keys}, MULTI_DISCRETE actions, NO mask
    (all actions legal). Obs round-trips numpy->GPU each step (the env is the slow
    emulator, not GPU-resident — so this won't hit bejeweled SPS; it removes SB3's
    framework overhead and lets us overlap/clean up the loop).
  * network: SAME as SB3 MultiInputPolicy — NatureCNN(frame, 256) + flatten(vector)
    -> concat -> separate pi/vf MLPs [64,64]; MultiDiscrete = two Categorical heads.
  * LR: linear-decay cap + KL-adaptive controller clamped to multiplier<=1.0 (the
    blowup fix), matching config_replicate.yaml.
  * metrics: diambra episode stats (ep_rew, max_stage, stage_progress, stage_reach,
    round_win_rate, win, dmg_ratio) + losses/approx_kl/SPS.
All hyperparameters are loaded from config_replicate.yaml (req: every param correct).
"""
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Same conv stack SB3 uses for image obs; features_dim=256 (CombinedExtractor default)."""
    def __init__(self, c, out=256):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, 128, 128)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n, out)), nn.ReLU())
        self.out = out

    def forward(self, x):
        return self.fc(self.cnn(x))


class Agent(nn.Module):
    """SB3-MultiInputPolicy-equivalent: shared CombinedExtractor (CNN+vector),
    separate pi/vf MLPs [64,64]. Actor = MultiDiscrete (independent Categoricals)."""
    def __init__(self, frame_c, vec_dim, nvec, net=(64, 64)):
        super().__init__()
        self.cnn = NatureCNN(frame_c, 256)
        feat = 256 + vec_dim
        self.nvec = list(nvec)
        def mlp():
            return nn.Sequential(layer_init(nn.Linear(feat, net[0])), nn.Tanh(),
                                 layer_init(nn.Linear(net[0], net[1])), nn.Tanh())
        self.pi = mlp(); self.vf = mlp()
        self.actor = layer_init(nn.Linear(net[1], int(sum(self.nvec))), std=0.01)
        self.critic = layer_init(nn.Linear(net[1], 1), std=1.0)

    def features(self, frame, vec):
        return torch.cat([self.cnn(frame), vec], dim=1)

    def get_value(self, frame, vec):
        return self.critic(self.vf(self.features(frame, vec)))

    def get_action_and_value(self, frame, vec, action=None):
        f = self.features(frame, vec)
        logits = self.actor(self.pi(f))
        splits = torch.split(logits, self.nvec, dim=1)
        dists = [torch.distributions.Categorical(logits=l) for l in splits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=1)
        logprob = torch.stack([d.log_prob(action[:, i]) for i, d in enumerate(dists)], dim=1).sum(1)
        entropy = torch.stack([d.entropy() for d in dists], dim=1).sum(1)
        return action, logprob, entropy, self.critic(self.vf(f))


def obs_to_tensor(obs, device, frame_key="frame"):
    """dict numpy obs -> (frame (N,C,128,128) float[0,1], vector (N,D) float)."""
    frame = torch.as_tensor(obs[frame_key], device=device).float()
    if frame.dim() == 4 and frame.shape[-1] in (1, 2, 3, 4, 5, 6, 8):  # NHWC -> NCHW
        frame = frame.permute(0, 3, 1, 2).contiguous()
    if frame.max() > 1.5:
        frame = frame / 255.0
    vkeys = sorted(k for k in obs if k != frame_key)
    if vkeys:
        vec = torch.cat([torch.as_tensor(np.asarray(obs[k]), device=device).float().reshape(obs[k].shape[0], -1)
                         for k in vkeys], dim=1)
    else:
        vec = torch.zeros((frame.shape[0], 0), device=device)
    return frame, vec


def main(cfg_file="config/config_replicate.yaml"):
    import yaml
    P = yaml.safe_load(open(cfg_file))
    pp = P["ppo_settings"]
    num_envs = P["num_envs"]
    spares = P.get("spares_per_worker", 1)
    num_steps = pp["n_steps"]
    batch_size = num_envs * num_steps
    # SB3 used batch_size=256 minibatches => num_minibatches = batch_size/256
    minibatch_size = pp["batch_size"]
    num_minibatches = max(1, batch_size // minibatch_size)
    total_timesteps = pp["time_steps"]
    update_epochs = pp["n_epochs"]
    gamma = pp["gamma"]; gae_lambda = 0.95
    ent_coef = pp["ent_coef"]; vf_coef = 0.5; max_grad_norm = 0.5
    clip_coef0, clip_coef1 = pp["clip_range"]      # linear-annealed like SB3
    lr0, lr1 = pp["learning_rate"]
    lr_warmup = int(pp.get("lr_warmup_steps", 0))   # ramp LR 0->base over N steps (gentle bootstrap)
    target_kl = pp.get("target_kl")
    seed = random.randint(0, 999999)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    from mp_vecenv import MPVecEnv, AsyncMPVecEnv
    double_buffer = bool(pp.get("double_buffer", False))
    # ASYNC collection: M=num_envs free-running engines stepped at their own pace; a
    # rollout = a fixed budget of num_steps*num_envs transitions, taken as engines
    # become ready -> a round/stage-transition stall on one engine never blocks the
    # batch (the dominant slowdown). Behavior logprob is stored per transition, so the
    # PPO clip = truncated-IS correction (boundary transitions are at most K=1 stale).
    async_collect = bool(pp.get("async_collect", False))
    sched = P.get("continue_game_schedule")
    per_env = [{"continue_game": float(c)} for c in sched] if sched else None
    if per_env: print(f"[env] per-env continue_game schedule: {sched}", flush=True)
    EnvCls = AsyncMPVecEnv if double_buffer else MPVecEnv
    ekw = {} if double_buffer else {"per_env_settings": per_env}   # async splits halves; per-env TBD
    envs = EnvCls(num_envs=num_envs, settings=P["settings"], wrappers=P["wrappers_settings"],
                  spares_per_worker=spares, base_port=P.get("base_port", 51000),
                  prefix=P.get("prefix", "cl"), game_id=P["settings"]["game_id"],
                  step_timeout=10.0, **ekw)
    if double_buffer: print(f"[env] DOUBLE-BUFFERED (2 halves of {num_envs//2})", flush=True)
    if async_collect: print(f"[env] ASYNC collection (M={num_envs} free-running, budget={num_steps*num_envs}/rollout)", flush=True)
    print(f"[env] ready launch={envs.launch_t:.1f}s make={envs.make_t:.1f}s", flush=True)
    nvec = list(np.asarray(envs.action_space.nvec).tolist())

    obs0 = envs.reset()
    frame0, vec0 = obs_to_tensor(obs0, device)
    agent = Agent(frame0.shape[1], vec0.shape[1], nvec).to(device)
    print(agent, flush=True)
    print(f"params={sum(p.numel() for p in agent.parameters()):,} nvec={nvec} vec_dim={vec0.shape[1]}", flush=True)
    optimizer = optim.Adam(agent.parameters(), lr=lr0, eps=1e-5)

    # bootstrap-from-checkpoint (finetune): load trained weights, then train a FRESH
    # budget with this config's (smaller) LR schedule. Loads agent weights only;
    # optimizer + global_step start fresh so the LR decay spans this run's time_steps.
    ckpt_path = str(pp.get("model_checkpoint", "0"))
    if ckpt_path and ckpt_path != "0":
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"] if "agent" in ck else ck)
        print(f"[bootstrap] loaded agent weights from {ckpt_path} "
              f"(trained to step {ck.get('global_step', '?')}); fresh optimizer, LR {lr0}->{lr1}", flush=True)

    import wandb
    run_name = time.strftime("%I:%M%p-%b-%d-%Y") + f"_cleanrl_{seed}"
    wandb.init(project="ppo-new-200M", name=run_name, config=P, save_code=True)
    model_dir = os.path.join(P["folders"]["parent_dir"], run_name, "model"); os.makedirs(model_dir, exist_ok=True)
    save_every = int(pp.get("autosave_freq", 1_000_000)); next_save = save_every
    def save_ckpt(step):
        torch.save({"agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                    "global_step": step, "nvec": nvec, "config": P},
                   os.path.join(model_dir, f"ckpt_{step}.pt"))

    # storage. async needs slack rows: a fast engine can record >num_steps transitions
    # in a rollout while a stalled one records fewer (budget is the total, not per-env).
    cap = int(num_steps * 2) if async_collect else num_steps
    f_buf = torch.zeros((cap, num_envs) + frame0.shape[1:], device=device)
    v_buf = torch.zeros((cap, num_envs, vec0.shape[1]), device=device)
    actions = torch.zeros((cap, num_envs, len(nvec)), dtype=torch.long, device=device)
    logprobs = torch.zeros((cap, num_envs), device=device)
    rewards = torch.zeros((cap, num_envs), device=device)
    dones = torch.zeros((cap, num_envs), device=device)
    values = torch.zeros((cap, num_envs), device=device)

    global_step = 0
    start = time.time()
    next_frame, next_vec = frame0, vec0
    next_done = torch.zeros(num_envs, device=device)
    num_iterations = total_timesteps // batch_size
    kl_mult = 1.0
    ep_stats = defaultdict(lambda: deque(maxlen=200))
    ep_rew_d = deque(maxlen=200); ep_len_d = deque(maxlen=200)   # normalized return (SB3-comparable)
    best_stage = 0; total_eps = 0

    def _collect_stats(infos):
        nonlocal best_stage, total_eps
        for info in infos:
            es = info.get("episode_stats")
            if es:
                for k, vv in es.items():
                    if isinstance(vv, (int, float, bool)): ep_stats[k].append(float(vv))
                best_stage = max(best_stage, es.get("max_stage", 0))
            eo = info.get("episode")
            if eo:
                ep_rew_d.append(eo["r"]); ep_len_d.append(eo["l"]); total_eps += 1

    if double_buffer:                       # persistent per-half collection state
        _half = num_envs // 2
        _cols = [slice(0, _half), slice(_half, num_envs)]
        held = [obs_to_tensor(obs0, device), obs_to_tensor(envs._reset_obs[1], device)]
        held_done = [torch.zeros(_half, device=device), torch.zeros(_half, device=device)]
        pend = [None, None]

    if async_collect:
        # persistent per-engine state. cur[i] = engine i's latest obs (1,*) tensors;
        # a_pend[i] = (frame,vec,action,logprob,value,done_of_obs) for its in-flight
        # action; the in-flight action straddles the rollout boundary (its result is
        # recorded next rollout -> at most K=1 stale, corrected by the stored logprob).
        cur_f = [frame0[i:i+1] for i in range(num_envs)]
        cur_v = [vec0[i:i+1] for i in range(num_envs)]
        cur_d = [0.0] * num_envs
        a_pend = [None] * num_envs
        budget = num_steps * num_envs
        def _send_actions(idxs):             # batched inference over the given engines
            if not idxs: return
            fb = torch.cat([cur_f[i] for i in idxs], 0)
            vb = torch.cat([cur_v[i] for i in idxs], 0)
            with torch.no_grad():
                a, lp, _, val = agent.get_action_and_value(fb, vb)
            ac = a.cpu().numpy()
            for j, i in enumerate(idxs):
                a_pend[i] = (cur_f[i], cur_v[i], a[j], lp[j], val[j].flatten(),
                             torch.as_tensor(cur_d[i], device=device).float())
                envs.send_action(i, ac[j])
        _send_actions(list(range(num_envs)))   # kick off all engines

    for it in range(1, num_iterations + 1):
        _t_iter = time.time()
        frac = 1.0 - (it - 1.0) / num_iterations
        base_lr = lr1 + (lr0 - lr1) * frac            # linear-decay cap
        clip_coef = clip_coef1 + (clip_coef0 - clip_coef1) * frac
        # LR warmup: ramp 0->base over the first lr_warmup_steps. Essential when
        # bootstrapping a CONVERGED policy with a fresh optimizer — without it the
        # first (undamped, fresh-Adam) updates kick the policy hard and destroy it.
        warm = 1.0 if lr_warmup <= 0 else min(1.0, global_step / lr_warmup)
        optimizer.param_groups[0]["lr"] = base_lr * kl_mult * warm
        _t_coll = time.time()

        if async_collect:
            # ── async collection: M engines free-run; collect a fixed budget of
            # num_steps*num_envs transitions as engines become ready. A stalled
            # engine (round/stage transition, reset) is simply not in `ready` and
            # never blocks the others. counts[i] = transitions recorded for engine i.
            counts = [0] * num_envs
            collected = 0
            while collected < budget:
                ready = envs.wait_ready()
                raw = []; ridx = []
                for i in ready:
                    o_i, r, d, info = envs.recv_one(i)
                    pf, pv, pa, plp, pval, pd = a_pend[i]; a_pend[i] = None
                    t = counts[i]
                    if t < cap:
                        f_buf[t, i] = pf[0]; v_buf[t, i] = pv[0]; actions[t, i] = pa
                        logprobs[t, i] = plp; values[t, i] = pval; dones[t, i] = pd
                        rewards[t, i] = float(r)
                        counts[i] += 1; collected += 1; global_step += 1
                        _collect_stats([info])
                    cur_d[i] = float(d); raw.append(o_i); ridx.append(i)
                if ridx:                       # ONE batched obs_to_tensor per wave
                    batch = {k: np.stack([o[k] for o in raw]) for k in raw[0]}
                    fb, vb = obs_to_tensor(batch, device)
                    for j, i in enumerate(ridx):
                        cur_f[i] = fb[j:j + 1]; cur_v[i] = vb[j:j + 1]
                    # ALWAYS re-send to every recv'd engine so each keeps exactly one
                    # action in flight (incl. past the budget -> those become next
                    # rollout's first transitions). Otherwise an engine recv'd in the
                    # final wave loses its in-flight status and is never serviced again
                    # -> wait_ready() eventually blocks forever (deadlock).
                    _send_actions(ridx)
            # per-engine bootstrap value+done from each engine's latest obs
            with torch.no_grad():
                bf = torch.cat(cur_f, 0); bv = torch.cat(cur_v, 0)
                boot_v = agent.get_value(bf, bv).flatten()
            next_value = boot_v.unsqueeze(0)                       # [1, num_envs]
            next_done = torch.as_tensor(cur_d, device=device).float()
        elif not double_buffer:
            # ── synchronous collection ──
            for step in range(num_steps):
                global_step += num_envs
                f_buf[step] = next_frame; v_buf[step] = next_vec; dones[step] = next_done
                with torch.no_grad():
                    a, lp, _, val = agent.get_action_and_value(next_frame, next_vec)
                    values[step] = val.flatten()
                actions[step] = a; logprobs[step] = lp
                obs, rew, done, infos = envs.step(a.cpu().numpy())
                next_frame, next_vec = obs_to_tensor(obs, device)
                rewards[step] = torch.as_tensor(rew, device=device)
                next_done = torch.as_tensor(done, device=device).float()
                _collect_stats(infos)
            with torch.no_grad():
                next_value = agent.get_value(next_frame, next_vec).reshape(1, -1)
        else:
            # ── double-buffered collection: two halves one phase out of sync ──
            # Each recorded transition fills its half's columns; bootstrap is captured
            # per half at the moment that half hits num_steps (s_{num_steps}); the
            # in-flight half carries across rollouts (its result is the next rollout's
            # first recorded transition) -> continuous trajectories, no gaps/waste.
            cnt = [0, 0]; boot_set = [False, False]
            next_value = torch.zeros(1, num_envs, device=device)
            next_done = torch.zeros(num_envs, device=device)
            while not (boot_set[0] and boot_set[1]):
                g = envs.last_group; fr, ve = held[g]
                with torch.no_grad():
                    a, lp, _, val = agent.get_action_and_value(fr, ve)
                pend[g] = (fr, ve, a, lp, val, held_done[g])
                obs, rew, done, infos = envs.step(a.cpu().numpy())
                gr = envs.last_group
                if pend[gr] is not None:
                    fr_p, ve_p, a_p, lp_p, val_p, done_p = pend[gr]; pend[gr] = None
                    held[gr] = obs_to_tensor(obs, device)
                    hd = torch.as_tensor(done, device=device).float()
                    if cnt[gr] < num_steps:
                        i = cnt[gr]; c = _cols[gr]
                        f_buf[i, c] = fr_p; v_buf[i, c] = ve_p; actions[i, c] = a_p
                        logprobs[i, c] = lp_p; values[i, c] = val_p.flatten()
                        dones[i, c] = done_p; rewards[i, c] = torch.as_tensor(rew, device=device)
                        cnt[gr] += 1; global_step += _half; _collect_stats(infos)
                        if cnt[gr] == num_steps and not boot_set[gr]:
                            with torch.no_grad():
                                next_value[0, c] = agent.get_value(held[gr][0], held[gr][1]).flatten()
                            next_done[c] = hd; boot_set[gr] = True
                    held_done[gr] = hd
                else:                          # first touch of this half (reset obs, no prior action)
                    held[gr] = obs_to_tensor(obs, device)
                    held_done[gr] = torch.as_tensor(done, device=device).float()

        coll_t = time.time() - _t_coll; _t_upd = time.time()
        if not async_collect:
            # ── GAE over the full [num_steps, num_envs] buffer ──
            with torch.no_grad():
                advantages = torch.zeros_like(rewards); lastgae = 0
                for t in reversed(range(num_steps)):
                    nnt = 1.0 - (next_done if t == num_steps - 1 else dones[t + 1])
                    nv = next_value if t == num_steps - 1 else values[t + 1]
                    delta = rewards[t] + gamma * nv * nnt - values[t]
                    advantages[t] = lastgae = delta + gamma * gae_lambda * nnt * lastgae
                returns = advantages + values
            bf = f_buf.reshape((-1,) + frame0.shape[1:]); bv = v_buf.reshape(-1, vec0.shape[1])
            ba = actions.reshape(-1, len(nvec)); blp = logprobs.reshape(-1)
            badv = advantages.reshape(-1); bret = returns.reshape(-1); bval = values.reshape(-1)
            B = batch_size
        else:
            # ── async: per-engine GAE over counts[i] steps (variable length), then
            # keep only the valid rows. Computed on CPU (tiny scalar loop, fast). ──
            rew_np = rewards.cpu().numpy(); val_np = values.cpu().numpy(); done_np = dones.cpu().numpy()
            nv_np = next_value[0].cpu().numpy(); nd_np = next_done.cpu().numpy()
            adv_np = np.zeros((cap, num_envs), dtype=np.float32)
            for i in range(num_envs):
                n = counts[i]; lastgae = 0.0
                for t in range(n - 1, -1, -1):
                    nd = nd_np[i] if t == n - 1 else done_np[t + 1, i]
                    nv = nv_np[i] if t == n - 1 else val_np[t + 1, i]
                    nnt = 1.0 - nd
                    delta = rew_np[t, i] + gamma * nv * nnt - val_np[t, i]
                    lastgae = delta + gamma * gae_lambda * nnt * lastgae
                    adv_np[t, i] = lastgae
            advantages = torch.as_tensor(adv_np, device=device)
            returns = advantages + values
            mask_np = np.zeros((cap, num_envs), dtype=bool)
            for i in range(num_envs): mask_np[:counts[i], i] = True
            flat = torch.as_tensor(mask_np.reshape(-1), device=device)
            bf = f_buf.reshape((-1,) + frame0.shape[1:])[flat]; bv = v_buf.reshape(-1, vec0.shape[1])[flat]
            ba = actions.reshape(-1, len(nvec))[flat]; blp = logprobs.reshape(-1)[flat]
            badv = advantages.reshape(-1)[flat]; bret = returns.reshape(-1)[flat]; bval = values.reshape(-1)[flat]
            B = int(flat.sum().item())
        inds = np.arange(B)
        # async B is variable (final collection wave overshoots budget); use only whole
        # minibatches (drop_last) so every minibatch is full -> no degenerate std()/NaN,
        # identical update shape to sync. Dropped tail (<minibatch_size) is random/epoch.
        B_eff = max(minibatch_size, (B // minibatch_size) * minibatch_size)
        early_stopped = False
        vls, pls, els, cfs, kls = [], [], [], [], []
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for s in range(0, B_eff, minibatch_size):
                mb = inds[s:s + minibatch_size]   # always exactly minibatch_size (drop_last)
                _, nlp, ent, nval = agent.get_action_and_value(bf[mb], bv[mb], ba[mb])
                logratio = nlp - blp[mb]; ratio = logratio.exp()
                with torch.no_grad():
                    akl = ((ratio - 1) - logratio).mean()
                    kls.append(akl.item())
                    cfs.append(((ratio - 1).abs() > clip_coef).float().mean().item())
                mb_adv = badv[mb]; mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                nval = nval.view(-1)
                v_un = (nval - bret[mb]) ** 2
                v_cl = bval[mb] + torch.clamp(nval - bval[mb], -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max(v_un, (v_cl - bret[mb]) ** 2).mean()
                ent_loss = ent.mean()
                loss = pg - ent_coef * ent_loss + vf_coef * v_loss
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm); optimizer.step()
                vls.append(v_loss.item()); pls.append(pg.item()); els.append(ent_loss.item())
                # SB3-aligned: early-stop the epoch loop when a minibatch KL > 1.5*target_kl
                if target_kl is not None and akl > 1.5 * target_kl:
                    early_stopped = True; break
            if early_stopped:
                break
        approx_kl = float(np.mean(kls)) if kls else 0.0   # SB3 logs/uses the MEAN over minibatches

        # adaptive KL-LR controller (clamped multiplier<=1.0 = the blowup fix)
        if target_kl is not None:
            if early_stopped:
                kl_mult = max(1e-5 / base_lr, kl_mult / 1.2)
            elif approx_kl > 2 * target_kl:
                kl_mult = max(1e-5 / base_lr, kl_mult / 1.5)
            elif approx_kl < 0.5 * target_kl:
                kl_mult = min(1.0, kl_mult * 1.5)

        sps = int(global_step / (time.time() - start))
        # metric names ALIGNED to SB3's native ones so SB3-vs-cleanrl overlay on the same charts
        log = {"charts/SPS": sps,
               "train/learning_rate": optimizer.param_groups[0]["lr"],
               "train/approx_kl": float(approx_kl), "train/kl_lr_multiplier": kl_mult,
               "train/early_stopped": int(early_stopped),
               "train/value_loss": float(np.mean(vls)), "train/policy_gradient_loss": float(np.mean(pls)),
               "train/entropy_loss": float(np.mean(els)), "train/clip_fraction": float(np.mean(cfs)),
               "progress/best_stage_ever": best_stage,
               "stats/episodes_total": total_eps}
        if ep_rew_d:
            log["rollout/ep_rew_mean"] = float(np.mean(ep_rew_d))   # normalized (SB3-comparable)
            log["rollout/ep_len_mean"] = float(np.mean(ep_len_d))
        if ep_stats.get("return"):
            log["stats/return_raw"] = float(np.mean(ep_stats["return"]))
        for k in ("clean_max_stage", "clean_stages_cleared", "stage_win_rate", "round_win_rate",
                  "dmg_ratio", "max_stage", "stage_progress", "game_cleared", "stages_cleared", "stage_attempts"):
            if ep_stats.get(k): log[f"stats/{k}"] = float(np.mean(ep_stats[k]))
        ms = np.array(ep_stats.get("max_stage", [0]))
        for kk in range(1, 11):
            log[f"stage_reach/ge_{kk}"] = float((ms >= kk).mean())
        wandb.log(log, step=global_step)
        if global_step >= next_save:
            save_ckpt(global_step); next_save += save_every
        upd_t = time.time() - _t_upd
        log["time/collect_s"] = coll_t; log["time/update_s"] = upd_t
        if it % 5 == 0 or it == 1:
            iter_t = time.time() - _t_iter
            print(f"it {it}/{num_iterations} step {global_step} SPS {sps} kl {float(approx_kl):.4f} "
                  f"rew {log.get('rollout/ep_rew_mean', 0):.2f} max_stage {log.get('stats/max_stage', 0):.2f} "
                  f"mult {kl_mult:.3f} | collect {coll_t:.2f}s update {upd_t:.2f}s "
                  f"({100*upd_t/max(1e-6,iter_t):.0f}% upd)", flush=True)

    save_ckpt(global_step); envs.close(); wandb.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgFile", default="config/config_replicate.yaml")
    main(ap.parse_args().cfgFile)
