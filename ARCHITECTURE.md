# SFIII (DIAMBRA) — Architecture & Running on the SLURM Cluster

_Last verified: 2026-06-14 on WPI Turing (`gpu-4-15.int.turing.wpi.edu`, A100-80GB, apptainer 1.4.0)._

This document explains how the training stack is wired, the **one bug that breaks
parallel environment spawning**, and the correct way to run the DIAMBRA engine
under apptainer on the SLURM cluster.

---

## 1. The big picture

DIAMBRA Arena is a **client/server** RL setup, unlike a normal in-process gym:

```
                          (gRPC over localhost:PORT)
  ┌────────────────────┐                         ┌─────────────────────────────┐
  │  Python trainer     │  actions ───────────▶  │  diambraEngineServer        │
  │  (main.py, SB3 PPO) │                         │  (MAME emulator + sfiii3n)  │
  │  SubprocVecEnv      │  ◀─────────── obs/rew   │  inside an apptainer        │
  │   ├ worker 0 ───────┼────▶ localhost:BASE+0   │  instance, 1 per env        │
  │   ├ worker 1 ───────┼────▶ localhost:BASE+1   └─────────────────────────────┘
  │   └ worker N-1 ─────┼────▶ localhost:BASE+N-1   (N independent containers)
  └────────────────────┘
```

- **Each environment is a separate apptainer container** running one
  `diambraEngineServer`, listening on its own TCP port.
- The Python side (`diambra.arena.make(...)`) is a **gRPC client** that connects
  to one engine by port. The list of engines is passed via the
  `DIAMBRA_ENVS="localhost:p0 localhost:p1 ..."` environment variable.
- N containers ⇒ N ports ⇒ N parallel envs. This is the only way to scale envs.

### Component map (this repo)

| File | Role |
|------|------|
| `startup.py` | Spawns the apptainer engine **containers + servers**, one per port. Blocks forever (keeps containers alive). |
| `start.sh`   | Thin wrapper: makes creds, `module load apptainer`, runs `startup.py`. |
| `make_sb3_env.py` | Builds the SB3 `SubprocVecEnv`; each worker calls `diambra.arena.make()` to connect to one engine. **Contains the `serial_wrapper` FileLock (the bottleneck).** |
| `main.py`    | Loads a YAML config, builds the vec env, trains PPO (stable-baselines3), logs to wandb. |
| `server.sh`  | Launches several `main.py` training jobs on different GPUs/ports. |
| `config/*.yaml` | Game settings + wrappers + PPO hyperparameters. |
| `entropy_decay.py` | SB3 callback: linear entropy-coefficient decay. |
| `engine_latest.sif` | **STOCK** diambra engine image — requires online login (see §3). |

---

## 2. The DIAMBRA engine image & the authorization problem

The stock engine (`docker://diambra/engine:latest`, baked into the committed
`engine_latest.sif`) **requires an online account login** before it will start.
Running it now hangs:

```
Stored credentials found.
Stored credentials check unsuccessful. (Unable to find key in json string.)
 ... DIAMBRA Arena ... register at https://diambra.ai/register/
Username (or Email):          ← blocks forever, never opens the port
```

The DIAMBRA project is effectively **unmaintained**, so the credential endpoint
can't be relied on. The fix is a **patched image that bypasses the auth check**:

```
docker://chenningcong227/diambra-engine:patched
```

**Verified behaviour (this is the correct image to use):**

```
Server listening on 0.0.0.0:51000     ← in ~2s, no login, no network round-trip
```

### Cheap existence check before pulling (~280 MB)
apptainer 1.4.0 has no `manifest` subcommand, so check via the Docker Hub API:

```bash
curl -s https://hub.docker.com/v2/repositories/chenningcong227/diambra-engine/tags/ \
  | python3 -c 'import sys,json;print([t["name"] for t in json.load(sys.stdin)["results"]])'
# -> ['patched']
```

### Pull it once (cached, machine-local)
```bash
module load apptainer
apptainer pull engine_patched.sif docker://chenningcong227/diambra-engine:patched
```

---

## 3. The parallel-spawn bug (the "lock") — ROOT CAUSE FOUND & FIXED

`make_sb3_env.py` wraps every env-creation in a global **FileLock**
(`serial_wrapper`, `lock_barrier.lock`) so only one worker proceeds at a time,
defeating the point of `SubprocVecEnv`. `startup.py` likewise creates containers
in a serial for-loop.

**Root cause (proven on this cluster):** apptainer, by default, **auto-mounts the
host `$HOME` and `/tmp` into every container**. The engine boots MAME + loads the
ROM during the **`EnvInit` RPC** (which fires inside `diambra.arena.make()`), and
that boot writes scratch/state into `$HOME`/`/tmp`. Because every engine shared the
*same host* `$HOME`/`/tmp`, concurrent `make()` calls **collided on those shared
files** and the engine dropped the connection:

```
StatusCode.UNAVAILABLE  details="Socket closed"
```

This is **not** auth, **not** container spawn, **not** a Python thread-global, and
**not** a TCP-port conflict — all ruled out empirically:

| make mode (N=3, healthy engines) | result |
|----------------------------------|--------|
| concurrent (threads, same proc)  | 1–2 / 3 — `Socket closed` |
| concurrent (separate processes)  | 1 / 3 — same error (so not a Python-global) |
| **staggered ~2s apart**          | 3 / 3 — spacing avoids the collision |
| port inspection during boot      | engine opens only its own port (no aux port) |
| roms mount after boot            | unchanged (engine doesn't write there) |

### The fix (verified)
Give each engine a **private `$HOME` and `/tmp`** via apptainer `--contain` +
per-engine `--workdir`/`--home`. The shared-mount collision disappears and
**concurrent make + concurrent step works with no lock and no stagger**:

```
spawn flag added:  --contain --workdir <scratch>/<eng_i> --home <scratch>/<eng_i>/home
                   (roms bound read-only; no credentials bind for the patched image)

N=8 parallel make :  8/8 OK in 5.5 s   (was 1/8 failing; serial was ~5.2 s × 8 = 40 s+)
N=8 step (threaded):  1,502 SPS aggregate (188/env)
```

Implemented in `engine_manager.py` (`_start_instance_cmd` + `_prep_workdir`).
**The `serial_wrapper` FileLock is now obsolete and should be removed.**

### Remaining bottleneck — reset (sync-blocking)
With the make race gone, the slow part is **`reset()`**: ~3 s/env, and a vec-step
blocks on the slowest env's reset (N=8 first-reset ≈ 25 s). Plan (per design): a
**hot-spare engine pool** — when an env episode ends, detach it into an async
reset/recycle queue and wire in a pre-warmed engine from the pool, so the vec-step
never blocks on reset. Combined with the fault-tolerant detach/attach layer (§3a).

---

## 4. Correct way to run on the SLURM cluster

apptainer is **not** on the default PATH — it is a module:

```bash
module load apptainer            # apptainer/1.4.0 on Turing
```

### 4a. Minimal: one engine by hand (smoke test)
```bash
module load apptainer
mkdir -p ~/.diambra && touch ~/.diambra/credentials   # patched image ignores contents
PORT=51000
apptainer instance start --userns \
  -B /usr/share/alsa -B /usr/bin/getopt -B /usr/bin/cut \
  --bind ~/.diambra/credentials:/tmp/.diambra/credentials,$PWD/roms:/opt/diambraArena/roms \
  engine_patched.sif engine_0
apptainer exec --userns instance://engine_0 \
  /bin/diambraEngineServer --envAddress 0.0.0.0:$PORT &
# expect: "Server listening on 0.0.0.0:51000"
ss -ltn | grep $PORT                 # confirm the port is up
# cleanup:
apptainer instance stop engine_0
```

Notes on the binds:
- `roms/sfiii3n.zip` → `/opt/diambraArena/roms` (the game ROM, required).
- the three `-B /usr/...` binds patch tools the engine shells out to.
- `--userns` runs rootless (correct for an unprivileged cluster user).

### 4b. N engines in parallel (what `startup.py` does — now safe to parallelize)
Launch all instances at once, then all servers; ports come up in seconds:
```bash
module load apptainer
N=16; BASE=51000
for i in $(seq 0 $((N-1))); do
  apptainer instance start --userns -B /usr/share/alsa -B /usr/bin/getopt -B /usr/bin/cut \
    --bind ~/.diambra/credentials:/tmp/.diambra/credentials,$PWD/roms:/opt/diambraArena/roms \
    engine_patched.sif engine_$i &
done; wait
for i in $(seq 0 $((N-1))); do
  apptainer exec --userns instance://engine_$i \
    /bin/diambraEngineServer --envAddress 0.0.0.0:$((BASE+i)) &
done
export DIAMBRA_ENVS=$(for i in $(seq 0 $((N-1))); do echo -n "localhost:$((BASE+i)) "; done)
```

### 4c. Running inside `sbatch` (the durable way)
Per `lc-cpp-engine/HANDOFF_GPU.md`: agent-spawned tmux/background processes die on
network drops — use the scheduler. Skeleton:
```bash
#!/bin/bash
#SBATCH --job-name=sfiii
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32      # >= num_envs threads for engines + trainer
#SBATCH --mem=64G
#SBATCH --time=24:00:00
module load apptainer
cd $SLURM_SUBMIT_DIR
# 1) bring up engines (background), 2) wait for ports, 3) run trainer, 4) cleanup on exit
trap 'for i in $(seq 0 $((N-1))); do apptainer instance stop engine_$i; done' EXIT
# ... (4b launch block) ...
python main.py --cfgFile config/<cfg>.yaml
```

### Gotchas verified on this cluster
- `module load apptainer` does **not persist** across separate shells — load it in
  the same script/step that calls apptainer.
- Network egress from compute nodes **is** open (diambra.ai:443 reachable) — the
  hang was the auth *logic*, not a firewall.
- Always **stop instances** at the end (`apptainer instance stop`) or they leak and
  hold ports; `startup.py --kill_all` / `server.sh` do this for old runs.

---

## 5. Current environment state (must fix before training runs)

- **The `minerl` conda env referenced by `server.sh` / `main.sh` no longer exists.**
  No conda env currently has `diambra-arena` installed.
  - `minecraft` env: has `stable_baselines3 2.1.0`, **no** `diambra`, **no** `sb3_contrib`.
- To run again you must recreate an env with: `diambra-arena`, `stable-baselines3`,
  `sb3-contrib` (for DQN/QR-DQN), `wandb`, `pyyaml`, `filelock`. Then point the run
  scripts at it instead of `minerl`.

---

## 6. wandb results (project `ningcong-chen/ppo-new-200M`)

19 runs total; **most crashed or were killed**. Best *finished* run
(`comic-dragon-12`, 50M steps) reached only `rollout/ep_rew_mean ≈ 9` (normalized).
Several runs climbed to ~13–19 then crashed. The picture is **instability**, not a
clean plateau. Per the lc-cpp-engine study, likely drivers are **game/reward
structure and untuned learning rate**, not just throughput.

---

## 7. Production pipeline (built & running)

Fresh **uv** env at `.venv` (pins from the old `minerl` requirements). New modules:

| file | role |
|------|------|
| `engine_manager.py` | parallel engine launch (`--contain` fix) + health + per-engine restart |
| `robust_env.py` | `RobustDiambraEnv`: watchdog timeout → detach hung engine → reattach (self-heal) |
| `sfiii_env.py` | `build_env` + `StatsWrapper` (max_stage, stage_progress, rounds, dmg_ratio, win, …) |
| `mp_vecenv.py` | `MPVecEnv`: multiprocess (linear scaling) + per-worker hot-spare (reset never stalls) |
| `sb3_mpvec.py` | `SB3MPVecEnv`: SB3 VecEnv adapter → drops into PPO/DQN |
| `adaptive_kl_lr.py` | KL-adaptive LR (Rudin et al.) **+ blowup fix** (`max_multiplier` clamp) |
| `train.py` | training entrypoint (PPO on `SB3MPVecEnv` + adaptive-LR + wandb stats) |
| `eval_video.py` + `video_saver.py` | **color** video via two seed-synced envs (gray policy / RGB frames) |
| `config/config_replicate.yaml` | exact best-run hparams (gamma 0.99, ent 1e-4, target_kl 0.01, norm_factor 0.2) + LR fix |

**Validated:** parallel spawn (8 engines/1s), fault recovery (hang→reattach, others unaffected),
hot-spare (5 episode-ends, 0 inline stalls), stats through full wrapper stack, multiprocess
scaling (raw 95% @ N=8 / 5.8k SPS @ N=16; with full wrappers+IPC ~75-85%).

### Run it
```bash
module load apptainer && source .venv/bin/activate && export $(cat wandb.key)
CUDA_VISIBLE_DEVICES=0 python train.py --cfgFile config/config_replicate.yaml
python eval_video.py --model results/<run>/model/<ckpt>.zip --out fight.mp4 --episodes 3
```
Engines auto-spawn from `engine_patched.sif` (no separate `startup.py`). Stop a run:
`pkill -f 'train.py'` then `apptainer instance stop` the `train_*` instances.

## 8. Why the best run died & the modelling fix

Best 2026 run `08:35-Mar-11` (max reward **21.5**) used a **KL-adaptive LR** (Rudin et al.):
LR×1.5 when KL<0.5·target. Near-optimal the gradient is tiny → KL stays low → LR ratchets to
`lr_cap_late` (8e-4, 8× base) → a later gradient spike destabilizes it → collapse (died at
9.5M/50M). Also **gamma 0.99 ≫ 0.94 ≫ 1.0** (the later `config_6` gamma=1.0 sweep tanked reward
to ~1). **Fix:** clamp the LR multiplier ≤ 1.0 (controller may only *cut* LR, never inflate above
the planned decay). The replication runs this fix on the crash-proof stack so it trains to
completion instead of dying early.
