# GAD B1 Runbook — 8×H200 SLURM cluster

*Updates the original team runbook to reflect how the pipeline actually runs on this cluster. Key difference from the sandbox version: this is a **SLURM cluster and the login node has no GPU** — every run goes through `sbatch` onto an `h200` compute node, not `bash ... &`.*

## 0. Cluster reality (design around these)
| Constraint | Consequence |
|---|---|
| Login node has **no GPU** | all training via `sbatch`; validate with `nvidia-smi` on the compute node, not the login node |
| SLURM QOS `h200_dev` (account `mrs_2`) | **no wall-time limit** (partition cap = 7 days; the "12h limit" is a myth here); **max 2 nodes per user** (`MaxTRESPU gpu=16,node=2`) — see the QOS table below to run more concurrently |
| Real compute nodes have normal `/dev/shm` and networking | **do NOT** set the sandbox hacks (`NCCL_SOCKET_IFNAME=lo`, `NCCL_SHM_DISABLE=1`) or source the sandbox `env.sh` proxies — they hurt or break on real nodes |
| H200 = **141 GB** HBM | 7B fits easily on one GPU → use **TP=1**; memory is never the bottleneck |
| **`/home` = FSx-OpenZFS, hard 1 TB per-user quota** (NOT for ckpts/models, per infra) | a single 7B ckpt ≈ 150 GB → 2–3 arms overflow 1 TB and saves die with **`Disk quota exceeded`** (killed the SeqKD arms 2026-07-27). **Write ckpts to Lustre instead** (see §5). `df -h ~` shows the 43 TB *filesystem*, NOT your quota — ignore it. ZFS snapshots keep deleted files counted against quota for **24–48 h**, so `rm` gives no instant relief — the fix is to stop writing to `/home`, not to delete |
| **Lustre project FS** `/checkpoints/$USER`, `/fsx/$USER` (200 T+, ~23 T free, **no 1 TB quota**) | **read-only from the login node, WRITABLE from compute nodes** — jobs checkpoint from compute, so point `trainer.default_local_dir` here. To copy existing data off `/home` you must run `cp`/`mv` from a compute node (`srun --overlap --jobid=<running> cp ...`) |

### QOS options (account `mrs_2`) — beating the 2-node cap
The default launchers hardcode `--qos=h200_dev`, which caps you at **2 nodes/user**. To run more experiments concurrently, override with `sbatch --qos=<name> ...` (account stays `mrs_2`):

| QOS | Priority | Per-user cap | Group cap | Use for |
|---|---|---|---|---|
| `h200_dev` | 100 | **gpu=16 / node=2** | — | default; single experiment |
| **`h200_mrs_2_high`** | 100 | **none** | gpu=296 (group) | **concurrent runs** — same priority as dev, no per-user limit |
| `h200_mrs_shared` (default) | 5 (low) | gpu=256 | gpu=80 (group) | overflow; low priority, may wait/preempt |
| `lowest` | 1 | — | gpu=0 | unusable for GPU jobs |

Example — two full A/B chains at once: keep one on `h200_dev`, put the other on `--qos=h200_mrs_2_high`. They draw from separate budgets. Inspect limits with `sacctmgr show qos <name> format=Name,Priority,MaxTRESPU,GrpTRES`.


## 1. Environment (once)
- venv at `~/gad_run/venv` (torch 2.6.0+cu124, vllm 0.8.5, verl editable). Python headers for Triton in `~/gad_run/pyinclude`.
- Repos: `~/LMOps/gad` (orchestration) and `~/LMOps/gad/verl` (fork `slahrichi/verl`, branches `warmup` / `gad` / `seqkd` / `eval`, plus `gad-d2-replay`).
- Do **not** source the sandbox `env.sh` in an sbatch job; the production launchers set only what a compute node needs (`TMPDIR`, `HF_HOME`, `C_INCLUDE_PATH`) and `unset` the proxies.

## 2. Winning performance config (adopt everywhere)
`actor_rollout_ref.rollout.tensor_model_parallel_size=1` + `VLLM_USE_V1=1` → ~212 s/step (≈1.9× faster than the TP=2/v0 reference). `gpu_memory_utilization=0.7` (higher gives nothing). Discriminator stays **fp32** (paper-faithful; bf16 gave no speedup). `+data.dataloader_num_workers=0`, `ray_init.num_cpus=32`, and **`export PYTHONUNBUFFERED=1`** (else the per-step metric lines — incl. `critic/d_acc_fresh` — buffer and don't appear in the `.out` until the job exits).

## 3. Data
`~/gad_run/data/lmsys_{train,test}-00000-of-00001.parquet` = `ytz20/LMSYS-Chat-GPT-5-Chat-Response` (192K train / 479 test). Verified to match the paper. Subsample for fast e2e tests: `mini_train.parquet` (3,072 rows = 12 steps/epoch).

## 4. Run the pipeline (all launchers in `~/gad_run/`, self-contained, `resume_mode=auto`)
```bash
# Stage 1 — WARMUP (warmup branch, ~2 days, 1 epoch = 750 steps)
sbatch ~/gad_run/run_warmup_prod.sh
#   -> checkpoints in ckpts/<EXP>/global_step_N ; keep-last-2

# Stage 2 — GAD adversarial (gad-d2-replay branch, ~2 days, 2 epochs)
#   merges the final warmup ckpt (FSDP->HF), resumes, runs with critic.replay.*
WARMUP_EXP=<warmup exp> RESUME_STEP=<final step> EXP=<gad exp> \
  REPLAY_CAPACITY=1024 sbatch ~/gad_run/run_gad_prod.sh
#   REPLAY_CAPACITY=0 for the vanilla-GAD baseline

# Fast end-to-end test on the subsample (~2 h instead of ~6 days)
EXP=mini-warmup TRAIN=$PWD/data/mini_train.parquet \
  EXTRA_ARGS="trainer.save_freq=6 trainer.test_freq=6" sbatch run_warmup_prod.sh
```
Resilience: if a job dies (node/preempt), just resubmit — `resume_mode=auto` continues from the last checkpoint. No job-chaining needed (no wall limit).

## 5. Checkpoints — where they go + you don't need them all
**Write to Lustre, not `/home`** (the 1 TB quota kills multi-arm runs). The launchers take a `CKPT_ROOT` override:
```bash
CKPT_ROOT=/checkpoints/$USER/gad_run/ckpts  EXP=<exp> ... sbatch run_gad_prod.sh   # (or run_seqkd_prod.sh)
#   default (unset) = ~/gad_run/ckpts on /home — only for tiny/debug runs
#   warmup is still READ from ~/gad_run/ckpts/$WARMUP_EXP; only the OUTPUT goes to CKPT_ROOT
```
GAD resumes from a **single** warmup checkpoint (the final one). Intermediates are only crash-insurance. Both launchers set `save_freq=200` (fs50=50) + `max_{actor,critic}_ckpt_to_keep=2` → ~300 GB/stage instead of TBs. After GAD starts (loads the merged HF), the warmup FSDP shards are deletable (keep only `{actor,critic}/huggingface/` ≈ 28 GB for re-init).

## 6. Monitor / manage
```bash
squeue -u $USER          # `squeue --me` intermittently returns empty here — use -u $USER or -j <ids>
tail -f ~/gad_run/logs/<stage>-<jobid>.out
grep -E "step:[0-9]|timing_s/step" ~/gad_run/logs/<stage>-<jobid>.out | tail
du -sh ~/gad_run/ckpts ; df -h ~   # watch OUR footprint AND shared-FS free space
```
`watch_fullscale.sh` runs a background milestone logger + **disk tripwire** (alerts if FSx free < 2 TB or ckpts > 2.5 TB).

## 7. Gotchas hit (and fixed)
- `ray_init.num_cpus=32` is **required** — without it Ray over-subscribes the node's cores and the raylet handshake fails.
- Inline the training config in the sbatch script (don't `bash` a second NFS script mid-job) — avoids an NFS "stale file handle" abort seen on a long run.
- `critic.replay.model_dtype`-style keys not in the config struct need Hydra's `+` prefix; `critic.replay.{capacity,rho,strategy}` are in-struct (plain override).
- **Frozen console log ≠ crashed job.** Ray's log capture can stall (esp. without `PYTHONUNBUFFERED=1`) while training continues. Verify with `srun --jobid=<id> --overlap -N1 nvidia-smi` (busy GPUs = alive) and checkpoint mtimes *before* killing anything.
- **Disk-full during a checkpoint save → corrupt ckpt** (critic missing `extra_state_*.pt`; a good save has 24 `.pt` per actor/ & critic/). Recovery: `rm -rf ckpts/<EXP>/global_step_N` and resubmit — empty/older ckpt dir → `resume_mode=auto` re-inits from the merged warmup HF. Prevent: **write ckpts to Lustre** (`CKPT_ROOT`, see §5) so the 1 TB `/home` quota is a non-issue; keep-last-2; prune completed arms to final `gs492`.
- **`h200_mrs_shared` is preemptible** (priority 5) — low-priority arms get bumped; on it, use a small `save_freq` (e.g. 50) or move to `h200_dev` for uninterrupted runs.
