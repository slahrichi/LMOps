# GAD B1 Runbook — 8×H200 SLURM cluster

*Updates the original team runbook to reflect how the pipeline actually runs on this cluster. Key difference from the sandbox version: this is a **SLURM cluster and the login node has no GPU** — every run goes through `sbatch` onto an `h200` compute node, not `bash ... &`.*

## 0. Cluster reality (design around these)
| Constraint | Consequence |
|---|---|
| Login node has **no GPU** | all training via `sbatch`; validate with `nvidia-smi` on the compute node, not the login node |
| SLURM QOS `h200_dev` (account `mrs_2`) | **no wall-time limit** (partition cap = 7 days; the "12h limit" is a myth here); **max 2 nodes per user** concurrently |
| Real compute nodes have normal `/dev/shm` and networking | **do NOT** set the sandbox hacks (`NCCL_SOCKET_IFNAME=lo`, `NCCL_SHM_DISABLE=1`) or source the sandbox `env.sh` proxies — they hurt or break on real nodes |
| H200 = **141 GB** HBM | 7B fits easily on one GPU → use **TP=1**; memory is never the bottleneck |
| Home = FSx, **per-user quota** | each 7B checkpoint ≈ 150 GB → **keep-last-2** or you WILL hit `EDQUOT` mid-run |

## 1. Environment (once)
- venv at `~/gad_run/venv` (torch 2.6.0+cu124, vllm 0.8.5, verl editable). Python headers for Triton in `~/gad_run/pyinclude`.
- Repos: `~/LMOps/gad` (orchestration) and `~/LMOps/gad/verl` (fork `slahrichi/verl`, branches `warmup` / `gad` / `seqkd` / `eval`, plus `gad-d2-replay`).
- Do **not** source the sandbox `env.sh` in an sbatch job; the production launchers set only what a compute node needs (`TMPDIR`, `HF_HOME`, `C_INCLUDE_PATH`) and `unset` the proxies.

## 2. Winning performance config (adopt everywhere)
`actor_rollout_ref.rollout.tensor_model_parallel_size=1` + `VLLM_USE_V1=1` → ~212 s/step (≈1.9× faster than the TP=2/v0 reference). `gpu_memory_utilization=0.7` (higher gives nothing). Discriminator stays **fp32** (paper-faithful; bf16 gave no speedup). `+data.dataloader_num_workers=0`, `ray_init.num_cpus=32`.

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

## 5. Checkpoints — you don't need them all
GAD resumes from a **single** warmup checkpoint (the final one). Intermediates are only crash-insurance. Both launchers set `save_freq=200` + `max_{actor,critic}_ckpt_to_keep=2` → ~300 GB/stage instead of TBs. After GAD starts (loads the merged HF), the warmup FSDP shards are deletable.

## 6. Monitor / manage
```bash
squeue --me
tail -f ~/gad_run/logs/<stage>-<jobid>.out
grep -E "step:[0-9]|timing_s/step" ~/gad_run/logs/<stage>-<jobid>.out | tail
du -sh ~/gad_run/ckpts/*                      # watch the quota
```

## 7. Gotchas hit (and fixed)
- `ray_init.num_cpus=32` is **required** — without it Ray over-subscribes the node's cores and the raylet handshake fails.
- Inline the training config in the sbatch script (don't `bash` a second NFS script mid-job) — avoids an NFS "stale file handle" abort seen on a long run.
- `critic.replay.model_dtype`-style keys not in the config struct need Hydra's `+` prefix; `critic.replay.{capacity,rho,strategy}` are in-struct (plain override).
