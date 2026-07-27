# GAD A/B Experiments — Full-Scale Replay-Buffer Study (SLURM / 8×H200)

**Status doc — last updated 2026-07-27.** Live job status is refreshed hourly into the
`gad-sbatch-ab-campaign` agent memory; this doc describes the *design and mechanics*.

---

## 1. Goal

Measure the effect of the **D2 discriminator replay buffer** on GAD training.

GAD (Generative Adversarial Distillation) distills a large teacher LLM into a smaller
student on-policy: a student generator is trained (via GRPO) to fool a co-trained
Bradley–Terry **discriminator/critic** that learns to tell teacher responses from student
responses — a GAN-style minimax loop. Training = **1-epoch warmup** (stabilize
generator+discriminator) → **2 epochs adversarial GAD** → eval.

The **D2 replay buffer** (`critic.replay.*`, branch `gad-d2-replay`) lets the discriminator
re-train on past student responses instead of only the current batch. The experiment is a
clean **A/B**:

| Arm | `critic.replay.capacity` | Meaning |
|-----|--------------------------|---------|
| **baseline** | `0` | exact vanilla GAD (replay OFF) |
| **replay**   | `4096` (`rho=0.5`) | discriminator replay ON |

Both arms in a pair **share the same warmup init and the same data subset**, so any
difference in outcome is attributable to the replay buffer alone.

## 2. Why stratified subsamples (33% / 50%)

Full data = `ytz20/LMSYS-Chat-GPT-5-Chat-Response` (192K LMSYS-Chat prompts + GPT-5-Chat
teacher responses). A full warmup+GAD run is **~5–6 days/arm** — over the 72h SLURM job
limit. So we run on **category-stratified subsamples** (stratified by prompt category,
proportion drift < 1e-4, smallest of 31 classes still ≥ 296 rows):

| Fraction | Train file | Rows | Purpose |
|----------|-----------|------|---------|
| **33%** | `data/lmsys_train_strat33.parquet` | 63,215 | primary A/B, fits comfortably in 72h |
| **50%** | `data/lmsys_train_strat50.parquet` | ~95K | scale check of the 33% result |

Val = `data/lmsys_test-00000-of-00001.parquet` (479 rows). Subsampling keeps GAD ≥ 400
adversarial steps (enough for the chase-cycle to develop) while fitting the job limit.

So the campaign is **4 GAD arms** (33-base, 33-replay, 50-base, 50-replay), each preceded by
a shared per-fraction warmup (33-warmup feeds both 33 arms; 50-warmup feeds both 50 arms).

## 3. Pipeline & launchers

Two SLURM-native launchers in `gad_run/`, both single-node 8×H200, TP=1 + vLLM v1, branch
`gad-d2-replay`:

| Stage | Script | Steps (33%) | Notes |
|-------|--------|-------------|-------|
| **1. Warmup** | `run_warmup_prod.sh` | 246 (~15h) | 1 epoch; initializes actor + critic |
| **2. Adversarial GAD** | `run_gad_prod.sh` | 492/arm (~40h) | 2 epochs; `critic_warmup=0`; merges warmup FSDP shards → HF, then resumes |

Both are **env-var parameterized** and chained via SLURM dependency (warmup → base + replay):

```bash
# baseline arm (replay OFF)
WARMUP_EXP=fs33-warmup RESUME_STEP=latest EXP=fs33-gad-base \
REPLAY_CAPACITY=0 REPLAY_RHO=0.5 \
TRAIN=/home/saadlahrichi/gad_run/data/lmsys_train_strat33.parquet \
VAL=/home/saadlahrichi/gad_run/data/lmsys_test-00000-of-00001.parquet \
sbatch run_gad_prod.sh

# replay arm — same, but EXP=fs33-gad-replay REPLAY_CAPACITY=4096
```

Key config (both arms): `algorithm.adv_estimator=grpo`, `data.prompt_key=content`,
`train_batch_size=256`, `max_prompt_length=2048`, `max_response_length=1536`,
`rollout.n=8`, `rollout.temperature=0.8`, `rollout.gpu_memory_utilization=0.7`, lr `1e-6`,
`grad_clip=0.2`, `kl_loss_coef=0.001`. The two GAD arms **flock-share** the warmup→HF merge
(`.merge.lock`); second job sees it's done and skips. `resume_mode=auto` reads each arm's
own `default_local_dir`, so a preempted/crashed arm resumes from its own last checkpoint.

## 4. Checkpointing policy

- **`save_freq` = 200** on all four arms, `max_actor_ckpt_to_keep = max_critic_ckpt_to_keep = 2`.
  (History: the 50% arms first ran on the preemptible `h200_mrs_shared` QOS with `save_freq=50`
  to bound requeue loss; after repeated preemption they were **moved to `h200_dev`
  (non-preemptible)** and reverted to `save_freq=200`.) Each 7B `actor+critic+optimizer` save
  ≈ **150 GB**; the verl default (keep all) once exhausted disk mid-run. `keep-last-2` bounds
  each stage to ~300 GB and leaves one known-good fallback if a save is corrupted mid-write.
  **Prune a completed arm to its final `gs492` only** (~165 GB) as soon as it finishes.
- **A save that dies on a full disk produces a corrupt checkpoint** (see §6). Watch free
  space; a save briefly holds ~3 checkpoints (write-then-prune) ≈ 450–540 GB transient.

## 5. Metrics & monitoring

**What counts as the A/B result (in priority order):**
1. **`critic/d_acc_fresh` band** — the *primary during-training readout*. Uncontaminated teacher-vs-current-student discriminator accuracy on fresh rows (see `GAD_B1_PROGRESS.md`/`INTEGRATION.md`). The replay hypothesis predicts a **tighter, higher band** (~[0.70,0.85]) vs. baseline's wide swing (~[0.55,0.95]); drift toward 0.5 = stale-dulling.
2. **LLM-judge win-rate** (Qwen2.5-72B chat, planned eval, §8) — the *actual student-quality verdict*, per the paper. This is the number that decides whether replay "wins."
3. **val rouge-L** — only a **cheap secondary sanity proxy** (longest-common-subsequence overlap of the student's output with the teacher reference). It's logged automatically each val, but correlates weakly with chat quality, so a rouge-L gap is **not** a result on its own — don't over-read it.

Tooling:
- `watch_fullscale.sh` → milestones to `logs/fs_watch.log` every 20 min; on completion prints per-arm `d_acc_fresh` bands + val rouge-L. It also has a **disk tripwire** — logs `free=…G ckpts=…G` each cycle and exits/alerts if FSx free < 2 TB or ckpts > 2.5 TB.
- Per-job stdout: `logs/{warmup,gad}-<jobid>.out` (grep `"Training Progress"` for step; `critic/d_acc_fresh:` for the metric — needs `PYTHONUNBUFFERED=1`, now set in the launchers, else it buffers until job exit).
- Hourly agent watch updates the `gad-sbatch-ab-campaign` memory and flags failures/disk.

## 6. Failed-arm recovery recipe

Failure mode already seen: a checkpoint save died on a full disk → the `global_step_N`
critic was missing its `extra_state_*.pt` files (unresumable), and with `save_freq=200`
there was no earlier checkpoint to fall back to. Recovery:

```bash
rm -rf ckpts/<EXP>/global_step_N            # remove the corrupt checkpoint
# resubmit the arm; empty ckpt dir => resume_mode=auto re-inits cleanly from merged warmup HF
WARMUP_EXP=fs33-warmup RESUME_STEP=latest EXP=<EXP> REPLAY_CAPACITY=<0|4096> \
TRAIN=.../lmsys_train_strat33.parquet sbatch run_gad_prod.sh
```

Sanity-check checkpoint integrity: each of `actor/` and `critic/` should have **24 `.pt`
files** (model + optim + extra_state, ×8 ranks). 16 = incomplete → corrupt.

Keep `ckpts/fs33-warmup/global_step_246` — it's the **shared re-init point** for both 33
arms and for iterating on the replay design after first results.

**Before declaring a job crashed, verify it — a frozen console log ≠ a dead job.** Ray's
log capture can stall (especially on jobs launched without `PYTHONUNBUFFERED=1`, whose
stdout is block-buffered) while training continues on the GPUs. Check, in order:
```bash
srun --jobid=<id> --overlap -N1 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ls -1dt ckpts/<EXP>/global_step_*/ | head -1 ; stat -c '%y' <that dir>   # recent = alive
```
High GPU util or a fresh checkpoint mtime ⇒ it's training; leave it alone. Only kill +
resume (from the last `global_step_N`) if GPUs are idle AND no recent checkpoint.

## 7. Current status (2026-07-27 ~16:34 UTC)

| Arm | Job | State | Progress |
|-----|-----|-------|----------|
| **33-replay** (cap 4096) | 1565446 | ✅ **COMPLETED** | 492/492; pruned to final `gs492`; val rouge-L **0.307** |
| **33-base** (cap 0) | 1569064 | RUNNING (h200-040-026) | ~step **299/492 (~61%)**, live `d_acc_fresh`, ~13h left; gs200 intact |
| **50-warmup** | 1571542 | RUNNING (**h200_dev**, h200-229-188) | ~step **169/374 (~45%)**, non-preemptible |
| **50-base** (cap 0) | 1571543 | PENDING (Dependency) | h200_dev; waits on 50-warmup |
| **50-replay** (cap 4096) | 1571544 | PENDING (Dependency) | h200_dev; waits on 50-warmup |

Notes:
- **33-replay finished clean** (ExitCode 0:0). Its buffered `d_acc_fresh` flushed on exit — full trajectory captured. `gs492` verified intact (24/24 `.pt`, 8 xstate), then pruned to gs492-only.
- **33-base** is the restart of 1565445 (which died on a disk-full checkpoint save). Runs with `PYTHONUNBUFFERED=1` → live metrics; its gs200 (the exact save the original corrupted on) landed **intact**.
- **50% chain was moved shared → `h200_dev`** after repeated preemption on shared; now non-preemptible, `save_freq=200`. IDs are the *current* ones above (earlier 1567822/823/824 were cancelled in the move).
- Disk healthy: ckpts ~521 GB, FSx ~13 TB free; tripwire armed.

### 7.1 Preliminary 33% read (steps 1–212 overlap; PRELIMINARY, n=1)

Comparing the two 33% arms over the steps both have completed (both resumed from the same `fs33-warmup` gs246, so it's a clean comparison):

| `d_acc_fresh`, steps 1–212 | base (cap 0) | replay (cap 4096) |
|---|---|---|
| mean | 0.815 | **0.864** |
| std (oscillation) | 0.173 | **0.118** |
| band | [0.051, 1.00] | [0.273, 1.00] |
| steps <0.5 / <0.7 | 15 / 37 | **6 / 15** |

- **Directional support for the replay hypothesis on discriminator stability:** replay's band is tighter + higher with far fewer instability dips (base even crashed to 0.05 once). Largest gap in the first ~140 steps; base converges toward replay (~0.89) by the last third.
- **Not a quality verdict:** val rouge-L is a wash (base ~0.32–0.34 vs replay 0.307 — a weak proxy anyway). Real read = win-rate at end.
- Confirm with: 33-base completion, the win-rate eval, and the 50% replicate.

## 8. Planned follow-ups

- **Replay sweep** (after first A/B results): `capacity ∈ {0, 1024, 4096, 16384}` ×
  `rho ∈ {0.25, 0.5}` × `strategy ∈ {uniform, recency-weighted (λ=1e-3)}`.
- **Eval**: Qwen2.5-72B chat win-rate (+ math accuracy for the multi-aspect line).

---
*Files: launchers `run_{warmup,gad}_prod.sh`; watcher `watch_fullscale.sh`; data
`data/lmsys_train_strat{33,50}.parquet`; checkpoints `ckpts/fs{33,50}-{warmup,gad-base,gad-replay}/`;
logs `logs/`. Related design docs: `GAD_B1_PROGRESS.md`, `GAD_B1_RUNBOOK_h200.md`,
`D3_MULTIASPECT_PLAN.md`.*
