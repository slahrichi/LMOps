#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=gad-ab-v1
#SBATCH --output=/home/%u/gad_run/logs/ab-v1-%j.out
# 3-STEP SMOKE TEST for the GAD warmup pipeline, adapted from haixuma's smoke
# script to Saad's environment (venv instead of conda; verl at ~/LMOps/gad/verl;
# staged 7B model; real teacher parquet; paths off /tmp). Verifies the full stack
# end-to-end before the 800-step warmup.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # sandbox proxies don't exist on compute nodes

WORKDIR=${GAD_WORK:-$HOME/gad_run}
LMOPS=${LMOPS:-$HOME/LMOps}
VERL=$LMOPS/gad/verl
MODEL=/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct

echo "===== 1/4: activate environment ====="
source "$WORKDIR/env.sh"
# needed exports from env.sh (big-disk TMPDIR + triton Python.h headers); NOT the sandbox proxies
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp
export HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
mkdir -p $TMPDIR
python -c "import torch, vllm, verl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl OK')"
echo ""

echo "===== 2/4: prepare patched launch script ====="
mkdir -p $WORKDIR/scripts
cp $LMOPS/gad/scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh \
   $WORKDIR/scripts/smoke_warmup.sh
# Repoint the reference script's /tmp data + checkpoint paths at Saad's real files.
sed -i \
  -e "s|/tmp/lmsys_gpt5_chat_filtered_train.parquet|$WORKDIR/data/lmsys_train-00000-of-00001.parquet|g" \
  -e "s|/tmp/lmsys_gpt5_chat_filtered_test.parquet|$WORKDIR/data/lmsys_test-00000-of-00001.parquet|g" \
  -e "s|/tmp/\${EXP_NAME}|$WORKDIR/checkpoints/\${EXP_NAME}|g" \
  $WORKDIR/scripts/smoke_warmup.sh
echo "  Patched paths:"
grep -n "$WORKDIR" $WORKDIR/scripts/smoke_warmup.sh | head -5
echo ""

echo "===== 3/4: verl branch (NO switch — preserve uncommitted D2 work) ====="
cd $VERL
echo "  branch: $(git branch --show-current)"
git status --short
git log --oneline -1
echo ""

echo "===== 4/4: launch 3-step A/B smoke (variant D: TP=1, gpu_mem 0.85, vLLM v1) ====="
# EXPERIMENT: enable the vLLM v1 engine. The runbook set VLLM_USE_V1=0 on the sandbox
# (v1 broke on 0.8.5 configs there); testing whether it works — and is faster — on a real node.
export VLLM_USE_V1=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=36000
# --- persist Ray logs to NFS so we can debug after the job exits ---
# RAY_TMPDIR must be SHORT (Ray adds ~70 char session subdirs; AF_UNIX cap is 107).
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}
mkdir -p $RAY_TMPDIR
# On exit (success or failure), copy Ray session logs to NFS for post-mortem
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
export RAY_BACKEND_LOG_LEVEL=info
# --- raise FD limit; Ray/vLLM open lots of sockets ---
ulimit -n 65536 || true
ulimit -n
cd $LMOPS/gad   # GAD stage needs tools/merge_model2hf.py; run from here per runbook
bash $WORKDIR/scripts/smoke_warmup.sh \
  --model $MODEL \
  --reward_model $MODEL \
  --exp_name ab-v1 \
  --nnodes 1 \
  trainer.total_training_steps=3 \
  trainer.val_before_train=False \
  trainer.test_freq=100000 \
  trainer.save_freq=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  "trainer.logger=[console]" \
  +data.dataloader_num_workers=0 \
  ray_init.num_cpus=32
echo ""
echo "===== per-step timing (this run) ====="
grep -aoE "timing_s/step:[0-9.]+|timing_s/gen:[0-9.]+|perf/throughput:[0-9.]+" "${SLURM_JOB_ID:+$WORKDIR/logs/ab-v1-${SLURM_JOB_ID}.out}" 2>/dev/null | tail -12 || true
echo "DONE"
