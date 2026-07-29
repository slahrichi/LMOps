#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --job-name=gad-evalgen
#SBATCH --output=/home/saadlahrichi/gad_run/logs/evalgen-%j.out
# GAD B1 EVAL stage-1: generate student outputs for a FINISHED arm on the 4 benchmark sets
# (verl `eval` branch, main_ppo val_only — matches scripts/generate/generate.sh: n=8, temp
# 0.8). Merges the arm's actor FSDP shards -> HF once, then loops the sets, dumping
# {set}_generation_results.jsonl to OUTDIR. Win-rate scoring is a separate step
# (judge_winrate.py, Qwen2.5-72B).
#   Usage: CKPT_ROOT=/checkpoints/$USER/gad_run/ckpts EXP=fs33-gad-replay STEP=492 \
#          sbatch run_eval_gen_prod.sh
#   Vars: VAL_SETS="lmsys dolly vicuna self-inst" (default all 4), N=8, GEN_TEMP=0.8
#   MODEL_PATH=<hf dir> to eval an external/pre-distillation model directly (skips FSDP merge),
#     e.g. base Qwen2.5-7B-Instruct: EXP=base-qwen7b STEP=0 MODEL_PATH=/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
EXP=${EXP:?set EXP (e.g. fs33-gad-replay / fs33-gad-base / fs33-seqkd)}
STEP=${STEP:?set STEP (checkpoint global_step, e.g. 492)}
CKPT_ROOT=${CKPT_ROOT:-$WORKDIR/ckpts}          # Lustre for prod arms: /checkpoints/$USER/gad_run/ckpts
VAL_SETS=${VAL_SETS:-"lmsys dolly vicuna self-inst"}
N=${N:-8}
GEN_TEMP=${GEN_TEMP:-0.8}   # NB: not TEMP — the env block below exports TEMP=$WORKDIR/tmp (tempdir), which would clobber it
CKPT=$CKPT_ROOT/$EXP/global_step_${STEP}
OUTDIR=$WORKDIR/eval/$EXP/global_step_${STEP}

# label -> eval parquet
valfile() { case "$1" in
  lmsys)     echo "$WORKDIR/data/lmsys_test-00000-of-00001.parquet" ;;
  dolly)     echo "$WORKDIR/data/dolly_test.parquet" ;;
  vicuna)    echo "$WORKDIR/data/vicuna_test.parquet" ;;
  self-inst) echo "$WORKDIR/data/self-inst_test.parquet" ;;
  gsm8k)     echo "$WORKDIR/data/gsm8k_test.parquet" ;;
  math500)   echo "$WORKDIR/data/math500_test.parquet" ;;
  humaneval) echo "$WORKDIR/data/humaneval_test.parquet" ;;
  *) echo "" ;; esac; }

echo "===== 1/3: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR $WORKDIR/tmp $OUTDIR $WORKDIR/logs
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true

echo "===== 2/3: verl branch = eval + resolve model ====="
cd $VERL && git checkout eval 2>&1 | tail -1
echo "verl branch: $(git branch --show-current)"
if [ -n "${MODEL_PATH:-}" ]; then
  # direct HF model (e.g. pre-distillation base Qwen2.5-7B-Instruct) — no FSDP merge
  echo "  using direct MODEL_PATH=$MODEL_PATH (no merge)"
else
  [ -d "$CKPT/actor" ] || { echo "ERROR: $CKPT/actor not found"; exit 1; }
  cd $LMOPS/gad
  if ls "$CKPT/actor/huggingface"/*.safetensors >/dev/null 2>&1; then
    echo "  actor already merged, skipping"
  else
    mkdir -p "$CKPT/actor/huggingface/"
    find "$CKPT/actor/" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$CKPT/actor/huggingface/" \;
    python tools/merge_model2hf.py --local_dir "$CKPT/actor"
  fi
  MODEL_PATH=$CKPT/actor/huggingface
fi
cd $LMOPS/gad

echo "===== 3/3: generate | sets='$VAL_SETS' | n=$N temp=$GEN_TEMP -> $OUTDIR ====="
for VAL_DATA in $VAL_SETS; do
  VAL=$(valfile "$VAL_DATA")
  [ -n "$VAL" ] && [ -f "$VAL" ] || { echo "SKIP $VAL_DATA: parquet not found ($VAL)"; continue; }
  echo "----- generating: $VAL_DATA ($VAL) -----"
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.prompt_key=content \
      data.train_files=$WORKDIR/data/lmsys_test-00000-of-00001.parquet \
      data.val_files=$VAL \
      data.validation_shuffle=False \
      data.train_batch_size=256 \
      data.val_batch_size=600 \
      data.max_prompt_length=2048 \
      data.max_response_length=1536 \
      data.truncation=right \
      +data.dataloader_num_workers=0 \
      actor_rollout_ref.model.path=$MODEL_PATH \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.grad_clip=0.2 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=256 \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.entropy_coeff=0.0 \
      actor_rollout_ref.actor.kl_loss_coef=0.0 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.temperature=$GEN_TEMP \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
      actor_rollout_ref.rollout.n=$N \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.kl_ctrl.kl_coef=0.0 \
      +trainer.val_data=$VAL_DATA \
      trainer.val_only=True \
      trainer.val_before_train=True \
      trainer.critic_warmup=0 \
      "trainer.logger=[console]" \
      trainer.project_name=gad_b1_eval \
      trainer.experiment_name=$EXP \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=1 \
      trainer.resume_mode=disable \
      trainer.default_hdfs_dir=null \
      trainer.total_epochs=1 \
      trainer.validation_data_dir=$OUTDIR \
      ray_init.num_cpus=32 \
      ${EXTRA_ARGS:-}
done
echo ""
echo "===== done; generations in $OUTDIR ====="; ls -la $OUTDIR/*generation_results.jsonl 2>&1 | tail