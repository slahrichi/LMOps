#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=gad-evalgen
#SBATCH --output=/home/saadlahrichi/gad_run/logs/evalgen-%j.out
# GAD B1 EVAL stage-1: generate student outputs for a FINISHED arm (verl `eval` branch,
# val_only). Merges the arm's actor FSDP shards -> HF, then generates on an eval set and
# dumps {val_data}generation_results.jsonl. Judge/win-rate scoring is a separate step
# (judge_winrate.py, Qwen2.5-72B). Mirrors the paper's scripts/generate/generate.sh.
#   Usage: EXP=fs33-gad-replay STEP=492 VAL_DATA=lmsys \
#          VAL=$PWD/data/lmsys_test-00000-of-00001.parquet sbatch run_eval_gen_prod.sh
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
EXP=${EXP:?set EXP (e.g. fs33-gad-replay / fs33-gad-base / fs33-seqkd)}
STEP=${STEP:?set STEP (checkpoint global_step, e.g. 492)}
VAL_DATA=${VAL_DATA:-lmsys}
VAL=${VAL:-$WORKDIR/data/lmsys_test-00000-of-00001.parquet}
CKPT=$WORKDIR/ckpts/$EXP/global_step_${STEP}
OUTDIR=$WORKDIR/eval/$EXP/global_step_${STEP}

echo "===== 1/3: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR $TMPDIR $OUTDIR $WORKDIR/logs
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true

echo "===== 2/3: verl branch = eval + merge actor FSDP->HF ====="
cd $VERL && git checkout eval 2>&1 | tail -1
echo "verl branch: $(git branch --show-current)"
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

echo "===== 3/3: generate on '$VAL_DATA' (val_only, n=1) -> $OUTDIR ====="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=$VAL \
    data.val_files=$VAL \
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
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
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
echo ""
echo "===== done; generations ====="; ls -la $OUTDIR/*generation_results.jsonl 2>&1 | tail