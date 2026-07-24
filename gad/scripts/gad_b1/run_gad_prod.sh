#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --job-name=gad-adv
#SBATCH --output=/home/saadlahrichi/gad_run/logs/gad-%j.out
# GAD B1 Stage-2 ADVERSARIAL — PRODUCTION launcher (7B, 8xH200, single node).
# SLURM-native rewrite of run_b1_gad.sh: TP=1 + vLLM v1 (winning config), no sandbox
# proxies/NCCL hacks, inline config, resume_mode=auto. Runs on the gad-d2-replay branch
# so the D2 discriminator replay buffer is active (critic.replay.*).
# Resumes from a warmup checkpoint: merges FSDP actor+critic shards -> HF, then trains
# with critic_warmup=0 for 2 epochs.
#   REPLAY_CAPACITY=0    -> exact vanilla-GAD baseline
#   REPLAY_CAPACITY=1024 -> replay ON (default here)
# Usage:
#   WARMUP_EXP=mini-warmup RESUME_STEP=12 EXP=mini-gad-replay \
#   TRAIN=$WORK/data/mini_train.parquet VAL=$WORK/data/lmsys_test-00000-of-00001.parquet \
#   EXTRA_ARGS="trainer.save_freq=8 trainer.test_freq=8" sbatch run_gad_prod.sh
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
MODEL=/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct
WARMUP_EXP=${WARMUP_EXP:-gpt5-chat-filtered-7b-warmup-lr1e-6}
EXP=${EXP:-gpt5-chat-filtered-7b-adversarial-lr1e-6}
RESUME_STEP=${RESUME_STEP:?set RESUME_STEP to the warmup global_step to merge/resume from}
TRAIN=${TRAIN:-$WORKDIR/data/lmsys_train-00000-of-00001.parquet}
VAL=${VAL:-$WORKDIR/data/lmsys_test-00000-of-00001.parquet}
CKPT=$WORKDIR/ckpts/$EXP
REPLAY_CAPACITY=${REPLAY_CAPACITY:-1024}
REPLAY_RHO=${REPLAY_RHO:-0.5}
REPLAY_STRATEGY=${REPLAY_STRATEGY:-uniform}

echo "===== 1/4: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp
export HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
mkdir -p $TMPDIR $CKPT $WORKDIR/logs
export VLLM_USE_V1=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true
python -c "import torch, vllm, verl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl OK')"

echo "===== 2/4: verl branch = gad-d2-replay (D2 replay active) ====="
cd $VERL
git checkout gad-d2-replay 2>&1 | tail -2
echo "verl branch: $(git branch --show-current)  ($(git log --oneline -1))"

echo "===== 3/4: merge warmup checkpoint (FSDP shards -> HF) ====="
WCKPT=$WORKDIR/ckpts/$WARMUP_EXP/global_step_${RESUME_STEP}
[ -d "$WCKPT" ] || { echo "ERROR: warmup checkpoint $WCKPT not found"; exit 1; }
cd $LMOPS/gad
for role in actor critic; do
  if [ -f "$WCKPT/$role/huggingface/model.safetensors" ] || ls "$WCKPT/$role/huggingface"/*.safetensors >/dev/null 2>&1; then
    echo "  $role already merged, skipping"
  else
    mkdir -p $WCKPT/$role/huggingface/
    find $WCKPT/$role/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} $WCKPT/$role/huggingface/ \;
    python tools/merge_model2hf.py --local_dir $WCKPT/$role
  fi
  echo "  $role/huggingface:"; ls $WCKPT/$role/huggingface | head
done
MODEL_PATH=$WCKPT/actor/huggingface
REWARD_PATH=$WCKPT/critic/huggingface

echo "===== 4/4: launch GAD adversarial (2 epochs) | replay capacity=$REPLAY_CAPACITY rho=$REPLAY_RHO ====="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=$TRAIN \
    data.val_files=$VAL \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$REWARD_PATH \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    critic.replay.capacity=$REPLAY_CAPACITY \
    critic.replay.rho=$REPLAY_RHO \
    critic.replay.strategy=$REPLAY_STRATEGY \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=gad_b1 \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.test_freq=50 \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$CKPT \
    ray_init.num_cpus=32 \
    ${EXTRA_ARGS:-}
echo ""
echo "===== done; checkpoints ====="; ls -1 $CKPT 2>&1 | tail
