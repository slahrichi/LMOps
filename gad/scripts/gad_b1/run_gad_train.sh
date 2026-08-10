#!/bin/bash
#SBATCH --qos=h200_mrs_2_high
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --job-name=gad-adv
#SBATCH --output=/home/saadlahrichi/gad_run/logs/gad-%j.out
# GAD B1 Stage-2 ADVERSARIAL — FLEET launcher (hardened variant of run_gad_prod.sh).
# Differences vs run_gad_prod.sh (kept separate so it can run concurrently in a fleet
# without editing the launcher the live 100% jobs are using):
#   1. RACE-HARDENED git checkout: skip if the shared verl tree is already on
#      gad-d2-replay (the common case for a fleet); retry with backoff on .git/index.lock
#      instead of dying (fixes the co-launch index.lock race).
#   2. SEED knob (referee A4 cross-seed): SEED=<n> threads the data-shuffle seed
#      (+data.seed), the vLLM rollout sampling seed (+actor_rollout_ref.rollout.seed),
#      and the replay sampling/eviction seed (critic.replay.seed). All three are read by
#      verl via .get() (main_ppo.py:254, vllm_rollout_spmd.py:166, config critic.replay.seed).
#   REPLAY_CAPACITY=0 -> exact vanilla-GAD baseline.
# Usage: SEED=1 EXP=... WARMUP_EXP=... RESUME_STEP=latest REPLAY_* ... sbatch run_gad_train.sh
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
CKPT_ROOT=${CKPT_ROOT:-$WORKDIR/ckpts}
CKPT=$CKPT_ROOT/$EXP
REPLAY_CAPACITY=${REPLAY_CAPACITY:-1024}
REPLAY_RHO=${REPLAY_RHO:-0.5}
REPLAY_STRATEGY=${REPLAY_STRATEGY:-uniform}
SEED=${SEED:-}

# SEED knob: only injected when SEED is set, so unset => byte-identical to run_gad_prod.sh defaults.
SEED_ARGS=""
if [ -n "$SEED" ]; then
  SEED_ARGS="+data.seed=$SEED +actor_rollout_ref.rollout.seed=$SEED critic.replay.seed=$SEED"
  echo "SEED=$SEED -> $SEED_ARGS"
fi

if [ "$RESUME_STEP" = "latest" ]; then
  RESUME_STEP=$(ls -1d $WORKDIR/ckpts/$WARMUP_EXP/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
  [ -n "$RESUME_STEP" ] || { echo "ERROR: no global_step_* checkpoint in $WORKDIR/ckpts/$WARMUP_EXP"; exit 1; }
  echo "resolved RESUME_STEP=latest -> $RESUME_STEP"
fi

echo "===== 1/4: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp
export HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
mkdir -p $TMPDIR $CKPT $WORKDIR/logs
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true
python -c "import torch, vllm, verl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl OK')"

echo "===== 2/4: verl branch = gad-d2-replay (race-hardened) ====="
cd $VERL
# Fleet-safe: shared tree is typically ALREADY on gad-d2-replay (another job put it there),
# so skip the checkout entirely (no index.lock touch). Only checkout if on another branch,
# retrying on lock contention rather than dying.
for i in $(seq 1 40); do
  cur=$(git symbolic-ref --short -q HEAD || echo DETACHED)
  [ "$cur" = "gad-d2-replay" ] && break
  if git checkout gad-d2-replay >/tmp/co.$$.$SLURM_JOB_ID 2>&1; then break; fi
  if grep -qiE "index.lock|another git process" /tmp/co.$$.$SLURM_JOB_ID; then
    echo "  checkout contended (attempt $i), backing off"; sleep $(( (RANDOM % 6) + 3 ))
  else
    echo "  checkout failed:"; cat /tmp/co.$$.$SLURM_JOB_ID; exit 1
  fi
done
rm -f /tmp/co.$$.$SLURM_JOB_ID
[ "$(git symbolic-ref --short -q HEAD || echo DETACHED)" = "gad-d2-replay" ] || { echo "ERROR: verl not on gad-d2-replay after retries"; exit 1; }
echo "verl branch: $(git branch --show-current)  ($(git log --oneline -1))"

echo "===== 3/4: merge warmup checkpoint (FSDP shards -> HF) ====="
WCKPT=$WORKDIR/ckpts/$WARMUP_EXP/global_step_${RESUME_STEP}
[ -d "$WCKPT" ] || { echo "ERROR: warmup checkpoint $WCKPT not found"; exit 1; }
cd $LMOPS/gad
exec 9>"$WORKDIR/ckpts/$WARMUP_EXP/.merge.lock"
flock 9
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
flock -u 9
MODEL_PATH=$WCKPT/actor/huggingface
REWARD_PATH=$WCKPT/critic/huggingface

echo "===== 4/4: launch GAD adversarial (2 epochs) | cap=$REPLAY_CAPACITY rho=$REPLAY_RHO strat=$REPLAY_STRATEGY seed=${SEED:-default} ====="
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
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.max_actor_ckpt_to_keep=${KEEP:-10} \
    trainer.max_critic_ckpt_to_keep=${KEEP:-10} \
    trainer.test_freq=50 \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$CKPT \
    ray_init.num_cpus=32 \
    ${SEED_ARGS} \
    ${EXTRA_ARGS:-}
echo ""
echo "===== done; checkpoints ====="; ls -1 $CKPT 2>&1 | tail
