#!/bin/bash
# B1 Stage 2 — GAD adversarial (7B, TP=1, 8xH200). Adapted from the reference
# gpt5-chat-filtered-7b-adversarial-lr1e-6.sh. Resumes from a warmup checkpoint:
# merges the FSDP actor+critic shards to HF format, then trains with critic_warmup=0.
# Checkout the gad branch first:  (cd verl && git checkout gad)
# Run from the LMOps/gad dir (merge_model2hf.py is under tools/). Pass --resume_step <N>.
set -x
source /home/saadlahrichi/gad_run/env.sh

export NCCL_TIMEOUT=36000
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1   # v1 engine: ~1.4x faster generation on real h200 nodes (validated). Use 0 only on the sandbox pod.
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

WORK=/home/saadlahrichi/gad_run
WARMUP_EXP=${WARMUP_EXP:-gpt5-chat-filtered-7b-warmup-lr1e-6}
EXP=${EXP:-gpt5-chat-filtered-7b-adversarial-lr1e-6}
RESUME_STEP=${RESUME_STEP:?set RESUME_STEP to the warmup global_step to resume from}
TRAIN=${TRAIN:-$WORK/data/lmsys_train-00000-of-00001.parquet}
VAL=${VAL:-$WORK/data/lmsys_test-00000-of-00001.parquet}
N_GPUS=${N_GPUS:-8}

CKPT=$WORK/ckpts/$WARMUP_EXP/global_step_${RESUME_STEP}
# merge FSDP shards -> HF for actor and critic (discriminator)
for role in actor critic; do
  mkdir -p $CKPT/$role/huggingface/
  find $CKPT/$role/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} $CKPT/$role/huggingface/ \;
  python tools/merge_model2hf.py --local_dir $CKPT/$role
  echo "Files in $CKPT/$role/huggingface:"; ls $CKPT/$role/huggingface
done
model_path=$CKPT/actor/huggingface
reward_model_path=$CKPT/critic/huggingface

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
    actor_rollout_ref.model.path=$model_path \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$reward_model_path \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=gad_b1 \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=$WORK/ckpts/$EXP
