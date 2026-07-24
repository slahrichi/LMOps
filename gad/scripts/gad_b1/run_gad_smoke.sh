#!/bin/bash
# GAD smoke run — validates the adversarial (gad-branch) machinery on a tiny model.
# Staged Qwen2.5-0.5B for BOTH actor (student/generator) and critic (discriminator).
# Synthetic 2-col parquet (content, teacher_response). Console logger, no wandb, no resume.
set -x
source /home/saadlahrichi/gad_run/env.sh

export NCCL_TIMEOUT=36000
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=0   # vllm 0.8.5 spmd rollout path used by this verl fork
# container networking: only lo/tap0 exist, /dev/shm is tiny (63M)
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1      # /dev/shm too small; use NVLink P2P / socket instead
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

MODEL=/checkpoints/xinyulin/pretrained_models/Qwen2.5-0.5B
WORK=/home/saadlahrichi/gad_run
EXP=gad_smoke_0p5b
N_GPUS=${N_GPUS:-2}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=$WORK/data/smoke_train.parquet \
    data.val_files=$WORK/data/smoke_val.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$MODEL \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.critic_warmup=2 \
    trainer.logger=['console'] \
    trainer.project_name=gad_smoke \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$WORK/ckpts/$EXP
