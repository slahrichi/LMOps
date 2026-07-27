#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=gad-seqkd
#SBATCH --output=/home/saadlahrichi/gad_run/logs/seqkd-%j.out
# GAD B1 — SeqKD baseline (paper's sequence-level KD). Teacher-forcing SFT of the BASE 7B
# student on GPT-5-Chat teacher responses (compute_sft_loss on the `seqkd` branch: no
# warmup, no discriminator/critic, no GRPO reward). This is the baseline GAD must beat.
# SLURM-native prod launcher: TP=1 + vLLM v1 (rollout used only for val generation), no
# sandbox proxies/NCCL hacks, PYTHONUNBUFFERED, keep-last-2, resume_mode=auto.
#   Usage: EXP=fs33-seqkd TRAIN=$PWD/data/lmsys_train_strat33.parquet sbatch run_seqkd_prod.sh
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
MODEL=/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct   # SeqKD trains the BASE student (no warmup)
EXP=${EXP:-fs33-seqkd}
TRAIN=${TRAIN:-$WORKDIR/data/lmsys_train_strat33.parquet}
VAL=${VAL:-$WORKDIR/data/lmsys_test-00000-of-00001.parquet}
CKPT_ROOT=${CKPT_ROOT:-$WORKDIR/ckpts}   # override to /checkpoints/saadlahrichi/gad_run/ckpts (Lustre, no 1TB quota; writable from compute)
CKPT=$CKPT_ROOT/$EXP
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-4}

echo "===== 1/3: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp
export HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
mkdir -p $TMPDIR $CKPT $WORKDIR/logs
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1   # stream metric lines live (else Ray buffers stdout until exit)
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true
python -c "import torch, vllm, verl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl OK')"

echo "===== 2/3: verl branch = seqkd (teacher-forcing SFT) ====="
cd $VERL
git checkout seqkd 2>&1 | tail -2
echo "verl branch: $(git branch --show-current)  ($(git log --oneline -1))"

echo "===== 3/3: launch SeqKD (SFT on teacher responses) | lr=$LR epochs=$EPOCHS train=$TRAIN ====="
cd $LMOPS/gad
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
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.val_before_train=True \
    trainer.critic_warmup=10 \
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
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$CKPT \
    ray_init.num_cpus=32 \
    ${EXTRA_ARGS:-}
echo ""
echo "===== done; checkpoints ====="; ls -1 $CKPT 2>&1 | tail