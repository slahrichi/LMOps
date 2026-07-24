#!/bin/bash
#SBATCH --qos=h200_dev
#SBATCH --account=mrs_2
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --job-name=gad-warmup
#SBATCH --output=/home/saadlahrichi/gad_run/logs/warmup-%j.out
# GAD B1 Stage-1 WARMUP — PRODUCTION launcher (7B, 8xH200, single node).
# Winning config from the A-E sweep: TP=1 + vLLM v1 engine (~212 s/step steady,
# ~2x faster than the TP=2/v0 reference). Discriminator stays fp32 = paper-faithful.
# Data: ytz20/LMSYS-Chat-GPT-5-Chat-Response (paper's teacher set). One epoch = 750 steps.
# Self-contained (config inline, no NFS intermediate script) + resume_mode=auto:
# if the job dies (node failure/preempt), just resubmit — it picks up from the last ckpt.
#   Full run:  sbatch run_warmup_prod.sh
#   Dry run :  EXTRA_ARGS="trainer.total_training_steps=3 trainer.save_freq=2" sbatch run_warmup_prod.sh
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # sandbox proxies don't exist on compute nodes

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
MODEL=/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct
EXP=${EXP:-gpt5-chat-filtered-7b-warmup-lr1e-6}
CKPT=$WORKDIR/ckpts/$EXP

echo "===== 1/3: env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp
export HF_HOME=$WORKDIR/hf
export C_INCLUDE_PATH=$WORKDIR/pyinclude:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$WORKDIR/pyinclude:${CPLUS_INCLUDE_PATH:-}
mkdir -p $TMPDIR $CKPT $WORKDIR/logs
python -c "import torch, vllm, verl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl OK')"

# --- winning perf config ---
export VLLM_USE_V1=1                 # v1 engine: cut generation ~118s -> ~85s (validated variant D)
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR
trap 'cp -r $RAY_TMPDIR $WORKDIR/logs/ray-${SLURM_JOB_ID} 2>/dev/null || true' EXIT
ulimit -n 65536 || true

echo "===== 2/3: verl branch = warmup (paper-faithful stage 1) + resume state ====="
cd $VERL
git checkout warmup 2>&1 | tail -2      # stage-1 code; D2 replay (gad-d2-replay) is a stage-2 concern
echo "verl branch: $(git branch --show-current)  ($(git log --oneline -1))"
CUR=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
echo "resume from checkpoint step: $CUR  (resume_mode=auto; ckpt dir: $CKPT)"

echo "===== 3/3: launch warmup (1 epoch = 750 steps) ====="
cd $LMOPS/gad   # GAD stage needs tools/merge_model2hf.py; run from here per runbook
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=$WORKDIR/data/lmsys_train-00000-of-00001.parquet \
    data.val_files=$WORKDIR/data/lmsys_test-00000-of-00001.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL \
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
    critic.model.path=$MODEL \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=10 \
    "trainer.logger=[console]" \
    trainer.project_name=gad_b1 \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$CKPT \
    ray_init.num_cpus=32 \
    ${EXTRA_ARGS:-}
echo ""
echo "===== done; checkpoints ====="
ls -1 $CKPT 2>&1 | tail -8
