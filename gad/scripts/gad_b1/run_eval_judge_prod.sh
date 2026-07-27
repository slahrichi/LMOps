#!/bin/bash
#SBATCH --qos=h200_mrs_2_high
#SBATCH --account=mrs_2
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=gad-judge
#SBATCH --output=/home/saadlahrichi/gad_run/logs/judge-%j.out
# GAD B1 EVAL stage-2: LLM-judge win-rate (Qwen2.5-72B, TP=2). Reuses the eval-branch judge
# code (get_online_transform_func + extract_judge) via judge_winrate.py. Reads the arm's
# {set}_generation_results.jsonl (produced by run_eval_gen_prod.sh) and writes winrate_*.json.
#   Usage: EXP=fs33-gad-replay STEP=492 REFERENCE=judge sbatch run_eval_judge_prod.sh
#   Vars: GEN_DIR, SETS="lmsys,dolly,vicuna,self-inst", REFERENCE={judge|teacher}
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
LMOPS=/home/saadlahrichi/LMOps
VERL=$LMOPS/gad/verl
EXP=${EXP:?set EXP (e.g. fs33-gad-replay)}
STEP=${STEP:?set STEP (e.g. 492)}
GEN_DIR=${GEN_DIR:-$WORKDIR/eval/$EXP/global_step_${STEP}}
SETS=${SETS:-lmsys,dolly,vicuna,self-inst}
REFERENCE=${REFERENCE:-judge}
JUDGE=${JUDGE:-/storage/home/saadlahrichi/models/Qwen2.5-72B-Instruct}

echo "===== env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp HF_HOME=$WORKDIR/hf
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR $WORKDIR/tmp $WORKDIR/logs
ulimit -n 65536 || true

echo "===== verl branch = eval (for get_online_transform_func) ====="
cd $VERL && git checkout eval 2>&1 | tail -1
echo "verl branch: $(git branch --show-current)"

echo "===== judge win-rate | gen_dir=$GEN_DIR sets=$SETS ref=$REFERENCE ====="
cd $LMOPS/gad   # so `import deepscaler` (local pkg) + `import verl` (pip) both resolve
python $WORKDIR/judge_winrate.py \
    --gen-dir "$GEN_DIR" \
    --sets "$SETS" \
    --reference "$REFERENCE" \
    --judge-model "$JUDGE" \
    --tp 2 \
    --out "$GEN_DIR/winrate_${REFERENCE}.json"
echo ""
echo "===== done ====="; cat "$GEN_DIR/winrate_${REFERENCE}.json" 2>/dev/null