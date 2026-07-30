#!/bin/bash
#SBATCH --qos=h200_mrs_2_high
#SBATCH --account=mrs_2
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=gad-judge
#SBATCH --output=/home/saadlahrichi/gad_run/logs/judge-%j.out
# GAD B1 EVAL stage-2: paper-faithful automatic score (Qwen2.5-72B, TP=2). Implements the
# paper App. A.3 GPT-4o eval (Fig-8 dual 1-10 scores; metric = student/(student+reference))
# via judge_winrate.py. Reads {set}_generation_results.jsonl and writes score_*.json.
#   Usage: EXP=fs33-gad-replay STEP=492 REFERENCE=judge TEACHER_CEILING=1 sbatch run_eval_judge_prod.sh
#   Vars: GEN_DIR, SETS="lmsys,dolly,vicuna,self-inst", REFERENCE={judge|teacher}, TEACHER_CEILING={0|1}
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
EXP=${EXP:?set EXP (e.g. fs33-gad-replay)}
STEP=${STEP:?set STEP (e.g. 492)}
GEN_DIR=${GEN_DIR:-$WORKDIR/eval/$EXP/global_step_${STEP}}
SETS=${SETS:-lmsys,dolly,vicuna,self-inst}
REFERENCE=${REFERENCE:-judge}
TEACHER_CEILING=${TEACHER_CEILING:-0}   # 1 = also score teacher_response vs judge ref (GPT-5 ceiling; meaningful on lmsys only)
JUDGE=${JUDGE:-/storage/home/saadlahrichi/models/Qwen2.5-72B-Instruct}

echo "===== env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp HF_HOME=$WORKDIR/hf
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 NCCL_TIMEOUT=36000
export RAY_TMPDIR=/tmp/rj${SLURM_JOB_ID}; mkdir -p $RAY_TMPDIR $WORKDIR/tmp $WORKDIR/logs
ulimit -n 65536 || true

echo "===== paper score | gen_dir=$GEN_DIR sets=$SETS ref=$REFERENCE teacher_ceiling=$TEACHER_CEILING ====="
python $WORKDIR/judge_winrate.py \
    --gen-dir "$GEN_DIR" \
    --sets "$SETS" \
    --reference "$REFERENCE" \
    --judge-model "$JUDGE" \
    --tp 2 \
    --data-dir "$WORKDIR/data" \
    $([ "$TEACHER_CEILING" = "1" ] && echo --teacher-ceiling) \
    --out "${OUT:-$GEN_DIR/score_${REFERENCE}.json}"
echo ""
echo "===== done ====="; cat "$GEN_DIR/score_${REFERENCE}.json" 2>/dev/null