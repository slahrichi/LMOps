#!/bin/bash
#SBATCH --qos=h200_mrs_shared
#SBATCH --account=mrs_2
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=gad-oodscore
#SBATCH --output=/home/saadlahrichi/gad_run/logs/oodscore-%j.out
# GAD B1 EVAL OOD-score: run the 5 verifiable OOD scorers over a FINISHED arm's generation
# dir and write <bench>_score.{txt,json}. Pure-CPU scorers (math_verify / code-exec / rule-check
# / letter-match); the 1 GPU is only to satisfy the GPU-partition scheduler. Idempotent &
# re-runnable — safe to resubmit without regenerating. Fail-LOUD: a missing gen file or a
# scorer crash/row-mismatch makes the job exit non-zero (so an afterok chain surfaces it).
#   Usage: EXP=fs50-gad-replay STEP=748 sbatch run_eval_ood_score.sh
#   Vars: GEN_DIR (default eval/$EXP/global_step_$STEP), OOD_SETS="gsm8k math500 humaneval ifeval mmlu"
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR=/home/saadlahrichi/gad_run
NB=$WORKDIR/followups/nonchat
EXP=${EXP:?set EXP (e.g. fs50-gad-replay)}
STEP=${STEP:?set STEP (e.g. 748)}
GEN_DIR=${GEN_DIR:-$WORKDIR/eval/$EXP/global_step_${STEP}}
OOD_SETS=${OOD_SETS:-"gsm8k math500 humaneval ifeval mmlu"}

echo "===== env ====="
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR/tmp TEMP=$WORKDIR/tmp TMP=$WORKDIR/tmp HF_HOME=$WORKDIR/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p $WORKDIR/tmp $WORKDIR/logs
[ -d "$GEN_DIR" ] || { echo "ERROR: GEN_DIR not found: $GEN_DIR" >&2; exit 1; }

echo "===== OOD score | $EXP@$STEP | sets='$OOD_SETS' | dir=$GEN_DIR ====="
fail=0
for b in $OOD_SETS; do
  scorer=$NB/score_$b.py
  gen=$GEN_DIR/${b}_generation_results.jsonl
  [ -f "$scorer" ] || { echo "!! no scorer for '$b' ($scorer)" >&2; fail=1; continue; }
  [ -f "$gen" ]    || { echo "!! MISSING gen for '$b' ($gen) -- gen step incomplete?" >&2; fail=1; continue; }
  echo "----- scoring $b -----"
  # tee stdout to <bench>_score.txt; JSON to <bench>_score.json; stderr to <bench>_score.err.
  # pipefail is set, so a scorer non-zero exit propagates through the tee and is caught.
  if python3 "$scorer" --gen "$gen" --out "$GEN_DIR/${b}_score.json" \
       2>"$GEN_DIR/${b}_score.err" | tee "$GEN_DIR/${b}_score.txt"; then
    :
  else
    echo "!! SCORER FAILED: $b (see $GEN_DIR/${b}_score.err)" >&2; fail=1
  fi
done

echo ""
echo "===== summary ($EXP@$STEP) ====="
for b in $OOD_SETS; do
  [ -f "$GEN_DIR/${b}_score.txt" ] && printf "  %-10s %s\n" "$b" "$(head -1 "$GEN_DIR/${b}_score.txt")"
done
[ $fail -eq 0 ] || { echo "ERROR: one or more OOD scorers failed/missing" >&2; exit 1; }
echo "===== done ====="
