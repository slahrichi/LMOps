#!/bin/bash
# GAD B1 EVAL batch driver (login-side submitter, NOT an sbatch job).
# For each arm "EXP:STEP", submits: (1) generation over all chat+OOD sets, then
# (2) the Qwen-72B chat judge and (3) the verifiable OOD scorers, both gated afterok on the gen.
# Prints every job id so the chain is auditable. Re-running is safe: gen overwrites its own
# jsonl, and the judge / OOD-score jobs are idempotent (re-runnable standalone if a dep breaks).
#
#   Usage:  ARMS="fs50-gad-base:748 fs50-gad-replay:748" bash run_eval_batch.sh
#   Vars:
#     ARMS          (required) space-separated EXP:STEP list
#     CKPT_ROOT     default /checkpoints/saadlahrichi/gad_run/ckpts (Lustre)
#     CHAT_SETS     default "lmsys dolly vicuna self-inst"
#     OOD_SETS      default "gsm8k math500 humaneval ifeval mmlu"
#     JUDGE_SETS    default "lmsys,dolly,vicuna,self-inst"  (comma-sep; judge_winrate.py)
#     REFERENCE     default judge
#     GEN_QOS       default h200_mrs_2_high   (8-GPU gen; priority-100 when group has room)
#     SCORE_QOS     default h200_mrs_shared   (2-GPU judge + 1-GPU OOD score)
#     DRY_RUN=1     print the sbatch commands without submitting
set -euo pipefail

HERE=/home/saadlahrichi/LMOps/gad/scripts/gad_b1
WORKDIR=/home/saadlahrichi/gad_run
CKPT_ROOT=${CKPT_ROOT:-/checkpoints/saadlahrichi/gad_run/ckpts}
ARMS=${ARMS:?set ARMS="EXP:STEP EXP:STEP ..." (e.g. "fs50-gad-base:748 fs50-gad-replay:748")}
CHAT_SETS=${CHAT_SETS:-"lmsys dolly vicuna self-inst"}
OOD_SETS=${OOD_SETS:-"gsm8k math500 humaneval ifeval mmlu"}
JUDGE_SETS=${JUDGE_SETS:-"lmsys,dolly,vicuna,self-inst"}
REFERENCE=${REFERENCE:-judge}
GEN_QOS=${GEN_QOS:-h200_mrs_2_high}
SCORE_QOS=${SCORE_QOS:-h200_mrs_shared}
DRY_RUN=${DRY_RUN:-0}

sub() {  # echo the command; run it unless DRY_RUN; return the captured jobid
  if [ "$DRY_RUN" = "1" ]; then echo "    DRY: $*" >&2; echo "DRYRUN"; else "$@"; fi
}

echo "==== eval batch | arms=[$ARMS] | ckpt_root=$CKPT_ROOT | gen_qos=$GEN_QOS score_qos=$SCORE_QOS ===="
for arm in $ARMS; do
  EXP=${arm%%:*}; STEP=${arm##*:}
  if [ "$EXP" = "$arm" ] || [ "$STEP" = "$arm" ]; then
    echo "!! bad arm spec '$arm' (need EXP:STEP)"; exit 1
  fi
  CKPT="$CKPT_ROOT/$EXP/global_step_${STEP}"
  echo "== $EXP @ gs$STEP =="
  if [ ! -d "$CKPT/actor" ]; then
    echo "  !! SKIP: $CKPT/actor not found (arm not finished / wrong CKPT_ROOT)"; continue
  fi

  jgen=$(CKPT_ROOT="$CKPT_ROOT" EXP="$EXP" STEP="$STEP" VAL_SETS="$CHAT_SETS $OOD_SETS" \
         sub sbatch --qos="$GEN_QOS" --parsable "$HERE/run_eval_gen_wt.sh")
  echo "  gen   : $jgen  (sets: $CHAT_SETS $OOD_SETS)"

  dep=""
  [ "$jgen" != "DRYRUN" ] && dep="--dependency=afterok:$jgen"

  jjudge=$(EXP="$EXP" STEP="$STEP" SETS="$JUDGE_SETS" REFERENCE="$REFERENCE" \
           sub sbatch --qos="$SCORE_QOS" --parsable $dep "$HERE/run_eval_judge_prod.sh")
  echo "  judge : $jjudge  (afterok:$jgen | sets=$JUDGE_SETS ref=$REFERENCE)"

  jood=$(EXP="$EXP" STEP="$STEP" OOD_SETS="$OOD_SETS" \
         sub sbatch --qos="$SCORE_QOS" --parsable $dep "$HERE/run_eval_ood_score.sh")
  echo "  ood   : $jood  (afterok:$jgen | sets=$OOD_SETS)"
done
echo "==== submitted. watch: squeue -u \$USER | grep -E 'evalgen|judge|oodscore' ===="
