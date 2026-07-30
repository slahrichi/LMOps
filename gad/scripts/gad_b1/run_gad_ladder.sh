#!/bin/bash
# GAD D2 capacity-ladder launcher (design doc C6) — thin wrapper over run_gad_prod.sh that
# sets the replay buffer capacity in *STEPS OF MEMORY* rather than raw rows, and preflights
# host RAM before launching a fat rung.
#
# WHY (design Feas-D1): the buffer stores the rows the critic actually sees, and
# ray_trainer repeats the batch by rollout.n BEFORE the critic, so
#   rows/rank/step (B_per_rank) = train_batch_size * rollout.n / (n_gpus_per_node * nnodes)
#                               = 256 * 8 / (8 * 1) = 256   (prod config)
# Therefore  REPLAY_CAPACITY = MEM_STEPS * B_per_rank.  (MEM_STEPS=16 -> 4096, the current
# replay arm; the v1 "capacity/32" formula was wrong by rollout.n=8x and mislabeled every rung.)
#
# Usage (rungs of the §3.1 capacity axis):
#   MEM_STEPS=1   EXP=fs33-gad-c1step   ... sbatch run_gad_ladder.sh     # 256   (~0.2% of run)
#   MEM_STEPS=16  EXP=fs33-gad-c16step  ... sbatch run_gad_ladder.sh     # 4096  (== current replay)
#   MEM_STEPS=246 EXP=fs33-gad-c1epoch  ... sbatch run_gad_ladder.sh     # ~1 epoch @33% (fat: RAM-checked)
#   MEM_STEPS=492 EXP=fs33-gad-call     ... sbatch run_gad_ladder.sh     # ~all training @33% (fat)
# All other vars pass straight through to run_gad_prod.sh (WARMUP_EXP, RESUME_STEP, EXP, TRAIN,
# CKPT_ROOT, REPLAY_RHO, SAVE_FREQ, KEEP, EXTRA_ARGS, ...). REPLAY_CAPACITY here is DERIVED from
# MEM_STEPS — do not also set it. Pass --dependency=... through to sbatch as usual.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MEM_STEPS=${MEM_STEPS:?set MEM_STEPS (steps of discriminator memory; capacity = MEM_STEPS * B_per_rank)}
TRAIN_BATCH=${TRAIN_BATCH:-256}
ROLLOUT_N=${ROLLOUT_N:-8}
NGPUS=${NGPUS:-8}          # n_gpus_per_node
NNODES=${NNODES:-1}
B_PER_RANK=$(( TRAIN_BATCH * ROLLOUT_N / (NGPUS * NNODES) ))
CAPACITY=$(( MEM_STEPS * B_PER_RANK ))

# ---- host-RAM preflight (design C6): a fat rung must fit in per-rank host RAM ----
# Each stored row is a compacted CPU DataProto (student + teacher), padded to
# max_prompt(2048)+max_response(1536)=3584 tok. Rough footprint: ~4 id/mask tensors x 2 (stu+tea)
# x 3584 tok, int32/uint8 after _compact -> conservatively ~120 KB/row/rank. Buffer is per-rank.
BYTES_PER_ROW=${BYTES_PER_ROW:-125000}
NEED_GB=$(( CAPACITY * BYTES_PER_ROW / 1024 / 1024 / 1024 ))
# node RAM (MB) from slurm alloc convention; fall back to /proc if run interactively
AVAIL_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
AVAIL_GB=$(( AVAIL_MB / 1024 ))
SAFETY=${RAM_SAFETY_GB:-64}   # leave headroom for the model/optim/activations off-GPU offload etc.

echo "===== capacity ladder ====="
echo "  MEM_STEPS=$MEM_STEPS  B_per_rank=$B_PER_RANK  ->  REPLAY_CAPACITY=$CAPACITY rows/rank"
echo "  = $(awk "BEGIN{printf \"%.1f\", $MEM_STEPS*100/492}")% of a 492-step (33%) run / $(awk "BEGIN{printf \"%.1f\", $MEM_STEPS*100/748}")% of a 748-step (50%) run"
echo "  est. buffer host RAM/rank ~= ${NEED_GB} GB (node MemTotal ~= ${AVAIL_GB} GB, safety ${SAFETY} GB)"
if [ "$AVAIL_GB" -gt 0 ] && [ "$NEED_GB" -gt "$(( AVAIL_GB - SAFETY ))" ]; then
  echo "  ✗ ABORT: estimated buffer RAM ${NEED_GB}GB exceeds node headroom (${AVAIL_GB}-${SAFETY}GB)."
  echo "    Lower MEM_STEPS, or override BYTES_PER_ROW/RAM_SAFETY_GB if you've measured actual usage."
  exit 1
fi
echo "  ✓ RAM preflight OK"
echo "  passing through to run_gad_prod.sh: EXP=${EXP:-<default>} CKPT_ROOT=${CKPT_ROOT:-<default>} REPLAY_RHO=${REPLAY_RHO:-0.5}"

# NB: sbatch #SBATCH directives live in run_gad_prod.sh; this wrapper just exports the derived
# capacity and execs sbatch on it, forwarding any extra sbatch flags ($@, e.g. --dependency=...).
export REPLAY_CAPACITY=$CAPACITY
if [ "${DRYRUN:-0}" = "1" ]; then
  echo "  [DRYRUN] would run: sbatch $* $HERE/run_gad_prod.sh  (REPLAY_CAPACITY=$REPLAY_CAPACITY)"
  exit 0
fi
exec sbatch "$@" "$HERE/run_gad_prod.sh"
