#!/bin/bash
# D2 memory-gate OFFLINE RD SCORING (design §5) — thin wrapper (no #SBATCH of its own; delegates to
# run_gad_prod.sh like run_gad_ladder.sh). Resume a discriminator checkpoint D@T and score an
# own-vintage store (step_<S>.pt from the push-tap) via CriticWorker.compute_audit_metrics, logging
# critic/audit/v<S>/bt_margin per vintage. recipe/gad/rd_aggregate then turns those into the
# regression discontinuity at the eviction horizon. Run once per D to score:
#   - the RD arm's own D@T  (primary)
#   - the c=0 reference D@T  (diff-in-diff corroboration) — SAME store, matched T.
#
# The audit-loader (fsdp_workers.init_model) expects audit_v<S>.pt, but the store saves step_<S>.pt,
# so we symlink store/step_<S>.pt -> $SCOREDIR/audit_v<S>.pt (on /home; ln -s never touches the
# read-only-from-login Lustre target) then point critic.audit.path there. val_only + val_before_train
# ⇒ one _validate pass ⇒ the audit fires ⇒ job exits.
#   Usage: STORE_DIR=/checkpoints/$USER/gad_run/vintage/c246gate EXP=fs33-gad-c246gate \
#          RESUME_DIR=/checkpoints/$USER/gad_run/ckpts/fs33-gad-c246gate/global_step_492 \
#          SCORE_TAG=c246gate-D492 bash run_rd_score.sh [--qos=... --dependency=...]
#   Reference: EXP=fs33-gad-base RESUME_DIR=.../fs33-gad-base/global_step_492 SCORE_TAG=c0-D492 (same STORE_DIR)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR=/home/saadlahrichi/gad_run
STORE_DIR=${STORE_DIR:?set STORE_DIR (vintage store with step_<S>.pt)}
RESUME_DIR=${RESUME_DIR:?set RESUME_DIR (global_step_<T> ckpt dir of the D to score with)}
SCORE_TAG=${SCORE_TAG:-rdscore}
EXP=${EXP:?set EXP (the arm whose ckpts CKPT_ROOT/EXP holds; used by run_gad_prod.sh)}
WARMUP_EXP=${WARMUP_EXP:-fs33-warmup}; RESUME_STEP=${RESUME_STEP:-246}
VAL=${VAL:-$WORKDIR/data/smoke_val.parquet}   # tiny; audit is the goal, val-gen is incidental

SCOREDIR=$WORKDIR/rdscore/$SCORE_TAG
mkdir -p "$SCOREDIR"
shopt -s nullglob
FILES=("$STORE_DIR"/step_*.pt)
[ ${#FILES[@]} -gt 0 ] || { echo "ERROR: no step_*.pt in $STORE_DIR (has the RD arm produced a store yet?)"; exit 1; }
VINTAGES=""
for f in "${FILES[@]}"; do
  S=$(basename "$f" .pt | sed 's/step_//')
  ln -sf "$f" "$SCOREDIR/audit_v${S}.pt"
  VINTAGES="${VINTAGES:+$VINTAGES,}$S"
done
echo "linked ${#FILES[@]} vintages from $STORE_DIR -> $SCOREDIR ; resume D from $RESUME_DIR"

export EXP WARMUP_EXP RESUME_STEP VAL
export TRAIN=$WORKDIR/data/mini_train.parquet
export CKPT_ROOT=/checkpoints/saadlahrichi/gad_run/ckpts
export REPLAY_CAPACITY=0
# The audit must score with the D@T critic in RESUME_DIR. trainer._load_checkpoint() does NOT
# restore weights (calls commented out), so resume_from_path can't swap D — instead hand the
# critic ckpt to run_gad_prod.sh's SCORE_CRITIC_CKPT hook, which merges it to HF and points
# critic.model.path at it. resume_mode=disable so no spurious global_step/dataloader restore.
export SCORE_CRITIC_CKPT="$RESUME_DIR/critic"
# NGPU (default 8): shrink the audit to fewer GPUs so it can pack onto partially-free nodes
# (val-only audit is world-size-agnostic — vintages are replicated). Hydra last-wins overrides
# run_gad_prod's hardcoded trainer.n_gpus_per_node=8. Pass --gpus=$NGPU via "$@" to match.
export EXTRA_ARGS="trainer.val_only=True trainer.val_before_train=True trainer.total_epochs=1 trainer.save_freq=-1 trainer.resume_mode=disable +critic.audit.path=$SCOREDIR +critic.audit.vintages=[] trainer.n_gpus_per_node=${NGPU:-8}"
exec sbatch "$@" "$HERE/run_gad_prod.sh"
