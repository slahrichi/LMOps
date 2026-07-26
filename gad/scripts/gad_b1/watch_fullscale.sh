#!/bin/bash
# Watches both full-scale A/B chains (33% on h200_dev, 50% on h200_mrs_2_high).
# Logs warmup->GAD milestones to fs_watch.log; on final completion prints per-job
# states + d_acc_fresh bands (baseline vs replay, both fractions) + val rouge-L.
LOGDIR=/home/saadlahrichi/gad_run/logs
MLOG=$LOGDIR/fs_watch.log
# jobid:label
JOBS=("1565444:33-warmup" "1565445:33-base" "1565446:33-replay" \
      "1565552:50-warmup" "1567812:50-base" "1567813:50-replay")
IDS=$(printf "%s," "${JOBS[@]%%:*}"); IDS=${IDS%,}

band() {  # summarize critic/d_acc_fresh from a job log
  local L=$LOGDIR/gad-$1.out
  [ -f "$L" ] || { echo "no log"; return; }
  grep -aoE "critic/d_acc_fresh:[0-9.]+" "$L" 2>/dev/null | grep -oE "[0-9.]+$" | \
    awk '{n++;s+=$1;if(min==""||$1<min)min=$1;if($1>max)max=$1} END{if(n)printf "n=%d mean=%.3f band=[%.3f,%.3f]",n,s/n,min,max; else printf "no d_acc_fresh yet"}'
}

prev=""
for i in $(seq 1 504); do   # ~7 days at 20min
  line=""
  for j in "${JOBS[@]}"; do
    id=${j%%:*}; lbl=${j##*:}
    st=$(squeue -h -j "$id" -o "%T" 2>/dev/null)
    [ -z "$st" ] && st=$(sacct -n -j "$id" --format=State 2>/dev/null | head -1 | tr -d ' ')
    case "$lbl" in *warmup*) f=$LOGDIR/warmup-$id.out;; *) f=$LOGDIR/gad-$id.out;; esac
    prog=$(grep -aoE "Training Progress: *[0-9]+%" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9]+%")
    line+="$lbl=$st${prog:+/$prog} "
  done
  if [ "$line" != "$prev" ]; then
    echo "[$(date '+%m-%d %H:%M')] $line" >> "$MLOG"
    prev="$line"
  fi
  n=$(squeue -h -j "$IDS" -o "%i" 2>/dev/null | grep -c .)
  [ "$n" = "0" ] && break
  sleep 1200
done

echo "################ FULL-SCALE A/B COMPLETE ################"
echo "=== final states ==="
for j in "${JOBS[@]}"; do
  id=${j%%:*}; lbl=${j##*:}
  echo "$lbl ($id): $(sacct -n -j "$id" --format=State,Elapsed,ExitCode 2>/dev/null | grep -vE 'batch|extern' | head -1)"
done
echo ""
echo "=== d_acc_fresh (clean signal) — replay vs baseline ==="
echo "33% baseline: $(band 1565445)"
echo "33% replay  : $(band 1565446)"
echo "50% baseline: $(band 1567812)"
echo "50% replay  : $(band 1567813)"
echo ""
echo "=== final val rouge-L ==="
for j in "1565445:33-base" "1565446:33-replay" "1567812:50-base" "1567813:50-replay"; do
  id=${j%%:*}; lbl=${j##*:}
  echo "$lbl: $(grep -aoE 'val/rouge-L/mean:[0-9.]+' "$LOGDIR/gad-$id.out" 2>/dev/null | tail -1)"
done
echo ""
echo "milestone log: $MLOG"; echo "--- milestones ---"; cat "$MLOG" 2>/dev/null | tail -30
