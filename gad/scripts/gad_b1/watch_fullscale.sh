#!/bin/bash
# Watches the GAD A/B chains + guards disk. Milestones -> logs/fs_watch.log every 20 min.
# DISK TRIPWIRE: exits early (notifying) if FSx free space drops below FREE_MIN_GB, so we can
# act before a save fails (the disk-full crash mode). On completion prints per-arm
# d_acc_fresh bands + val rouge-L. Job IDs current as of 2026-07-27 (50% moved to h200_dev).
LOGDIR=/home/saadlahrichi/gad_run/logs
MLOG=$LOGDIR/fs_watch.log
FS=/storage/home/saadlahrichi
FREE_MIN_GB=2000          # alert if FSx free < 2 TB
CKPT_MAX_GB=2500          # alert if our ckpts exceed 2.5 TB
JOBS=("1569064:33-base" "1565446:33-replay" \
      "1571542:50-warmup" "1571543:50-base" "1571544:50-replay")
IDS=$(printf "%s," "${JOBS[@]%%:*}"); IDS=${IDS%,}

band() { local L=$LOGDIR/gad-$1.out; [ -f "$L" ] || { echo "no log"; return; }
  grep -aoE "critic/d_acc_fresh:[0-9.]+" "$L" 2>/dev/null | grep -oE "[0-9.]+$" | \
    awk '{n++;s+=$1;if(min==""||$1<min)min=$1;if($1>max)max=$1} END{if(n)printf "n=%d mean=%.3f band=[%.3f,%.3f]",n,s/n,min,max; else printf "none"}'; }

prev=""; alert=""
for i in $(seq 1 504); do
  free_gb=$(df -B1G --output=avail "$FS" 2>/dev/null | tail -1 | tr -dc '0-9')
  ck_gb=$(du -sB1G /home/saadlahrichi/gad_run/ckpts 2>/dev/null | grep -oE '^[0-9]+')
  line=""
  for j in "${JOBS[@]}"; do
    id=${j%%:*}; lbl=${j##*:}
    st=$(squeue -h -j "$id" -o "%T" 2>/dev/null); [ -z "$st" ] && st=$(sacct -n -j "$id" --format=State 2>/dev/null|head -1|tr -d ' ')
    case "$lbl" in *warmup*) f=$LOGDIR/warmup-$id.out;; *) f=$LOGDIR/gad-$id.out;; esac
    prog=$(grep -aoE "Training Progress: *[0-9]+%" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9]+%")
    line+="$lbl=$st${prog:+/$prog} "
  done
  line+="| free=${free_gb}G ckpts=${ck_gb}G"
  if [ "$line" != "$prev" ]; then echo "[$(date '+%m-%d %H:%M')] $line" >> "$MLOG"; prev="$line"; fi
  # disk tripwire
  if [ -n "$free_gb" ] && [ "$free_gb" -lt "$FREE_MIN_GB" ]; then alert="FSx free ${free_gb}G < ${FREE_MIN_GB}G"; fi
  if [ -n "$ck_gb" ] && [ "$ck_gb" -gt "$CKPT_MAX_GB" ]; then alert="ckpts ${ck_gb}G > ${CKPT_MAX_GB}G"; fi
  [ -n "$alert" ] && { echo "[$(date '+%m-%d %H:%M')] *** DISK ALERT: $alert ***" >> "$MLOG"; break; }
  n=$(squeue -h -j "$IDS" -o "%i" 2>/dev/null | grep -c .); [ "$n" = "0" ] && break
  sleep 1200
done

echo "################ WATCH EXIT ################"
[ -n "$alert" ] && echo "!!! DISK ALERT: $alert  — prune intermediate ckpts of completed arms (keep final gs492 only)."
echo "=== disk ==="; df -h "$FS" | tail -1; du -sh /home/saadlahrichi/gad_run/ckpts 2>/dev/null
echo "=== final states ==="
for j in "${JOBS[@]}"; do id=${j%%:*}; lbl=${j##*:}
  echo "$lbl ($id): $(sacct -n -j "$id" --format=State,Elapsed,ExitCode 2>/dev/null|grep -vE 'batch|extern'|head -1)"; done
echo "=== d_acc_fresh bands ==="
echo "33-base: $(band 1569064)"; echo "33-replay: $(band 1565446)"
echo "50-base: $(band 1571543)"; echo "50-replay: $(band 1571544)"
echo "=== val rouge-L ==="
for j in 1569064:33-base 1565446:33-replay 1571543:50-base 1571544:50-replay; do id=${j%%:*}
  echo "${j##*:}: $(grep -aoE 'val/rouge-L/mean:[0-9.]+' "$LOGDIR/gad-$id.out" 2>/dev/null|tail -1)"; done
