#!/usr/bin/env python
"""Aggregate the GAD 33% comparison to fully test the replay hypothesis (stability -> quality):
  1) GPT-4o-style score  = student/(student+ref)      [from eval score_judge.json]
  2) val ROUGE-L         = rougeL(output, teacher_output)  [from eval gens; lmsys = vs GPT-5]
     (+ final training val/rouge-L/mean from the arm's log, where trained)
  3) d_acc_fresh          = discriminator teacher-vs-student acc  [GAD arms only, from training log]
Run on login (CPU). Missing pieces (arms not yet eval'd) show as '--'."""
import json, os, re, statistics as st
from rouge_score import rouge_scorer

EVAL = "/home/saadlahrichi/gad_run/eval"
LOGS = "/home/saadlahrichi/gad_run/logs"
SETS = ["lmsys", "dolly", "vicuna", "self-inst"]
# (label, eval_subdir, training_log or None)
ARMS = [
    ("base-Qwen(floor)", "base-qwen7b/global_step_0",   None),
    ("SeqKD",            "fs33-seqkd/global_step_984",  "seqkd-1575684.out"),
    ("GAD-base",         "fs33-gad-base/global_step_492", "gad-1569064.out"),
    ("GAD-replay",       "fs33-gad-replay/global_step_492", "gad-1565446.out"),
]
_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def score_row(d):
    p = f"{EVAL}/{d}/score_judge.json"
    return json.load(open(p))["results"] if os.path.exists(p) else {}

def rougeL_set(d, s):
    f = f"{EVAL}/{d}/{s}_generation_results.jsonl"
    if not os.path.exists(f):
        return None
    v = []
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line); o = r["output"]; t = r.get("teacher_output", "")
        o = o[0] if isinstance(o, list) else o
        if t:
            v.append(_sc.score(str(t), str(o))["rougeL"].fmeasure)
    return st.mean(v) if v else None

def log_text(logf):
    p = f"{LOGS}/{logf}"
    return open(p, errors="ignore").read() if (logf and os.path.exists(p)) else ""

def d_acc_series(logf):
    """Parse UNIQUE {step: d_acc_fresh} (dedupes Ray '[repeated Nx]' duplicate log lines)."""
    p = f"{LOGS}/{logf}"
    if not (logf and os.path.exists(p)):
        return {}
    d = {}
    for line in open(p, errors="ignore"):
        if "critic/d_acc_fresh" not in line:
            continue
        ms = re.search(r"training/global_step:([0-9]+)", line) or re.search(r"\bstep:([0-9]+)\b", line)
        md = re.search(r"critic/d_acc_fresh:([0-9.]+)", line)
        if ms and md:
            d[int(ms.group(1))] = float(md.group(1))
    return d

def d_acc_stats(series, keys=None):
    keys = sorted(series) if keys is None else keys
    v = [series[k] for k in keys if k in series]
    if not v:
        return None
    return dict(n=len(v), mean=st.mean(v), std=st.pstdev(v), mn=min(v),
                lt5=sum(x < 0.5 for x in v), lt7=sum(x < 0.7 for x in v),
                lo=(min(keys) if keys else None), hi=(max(keys) if keys else None))

def final_val_rouge(txt):
    m = re.findall(r"val/rouge-L/mean:([0-9.]+)", txt)
    return float(m[-1]) if m else None

def fmt(x, w=9, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, float) else f"{'--':>{w}s}"

print("=== 1) GPT-4o-style score  (student/(student+ref); higher=better; 0.5=parity w/ Qwen-72B ref) ===")
print(f"{'arm':18s}" + "".join(f"{s:>10s}" for s in SETS))
for label, d, _ in ARMS:
    row = score_row(d)
    print(f"{label:18s}" + "".join(fmt(row.get(s, {}).get("score"), 10) for s in SETS))

print("\n=== 2) val ROUGE-L  (eval output vs teacher_output; lmsys = vs GPT-5-Chat) ===")
print(f"{'arm':18s}" + "".join(f"{s:>10s}" for s in SETS) + "   train-val/rougeL(final)")
for label, d, logf in ARMS:
    rl = "".join(fmt(rougeL_set(d, s), 10) for s in SETS)
    fv = final_val_rouge(log_text(logf))
    print(f"{label:18s}{rl}   " + (f"{fv:.3f}" if fv is not None else "--"))

print("\n=== 3) d_acc_fresh  (discriminator teacher-vs-student acc; GAD arms only; replay hyp: tighter+higher) ===")
series = {label: d_acc_series(logf) for label, d, logf in ARMS if logf and "gad-" in (logf or "")}
gad_arms = {k: v for k, v in series.items() if v}
for label, s in gad_arms.items():
    st_ = d_acc_stats(s)
    print(f"{label:18s}own-range[{st_['lo']}-{st_['hi']}]: mean={st_['mean']:.3f} std={st_['std']:.3f} min={st_['mn']:.3f} <0.5:{st_['lt5']} <0.7:{st_['lt7']} n={st_['n']}")
if len(gad_arms) == 2:
    (la, sa), (lb, sb) = list(gad_arms.items())
    common = sorted(set(sa) & set(sb))
    if common:
        print(f"  --- ALIGNED over common steps {common[0]}-{common[-1]} (n={len(common)}) [the paper-grade comparison] ---")
        for label, s in gad_arms.items():
            a = d_acc_stats(s, common)
            print(f"  {label:16s}mean={a['mean']:.3f} std={a['std']:.3f} min={a['mn']:.3f} <0.5:{a['lt5']} <0.7:{a['lt7']}")
        print("  NOTE: metrics logging froze mid-run (Ray stdout stall) at different steps per arm; both trained fully to gs492.")
