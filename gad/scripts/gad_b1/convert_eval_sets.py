#!/usr/bin/env python
"""Convert MiniLLM-style eval jsonl (dolly/vicuna/self_instruct) into the parquet schema
our generation launcher expects: `content` = chat list [{system}, {user: prompt}], plus
`teacher_response` (dataset's own reference, for schema + secondary rouge-L). The win-rate
scorer generates its own Qwen-72B reference, so only the prompts matter here.
Matches the lmsys parquet formatting (Alpaca-formatted prompt as the user turn)."""
import json, os, sys
import pandas as pd

EVAL = "/storage/home/saadlahrichi/eval"
OUT = "/home/saadlahrichi/gad_run/data"
SYS = "You are a helpful assistant."

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def prompt_of(rec):
    # prefer the pre-formatted Alpaca `prompt`; else build from instruction(+input)
    if rec.get("prompt"):
        return rec["prompt"]
    instr, inp = rec.get("instruction", ""), rec.get("input", "")
    return (instr + ("\n\n" + inp if inp else "")).strip()

SETS = {
    "dolly":  f"{EVAL}/dolly/valid.jsonl",
    "vicuna": f"{EVAL}/vicuna/valid.jsonl",
    "self-inst": f"{EVAL}/self_instruct/user_oriented_instructions.jsonl",  # if materialized
}
for name, path in SETS.items():
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} not found"); continue
    recs = load_jsonl(path)
    rows = []
    for i, r in enumerate(recs):
        p = prompt_of(r)
        if not p:
            continue
        rows.append({
            "id": f"{name}-{i}",
            "content": [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            "teacher_response": r.get("output", "") or "",
            "category": r.get("category", r.get("topic", name)),
        })
    df = pd.DataFrame(rows)
    outp = f"{OUT}/{name}_test.parquet"
    df.to_parquet(outp, index=False)
    print(f"{name}: {len(df)} rows -> {outp}  (teacher_response nonempty: {(df['teacher_response'].str.len()>0).sum()})")
