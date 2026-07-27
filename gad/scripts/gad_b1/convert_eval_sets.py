#!/usr/bin/env python
"""Convert MiniLLM-style eval jsonl (dolly/vicuna/self_instruct) into the parquet schema
main_ppo val_only / main_generation expects: `content` = chat list [{system},{user: prompt}],
plus `teacher_response` (dataset reference, for schema + secondary rouge-L). The win-rate
scorer generates its own Qwen-72B reference, so only the prompts must be faithful.
lmsys already ships in this schema (real GPT-5-Chat teacher_response)."""
import json, os
import pandas as pd

EVAL = "/storage/home/saadlahrichi/eval"
OUT = "/home/saadlahrichi/gad_run/data"
SYS = "You are a helpful assistant."
ALPACA_INP = ("Below is an instruction that describes a task, paired with an input that provides "
              "further context. Write a response that appropriately completes the request.\n\n"
              "### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:")
ALPACA_NOINP = ("Below is an instruction that describes a task. Write a response that appropriately "
                "completes the request.\n\n### Instruction:\n{instr}\n\n### Response:")

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def extract(rec):
    """-> (prompt_text, reference_output). Handles dolly/vicuna (pre-formatted `prompt`)
    and self-instruct (instruction + instances[0].input/output)."""
    if rec.get("prompt"):                       # dolly / vicuna: already Alpaca-formatted
        return rec["prompt"], rec.get("output", "") or ""
    instr = rec.get("instruction", "")
    inp, out = rec.get("input", "") or "", rec.get("output", "") or ""
    if rec.get("instances"):                    # self-instruct user_oriented
        inst0 = rec["instances"][0]
        inp = inst0.get("input", "") or ""
        out = inst0.get("output", "") or out
    p = (ALPACA_INP if inp else ALPACA_NOINP).format(instr=instr, inp=inp)
    return p, out

SETS = {
    "lmsys":     None,  # already in schema (skip)
    "dolly":     f"{EVAL}/dolly/valid.jsonl",
    "vicuna":    f"{EVAL}/vicuna/valid.jsonl",
    "self-inst": f"{EVAL}/self_instruct/user_oriented_instructions.jsonl",
}
for name, path in SETS.items():
    if path is None:
        print(f"SKIP {name}: already in target schema"); continue
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} not found"); continue
    rows = []
    for i, r in enumerate(load_jsonl(path)):
        p, out = extract(r)
        if not p.strip():
            continue
        rows.append({
            "id": f"{name}-{i}",
            "content": [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            "teacher_response": out,
            "category": r.get("category", r.get("topic", r.get("motivation_app", name))),
        })
    df = pd.DataFrame(rows)
    outp = f"{OUT}/{name}_test.parquet"
    df.to_parquet(outp, index=False)
    print(f"{name}: {len(df)} rows -> {outp}  (teacher_response nonempty: {(df['teacher_response'].str.len()>0).sum()})")
