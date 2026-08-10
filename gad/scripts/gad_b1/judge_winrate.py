#!/usr/bin/env python
"""GAD B1 eval stage-2: paper-faithful automatic score (paper App. A.3, Figures 7-8; [GDWH24]).

The judge (GPT-4o in the paper; Qwen2.5-72B here) rates the student and a reference answer,
each on a 1-10 scale (helpfulness/relevance/accuracy/detail), via the Figure-8 prompt. The
reported score = mean over examples of  student_score / (student_score + reference_score).
Reference = judge-model-generated answer (README protocol) or teacher_response (--reference teacher).
Generation is greedy (matches paper). Assistant order is randomized per example to remove order bias.

NOTE: this is a Qwen-72B reimplementation of the paper's GPT-4o eval — absolute scores are not
byte-comparable to the paper (different judge model), but the arm-vs-arm A/B is internally consistent.
"""
import argparse, json, os, re, random
import statistics as st
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Figure 7: prompt wrapper (how the instruction is presented) — used to (re)generate the reference.
FIG7 = ("Below is an instruction that describes a task.\n"
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n{instr}\n### Response:")

# Figure 8: the GPT-4o feedback request (verbatim from paper App. A.3).
FIG8 = (
    "We would like to request your feedback on the performance of two AI assistants in response "
    "to the user instruction and input displayed above.\n"
    "Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. "
    "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates "
    "better overall performance.\n"
    "Please first output a single line containing only two values indicating the scores for "
    "Assistant 1 and 2, respectively. The two scores are separated by a space.\n"
    "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding "
    "any potential bias and ensuring that the order in which the responses were presented does not "
    "affect your judgment."
)


def load_gen(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out, teach = r["output"], r.get("teacher_output", "")
            if isinstance(out, list):
                out = out[0] if out else ""
            if isinstance(teach, list):
                teach = teach[0] if teach else ""
            rows.append({"input": r["input"], "student": out or "", "teacher": teach or ""})
    return rows


def clean_instruction(inp):
    """Extract the instruction (+input) from the flattened gen prompt for the [Question] slot."""
    if "### Instruction:" in inp:
        s = inp.split("### Instruction:", 1)[1]
        s = s.split("### Response:", 1)[0]
        return s.strip()
    # fallback: strip role markers
    return inp.replace("system\nYou are a helpful assistant.\nuser\n", "").strip()


def judge_prompt(tok, question, ans1, ans2):
    body = (f"[Question]\n{question}\n\n"
            f"[The Start of Assistant 1's Answer]\n{ans1}\n[The End of Assistant 1's Answer]\n\n"
            f"[The Start of Assistant 2's Answer]\n{ans2}\n[The End of Assistant 2's Answer]\n\n"
            f"{FIG8}")
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   add_generation_prompt=True, tokenize=False)


def parse_scores(text):
    """Extract (s1, s2) robustly. Prefers the instructed 'N M' first line."""
    t = text.strip()
    # 1) instructed format: a line that is exactly two numbers
    for line in t.splitlines():
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$", line.strip())
        if m:
            return float(m.group(1)), float(m.group(2))
    # 2) explicit 'Assistant 1: X ... Assistant 2: Y'
    m1 = re.search(r"[Aa]ssistant\s*1\D{0,4}(\d+(?:\.\d+)?)", t)
    m2 = re.search(r"[Aa]ssistant\s*2\D{0,4}(\d+(?:\.\d+)?)", t)
    if m1 and m2:
        return float(m1.group(1)), float(m2.group(1))
    # 3) fallback: first line containing >=2 numbers
    for line in t.splitlines():
        nums = re.findall(r"\d+(?:\.\d+)?", line)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def score_candidates(llm, tok, questions, cands, refs, rng, max_tokens=1024):
    """Paper score for `cands` vs `refs`: mean student/(student+ref), order-randomized."""
    prompts, cand_is_a1 = [], []
    for q, c, r in zip(questions, cands, refs):
        a1_is_cand = rng.random() < 0.5
        a1, a2 = (c, r) if a1_is_cand else (r, c)
        prompts.append(judge_prompt(tok, q, a1, a2))
        cand_is_a1.append(a1_is_cand)
    gen = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    ratios, csc, rsc, bad = [], [], [], 0
    pp = []  # per-prompt records (B1 length-controlled analysis)
    for idx, (g, a1_is_cand) in enumerate(zip(gen, cand_is_a1)):
        sc = parse_scores(g.outputs[0].text)
        if sc is None:
            bad += 1; continue
        s1, s2 = sc
        cs, rs = (s1, s2) if a1_is_cand else (s2, s1)
        if cs + rs <= 0:
            bad += 1; continue
        ratios.append(cs / (cs + rs)); csc.append(cs); rsc.append(rs)
        pp.append({"i": idx, "cs": cs, "rs": rs, "ratio": cs / (cs + rs)})
    return {"n_scored": len(ratios), "n_unparsed": bad,
            "score": (st.mean(ratios) if ratios else float("nan")),
            "mean_student_1to10": (st.mean(csc) if csc else float("nan")),
            "mean_ref_1to10": (st.mean(rsc) if rsc else float("nan")),
            "perprompt": pp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--sets", default="lmsys,dolly,vicuna,self-inst")
    ap.add_argument("--judge-model", default="/storage/home/saadlahrichi/models/Qwen2.5-72B-Instruct")
    ap.add_argument("--reference", choices=["judge", "teacher"], default="judge",
                    help="judge=judge-model-generated ref (paper README); teacher=teacher_output col (GPT-5 on lmsys)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-ceiling", action="store_true",
                    help="also score teacher_response vs same refs (GPT-5 ceiling; lmsys only meaningful). Needs --reference judge.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.90, max_model_len=args.max_model_len, dtype="bfloat16")
    rng = random.Random(args.seed)

    results = {}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        path = os.path.join(args.gen_dir, f"{name}_generation_results.jsonl")
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found"); continue
        rows = load_gen(path)
        questions = [clean_instruction(r["input"]) for r in rows]

        if args.reference == "judge":  # greedy reference from the judge model (paper: GPT-4o generates it)
            ref_prompts = [tok.apply_chat_template([{"role": "user", "content": FIG7.format(instr=q)}],
                                                   add_generation_prompt=True, tokenize=False) for q in questions]
            ref_gen = llm.generate(ref_prompts, SamplingParams(temperature=0.0, max_tokens=1536))
            refs = [g.outputs[0].text for g in ref_gen]
        else:
            refs = [r["teacher"] for r in rows]

        res = score_candidates(llm, tok, questions, [r["student"] for r in rows], refs, rng)
        pp_recs = res.pop("perprompt", [])
        # per-prompt dump for the B1 length-controlled analysis (student length + judge scores)
        pp_path = os.path.join(args.gen_dir, f"{name}_perprompt.jsonl")
        with open(pp_path, "w") as pf:
            for rec in pp_recs:
                s = rows[rec["i"]]["student"]
                pf.write(json.dumps({"set": name, "i": rec["i"],
                    "student_words": len(s.split()), "student_chars": len(s),
                    "cs": rec["cs"], "rs": rec["rs"], "ratio": rec["ratio"]}) + "\n")
        results[name] = {"n": len(rows), **res}
        print(f"{name:10s} n={len(rows):4d}  score={res['score']:.3f}  "
              f"(student {res['mean_student_1to10']:.2f} vs ref {res['mean_ref_1to10']:.2f} /10, "
              f"unparsed={res['n_unparsed']})")

        if args.teacher_ceiling:
            if args.reference != "judge":
                print(f"{name:10s} teacher-ceiling SKIPPED (needs --reference judge)")
            elif not all(r["teacher"].strip() for r in rows):
                print(f"{name:10s} teacher-ceiling SKIPPED (empty teacher_response)")
            else:
                tc = score_candidates(llm, tok, questions, [r["teacher"] for r in rows], refs, rng)
                results[name]["teacher_ceiling"] = tc
                note = "" if name == "lmsys" else "  [NB: teacher_response=orig dataset answer, NOT GPT-5]"
                print(f"{name:10s} TEACHER-CEILING score={tc['score']:.3f} "
                      f"(teacher {tc['mean_student_1to10']:.2f} vs ref {tc['mean_ref_1to10']:.2f} /10){note}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metric": "student/(student+reference), 1-10 dual-score (paper App.A.3 Fig8)",
                       "judge": os.path.basename(args.judge_model), "reference": args.reference,
                       "teacher_ceiling": args.teacher_ceiling, "seed": args.seed, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
