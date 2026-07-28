#!/usr/bin/env python
"""GAD B1 eval stage-2: LLM-judge win-rate scorer (Qwen2.5-72B).

REUSES the eval-branch judge code rather than reinventing it:
  - verl.utils.dataset.prompt_templates.get_online_transform_func  -> exact judge prompt
    (MYPROMPT2: order/length-bias-mitigated "\\boxed{Assistant N}") + internal position-shuffle
  - deepscaler.rewards.judge_extractor.extract_judge               -> verdict parser (1/2/-1)

Pipeline per eval set (reads {set}_generation_results.jsonl = {input, output, teacher_output}):
  student answer  = output[sample_idx]
  reference       = teacher_output (GPT-5-Chat; --reference teacher)  OR
                    a Qwen-72B-generated answer (--reference judge, README protocol)
  judge           = Qwen-72B scores student-vs-reference; the transform shuffles positions and
                    tracks the student's side via reward_model.ground_truth, so
                    student_win iff extract_judge(verdict) == that side.
Win-rate = wins / (wins + losses); ties/invalid reported separately.
"""
import argparse, json, os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.dataset.prompt_templates import get_online_transform_func
from deepscaler.rewards.judge_extractor import extract_judge


def load_gen(path, sample_idx):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out, teach = r["output"], r.get("teacher_output", "")
            if isinstance(out, list):
                out = out[sample_idx] if sample_idx < len(out) else out[0]
            if isinstance(teach, list):
                teach = teach[0] if teach else ""
            rows.append({"question": r["input"], "student": out or "", "teacher": teach or ""})
    return rows


def chat_str(tok, chat):
    # transform returns [{user},{system}]; put system first for correct rendering
    chat = sorted(chat, key=lambda m: 0 if m["role"] == "system" else 1)
    return tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


def judge_vs_refs(llm, tok, transform, questions, candidates, refs):
    """Win-rate of `candidates` against `refs` under the reused arena judge (position-shuffled)."""
    prompts, side = [], []
    for q, c, ref in zip(questions, candidates, refs):
        ex = transform({"question": q, "response1": c, "response2": ref,
                        "answer": "1", "extra_info": {}, "data_source": "_all"})
        prompts.append(chat_str(tok, ex["prompt"]))
        side.append(int(ex["reward_model"]["ground_truth"]))
    gen = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))
    inv = {"n_judge": 0, "n_invalid_judge": 0}
    w = l = iv = 0
    for g, s in zip(gen, side):
        pick = extract_judge(g.outputs[0].text, inv)
        if pick == -1:
            iv += 1
        elif pick == s:
            w += 1
        else:
            l += 1
    return {"wins": w, "losses": l, "invalid": iv,
            "win_rate": (w / (w + l) if (w + l) else float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, help="dir with {set}_generation_results.jsonl")
    ap.add_argument("--sets", default="lmsys,dolly,vicuna,self-inst")
    ap.add_argument("--judge-model", default="/storage/home/saadlahrichi/models/Qwen2.5-72B-Instruct")
    ap.add_argument("--reference", choices=["teacher", "judge"], default="judge",
                    help="teacher=teacher_output col (GPT-5 on lmsys); judge=Qwen-72B-generated (paper README)")
    ap.add_argument("--template", default="my_prompt2")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default=None, help="write per-set results JSON here")
    ap.add_argument("--teacher-ceiling", action="store_true",
                    help="also score teacher_response vs the SAME judge reference (upper bound). "
                         "Only meaningful where teacher_response is the real GPT-5-Chat teacher (lmsys). "
                         "Requires --reference judge.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.90, max_model_len=args.max_model_len, dtype="bfloat16")
    transform = get_online_transform_func(args.template, "_all", shuffle_response_order=True, random_seed=42)

    results = {}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        path = os.path.join(args.gen_dir, f"{name}_generation_results.jsonl")
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found"); continue
        rows = load_gen(path, args.sample_idx)

        # reference answers
        if args.reference == "judge":
            ref_prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                                   add_generation_prompt=True, tokenize=False) for r in rows]
            ref_gen = llm.generate(ref_prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1536))
            refs = [g.outputs[0].text for g in ref_gen]
        else:
            refs = [r["teacher"] for r in rows]

        # judge student vs reference (position-shuffle handled inside the transform)
        questions = [r["question"] for r in rows]
        res = judge_vs_refs(llm, tok, transform, questions, [r["student"] for r in rows], refs)
        results[name] = {"n": len(rows), **res}
        print(f"{name:10s} n={len(rows):4d}  win={res['wins']:4d} loss={res['losses']:4d} "
              f"invalid={res['invalid']:3d}  win_rate={res['win_rate']:.3f}  (ref={args.reference})")

        # teacher ceiling: score the teacher_response vs the SAME judge refs (comparable upper bound).
        # Only a true GPT-5 ceiling on lmsys; elsewhere teacher_response is the original dataset answer.
        if args.teacher_ceiling:
            if args.reference != "judge":
                print(f"{name:10s} teacher-ceiling SKIPPED (needs --reference judge)")
            elif not all(r["teacher"].strip() for r in rows):
                print(f"{name:10s} teacher-ceiling SKIPPED (empty teacher_response)")
            else:
                tres = judge_vs_refs(llm, tok, transform, questions, [r["teacher"] for r in rows], refs)
                results[name]["teacher_ceiling"] = tres
                note = "" if name == "lmsys" else "  [NB: teacher_response=orig dataset answer, NOT GPT-5]"
                print(f"{name:10s} TEACHER-CEILING win={tres['wins']:4d} loss={tres['losses']:4d} "
                      f"invalid={tres['invalid']:3d}  win_rate={tres['win_rate']:.3f}{note}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"reference": args.reference, "template": args.template,
                       "sample_idx": args.sample_idx, "teacher_ceiling": args.teacher_ceiling,
                       "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
