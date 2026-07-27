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

        # judge prompts (student=response1, reference=response2, answer="1" -> transform tracks student side)
        j_prompts, student_side = [], []
        for r, ref in zip(rows, refs):
            ex = transform({"question": r["question"], "response1": r["student"], "response2": ref,
                            "answer": "1", "extra_info": {}, "data_source": "_all"})
            j_prompts.append(chat_str(tok, ex["prompt"]))
            student_side.append(int(ex["reward_model"]["ground_truth"]))
        j_gen = llm.generate(j_prompts, SamplingParams(temperature=0.0, max_tokens=2048))

        inv = {"n_judge": 0, "n_invalid_judge": 0}
        wins = losses = invalid = 0
        for g, side in zip(j_gen, student_side):
            pick = extract_judge(g.outputs[0].text, inv)
            if pick == -1:
                invalid += 1
            elif pick == side:
                wins += 1
            else:
                losses += 1
        wr = wins / (wins + losses) if (wins + losses) else float("nan")
        results[name] = {"n": len(rows), "wins": wins, "losses": losses, "invalid": invalid, "win_rate": wr}
        print(f"{name:10s} n={len(rows):4d}  win={wins:4d} loss={losses:4d} invalid={invalid:3d}  "
              f"win_rate={wr:.3f}  (ref={args.reference})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"reference": args.reference, "template": args.template,
                       "sample_idx": args.sample_idx, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
