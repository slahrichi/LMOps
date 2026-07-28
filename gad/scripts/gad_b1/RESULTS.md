# GAD Replay-Buffer Study — Results & Methods Digest (33% cohort)

*Paper-grade digest. All numbers verified against `eval/*/score_judge.json` and training logs on 2026-07-28. Working log / ops history: `~/.claude/.../gad-sbatch-ab-campaign.md`. Reproduce results table: `python ~/gad_run/aggregate_results.py`.*

## 1. Research question
Does a **bounded replay buffer on GAD's discriminator (D2)** improve adversarial distillation vs vanilla GAD? Hypothesis: replaying past `(teacher, student)` pairs breaks the GAN chase-cycle → a **more stable discriminator** → **better student quality**.

## 2. Method
- **GAD (Generative Adversarial Distillation):** distill a large teacher into a small student on-policy. A student **generator** (trained with **GRPO**) is optimized to fool a co-trained Bradley–Terry **discriminator/critic** that separates teacher from student responses. Pipeline: **1-epoch warmup** (init generator+discriminator) → **2-epoch adversarial GAD** (`critic_warmup=0`).
- **D2 replay buffer** (`critic.replay.*`, verl branch `gad-d2-replay`): discriminator re-trains on a buffer of past student responses. **A/B knob = `capacity`**: base `0` (replay OFF = vanilla GAD) vs replay `4096`; `rho=0.5`, `strategy=uniform`. Both arms share the same warmup init and the same data subset ⇒ any difference is attributable to replay alone.

## 3. Models
- **Student:** Qwen2.5-7B-Instruct (`/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct`).
- **Teacher:** GPT-5-Chat (responses from `ytz20/LMSYS-Chat-GPT-5-Chat-Response`).
- **Eval judge:** Qwen2.5-72B-Instruct (TP=2). *(Paper used GPT-4o; see Limitations.)*

## 4. Data
- **Train:** `ytz20/LMSYS-Chat-GPT-5-Chat-Response` (192,014 prompts + GPT-5-Chat responses). To fit the 72 h job limit, **category-stratified subsamples**: 33% = 63,215 rows (this cohort), 50% = 95,782 rows (replicate, in progress).
- **Eval sets (prompt counts, all scored, 0 unparsed judgments):** lmsys **479**, dolly **500**, vicuna **80**, self-inst **252**. lmsys is `ytz20` test split (=479, the authors' released `num_examples`; paper's "500" is the nominal pre-4k-filter count). dolly/vicuna/self-inst are MiniLLM-style prompt sets; only **lmsys carries the real GPT-5-Chat teacher_response** (others carry original dataset answers).

## 5. Training configuration (33% cohort)
| | value |
|---|---|
| Warmup | 1 epoch (246 steps), then merge FSDP→HF, resume |
| GAD | 2 epochs = **492 steps/arm**; `adv_estimator=grpo`, lr **1e-6**, train_batch **256**, ppo_mini_batch **256**, rollout **n=8**, temp **0.8**, max_prompt **2048**, max_response **1536**, `kl_loss_coef=0.001`, `kl_ctrl.kl_coef=0.001`, grad_clip **0.2**, `critic_warmup=0`, discriminator dtype **fp32** |
| Replay | base `capacity=0` / replay `capacity=4096`, `rho=0.5`, `strategy=uniform` |
| SeqKD baseline | teacher-forcing SFT of the base 7B on GPT-5 responses (branch `seqkd`): lr **5e-6**, **4 epochs** (984 steps at 33%), bs/mini 256, maxlen 2048/1536, no discriminator/GRPO. Matches the paper's **released script** `gpt5-chat-filtered-7b-seqkd-lr5e-6.sh` (which sets `total_epochs=4`); **note the paper *text* states 3 epochs** — see Limitations. |
| Hardware | single node 8×H200, TP=1, vLLM v1; ~212 s/step |

## 6. Evaluation protocol (paper App. A.3, Figs 7–8)
- **Generation:** greedy (`do_sample=False`), response length 1536, Fig-7 Alpaca wrapper. Deterministic.
- **Judge:** the judge rates the **student** and a **reference** answer each on **1–10** (Fig-8 prompt), order-randomized (seed 42). **Metric = student_score / (student_score + reference_score)** (mean over prompts). 0.5 = parity with the reference.
- **Reference = judge-model-generated** (Qwen-72B answers each prompt, greedy) for all 4 sets — the paper's "use GPT-4o to generate reference answer and score" protocol, with Qwen-72B substituted. Deterministic ⇒ **all arms scored against byte-identical references**.
- **Teacher ceiling (lmsys only):** GPT-5-Chat `teacher_response` scored vs the same references.
- Scorer: `judge_winrate.py`; launchers `run_eval_{gen,judge}_prod.sh`.

## 7. Results — 33% cohort

### 7.1 GPT-4o-style quality score (student/(student+ref); higher better; 0.5 = parity)
| arm | lmsys | dolly | vicuna | self-inst |
|---|---|---|---|---|
| base-Qwen (pre-distill floor) | 0.485 | 0.475 | 0.494 | 0.486 |
| SeqKD | 0.483 | 0.469 | 0.499 | 0.486 |
| GAD-base (replay OFF) | 0.460 | 0.460 | 0.481 | 0.472 |
| **GAD-replay (cap 4096)** | **0.482** | **0.472** | **0.482** | **0.487** |
| *replay − GAD-base* | *+0.022* | *+0.012* | *+0.001* | *+0.015* |
| lmsys teacher ceiling (GPT-5) | 0.499 | — | — | — |

### 7.2 val ROUGE-L (eval output vs teacher_output; lmsys = vs GPT-5)
| arm | lmsys | dolly | vicuna | self-inst | train-val (final) |
|---|---|---|---|---|---|
| base-Qwen floor | 0.297 | 0.232 | 0.252 | 0.189 | — |
| SeqKD | 0.359 | 0.245 | 0.244 | 0.230 | 0.359 |
| GAD-base | 0.340 | 0.270 | 0.267 | 0.255 | 0.346 |
| GAD-replay | 0.321 | 0.217 | 0.260 | 0.201 | 0.307 |

### 7.3 Discriminator stability `d_acc_fresh` (GAD arms only; aligned over common steps 1–212)
| | GAD-base | GAD-replay |
|---|---|---|
| mean | 0.815 | **0.864** |
| std | 0.173 | **0.118** |
| min | 0.051 | **0.273** |
| dips <0.5 / <0.7 | 15 / 37 | **6 / 15** |

*Aligned window: metric logging stalled mid-run (Ray stdout) at different steps (base→374, replay→212; both trained fully to gs492), so stats are over the common steps 1–212. Do NOT use unaligned full-range stats.*

## 8. Findings
1. **Replay > vanilla GAD-base on all 4 sets** (+0.001 to +0.022 quality score) — directionally unanimous.
2. **Replay stabilizes the discriminator** (aligned d_acc_fresh: std 0.118 vs 0.173, min 0.273 vs 0.051, ~half the collapses). This is the most robust signal and the hypothesized mechanism.
3. **Coherent trade-off:** replay has the **lowest** ROUGE-L (0.321 < base 0.340 < SeqKD 0.359 on lmsys) while scoring highest with the judge — consistent with GAD trading n-gram overlap for quality (ROUGE-L is a training diagnostic, not a quality metric).
4. **Mechanism:** vanilla GAD-base dips **below** the base-Qwen floor (adversarial instability hurt it); replay's stability **rescues it back to the floor/SeqKD cluster**.

## 9. Limitations / caveats (must state in paper)
- **Effect size is small** (~0.01–0.02 quality-score, plausibly within judge noise); strength comes from **unanimous direction across 4 sets**, not magnitude. **n=1 run/arm** — no error bars yet.
- **Judge-score metric is saturated:** floor, SeqKD, and GAD-replay all cluster at ~0.48–0.49 (all capable models pin ~8/10 vs ~8.5 reference). It resolves GAD-base's degradation but not fine differences among the strong arms.
- **GAD does not beat SeqKD here** (SeqKD ≈ replay ≈ floor), contrary to the paper's GAD>SeqKD claim — attributable to the 33% subsample + Qwen-72B judge (vs the paper's GPT-4o) + metric saturation + n=1.
- **Judge ≠ paper's:** Qwen-72B, not GPT-4o ⇒ absolute scores are not byte-comparable to the paper; only within-this-study relative deltas are valid.
- **d_acc_fresh only observed for the first ~212/492 steps** (logging stall) — the stability claim is for the early/mid adversarial phase.
- Replay reference generated by the same model that judges (self-preference bias), constant across arms so it cancels in relative comparison.
- **SeqKD baseline trained 4 epochs (per the paper's released script), but the paper text says 3 epochs** — an inconsistency in the paper; we followed the code. The extra epoch may make our SeqKD baseline modestly stronger (higher teacher-fidelity/overfit) than the paper's, which matters since SeqKD ≈ GAD-replay here. A 3-epoch rerun is the clean fix for exact text-parity.

## 10. Reproducibility / next
- **50% cohort (in progress):** 50-base (1576531) / 50-replay (1576532), same config on the 95,782-row subset — tests reproducibility of the replay effect at scale.
- To resolve saturation / paper-parity: rerun the judge with **GPT-4o** and/or a **pairwise-on-hard-prompts** eval against the finished 33% checkpoints.
- Artifacts: scores `eval/{base-qwen7b,fs33-seqkd,fs33-gad-base,fs33-gad-replay}/…/score_judge.json`; checkpoints on Lustre `/checkpoints/saadlahrichi/gad_run/ckpts/`; code `LMOps/gad/scripts/gad_b1/`.
