# Direction 3 — Multi-Aspect Discriminator: Implementation Plan

*Decompose GAD's single scalar reward into per-aspect heads (helpfulness / correctness / style) so the generator sees **which axis** it is failing on, and we get interpretable per-aspect training curves. Self-contained experiment on the `gad` branch; reuses the measurement discipline from Direction 2 (a config that reduces to vanilla GAD as the correctness gate, plus a clean per-aspect `d_acc_fresh`).*

## 1. Hypothesis & success criteria
- **Hypothesis:** a single scalar entangles helpfulness/correctness/style; the student optimizes the easy axis (style) while correctness stalls. Per-aspect heads sharpen the signal → win-rate ≥ vanilla, plus the diagnostic figure.
- **Success:** per-aspect student-score curves that *separate* (e.g. style transfers first, correctness lags) + LMSYS GPT-4o win-rate ≥ vanilla GAD.
- **Primary failure mode:** head collapse — if `corr(D_a, D_b) > 0.9` mid-training the aspects are one signal in disguise (abort or add decorrelation loss).

## 2. Core design decision — `num_labels = A`, not a custom module
The critic is `AutoModelForTokenClassification` (`verl/utils/model.py:503`), head = `Linear(hidden, num_labels)`, with `num_labels=1` set at `fsdp_workers.py:919` and `:1237`; `dp_critic` reads per-token value from `output.logits[..., -resp_len:]`. Setting **`num_labels = A`** makes `output.logits` `[B, T, A]` — exactly A per-token aspect scores, with no new nn.Module, so FSDP wrapping, gradient checkpointing, and the FSDP→HF merge all keep working unchanged. (Equivalent to A separate `Linear(H,1)` heads.)
- New config: `critic.num_aspects` (default **1 ⇒ byte-identical to current single-head GAD** = the correctness gate).

## 3. Edit map
| File | Change |
|---|---|
| `verl/workers/fsdp_workers.py` (~919, ~1237) | `critic_model_config.num_labels = config.model.get("num_aspects", 1)` |
| `verl/workers/critic/dp_critic.py` `_forward_micro_batch` | stop squeezing the label dim; return per-token `[., A]` (last-token gather over each aspect) |
| `verl/workers/critic/dp_critic.py` `update_critic` | per-aspect BT loss (sum/mean over aspects); log per-aspect `d_acc_fresh` (reuse our helper per column) + `corr(D_a,D_b)` diagnostic; optional `freeze_heads` |
| `verl/trainer/ppo/core_algos.py` `compute_discriminator_loss` | accept `[., A]`; `L_D = mean_a −logσ(Σ t_{i,a} − Σ s_{i,a})` |
| reward path (`compute_values` → GRPO reward) | **rank-normalized aggregation** `r = mean_a whiten_group(D_a)` → scalar reward for GRPO |
| `verl/trainer/config/ppo_trainer.yaml` (`critic:`) | `num_aspects`, `aspect_agg={uniform,rank_norm}`, `freeze_heads={true,false}`, `decorrelation_coef` |

## 4. Two-phase structure
1. **Head warmup (new):** initialize the A heads to *mean semantically distinct aspects* using weak labels (§5), Bradley-Terry per aspect. Everything else = normal GAD warmup.
2. **Adversarial:** **freeze the aspect heads** (recommended first cut) — only the shared trunk updates, so aspect scores stay interpretable and the paper figure ("aspect_a over training") is meaningful. Co-evolving heads is the fallback if frozen underfits as the trunk drifts.

## 5. Aspect labels — the crux (decision needed, see §8)
Three options from the manager doc, in order of recommendation:
- **A — LLM-judge weak labels (recommended):** ~10K warmup prompts × (teacher, snapshot-student) × 3 aspects, ask a judge per aspect "which is better, A or B?" → binary labels → BT head-warmup. Cost ≈ $150 on GPT-4o; **~$0 if we use Qwen2.5-72B on our own H200s** (no API, no data egress — likely preferable here).
- **B — rule-based proxies:** length/format features for style, executable/answer-match for correctness, prompt-response similarity for helpfulness. Auxiliary only.
- **C — decorrelation-only (no labels):** force heads apart via `L_decor = mean_{a≠b} |corr(D_a−D_a^t, D_b−D_b^t)|`. Cheap but heads may not be semantically meaningful.

## 6. Reward aggregation (feeds GRPO)
`rank_norm` (recommended): whiten each aspect's scores within the GRPO group (n=8) before averaging, so one large-scale aspect can't dominate:
```
D_all  = critic(...)                     # [B, A]  (last-token, per aspect)
D_norm = (D_all - D_all.mean(0)) / (D_all.std(0) + 1e-6)
r      = D_norm.mean(dim=-1)             # [B] -> GRPO as usual
```
`uniform` (`w_a = 1/A`) is the simpler baseline and the natural bridge to the gate.

## 7. Correctness gate + diagnostics (reuse D2 discipline)
- **Gate:** `num_aspects=1` must be byte-identical to current single-head GAD (same pattern as cap=0). Add an in-loop invariant that the aggregated reward at `num_aspects=1`/uniform equals the single-head value.
- **Clean signal:** per-aspect `d_acc_fresh` (our existing helper, applied per column) — uncontaminated, comparable across runs.
- **Head-collapse detector:** log `corr(D_a, D_b)` every N steps; alarm at >0.9.
- **Paper figure:** per-aspect mean student score vs. step.

## 8. Decisions (locked)
1. **Label source — Qwen2.5-72B on-cluster.** ~$0, no API key, no data egress off the cluster. Build a batch-inference job that, per warmup prompt, asks Qwen2.5-72B per aspect "which response is better, A or B?" over (teacher, snapshot-student) pairs → binary aspect labels → BT head-warmup. Runs as an H200 batch job (queues behind the full-scale D2 jobs, since it also needs GPUs).
2. **Head behavior — frozen after warmup.** Only the shared trunk updates in the adversarial phase; aspect heads fixed → interpretable per-aspect curves. Co-evolve is the fallback if frozen underfits.
3. **Snapshot student for labeling — default: the warmup checkpoint** (harder, more informative pairs than an untrained model, so the heads must actually discriminate per aspect). *Minor, revisit if labels are too teacher-skewed.*

## 9. Rollout
1. Subsampled pilot (like D2): `num_aspects=1` gate → 3-aspect frozen on the 3K subset → verify `corr(heads)` stays < 0.9 and curves separate.
2. Full-scale: recommended config (3 aspects, weak labels, frozen, rank-normalized), 400+ adversarial steps, judged by LMSYS GPT-4o win-rate vs. the Direction-2 baseline.
