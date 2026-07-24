# GAD Track B1 — Progress Report

*Reproducing Generative Adversarial Distillation (GAD, arXiv:2511.10643) on an 8×H200 SLURM cluster, and extending it with a discriminator replay buffer (D2). Student = Qwen2.5-7B-Instruct; teacher = GPT-5-Chat.*

## 1. Background

GAD distills a large teacher LLM into a smaller student **on-policy and black-box**: a *student generator* produces responses and a co-trained *discriminator* (a Bradley–Terry reward model) learns to tell teacher responses from student responses. The generator is optimized (via GRPO, an RL algorithm) to fool the discriminator, and the discriminator keeps adapting — a GAN-style minimax loop. Training has three phases: a **1-epoch warmup** (initialize generator + discriminator so the adversarial loop starts from a stable point), then **2 epochs of adversarial GAD**, then evaluation. The dataset is the paper's released teacher set (`ytz20/LMSYS-Chat-GPT-5-Chat-Response`, 192K LMSYS-Chat-1M-Clean prompts with GPT-5-Chat teacher responses) — verified to match the paper exactly.

**Our extension (D2):** a *replay buffer* for the discriminator. In a GAN, the discriminator can "chase" a moving generator and forget earlier failure modes. D2 mixes a fraction of **stale** (teacher, past-student) comparison pairs into each discriminator update to counter this. `capacity` is the buffer size in rows; `capacity=0` disables it (exact vanilla-GAD baseline).

## 2. Hypotheses

- **H1 (throughput):** the reference configuration is not compute-optimal on H200; per-step time can be materially reduced without changing the method.
- **H2 (D2 correctness):** the replay buffer can be integrated so it affects **only** the discriminator update, never the generator's policy-gradient signal.
- **H3 (D2 efficacy):** mixing stale samples stabilizes discriminator training and improves the distilled student. *(Not yet tested — see §5.)*

## 3. Experiments and results

### 3.1 Throughput optimization (H1 — confirmed)

We swept rollout configuration with fixed 3-step smoke runs. Two levers dominated:

- **Tensor parallelism (TP).** TP splits a single model's weights across N GPUs so they cooperate on one forward/backward; it's needed when a model doesn't fit on one GPU, but it adds a per-token cross-GPU synchronization cost. A 7B model fits comfortably on one 141 GB H200, so **TP=2 was pure overhead**. Moving to **TP=1** (each GPU runs an independent generator replica — data parallelism instead) removed the synchronization and cut generation time ~2.4×.
- **Inference engine.** The rollout uses vLLM. Switching from its legacy engine to the newer **v1 engine** cut generation a further ~1.4×.

| Config | per-step | note |
|---|---|---|
| TP=2, engine v0 (reference) | ~405 s | baseline |
| TP=1, v0 | ~270 s | |
| TP=1, **v1** | **~212 s** | adopted |

We also tested raising the GPU-memory fraction for the KV cache (no effect — memory was never the bottleneck) and forcing the discriminator to bf16 to enable FlashAttention (no measurable gain — the discriminator update is not attention-bound; **negative result, and it deviates from the paper's fp32 setup, so rejected**). Net: **~405 → ~212 s/step (~1.9×)**, cutting the warmup from ~3.8 days to ~2 days. Adopted for all production runs.

### 3.2 D2 replay — end-to-end validation (H2/H3)

To avoid a ~6-day full run before knowing the code works, we **subsampled the training set to 3,072 examples** (12 steps/epoch) and ran the entire pipeline (warmup → merge → GAD) in ~2 hours, comparing replay ON (capacity=1024) vs. baseline (capacity=0).

**Findings:**
1. **The pipeline runs end-to-end.** Both configs completed 24 GAD steps.
2. **Found and fixed a GPU-only bug.** On the first *mixed* discriminator update, training crashed: buffered rows are stored on CPU (a memory optimization) and were not moved back to the GPU before being concatenated with the fresh batch. The CPU-only unit tests could not surface this. Fixed by moving the buffered tensor block to the live device before concatenation.
3. **The naive comparison is not interpretable.** Final val rouge-L was replay 0.272 vs baseline 0.280 — within noise at this scale — and the discriminator's own `d_loss`/`d_acc` are **contaminated** when replay is on (see §4).

**Status:** H2 supported — the gate showed step-1 generator quantities byte-identical (cap=0 vs. cap=4096), now hardened into a deterministic in-loop invariant (row count preserved, fresh batch never mutated, checked every step). The clean `d_acc_fresh` metric is implemented and an interpretable A/B re-run confirmed it (§3.3). H3 (efficacy) still needs the full-scale run.

### 3.3 A/B result (subsampled, clean metric + live invariants)

Re-ran baseline (cap=0) vs. replay (cap=4096, rho=0.5), 24 GAD steps each, with `d_acc_fresh` and the in-loop correctness assertions active.

- **Correctness (H2) — passed deterministically.** The replay run completed all 24 steps with zero assertion failures: batch size preserved and the fresh batch unmutated on every mixed update (step 2 onward). Single-run, so no cross-node nondeterminism confound.
- **Contamination — demonstrated.** On the replay arm the old `d_acc` (last 12 steps: band [0.921, 0.984], spread 0.063) is ~40% narrower than the clean `d_acc_fresh` ([0.895, 1.000], spread 0.105) — stale rows make the discriminator look more stable than it actually is on current students. Confirms the clean metric was necessary.
- **Efficacy (H3) — inconclusive at this scale, as expected.** `d_acc_fresh` is statistically indistinguishable: baseline mean 0.941 (band [0.863, 1.000]) vs. replay 0.940 ([0.824, 1.000]). Replay targets the mode-collapse chase-cycle, which only emerges over hundreds of steps; 24 steps gives it nothing to fix. rouge-L (0.285 vs. 0.267) is noise. Efficacy is deferred to the full-scale run (§6.1).

## 4. Why `d_acc`/`d_loss` are contaminated, and why the correctness gate is the right first measure

**Contamination.** `d_acc` (discriminator accuracy) and `d_loss` are computed *over the exact batch the discriminator trains on*. With replay on, that batch is a **mixture of fresh current-policy student responses and stale past-policy ones**. Stale responses are typically easier to classify (the discriminator has seen similar before, and the policy has since moved on), so the metric is measured on a *different, easier distribution* than the baseline's fresh-only batch. Any difference therefore conflates the replay's effect with the change in what's being measured — it cannot be read as "replay makes the discriminator better." The clean signal measures teacher vs. *current-policy* student on **fresh rows only**, with the pre-update discriminator (`d_acc_fresh`) — identical in definition across both arms, so it is directly comparable. We implemented this in-loop form rather than the held-out-generation variant sketched in the integration notes: it needs no extra rollout, and because both arms score the same fresh distribution it is arguably the cleaner comparison.

**Why the gate comes first.** Before any efficacy number is meaningful, we must know the intervention is *correctly scoped*: replay must change only the discriminator update, never leak stale, off-policy data into the generator's policy-gradient (GRPO advantages). If it leaked, we'd be training the generator on mislabeled stale data — silently corrupting the policy. The **correctness gate** tests exactly this: run one step with `capacity=0` and with `capacity=4096` from the same checkpoint, and confirm the generator-side quantities (input ids, responses, advantages) are **identical**. Replay runs *after* reward/advantage computation and on a private copy, so within a step the generator path must be untouched; the buffer can only influence future steps *indirectly through the discriminator*. The gate is deterministic, cheap, and falsifiable — unlike a noisy quality score — which is why it's the right success criterion at this stage. **Result:** step-1 quantities were byte-identical across cap=0 and cap=4096; because a cross-run metric diff is confounded by ordinary GPU/cross-node nondeterminism at later steps, we hardened the gate into a deterministic, single-run in-loop assertion — `build_batch` preserves the row count (so the D:G update ratio matches the baseline) and never mutates the fresh batch, checked every step.

## 5. Lessons learned

- **Validate end-to-end on the target hardware before scaling.** The device bug and a disk-quota blowup were both invisible to CPU unit tests and only appeared on GPU/at scale. The 2-hour subsampled run paid for itself immediately.
- **Subsampling >> step-capping for pipeline tests.** A small dataset exercises real epoch boundaries, checkpointing, and the merge/resume handoff — a step cap on full data does not.
- **Checkpoints dominate storage, and you rarely need them all.** Each 7B actor+critic+optimizer save is ~150 GB; the default (keep all) exhausted the FSx quota mid-run (which even truncated a source file — recovered from git). GAD resumes from a *single* warmup checkpoint, so `keep-last-2` bounds warmup/GAD to ~300 GB each. **Commit early**: the truncated file was restored from a commit made minutes earlier.
- **Match measurement to the intervention.** A metric computed on data the intervention *changes* (the mixed batch) can't measure the intervention. Separate correctness (gate) from efficacy (held-out metric).
- **Verify the environment, don't inherit assumptions.** The provided scripts targeted a different machine; the "12-hour job limit" turned out to be false (7-day cap); the sandbox's networking hacks were unnecessary (and harmful) on real compute nodes.

## 6. Planned next steps

Three discriminator research directions are scoped for this track (source: `manager_ideas.md`); each is a self-contained experiment on the `gad` branch. We execute them in order **2 → 3 → 1** (lowest-risk stability trick first, then interpretability, then the most invasive change to the rollout/advantage path). Every direction reuses the measurement discipline established on Direction 2: a **deterministic correctness gate** — a configuration that must reduce *exactly* to vanilla GAD — before any efficacy claim, and a **clean, contamination-free metric** as the primary signal.

### 6.1 Direction 2 — Replay-Buffer Discriminator *(in progress)*
Mixes stale (teacher, past-student) BT pairs into the D update to break the GAN chase-cycle / mode collapse.
- **Done:** integration + GPU device fix; deterministic in-loop correctness invariants (row-count preserved ⇒ D:G ratio unchanged; fresh batch never mutated); clean `d_acc_fresh` diagnostic; subsampled A/B (baseline vs. cap=4096) running.
- **Next:** full-scale A/B on a **33% category-stratified subsample** (`make_subsample.py`: 192K → 63,215 rows, proportion drift < 1e-4, smallest of 31 classes still 296). Subsampling keeps GAD ≥ 400 adversarial steps (enough for the chase-cycle to develop) while fitting the 72 h job limit — full data would need ~5 d/arm. TP=1 + v1, keep-last-2: warmup (246 st, ~15 h) → baseline + replay GAD in parallel (492 st/arm, ~40 h). Both arms share the subset, so the comparison stays clean. Then the **replay sweep** — capacity ∈ {0, 1024, 4096, 16384} × rho ∈ {0.25, 0.5} × strategy ∈ {uniform, recency-weighted (λ=1e-3)}.
- **Success:** `d_acc_fresh` settles into a tight ~[0.70, 0.85] band vs. the baseline's wide ~[0.55, 0.95] swing, and LMSYS GPT-4o judge win-rate ≥ baseline. Drift toward 0.5 ⇒ stale-dulling ⇒ lower capacity/rho.

### 6.2 Direction 3 — Multi-Aspect Discriminator Head *(next)*
A single scalar reward entangles *helpfulness*, *correctness*, and *style*, so the generator can't see which axis it is failing — and tends to grab the easy one (style/formatting) while correctness stalls.
- **Hypothesis:** decomposing D into per-aspect heads on a shared trunk sharpens the training signal and yields interpretable per-aspect curves (a strong paper figure).
- **Experiment:** 3 heads (helpful / correct / style); warm the heads with **LLM-judge weak labels** (GPT-4o or Qwen2.5-72B, per-aspect A/B preference on ~10K warmup prompts, ≈ $150); **frozen** heads in the adversarial phase (cleaner interpretability); **rank-normalized** aggregation `r = mean_a whiten(D_a)` fed to GRPO.
- **Correctness gate:** with uniform weights the aggregate reward must reproduce vanilla single-head GAD (heads are additive, not path-changing).
- **Success / risk:** per-aspect curves that separate (e.g. style transfers first, correctness lags) + win-rate ≥ vanilla. **Head collapse** is the main failure mode — log `corr(D_a, D_b)`; if > 0.9 mid-training the aspects are one signal in disguise (abort or add a decorrelation loss).

### 6.3 Direction 1 — Sliced / Hierarchical Discriminator *(after)*
A single score per response collapses credit assignment on long (300–500-token) chat answers: every token of a mostly-good response gets the same penalty for one bad paragraph.
- **Hypothesis:** scoring per slice (paragraph) lets the generator localize and fix the specific bad span.
- **Experiment:** emit per-slice scores at slice-end token positions (one causal forward, K extra projections); loss `L_D = α·BT_agg + (1−α)·BT_per_slice` (start α=0.5); per-slice-position group-normalized GRPO advantages broadcast to that slice's tokens. Pilot grid: slicing ∈ {paragraph, 256-tok} × α ∈ {0, 0.5, 1.0} × min-slices-to-normalize ∈ {3, 4}, 400 steps, LMSYS GPT-4o. *(Most invasive: touches the rollout data pipeline, `compute_grpo_advantage`, and the actor GRPO loss to carry per-token advantages.)*
- **Correctness gate:** at α=1.0 (aggregate only) the run must match vanilla GAD exactly (the per-slice path is a no-op).
- **Success / risk:** +0.5 to +1.5 GPT-4o points on LMSYS + OOD gains. If per-slice loss destabilizes D (teacher-vs-student accuracy oscillating > 0.15/step), lower α.

### 6.4 Shared measurement protocol
For each direction: (1) a config that reduces to vanilla GAD (cap=0 / uniform weights / α=1.0), verified byte-identical or via in-loop invariants — the correctness gate — *before* any efficacy number; (2) a clean, uncontaminated primary metric, adversarially interpreted before we trust it; (3) a short subsampled pilot to shake out plumbing before committing a multi-day full run.
