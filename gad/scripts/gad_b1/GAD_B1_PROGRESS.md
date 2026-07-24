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

**Status:** H2 pending the correctness gate (running); H3 not yet tested (needs a clean metric + full scale).

## 4. Why `d_acc`/`d_loss` are contaminated, and why the correctness gate is the right first measure

**Contamination.** `d_acc` (discriminator accuracy) and `d_loss` are computed *over the exact batch the discriminator trains on*. With replay on, that batch is a **mixture of fresh current-policy student responses and stale past-policy ones**. Stale responses are typically easier to classify (the discriminator has seen similar before, and the policy has since moved on), so the metric is measured on a *different, easier distribution* than the baseline's fresh-only batch. Any difference therefore conflates the replay's effect with the change in what's being measured — it cannot be read as "replay makes the discriminator better." The clean signal is a **held-out** accuracy: teacher vs. *freshly generated current-policy* student on data never in the buffer (`d_acc_heldout`). This is specified in our integration notes but **not yet implemented** — it's the next diagnostic to add.

**Why the gate comes first.** Before any efficacy number is meaningful, we must know the intervention is *correctly scoped*: replay must change only the discriminator update, never leak stale, off-policy data into the generator's policy-gradient (GRPO advantages). If it leaked, we'd be training the generator on mislabeled stale data — silently corrupting the policy. The **correctness gate** tests exactly this: run one step with `capacity=0` and with `capacity=4096` from the same checkpoint, and confirm the generator-side quantities (input ids, responses, advantages) are **identical**. Replay runs *after* reward/advantage computation and on a private copy, so within a step the generator path must be untouched; the buffer can only influence future steps *indirectly through the discriminator*. The gate is deterministic, cheap, and falsifiable — unlike a noisy quality score — which is why it's the right success criterion at this stage. *(Running now on the subsample.)*

## 5. Lessons learned

- **Validate end-to-end on the target hardware before scaling.** The device bug and a disk-quota blowup were both invisible to CPU unit tests and only appeared on GPU/at scale. The 2-hour subsampled run paid for itself immediately.
- **Subsampling >> step-capping for pipeline tests.** A small dataset exercises real epoch boundaries, checkpointing, and the merge/resume handoff — a step cap on full data does not.
- **Checkpoints dominate storage, and you rarely need them all.** Each 7B actor+critic+optimizer save is ~150 GB; the default (keep all) exhausted the FSx quota mid-run (which even truncated a source file — recovered from git). GAD resumes from a *single* warmup checkpoint, so `keep-last-2` bounds warmup/GAD to ~300 GB each. **Commit early**: the truncated file was restored from a commit made minutes earlier.
- **Match measurement to the intervention.** A metric computed on data the intervention *changes* (the mixed batch) can't measure the intervention. Separate correctness (gate) from efficacy (held-out metric).
- **Verify the environment, don't inherit assumptions.** The provided scripts targeted a different machine; the "12-hour job limit" turned out to be false (7-day cap); the sandbox's networking hacks were unnecessary (and harmful) on real compute nodes.

## 6. Planned next steps

1. **Correctness gate** (running): confirm cap=0 vs cap=4096 leave the generator path identical → H2.
2. **Implement `d_acc_heldout`** — the clean discriminator diagnostic — and re-run the subsampled A/B to get an *interpretable* efficacy signal.
3. **Full-scale run** with the optimized config (TP=1 + v1, keep-last-2): warmup (~2 d) → GAD-with-replay (~2 d) → evaluation vs. the SeqKD baseline and the teacher.
4. **Replay sweep** once the clean metric is in place: capacity ∈ {0, 1024, 4096}, rho ∈ {0.25, 0.5}, strategy ∈ {uniform, recency}.
