# GAD D2 replay — first end-to-end run on H200 (manager update)

**TL;DR:** We ran the D2 discriminator-replay buffer through the full GAD pipeline for the first time on real hardware (8×H200), using a subsampled dataset for a fast turnaround. It runs end-to-end and we fixed a GPU-only bug found in the process. **This was a plumbing/correctness milestone, not a quality result** — the numbers below are not yet a verdict on whether replay helps.

## What we ran
A fast end-to-end test on a 3,072-example subsample (vs. the full 192K): warmup → checkpoint merge → GAD adversarial training, comparing **replay ON** (buffer capacity 1024) against a **baseline** (capacity 0 = vanilla GAD). 24 GAD steps each, run in parallel.

## Results
- Both runs completed cleanly.
- **Final validation rouge-L: replay 0.272, baseline 0.280.**
- Discriminator trajectory (last 10 steps, d_loss / d_acc):

| step | replay | baseline |
|---|---|---|
| 18 | 0.058 / 0.98 | 0.103 / 0.94 |
| 19 | 0.064 / 0.96 | 0.150 / 0.95 |
| 20 | 0.109 / 0.92 | 0.168 / 0.89 |
| 22 | 0.085 / 0.97 | 0.077 / 0.96 |
| 24 | 0.076 / 0.96 | 0.158 / 0.96 |

## What this means for our experiments
- **Positive:** the replay integration works end-to-end on GPU. We found and fixed a device-placement bug (stale buffered rows stayed on CPU and crashed training on the first mixed update) that our CPU-only unit tests could not have caught — exactly why an end-to-end GPU run mattered.
- **Not conclusive:** the rouge-L gap (0.272 vs 0.280) is **noise** at this scale (a few dozen steps on 3K examples), and the discriminator's `d_loss`/`d_acc` are **not directly comparable** when replay is on (they are measured on a fresh+stale mixture, i.e., a different data distribution than the baseline). One directional *hint*: the baseline's discriminator loss spikes harder mid-run while replay stays smoother — consistent with replay's intended stabilizing effect, but far too small a run to claim it.
- **Bottom line:** we can now trust the machinery. To measure whether replay actually *helps*, we need (1) a correctness check that replay only touches the discriminator (running now), (2) a clean held-out discriminator metric, and (3) a full-scale run. We also cut per-step cost ~2× (engine/parallelism tuning), so a full run is ~2 days rather than ~4.

*Prepared from run logs; jobs 1564466 (replay) / 1564467 (baseline).*

---

## Update (2026-07-27) — full-scale A/B underway

The correctness check passed (deterministically) and the clean discriminator metric (`d_acc_fresh`) is implemented. We're now running the **full-scale A/B** on a **33% category-stratified subsample** (63,215 examples; keeps GAD ≥ 400 steps while fitting the 72 h job limit), baseline vs. replay, both from the *same* warmup init.

- **33-replay arm complete** (val rouge-L 0.307); **33-base ~60%**; a **50% replicate** is queued.
- **Preliminary (n=1, discriminator *stability*):** over the overlapping steps, replay holds a **tighter, higher** teacher-vs-current-student accuracy band (std **0.118 vs 0.173**, mean 0.864 vs 0.815) with far fewer instability dips (baseline even collapsed to ~0.05 once) — the predicted "healthier adversarial equilibrium."
- **Caveat:** this is *not* a quality verdict. val rouge-L is a wash and is only a weak lexical proxy; the real test is the **LLM-judge win-rate** once base finishes (+ the 50% replicate to confirm, n=1 today).

Full mechanics/status: `GAD_AB_EXPERIMENTS.md`.
