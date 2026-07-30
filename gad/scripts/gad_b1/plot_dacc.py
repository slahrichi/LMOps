#!/usr/bin/env python
"""Discriminator-stability plot for the GAD 33% replay-buffer study.

Overlays critic/d_acc_fresh (discriminator teacher-vs-student accuracy) vs training
step for GAD-base vs GAD-replay. Data is parsed from the Ray session logs (the SLURM
.out froze mid-run; the ray copies are the FULL stream through gs492) using the exact
parse in aggregate_results.d_acc_series() (regex training/global_step + d_acc_fresh,
deduped by step). Both arms cover steps 1-492.

Run on login/CPU:  ./venv/bin/python plot_dacc.py
Writes: eval/dacc_stability_33pct.png   (matplotlib Agg, no display)
"""
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")  # headless: no display, savefig only
import matplotlib.pyplot as plt

# Reuse the exact parse used by the aggregate report (Ray-log-aware, deduped by step).
from aggregate_results import d_acc_series

OUT = "/home/saadlahrichi/gad_run/eval/dacc_stability_33pct.png"
ROLL = 15  # rolling-mean window (steps)

# (label, training .out -> resolves to its logs/ray-<jobid>/ session copy, color)
ARMS = [
    ("GAD-base",   "gad-1569064.out", "#2a78d6"),  # categorical slot 1 (blue)
    ("GAD-replay", "gad-1565446.out", "#eb6834"),  # categorical slot 8 (orange)
]

# --- chart chrome / ink (dataviz reference palette, light surface) ---
SURFACE = "#fcfcfb"
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
AXIS     = "#c3c2b7"


def rolling_mean(xs, ys, w):
    """Centered rolling mean over y, aligned to x (list order = sorted step order)."""
    out = []
    half = w // 2
    n = len(ys)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(st.mean(ys[lo:hi]))
    return out


def main():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    legend_handles = []
    for label, logf, color in ARMS:
        series = d_acc_series(logf)
        if not series:
            raise SystemExit(f"No d_acc_fresh parsed for {label} ({logf}) -- check logs/ray-*")
        steps = sorted(series)
        vals = [series[s] for s in steps]
        mean, std, mn = st.mean(vals), st.pstdev(vals), min(vals)

        # raw: thin, translucent
        ax.plot(steps, vals, color=color, lw=0.8, alpha=0.30, zorder=2)
        # rolling-mean overlay: solid, prominent (carries identity)
        roll = rolling_mean(steps, vals, ROLL)
        line, = ax.plot(steps, roll, color=color, lw=2.2, alpha=0.95, zorder=3,
                        label=f"{label}   mean={mean:.3f}  std={std:.3f}  min={mn:.3f}  (n={len(vals)})")
        legend_handles.append(line)

    ax.set_title("Discriminator stability: d_acc_fresh vs training step  (GAD 33% cohort)",
                 color=INK, fontsize=13, pad=12, loc="left")
    ax.set_xlabel("training step", color=INK2, fontsize=11)
    ax.set_ylabel("critic/d_acc_fresh  (teacher-vs-student acc)", color=INK2, fontsize=11)
    ax.text(0.0, 1.005,
            f"raw (thin) + {ROLL}-step rolling mean (bold);  lower std = steadier discriminator",
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")

    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":", zorder=1)  # chance line
    ax.text(ax.get_xlim()[1] if False else 492, 0.5, " chance (0.5)",
            color=MUTED, fontsize=8, va="bottom", ha="right")

    # recessive chrome
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)

    leg = ax.legend(handles=legend_handles, loc="lower right", frameon=True,
                    fontsize=9, facecolor=SURFACE, edgecolor=GRID)
    for txt in leg.get_texts():
        txt.set_color(INK)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
