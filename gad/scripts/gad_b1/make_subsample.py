#!/usr/bin/env python
"""Stratified subsample of the GAD training set, preserving the `category` distribution.

Used for the full rerun A/B: subsampling to 33% keeps GAD >= 400 adversarial steps
(the manager's pilot length, enough for replay's chase-cycle to develop) while bringing
each stage inside the 72h SLURM job limit (full data would need ~5 days/arm). Both A/B
arms train on the SAME subset, so the replay-vs-baseline comparison stays clean.

Usage: python make_subsample.py [frac] [seed]   (defaults: 0.33, 42)
"""
import sys
import pandas as pd

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
SRC = "data/lmsys_train-00000-of-00001.parquet"
OUT = f"data/lmsys_train_strat{int(FRAC*100)}.parquet"

df = pd.read_parquet(SRC)
sub = (
    df.groupby("category", group_keys=False)
    .apply(lambda g: g.sample(frac=FRAC, random_state=SEED), include_groups=True)
    .sample(frac=1.0, random_state=SEED)  # shuffle
    .reset_index(drop=True)
)
sub.to_parquet(OUT, index=False)

p_full = df["category"].value_counts(normalize=True)
p_sub = sub["category"].value_counts(normalize=True).reindex(p_full.index)
print(f"{SRC} ({len(df)}) -> {OUT} ({len(sub)}, {len(sub)/len(df):.3f})")
print(f"steps/epoch @ bs=256: {len(sub)//256} | max proportion drift: {(p_sub-p_full).abs().max():.5f}")
