#!/usr/bin/env python3
import numpy as np
from pathlib import Path

cache_dir = Path("output/eval_memorization/diversity_cache")
for s in [101, 202, 606, 707]:
    v = np.load(cache_dir / f"vol_T1W_real_seed{s}.npy")
    pts = np.where(v > 0.08)
    if len(pts[0]) > 0:
        print(f"T1W Seed {s}: x=[{pts[0].min()}, {pts[0].max()}], y=[{pts[1].min()}, {pts[1].max()}], z=[{pts[2].min()}, {pts[2].max()}], total_pts={len(pts[0])}")
    else:
        print(f"T1W Seed {s}: EMPTY")

for s in [101, 202, 606, 707]:
    v = np.load(cache_dir / f"vol_T2W_real_seed{s}.npy")
    pts = np.where(v > 0.08)
    if len(pts[0]) > 0:
        print(f"T2W Seed {s}: x=[{pts[0].min()}, {pts[0].max()}], y=[{pts[1].min()}, {pts[1].max()}], z=[{pts[2].min()}, {pts[2].max()}], total_pts={len(pts[0])}")
