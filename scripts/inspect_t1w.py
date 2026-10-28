#!/usr/bin/env python3
import numpy as np
from pathlib import Path

cache_dir = Path("output/eval_memorization/diversity_cache")
seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

for s in seeds:
    v = np.load(cache_dir / f"vol_T1W_real_seed{s}.npy")
    print(f"Seed {s}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}")
    # find where ventricles/mid-brain are:
    for z in [60, 65, 70, 75]:
        fg = np.sum(v[:, :, z] > 0.08)
        print(f"   z={z}: fg={fg}")
