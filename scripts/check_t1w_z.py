#!/usr/bin/env python3
import numpy as np
from pathlib import Path

cache_dir = Path("output/eval_memorization/diversity_cache")
seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

print("--- Checking T1W slices ---")
for s in seeds:
    v = np.load(cache_dir / f"vol_T1W_real_seed{s}.npy")
    # check where lateral ventricles are: usually where z is higher (z ~ 62-72)
    # let's find z with max tissue in range [62, 75]
    sub = [np.sum(v[:, :, z] > 0.06) for z in range(60, 76)]
    best_sub_z = 60 + int(np.argmax(sub))
    print(f"Seed {s}: best_z in [60,75] = {best_sub_z}, tissue = {sub[best_sub_z-60]}")
