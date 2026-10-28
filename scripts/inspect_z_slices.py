#!/usr/bin/env python3
import numpy as np
from pathlib import Path

cache_dir = Path("output/eval_memorization/diversity_cache")
seeds = [101, 202, 303, 404, 505]

for mod in ["T1W", "T2W", "PDW", "FLAIR"]:
    print(f"\n--- {mod} ---")
    for s in seeds:
        f = cache_dir / f"vol_{mod}_real_seed{s}.npy"
        if f.exists():
            v = np.load(f)
            # Find z range with brain tissue
            bg_thresh = 0.08
            z_tissue = [np.sum(v[:, :, z] > bg_thresh) for z in range(v.shape[2])]
            nonzero_z = [z for z, cnt in enumerate(z_tissue) if cnt > 1000]
            max_z = int(np.argmax(z_tissue))
            print(f"Seed {s}: brain z range [{nonzero_z[0]}, {nonzero_z[-1]}], max area at z={max_z}")
