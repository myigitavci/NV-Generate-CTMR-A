#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_10seeds_separate_figures import ALL_SEEDS

cache_dir = Path("output/eval_memorization/diversity_cache")
for mod in ["T2W", "FLAIR", "PDW", "T1W"]:
    print(f"\n=== {mod} ===")
    for s in ALL_SEEDS:
        v = np.load(cache_dir / f"vol_{mod}_real_seed{s}.npy")
        bg_thresh = 0.08
        z_tissue = [np.sum(v[:, :, z] > bg_thresh) for z in range(v.shape[2])]
        nonzero = [z for z, c in enumerate(z_tissue) if c > 1200]
        if nonzero:
            z_min, z_max = nonzero[0], nonzero[-1]
            h = z_max - z_min
            # target 54% of brain
            target_z = int(z_min + 0.54 * h)
            print(f"Seed {s}: z=[{z_min}, {z_max}], h={h}, 54%={target_z}")
