#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mrclip_validation_experiments import get_middle_slices, normalize_for_display

cache_dir = Path("output/eval_memorization/diversity_cache")
v = np.load(cache_dir / "vol_T2W_real_seed101.npy")
print(f"Volume shape: {v.shape}")

# Let's check how get_middle_slices extracts them:
# In run_mrclip_validation_experiments.py:
# let's view get_middle_slices implementation
