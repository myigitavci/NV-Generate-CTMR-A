#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mrclip_validation_experiments import normalize_for_display

cache_dir = Path("output/eval_memorization/diversity_cache")
v = np.load(cache_dir / "vol_T2W_real_seed101.npy")
norm_vol = normalize_for_display(v)

sx, sy, sz = norm_vol.shape
print(f"Shape: {sx}, {sy}, {sz}")

# Let's inspect coronal at y=sy//2, and sagittal at x=sx//2
# Note: spacing is [0.94, 0.94, 1.36] mm or [1.0, 1.0, 2.0]
# In voxels: 128 slices along Z covers ~174 mm (128 * 1.36 mm)
# 256 voxels along X/Y covers ~240 mm (256 * 0.94 mm)
# Ratio of voxel size Z / X = 1.36 / 0.94 = 1.4468

# Extract raw
raw_coronal = np.rot90(norm_vol[:, sy // 2, :])
raw_sagittal = np.rot90(norm_vol[sx // 2, :, :])

print(f"Raw coronal shape: {raw_coronal.shape}")
print(f"Raw sagittal shape: {raw_sagittal.shape}")

# Save raw test images to check orientation
Image.fromarray((raw_coronal * 255).astype(np.uint8)).save("coronal_test_raw.png")
Image.fromarray((raw_sagittal * 255).astype(np.uint8)).save("sagittal_test_raw.png")
print("Saved test raw images.")
