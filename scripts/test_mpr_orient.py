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

# Spacing: dx = 0.94, dy = 0.94, dz = 1.36
# Ratio of voxel size dz / dx = 1.36 / 0.94 = 1.4468
z_scale = 1.36 / 0.94

# 1. Coronal: slice through y_mid
# In norm_vol[:, y_mid, :], axis 0 is X (LR, 256), axis 1 is Z (IS, 128)
# We want Z (IS) vertical (Up = Superior, Down = Inferior)
# And X (LR) horizontal.
# So we need array of shape (Z_height, X_width)
# Transpose norm_vol[:, y_mid, :].T -> shape is (128, 256)
# But in Z, 0 is Inferior and 127 is Superior.
# In an image array, row 0 is top (Superior).
# So we invert row axis: norm_vol[:, y_mid, :].T[::-1, :]!
coronal_arr = norm_vol[:, sy // 2, :].T[::-1, :] # shape (128, 256)
# Resize with z_scale along vertical:
new_h = int(128 * z_scale) # ~185
im_cor = Image.fromarray((coronal_arr * 255).astype(np.uint8)).resize((256, new_h), resample=Image.Resampling.LANCZOS)
im_cor.save("coronal_test_oriented.png")

# 2. Sagittal: slice through x_mid
# In norm_vol[x_mid, :, :], axis 0 is Y (AP, 256), axis 1 is Z (IS, 128)
# Transpose norm_vol[x_mid, :, :].T[::-1, :] -> shape (128, 256)
sagittal_arr = norm_vol[sx // 2, :, :].T[::-1, :]
# Usually Anterior is facing Left or Right. Let's check:
im_sag = Image.fromarray((sagittal_arr * 255).astype(np.uint8)).resize((256, new_h), resample=Image.Resampling.LANCZOS)
im_sag.save("sagittal_test_oriented.png")

print("Saved coronal_test_oriented.png and sagittal_test_oriented.png")
