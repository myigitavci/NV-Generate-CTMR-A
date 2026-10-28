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

# Find brain foreground bounding box
bg_thresh = 0.08
pts = np.where(norm_vol > bg_thresh)
x_min, x_max = int(np.percentile(pts[0], 1.0)), int(np.percentile(pts[0], 99.0))
y_min, y_max = int(np.percentile(pts[1], 1.0)), int(np.percentile(pts[1], 99.0))
z_min, z_max = int(np.percentile(pts[2], 1.0)), int(np.percentile(pts[2], 99.0))

print(f"X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}], Z: [{z_min}, {z_max}]")

# Midline sagittal: x = (x_min + x_max) // 2
x_mid = (x_min + x_max) // 2
# Mid coronal: y = (y_min + y_max) // 2
y_mid = int(y_min + 0.48 * (y_max - y_min))

z_scale = 1.36 / 0.94

# Coronal slice
cor_raw = norm_vol[:, y_mid, :].T[::-1, :] # (128, 256)
# Sagittal slice: Anterior facing Left
sag_raw = np.fliplr(norm_vol[x_mid, :, :].T[::-1, :]) # (128, 256)

# Test square cropping with aspect ratio correction
def process_mpr_slice(raw_2d, z_scale):
    # raw_2d is (128, 256) where axis 0 is Z (IS) and axis 1 is X or Y
    # Resize to physical isotropic dimensions: (int(128 * z_scale), 256)
    new_h = int(raw_2d.shape[0] * z_scale)
    im = Image.fromarray((raw_2d * 255).astype(np.uint8)).resize((raw_2d.shape[1], new_h), resample=Image.Resampling.LANCZOS)
    arr = np.array(im).astype(np.float32) / 255.0

    # Center crop to square around brain
    fg = np.where(arr > 0.04)
    if len(fg[0]) > 200:
        y0, y1 = int(np.min(fg[0])), int(np.max(fg[0]))
        x0, x1 = int(np.min(fg[1])), int(np.max(fg[1]))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half_dim = int(max(y1 - y0, x1 - x0) * 0.54)

        y_start = max(0, cy - half_dim)
        y_end = min(arr.shape[0], cy + half_dim)
        x_start = max(0, cx - half_dim)
        x_end = min(arr.shape[1], cx + half_dim)

        crop = arr[y_start:y_end, x_start:x_end]
        res = Image.fromarray((crop * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
        out = np.array(res).astype(np.float32) / 255.0
    else:
        out = arr

    p1, p99 = np.percentile(out[out > 0.02], [0.5, 99.5]) if np.any(out > 0.02) else (0, 1)
    return np.clip((out - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

cor_final = process_mpr_slice(cor_raw, z_scale)
sag_final = process_mpr_slice(sag_raw, z_scale)

Image.fromarray((cor_final * 255).astype(np.uint8)).save("coronal_sample.png")
Image.fromarray((sag_final * 255).astype(np.uint8)).save("sagittal_sample.png")
print("Saved coronal_sample.png and sagittal_sample.png")
