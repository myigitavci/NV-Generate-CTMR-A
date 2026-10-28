#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_10seeds_separate_figures import (
    ALL_SEEDS,
    MODALITIES_INFO,
    plot_single_modality_10seeds,
)
from PIL import Image

cache_dir = Path("output/eval_memorization/diversity_cache")
out_dir = Path("output/eval_memorization")

def extract_canonical_ventricular_slice(vol: np.ndarray) -> np.ndarray:
    norm_vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    # Search in standard anatomical ventricular range z in [60, 70]
    z_areas = [np.sum(norm_vol[:, :, z] > 0.08) for z in range(60, 71)]
    best_z = 60 + int(np.argmax(z_areas))
    raw_slice = np.rot90(norm_vol[:, :, best_z])

    fg_pts = np.where(raw_slice > 0.04)
    if len(fg_pts[0]) > 200:
        y0, y1 = int(np.min(fg_pts[0])), int(np.max(fg_pts[0]))
        x0, x1 = int(np.min(fg_pts[1])), int(np.max(fg_pts[1]))
        h, w = y1 - y0, x1 - x0
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half_dim = int(max(h, w) * 0.55)

        y_start = max(0, cy - half_dim)
        y_end = min(raw_slice.shape[0], cy + half_dim)
        x_start = max(0, cx - half_dim)
        x_end = min(raw_slice.shape[1], cx + half_dim)

        cropped = raw_slice[y_start:y_end, x_start:x_end]
        pil_im = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
            (512, 512), resample=Image.Resampling.LANCZOS
        )
        norm = np.array(pil_im).astype(np.float32) / 255.0
    else:
        norm = raw_slice

    p1, p99 = np.percentile(norm[norm > 0.02], [0.5, 99.5]) if np.any(norm > 0.02) else (0, 1)
    norm = np.clip((norm - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return norm

# Re-slice all 4 modalities with this canonical ventricular definition
for mod_id, info in MODALITIES_INFO.items():
    slices = []
    for s in ALL_SEEDS:
        vol = np.load(cache_dir / f"vol_{info['key']}_real_seed{s}.npy")
        slc = extract_canonical_ventricular_slice(vol)
        slices.append(slc)
    plot_single_modality_10seeds(mod_id, info, slices, ALL_SEEDS, out_dir)

print("All 4 modalities re-rendered with canonical ventricular slice extractor!")
