#!/usr/bin/env python3
"""
create_compact_scanner_cross_matrix.py
======================================
Compact publication comparison matrix showing T2-Weighted and T2-FLAIR
across Axial, Coronal, and Sagittal planes for the 4 exact scanner models
available in the MIDI dataset:
  1. Siemens MAGNETOM Vida (3.0T)
  2. Siemens MAGNETOM Sola (1.5T)
  3. Philips Achieva dStream (3.0T)
  4. Philips Ingenia (1.5T)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

slice_dir = Path("output/eval_hardware/scanner_variation_brains/slices_512x512")
out_dir = Path("output/eval_hardware/scanner_variation_brains/panels")
out_dir.mkdir(parents=True, exist_ok=True)

# The 4 true scanners in the training dataset
scanners = [
    {"label": "Siemens Vida (3.0T)", "stem": "siemens_magnetom_vida_3.0T"},
    {"label": "Siemens Sola (1.5T)", "stem": "siemens_magnetom_sola_1.5T"},
    {"label": "Philips Achieva (3.0T)", "stem": "philips_achieva_dstream_3.0T"},
    {"label": "Philips Ingenia (1.5T)", "stem": "philips_ingenia_1.5T"},
]

rows = [
    {"mod": "t2w", "plane": "axial", "label": "T2w (Axial)", "color": "#14532d"},
    {"mod": "t2w", "plane": "coronal", "label": "T2w (Coronal)", "color": "#14532d"},
    {"mod": "t2w", "plane": "sagittal", "label": "T2w (Sagittal)", "color": "#14532d"},
    {"mod": "flair", "plane": "axial", "label": "FLAIR (Axial)", "color": "#7c2d12"},
    {"mod": "flair", "plane": "coronal", "label": "FLAIR (Coronal)", "color": "#7c2d12"},
    {"mod": "flair", "plane": "sagittal", "label": "FLAIR (Sagittal)", "color": "#7c2d12"},
]

n_rows = len(rows)
n_cols = len(scanners)

# Compact figure: no title, tight spacing
fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 2.7 * n_rows), dpi=220)
plt.subplots_adjust(wspace=0.015, hspace=0.02, left=0.10, right=0.98, top=0.96, bottom=0.01)

for r_idx, r in enumerate(rows):
    mod = r["mod"]
    plane = r["plane"]
    for c_idx, sc in enumerate(scanners):
        ax = axes[r_idx, c_idx]
        img_p = slice_dir / f"{mod}_{sc['stem']}_{plane}.png"
        if img_p.exists():
            im = Image.open(img_p)
            ax.imshow(im, cmap="gray")
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", color="red")
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        
        # Column title only on top row
        if r_idx == 0:
            ax.set_title(sc["label"], fontsize=12, fontweight="bold", pad=8)
            
        # Row label on leftmost column
        if c_idx == 0:
            ax.text(-0.06, 0.5, r["label"], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", color=r["color"],
                    rotation=90, va="center", ha="right")

out_panel_path = out_dir / "master_vendor_cross_contrast_axial_matrix.png"
compact_panel_path = out_dir / "compact_dataset_scanners_matrix.png"

plt.savefig(out_panel_path, dpi=220, bbox_inches="tight")
plt.savefig(compact_panel_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved dataset scanner compact figure to:\n  - {out_panel_path}\n  - {compact_panel_path}")
