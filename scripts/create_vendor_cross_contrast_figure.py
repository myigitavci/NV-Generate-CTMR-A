#!/usr/bin/env python3
"""
create_vendor_cross_contrast_figure.py
======================================
Compiles a sleek, publication-ready multi-scanner & multi-vendor comparison
matrix across T1w, T2w, and FLAIR modalities showing the exact same anatomical
seed across:
  - Siemens (3.0T Vida & 1.5T Avanto)
  - GE Healthcare (3.0T Discovery & 1.5T Signa)
  - Philips Healthcare (3.0T Achieva & 1.5T Ingenia)
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

scanners = [
    {"label": "Siemens Vida (3.0T)", "stem": "siemens_magnetom_vida_3.0T"},
    {"label": "Siemens Avanto (1.5T)", "stem": "siemens_avanto_1.5T"},
    {"label": "GE Discovery (3.0T)", "stem": "ge_discovery_mr750_3.0T"},
    {"label": "GE Signa HDxt (1.5T)", "stem": "ge_signa_hdxt_1.5T"},
    {"label": "Philips Achieva (3.0T)", "stem": "philips_achieva_dstream_3.0T"},
    {"label": "Philips Ingenia (1.5T)", "stem": "philips_ingenia_1.5T"},
]

modalities = [
    {"mod": "T1W", "name": "T1-Weighted (T1w)", "color": "#1e3a8a"},
    {"mod": "T2W", "name": "T2-Weighted (T2w)", "color": "#14532d"},
    {"mod": "FLAIR", "name": "T2-FLAIR", "color": "#7c2d12"},
]

# 1. Master Axial Cross-Scanner Matrix (3 rows x 6 columns)
fig, axes = plt.subplots(len(modalities), len(scanners), figsize=(18, 9.5), dpi=220)

for row_idx, m in enumerate(modalities):
    mod_code = m["mod"].lower()
    for col_idx, sc in enumerate(scanners):
        ax = axes[row_idx, col_idx]
        img_p = slice_dir / f"{mod_code}_{sc['stem']}_axial.png"
        if img_p.exists():
            im = Image.open(img_p)
            ax.imshow(im, cmap="gray")
        ax.axis("off")
        
        # Column title only on top row
        if row_idx == 0:
            ax.set_title(sc["label"], fontsize=11, fontweight="bold", pad=8)
            
        # Row label on leftmost column
        if col_idx == 0:
            ax.text(-0.08, 0.5, m["name"], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", color=m["color"],
                    rotation=90, va="center", ha="right")

fig.suptitle(
    "Multi-Scanner Brain Synthesis Across Vendors & Field Strengths (Fixed Latent Seed: 6256)",
    fontsize=14, fontweight="bold", y=0.98
)
plt.tight_layout(rect=[0.03, 0.02, 0.98, 0.95])
master_panel_path = out_dir / "master_vendor_cross_contrast_axial_matrix.png"
plt.savefig(master_panel_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved master axial matrix -> {master_panel_path}")


# 2. Orthogonal Tri-Planar Panel per Scanner for each Modality
for m in modalities:
    mod_code = m["mod"].lower()
    fig, axes = plt.subplots(len(scanners), 3, figsize=(9.5, 2.7 * len(scanners)), dpi=200)
    for row_idx, sc in enumerate(scanners):
        for col_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
            ax = axes[row_idx, col_idx]
            img_p = slice_dir / f"{mod_code}_{sc['stem']}_{plane}.png"
            if img_p.exists():
                im = Image.open(img_p)
                ax.imshow(im, cmap="gray")
            ax.axis("off")
            
            if row_idx == 0:
                ax.set_title(plane.capitalize(), fontsize=11, fontweight="bold", pad=6)
                
        # Row label on left
        axes[row_idx, 0].text(-0.12, 0.5, sc["label"], transform=axes[row_idx, 0].transAxes,
                              fontsize=10, fontweight="bold", rotation=90, va="center", ha="right")

    fig.suptitle(f"{m['name']} Synthesis Across MRI Scanners (Orthogonal Slices)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.05, 0.02, 0.98, 0.97])
    tri_path = out_dir / f"triplanar_scanner_comparison_{mod_code}.png"
    plt.savefig(tri_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved tri-planar comparison -> {tri_path}")
