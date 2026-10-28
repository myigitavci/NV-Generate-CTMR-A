#!/usr/bin/env python3
"""
build_systematic_real_vs_synth_figure.py
========================================
Builds a high-impact publication figure comparing Real vs Synthetic MRI scans
across 6 subjects covering T2w, PDw, and FLAIR modalities using high-quality
systematic evaluation results from output/systematic_evaluation_30casesnew2.

Layout:
  - 2 Rows (3 subjects per row).
  - Each subject has [Real, Synthetic] side-by-side (3 x 2 pairs = 6 columns per row).
  - Row 1: Subject 1 (T2w), Subject 2 (PDw), Subject 3 (FLAIR)
  - Row 2: Subject 4 (T2w), Subject 5 (PDw), Subject 6 (FLAIR)
  - Clean headers: Subject label on top, bold "Real" & "Synthetic" sub-labels.
  - No verbose scanner specification text.
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

# Root directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYS_EVAL_DIR = PROJECT_ROOT / "output" / "systematic_evaluation_30casesnew2"
OUT_DIR = PROJECT_ROOT / "output" / "eval_real_synth"
PANEL_DIR = OUT_DIR / "panels"
SLICE_DIR = OUT_DIR / "slices_systematic"
for d in [PANEL_DIR, SLICE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 6 Curated Subjects from Systematic Evaluation
CASES_CONFIG = [
    # Row 1: T2w, PDw, FLAIR
    {
        "id": "sub_01_t2w",
        "label": "Subject 1 (T2w)",
        "comp_file": "case_13_row171_T2W_comparison.png",
        "sample_row": 1, # Sample 1
    },
    {
        "id": "sub_02_pdw",
        "label": "Subject 2 (PDw)",
        "comp_file": "case_14_row1302_PDW_comparison.png",
        "sample_row": 1, # Sample 1
    },
    {
        "id": "sub_03_flair",
        "label": "Subject 3 (FLAIR)",
        "comp_file": "case_03_row72_FLAIR_comparison.png",
        "sample_row": 3, # Sample 3
    },
    # Row 2: T2w, PDw, FLAIR
    {
        "id": "sub_04_t2w",
        "label": "Subject 4 (T2w)",
        "comp_file": "case_77_row139_T2W_comparison.png",
        "sample_row": 1, # Sample 1
    },
    {
        "id": "sub_05_pdw",
        "label": "Subject 5 (PDw)",
        "comp_file": "case_22_row1338_PDW_comparison.png",
        "sample_row": 3, # Sample 3
    },
    {
        "id": "sub_06_flair",
        "label": "Subject 6 (FLAIR)",
        "comp_file": "case_19_row444_FLAIR_comparison.png",
        "sample_row": 1, # Sample 1
    },
]


def extract_axial_tile(comp_img_path: Path, row_idx: int) -> Image.Image:
    """
    Extracts the clean Axial brain square tile from the systematic comparison figure,
    isolating the black imshow bounding box and removing surrounding white matplotlib margins/text.
    row_idx = 0 for Ground Truth Real, 1..3 for Synthetic Samples 1..3.
    """
    im = Image.open(comp_img_path).convert("RGB")
    w, h = im.size

    # The systematic comparison image has 4 main rows:
    # Row 0: Real Ground Truth (y roughly 0.0 to 0.25)
    # Row 1: Sample 1 (y roughly 0.25 to 0.50)
    # Row 2: Sample 2 (y roughly 0.50 to 0.75)
    # Row 3: Sample 3 (y roughly 0.75 to 1.00)
    row_height = h / 4.0
    y_start = int(row_idx * row_height)
    y_end = int((row_idx + 1) * row_height)
    x_start = 0
    x_end = int(w * 0.36)

    sub_area = im.crop((x_start, y_start, x_end, y_end))
    arr = np.array(sub_area)
    gray = np.mean(arr, axis=-1)

    # In matplotlib, the imshow box is a dark/black area surrounded by white (gray > 240)
    # Find rows and cols that are predominantly dark (the imshow region)
    is_dark = gray < 20
    row_dark_counts = np.sum(is_dark, axis=1)
    col_dark_counts = np.sum(is_dark, axis=0)

    valid_rows = np.where(row_dark_counts > 50)[0]
    valid_cols = np.where(col_dark_counts > 50)[0]

    if len(valid_rows) > 20 and len(valid_cols) > 20:
        y_box_min, y_box_max = int(valid_rows.min()), int(valid_rows.max())
        x_box_min, x_box_max = int(valid_cols.min()), int(valid_cols.max())
        
        # Crop the clean imshow area
        imshow_crop = arr[y_box_min:y_box_max, x_box_min:x_box_max]
        crop_gray = np.mean(imshow_crop, axis=-1)
        h_c, w_c = crop_gray.shape

        # Matplotlib titles/labels are near edges. Brain is in the center.
        # Zero out margins where text might exist: top 18%, left 10%, bottom 5%, right 5%
        inner_mask = np.zeros_like(crop_gray, dtype=bool)
        inner_mask[int(h_c * 0.18):int(h_c * 0.95), int(w_c * 0.10):int(w_c * 0.95)] = True
        
        brain_pts = np.where((crop_gray > 15) & inner_mask)
        
        if len(brain_pts[0]) > 50:
            by_min, by_max = int(brain_pts[0].min()), int(brain_pts[0].max())
            bx_min, bx_max = int(brain_pts[1].min()), int(brain_pts[1].max())
            
            bh = by_max - by_min
            bw = bx_max - bx_min
            side = max(bh, bw)
            pad = int(side * 0.12)
            side_padded = side + 2 * pad
            
            bcy = (by_min + by_max) // 2
            bcx = (bx_min + bx_max) // 2
            
            y0 = bcy - side_padded // 2
            y1 = y0 + side_padded
            x0 = bcx - side_padded // 2
            x1 = x0 + side_padded
            
            pad_top = max(0, -y0)
            pad_bottom = max(0, y1 - imshow_crop.shape[0])
            pad_left = max(0, -x0)
            pad_right = max(0, x1 - imshow_crop.shape[1])
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                imshow_crop = np.pad(imshow_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
                y0 += pad_top
                y1 += pad_top
                x0 += pad_left
                x1 += pad_left
                
            from scipy.ndimage import label, binary_fill_holes

            final_tile = imshow_crop[y0:y1, x0:x1].copy()
            final_gray = np.mean(final_tile, axis=-1)
            
            # Connected component of the brain:
            brain_mask_init = final_gray > 18
            labeled_arr, num_feats = label(brain_mask_init)
            if num_feats > 0:
                # The brain is the largest component near the center
                sizes = [np.sum(labeled_arr == k) for k in range(1, num_feats + 1)]
                largest_lbl = np.argmax(sizes) + 1
                brain_comp = (labeled_arr == largest_lbl)
                brain_comp = binary_fill_holes(brain_comp)
                
                # Dilate mask slightly to not clip delicate cortical edges
                from scipy.ndimage import binary_dilation
                brain_comp_dilated = binary_dilation(brain_comp, iterations=3)
                
                # Zero out background outside brain
                for c in range(final_tile.shape[-1]):
                    final_tile[~brain_comp_dilated, c] = 0

            tile_img = Image.fromarray(final_tile)
            return tile_img.resize((512, 512), resample=Image.Resampling.LANCZOS)
        
        tile_img = Image.fromarray(imshow_crop)
        return tile_img.resize((512, 512), resample=Image.Resampling.LANCZOS)

    return sub_area.resize((512, 512), resample=Image.Resampling.LANCZOS)


def generate_master_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Extract tiles
    tiles = {}
    for case in CASES_CONFIG:
        cid = case["id"]
        comp_p = SYS_EVAL_DIR / case["comp_file"]
        
        real_tile = extract_axial_tile(comp_p, row_idx=0)
        syn_tile = extract_axial_tile(comp_p, row_idx=case["sample_row"])
        
        real_path = SLICE_DIR / f"{cid}_real.png"
        syn_path = SLICE_DIR / f"{cid}_syn.png"
        real_tile.save(real_path)
        syn_tile.save(syn_path)
        
        tiles[cid] = {"real": real_tile, "syn": syn_tile, "label": case["label"]}

    # Create master figure: 2 rows x 3 pairs (Real, Syn)
    # 8 grid columns: pair1 (0,1), spacer (2), pair2 (3,4), spacer (5), pair3 (6,7)
    fig = plt.figure(figsize=(18.0, 7.2), dpi=220)
    gs = gridspec.GridSpec(
        2, 8,
        width_ratios=[1.0, 1.0, 0.12, 1.0, 1.0, 0.12, 1.0, 1.0],
        wspace=0.015, hspace=0.18,
        left=0.01, right=0.99, top=0.91, bottom=0.02
    )

    pair_cols = [(0, 1), (3, 4), (6, 7)]
    row1_cases = CASES_CONFIG[:3]
    row2_cases = CASES_CONFIG[3:]

    for r_idx, row_cases in enumerate([row1_cases, row2_cases]):
        for p_idx, case in enumerate(row_cases):
            c_real, c_syn = pair_cols[p_idx]
            cid = case["id"]

            # Real tile
            ax_r = fig.add_subplot(gs[r_idx, c_real])
            ax_r.imshow(tiles[cid]["real"])
            ax_r.axis("off")
            ax_r.set_title("Real", fontsize=13.0, fontweight="bold", pad=6)

            # Synthetic tile
            ax_s = fig.add_subplot(gs[r_idx, c_syn])
            ax_s.imshow(tiles[cid]["syn"])
            ax_s.axis("off")
            ax_s.set_title("Synthetic", fontsize=13.0, fontweight="bold", pad=6)

    # Super-labels above each subject pair
    # Row 1 Subject labels (T2w, PDw, FLAIR)
    fig.text(0.165, 0.965, CASES_CONFIG[0]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.500, 0.965, CASES_CONFIG[1]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.835, 0.965, CASES_CONFIG[2]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")

    # Row 2 Subject labels (T2w, PDw, FLAIR)
    fig.text(0.165, 0.475, CASES_CONFIG[3]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.500, 0.475, CASES_CONFIG[4]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.835, 0.475, CASES_CONFIG[5]["label"], fontsize=14.5, fontweight="bold", color="black", ha="center")

    out_file = PANEL_DIR / "real_vs_synthetic_6subjects_comparison.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated master figure -> {out_file}")


if __name__ == "__main__":
    generate_master_figure()
