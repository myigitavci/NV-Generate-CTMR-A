#!/usr/bin/env python3
"""
generate_3row_scanner_matrix_figure.py
======================================
Publication figure comparing Real vs Synthetic MRI scans across 3 Scanners (3 rows)
and 3 Modalities (T2w, PDw, FLAIR):
  - Row 1: Siemens Avanto (1.5T)
  - Row 2: GE Signa HDxt (1.5T)
  - Row 3: Siemens Aera (1.5T)

Layout:
  - 3 Rows x 3 Modality Pairs (Real, Synthetic side-by-side) = 6 image columns.
  - Modality Titles on top: "Synthetic T2w", "Synthetic PDw", "Synthetic FLAIR".
  - Sub-column titles "Real" and "Synthetic" placed ONLY at the top (above row 1).
  - Scanner names on the left of each row.
  - No "Subject X" text.
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
from scipy.ndimage import label, binary_fill_holes, binary_dilation

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYS_EVAL_DIR = PROJECT_ROOT / "output" / "systematic_evaluation_30casesnew2"
OUT_DIR = PROJECT_ROOT / "output" / "eval_real_synth"
PANEL_DIR = OUT_DIR / "panels"
SLICE_DIR = OUT_DIR / "slices_scanner_matrix"
for d in [PANEL_DIR, SLICE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 3 Rows x 3 Modalities = 9 Cases (Real + Syn)
SCANNER_ROWS = [
    {
        "scanner_name": "Siemens Avanto (1.5T)",
        "t2w": {
            "comp_file": "case_197_row1104_T2W_comparison.png",
            "sample_row": 1,
        },
        "pdw": {
            "comp_file": "case_18_row215_PDW_comparison.png",
            "sample_row": 1,
        },
        "flair": {
            "comp_file": "case_19_row444_FLAIR_comparison.png",
            "sample_row": 1,
        },
    },
    {
        "scanner_name": "GE Signa HDx (1.5T)",
        "t2w": {
            "comp_file": "case_65_row155_T2W_comparison.png",
            "sample_row": 1,
        },
        "pdw": {
            "comp_file": "case_14_row1302_PDW_comparison.png",
            "sample_row": 1,
        },
        "flair": {
            "comp_file": "case_35_row1791_FLAIR_comparison.png",
            "sample_row": 1,
        },
    },
    {
        "scanner_name": "Siemens Aera (1.5T)",
        "t2w": {
            "comp_file": "case_13_row171_T2W_comparison.png",
            "sample_row": 1,
        },
        "pdw": {
            "comp_file": "case_78_row109_PDW_comparison.png",
            "sample_row": 1,
        },
        "flair": {
            "comp_file": "case_03_row72_FLAIR_comparison.png",
            "sample_row": 3,
        },
    },
]


def extract_clean_axial_tile(comp_img_path: Path, row_idx: int) -> Image.Image:
    """
    Extracts the clean Axial brain tile from the systematic comparison figure.
    """
    im = Image.open(comp_img_path).convert("RGB")
    w, h = im.size

    row_height = h / 4.0
    y_start = int(row_idx * row_height)
    y_end = int((row_idx + 1) * row_height)
    x_start = 0
    x_end = int(w * 0.36)

    sub_area = im.crop((x_start, y_start, x_end, y_end))
    arr = np.array(sub_area)
    gray = np.mean(arr, axis=-1)

    is_dark = gray < 20
    row_dark_counts = np.sum(is_dark, axis=1)
    col_dark_counts = np.sum(is_dark, axis=0)

    valid_rows = np.where(row_dark_counts > 50)[0]
    valid_cols = np.where(col_dark_counts > 50)[0]

    if len(valid_rows) > 20 and len(valid_cols) > 20:
        y_box_min, y_box_max = int(valid_rows.min()), int(valid_rows.max())
        x_box_min, x_box_max = int(valid_cols.min()), int(valid_cols.max())
        
        imshow_crop = arr[y_box_min:y_box_max, x_box_min:x_box_max]
        crop_gray = np.mean(imshow_crop, axis=-1)
        h_c, w_c = crop_gray.shape

        inner_mask = np.zeros_like(crop_gray, dtype=bool)
        inner_mask[int(h_c * 0.18):int(h_c * 0.95), int(w_c * 0.10):int(w_c * 0.95)] = True
        
        brain_pts = np.where((crop_gray > 15) & inner_mask)
        
        if len(brain_pts[0]) > 50:
            by_min, by_max = int(brain_pts[0].min()), int(brain_pts[0].max())
            bx_min, bx_max = int(brain_pts[1].min()), int(brain_pts[1].max())
            
            bh = by_max - by_min
            bw = bx_max - bx_min
            side = max(bh, bw)
            pad = int(side * 0.10)
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
                
            final_tile = imshow_crop[y0:y1, x0:x1].copy()
            final_gray = np.mean(final_tile, axis=-1)
            
            # Brain connected component isolation
            brain_mask_init = final_gray > 18
            labeled_arr, num_feats = label(brain_mask_init)
            if num_feats > 0:
                sizes = [np.sum(labeled_arr == k) for k in range(1, num_feats + 1)]
                largest_lbl = np.argmax(sizes) + 1
                brain_comp = (labeled_arr == largest_lbl)
                brain_comp = binary_fill_holes(brain_comp)
                brain_comp_dilated = binary_dilation(brain_comp, iterations=3)
                
                for c in range(final_tile.shape[-1]):
                    final_tile[~brain_comp_dilated, c] = 0

            tile_img = Image.fromarray(final_tile)
            return tile_img.resize((512, 512), resample=Image.Resampling.LANCZOS)
        
        tile_img = Image.fromarray(imshow_crop)
        return tile_img.resize((512, 512), resample=Image.Resampling.LANCZOS)

    return sub_area.resize((512, 512), resample=Image.Resampling.LANCZOS)


def generate_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 18.0 x 10.2 gives ample height for vertical scanner labels without overlap
    fig = plt.figure(figsize=(18.0, 10.2), dpi=250)
    
    # 3 rows x 8 columns (3 pairs of Real/Syn with spacing)
    # pair 1 (0, 1), spacer (2), pair 2 (3, 4), spacer (5), pair 3 (6, 7)
    # Minimal hspace between rows (0.018) to remove extra row gap
    gs = gridspec.GridSpec(
        3, 8,
        width_ratios=[1.0, 1.0, 0.08, 1.0, 1.0, 0.08, 1.0, 1.0],
        wspace=0.012, hspace=0.018,
        left=0.065, right=0.99, top=0.915, bottom=0.015
    )

    pair_cols = [(0, 1), (3, 4), (6, 7)]
    modalities = ["t2w", "pdw", "flair"]
    mod_titles = ["Synthetic T2w", "Synthetic PDw", "Synthetic FLAIR"]

    axes_dict = {}

    for r_idx, row_data in enumerate(SCANNER_ROWS):
        for m_idx, mod_key in enumerate(modalities):
            case_info = row_data[mod_key]
            comp_path = SYS_EVAL_DIR / case_info["comp_file"]
            
            real_tile = extract_clean_axial_tile(comp_path, row_idx=0)
            syn_tile = extract_clean_axial_tile(comp_path, row_idx=case_info["sample_row"])
            
            c_real, c_syn = pair_cols[m_idx]
            
            # Real Tile
            ax_r = fig.add_subplot(gs[r_idx, c_real])
            ax_r.imshow(real_tile)
            ax_r.axis("off")
            if r_idx == 0:
                ax_r.set_title("Real", fontsize=16.5, fontweight="bold", pad=8)
            
            # Synthetic Tile
            ax_s = fig.add_subplot(gs[r_idx, c_syn])
            ax_s.imshow(syn_tile)
            ax_s.axis("off")
            if r_idx == 0:
                ax_s.set_title("Synthetic", fontsize=16.5, fontweight="bold", pad=8)

            axes_dict[(r_idx, m_idx)] = (ax_r, ax_s)

    # Force drawing to resolve exact axes bounding boxes
    fig.canvas.draw()

    # Place Vertical Scanner Names on the left of each row (centered to each row's bounding box)
    for r_idx, row_data in enumerate(SCANNER_ROWS):
        sname = row_data["scanner_name"]
        ax_first = axes_dict[(r_idx, 0)][0]
        bbox = ax_first.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        
        fig.text(
            0.026, y_center, sname,
            fontsize=15.0, fontweight="bold", color="black",
            ha="center", va="center", rotation=90
        )

    # Modality super-headers centered exactly above each (Real, Synthetic) column pair
    for m_idx, m_title in enumerate(mod_titles):
        ax_r, ax_s = axes_dict[(0, m_idx)]
        bbox_r = ax_r.get_position()
        bbox_s = ax_s.get_position()
        x_center = (bbox_r.x0 + bbox_s.x1) / 2.0
        
        fig.text(
            x_center, 0.968, m_title,
            fontsize=19.5, fontweight="bold", color="black",
            ha="center", va="center"
        )

    out_file = PANEL_DIR / "scanner_matrix_real_vs_synth_3rows.png"
    plt.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved master 3-row figure -> {out_file}")


if __name__ == "__main__":
    generate_figure()
