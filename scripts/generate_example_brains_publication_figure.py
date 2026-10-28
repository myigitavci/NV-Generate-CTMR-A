#!/usr/bin/env python3
"""
generate_example_brains_publication_figure.py
=============================================
Publication-grade, compact comparison figure demonstrating metadata-conditioned
3D brain MRI synthesis across 3 major clinical contrasts:
  1. T1-Weighted (T1w)
  2. T2-Weighted (T2w)
  3. T2-FLAIR

For each contrast:
  - Displays full acquisition and scanner metadata badge
  - Real Ground Truth Scan (Axial, Coronal, Sagittal)
  - Synthesized MRI (Axial, Coronal, Sagittal)
  - Clean anatomical crops focusing on cerebrum / ventricles / cortical ribbon.
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Global Styling
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

out_dir = Path("output/eval_physics")
out_dir.mkdir(parents=True, exist_ok=True)


def normalize_volume(vol: np.ndarray, lower_pct: float = 1.5, upper_pct: float = 99.0) -> np.ndarray:
    """Robust percentile intensity windowing to [0, 1]."""
    v = vol.astype(np.float32)
    bg_thresh = np.percentile(v, 8.0)
    fg = v[v > bg_thresh]
    if len(fg) > 100:
        p_low = np.percentile(fg, lower_pct)
        p_high = np.percentile(fg, upper_pct)
    else:
        p_low = np.percentile(v, lower_pct)
        p_high = np.percentile(v, upper_pct)

    if p_high > p_low:
        v_norm = np.clip((v - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        v_norm = np.clip(v, 0.0, 1.0)
    return v_norm


def crop_brain_cerebrum(slice_2d: np.ndarray, is_real: bool = False, plane: str = "axial", pad_ratio: float = 0.035) -> np.ndarray:
    """Crop brain slice to a tight square focusing on the cranial/cerebral region."""
    gray = slice_2d.astype(np.float32)
    thresh = max(0.06, float(np.percentile(gray[gray > 0], 6.0)) if (gray > 0).any() else 0.06)
    pts = np.where(gray > thresh)

    if len(pts[0]) > 50:
        y_min, y_max = int(np.min(pts[0])), int(np.max(pts[0]))
        x_min, x_max = int(np.min(pts[1])), int(np.max(pts[1]))

        if is_real and plane == "coronal":
            # Real coronal: head top is y_min, neck/chin is y_max. Crop to upper 72%
            y_max = int(y_min + (y_max - y_min) * 0.72)
        elif is_real and plane == "sagittal":
            # Real sagittal: head top is y_min, neck is y_max. Crop to upper 73%
            y_max = int(y_min + (y_max - y_min) * 0.73)
            # Anterior face profile on right, trim facial profile to center brain
            x_max = int(x_min + (x_max - x_min) * 0.88)

        h = y_max - y_min
        w = x_max - x_min
        side = max(h, w)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        y0 = cy - side_padded // 2
        y1 = y0 + side_padded
        x0 = cx - side_padded // 2
        x1 = x0 + side_padded

        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - slice_2d.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - slice_2d.shape[1])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            slice_2d = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left

        return slice_2d[y0:y1, x0:x1]
    return slice_2d


def extract_slices(vol: np.ndarray, is_synthetic: bool = False, modality: str = "T1w") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract optimal Axial, Coronal, and Sagittal slices."""
    sx, sy, sz = vol.shape

    is_real_case = not is_synthetic
    if not is_real_case:
        # Synthetic volume (256, 256, 128)
        if modality == "T1w":
            z_idx = int(sz * 0.48)
            y_idx = int(sy * 0.50)
            x_idx = int(sx * 0.50)
        elif modality == "T2w":
            z_idx = int(sz * 0.53)
            y_idx = int(sy * 0.50)
            x_idx = int(sx * 0.50)
        else: # FLAIR
            z_idx = int(sz * 0.50)
            y_idx = int(sy * 0.50)
            x_idx = int(sx * 0.50)

        ax = np.rot90(vol[:, :, z_idx])
        cor = np.rot90(vol[:, y_idx, :])
        sag = np.rot90(vol[x_idx, :, :])

        # Anisotropic aspect ratio correction
        z_aspect = 1.36 / 0.94
        im_cor = Image.fromarray((np.clip(cor, 0, 1) * 255).astype(np.uint8))
        cor = np.array(im_cor.resize((cor.shape[1], int(round(cor.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

        im_sag = Image.fromarray((np.clip(sag, 0, 1) * 255).astype(np.uint8))
        sag = np.array(im_sag.resize((sag.shape[1], int(round(sag.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    else:
        # Real OASIS3 volume (isotropic ~256x256x256)
        if modality == "T1w":
            z_idx = int(sz * 0.65)
            y_idx = int(sy * 0.52)
            x_idx = int(sx * 0.50)
        elif modality == "T2w":
            z_idx = int(sz * 0.64)
            y_idx = int(sy * 0.52)
            x_idx = int(sx * 0.50)
        else: # FLAIR
            z_idx = int(sz * 0.64)
            y_idx = int(sy * 0.52)
            x_idx = int(sx * 0.50)

        ax = np.rot90(vol[:, :, z_idx])
        cor = np.rot90(vol[:, y_idx, :])
        sag = np.rot90(vol[x_idx, :, :])

    return (
        crop_brain_cerebrum(ax, is_real=is_real_case, plane="axial"),
        crop_brain_cerebrum(cor, is_real=is_real_case, plane="coronal"),
        crop_brain_cerebrum(sag, is_real=is_real_case, plane="sagittal"),
    )


def main():
    print("Generating master publication figure...", flush=True)

    oasis_base = Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS31000/OAS31000_MR_d0072")

    cases = [
        {
            "id": "T1w",
            "title": "T1-Weighted (T1w)",
            "color": "#1e3a8a",  # Deep Navy Blue
            "accent": "#0284c7", # Sky Blue
            "scanner": "Siemens MAGNETOM Vida (3.0T)",
            "sequence": "Axial Spin Echo (SE)",
            "te": "12 ms",
            "tr": "550 ms",
            "ti": "None",
            "fa": "90°",
            "real_nii": str(oasis_base / "anat4/NIFTI/ur_sub-OAS31000_ses-d0072_T1w.nii.gz"),
            "syn_nii": "output/validation_experiments/exp2_multimodal_panel/subject_02_T1_Weighted.nii.gz",
        },
        {
            "id": "T2w",
            "title": "T2-Weighted (T2w)",
            "color": "#14532d",  # Forest Green
            "accent": "#16a34a", # Vibrant Green
            "scanner": "Siemens MAGNETOM Vida (3.0T)",
            "sequence": "Axial Spin Echo (SE)",
            "te": "95 ms",
            "tr": "4500 ms",
            "ti": "None",
            "fa": "90°",
            "real_nii": str(oasis_base / "anat2/NIFTI/ur_sub-OAS31000_ses-d0072_acq-TSE_T2w.nii.gz"),
            "syn_nii": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_Weighted.nii.gz",
        },
        {
            "id": "FLAIR",
            "title": "T2-FLAIR",
            "color": "#581c87",  # Deep Purple
            "accent": "#9333ea", # Vibrant Purple
            "scanner": "Philips Achieva dStream (3.0T)",
            "sequence": "Axial Inversion Recovery (IR)",
            "te": "340 ms",
            "tr": "4800 ms",
            "ti": "1650 ms",
            "fa": "90°",
            "real_nii": str(oasis_base / "anat3/NIFTI/ur_sub-OAS31000_ses-d0072_FLAIR.nii.gz"),
            "syn_nii": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_FLAIR.nii.gz",
        },
    ]

    processed = []
    for c in cases:
        print(f"Loading {c['title']}...", flush=True)
        vol_r = normalize_volume(nib.load(c["real_nii"]).get_fdata())
        vol_s = normalize_volume(nib.load(c["syn_nii"]).get_fdata())

        r_slices = extract_slices(vol_r, is_synthetic=False, modality=c["id"])
        s_slices = extract_slices(vol_s, is_synthetic=True, modality=c["id"])

        processed.append({
            "info": c,
            "real": r_slices,
            "syn": s_slices,
        })

    # -----------------------------------------------------------------------
    # Master Canvas Layout (Compact, Sleek, Publication-Quality)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(13.2, 7.2), dpi=300, facecolor="#ffffff")

    outer_gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.28,
        left=0.03,
        right=0.985,
        top=0.905,
        bottom=0.03,
    )

    plane_labels = ["Axial", "Coronal", "Sagittal"]
    row_axes_registry = []

    for row_idx, p_case in enumerate(processed):
        info = p_case["info"]
        row_gs = outer_gs[row_idx].subgridspec(
            1, 3,
            width_ratios=[1.15, 1.45, 1.45],  # [Card, Real 3-plane, Syn 3-plane]
            wspace=0.18,
        )

        # 1. Metadata Info Card
        ax_card = fig.add_subplot(row_gs[0, 0])
        ax_card.set_xticks([])
        ax_card.set_yticks([])
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.axis("off")

        # Outer card box
        rect = patches.FancyBboxPatch(
            (0.02, 0.04), 0.96, 0.92,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            transform=ax_card.transAxes,
        )
        ax_card.add_patch(rect)

        # Contrast banner header
        banner = patches.FancyBboxPatch(
            (0.02, 0.68), 0.96, 0.28,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=0,
            facecolor=info["color"],
            transform=ax_card.transAxes,
        )
        ax_card.add_patch(banner)

        # Header title text
        ax_card.text(
            0.50, 0.81, info["title"],
            fontsize=11.5, fontweight="bold", color="#ffffff",
            ha="center", va="center", transform=ax_card.transAxes
        )

        # Parameter fields
        meta_lines = [
            f"Scanner: {info['scanner']}",
            f"Sequence: {info['sequence']}",
            f"TE / TR: {info['te']} / {info['tr']}",
            f"TI: {info['ti']}  |  Flip Angle: {info['fa']}",
        ]
        
        y_pos = 0.54
        for line in meta_lines:
            ax_card.text(
                0.08, y_pos, line,
                fontsize=8.2, color="#1e293b", fontweight="semibold" if "TE" in line or "Scanner" in line else "normal",
                ha="left", va="center", transform=ax_card.transAxes
            )
            y_pos -= 0.14

        # 2. Real Ground Truth Scan (3 Planes)
        real_container = row_gs[0, 1].subgridspec(1, 3, wspace=0.03)
        real_axes = []
        for p_i in range(3):
            ax_r = fig.add_subplot(real_container[0, p_i])
            real_axes.append(ax_r)
            slc = p_case["real"][p_i]
            ax_r.imshow(slc, cmap="gray", aspect="equal", vmin=0.0, vmax=1.0)
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            for spine in ax_r.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.0)

            if row_idx == 0:
                ax_r.set_title(plane_labels[p_i], fontsize=10.0, fontweight="bold", color="#334155", pad=4)

        # 3. Model Synthesized Scan (3 Planes)
        syn_container = row_gs[0, 2].subgridspec(1, 3, wspace=0.03)
        syn_axes = []
        for p_i in range(3):
            ax_s = fig.add_subplot(syn_container[0, p_i])
            syn_axes.append(ax_s)
            slc = p_case["syn"][p_i]
            ax_s.imshow(slc, cmap="gray", aspect="equal", vmin=0.0, vmax=1.0)
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            for spine in ax_s.spines.values():
                spine.set_edgecolor(info["accent"])
                spine.set_linewidth(1.3)

            if row_idx == 0:
                ax_s.set_title(plane_labels[p_i], fontsize=10.0, fontweight="bold", color="#334155", pad=4)

        row_axes_registry.append({
            "card": ax_card,
            "real_axes": real_axes,
            "syn_axes": syn_axes,
        })

    fig.canvas.draw()

    # Dynamic midpoints for main column headers
    x_card_mid = (row_axes_registry[0]["card"].get_position().x0 + row_axes_registry[0]["card"].get_position().x1) / 2.0
    x_real_mid = (row_axes_registry[0]["real_axes"][0].get_position().x0 + row_axes_registry[0]["real_axes"][2].get_position().x1) / 2.0
    x_syn_mid = (row_axes_registry[0]["syn_axes"][0].get_position().x0 + row_axes_registry[0]["syn_axes"][2].get_position().x1) / 2.0

    fig.text(x_card_mid, 0.955, "Acquisition Metadata", fontsize=14.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_real_mid, 0.955, "Real Ground Truth Scan", fontsize=14.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_syn_mid, 0.955, "Synthesised MRI (Ours)", fontsize=14.0, fontweight="bold", color="#0f172a", ha="center")

    out_file = out_dir / "metadata_conditioned_real_vs_synthetic_brains.png"
    fig.savefig(str(out_file), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Successfully generated publication figure: {out_file}", flush=True)


if __name__ == "__main__":
    main()
