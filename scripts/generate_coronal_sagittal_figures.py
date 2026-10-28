#!/usr/bin/env python3
"""
generate_coronal_sagittal_figures.py
====================================
Generates a separate set of publication figures showing Coronal and Sagittal
multi-planar reconstructions (MPR) for 5 diverse samples per modality:
  1. T1-Weighted (T1w): intra_prompt_diversity_coronal_sagittal_t1w.png / .pdf
  2. T2-Weighted (T2w): intra_prompt_diversity_coronal_sagittal_t2w.png / .pdf
  3. Proton Density Weighted (PDw): intra_prompt_diversity_coronal_sagittal_pdw.png / .pdf
  4. T2-FLAIR: intra_prompt_diversity_coronal_sagittal_flair.png / .pdf

Key Features:
  - 5 diverse anatomical samples across seeds: [101, 202, 303, 404, 505].
  - 2 rows:
      Row 1: Coronal slices
      Row 2: Sagittal slices
  - Correct anatomical orientation and isotropic voxel spacing correction (z_scale = 1.36 / 0.94).
  - Clean top margin with zero redundant title clutter.
  - Column headers: "Sample 01 (Seed 101)" through "Sample 05 (Seed 505)".
  - Row labels: "Coronal" and "Sagittal".
  - Real Clinical MIDI Prompt formatted cleanly below the images across 3 well-proportioned lines.
  - Exports both 300 DPI PNG and vector PDF.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.generate_10seeds_separate_figures import MODALITIES_INFO


SEEDS = [101, 202, 303, 404, 505]


def extract_coronal_and_sagittal(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts canonical coronal and sagittal slices from 3D volume (256, 256, 128).
    Applies isotropic physical voxel spacing adjustment (z_scale = 1.36 / 0.94),
    standard radiological orientation (Superior at Top, Anterior facing Left in Sagittal),
    and tight center cropping.
    """
    norm_vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    bg_thresh = 0.08
    pts = np.where(norm_vol > bg_thresh)

    if len(pts[0]) > 500:
        x_min, x_max = int(np.percentile(pts[0], 2.0)), int(np.percentile(pts[0], 98.0))
        y_min, y_max = int(np.percentile(pts[1], 2.0)), int(np.percentile(pts[1], 98.0))
        # Midline sagittal (X) and mid-cerebral coronal (Y)
        x_mid = (x_min + x_max) // 2
        y_mid = int(y_min + 0.48 * (y_max - y_min))
    else:
        x_mid = norm_vol.shape[0] // 2
        y_mid = norm_vol.shape[1] // 2

    z_scale = 1.36 / 0.94

    # 1. Coronal: Superior at top, Left/Right horizontal
    raw_cor = norm_vol[:, y_mid, :].T[::-1, :]

    # 2. Sagittal: Superior at top, Anterior facing Left
    raw_sag = np.fliplr(norm_vol[x_mid, :, :].T[::-1, :])

    def process_plane(raw_2d: np.ndarray) -> np.ndarray:
        # Scale vertical dimension to match physical isotropic aspect ratio
        new_h = int(raw_2d.shape[0] * z_scale)
        im = Image.fromarray((raw_2d * 255).astype(np.uint8)).resize(
            (raw_2d.shape[1], new_h), resample=Image.Resampling.LANCZOS
        )
        arr = np.array(im).astype(np.float32) / 255.0

        # Square center crop around brain tissue
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
            res = Image.fromarray((crop * 255).astype(np.uint8)).resize(
                (512, 512), resample=Image.Resampling.LANCZOS
            )
            out = np.array(res).astype(np.float32) / 255.0
        else:
            out = arr

        p1, p99 = np.percentile(out[out > 0.02], [0.5, 99.5]) if np.any(out > 0.02) else (0, 1)
        return np.clip((out - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    cor_slice = process_plane(raw_cor)
    sag_slice = process_plane(raw_sag)
    return cor_slice, sag_slice


def plot_coronal_sagittal_modality(
    mod_id: str,
    info: dict,
    coronals: list[np.ndarray],
    sagittals: list[np.ndarray],
    seeds: list[int],
    out_dir: Path,
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

    fig = plt.figure(figsize=(15.5, 7.5), facecolor="#ffffff")

    # Grid for 2 rows x 5 columns
    # Left at 0.075 to give clean space for row labels ("Coronal", "Sagittal")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        left=0.075,
        right=0.97,
        top=0.95,
        bottom=0.20,
        wspace=0.06,
        hspace=0.18,
    )

    # Row 0: Coronal
    for c in range(5):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(coronals[c], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)

        ax.set_title(
            f"Sample {c+1:02d}  (Seed {seeds[c]})",
            fontsize=9.0,
            fontweight="bold",
            color="#1e293b",
            pad=5,
        )

    # Row 1: Sagittal
    for c in range(5):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(sagittals[c], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)

    # Row Headers on the left (clean vertical text or badges)
    fig.text(
        0.035,
        0.75,
        "Coronal",
        fontsize=10.5,
        fontweight="bold",
        color=info["badge_color"],
        ha="center",
        va="center",
        rotation=90,
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.25",
            facecolor=info["bg_badge"],
            edgecolor=info["badge_color"],
            linewidth=1.0,
        ),
    )

    fig.text(
        0.035,
        0.40,
        "Sagittal",
        fontsize=10.5,
        fontweight="bold",
        color=info["badge_color"],
        ha="center",
        va="center",
        rotation=90,
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.25",
            facecolor=info["bg_badge"],
            edgecolor=info["badge_color"],
            linewidth=1.0,
        ),
    )

    # Prompt card below images (formatted over 3 lines)
    lines = info["lines"]
    formatted_prompt = (
        f"Real Clinical MIDI Prompt:\n"
        f"  1. {lines[0]}\n"
        f"  2. {lines[1]}\n"
        f"  3. {lines[2]}"
    )

    fig.text(
        0.035,
        0.085,
        formatted_prompt,
        fontsize=8.5,
        fontfamily="monospace",
        fontweight="medium",
        color="#1e293b",
        ha="left",
        va="center",
        linespacing=1.45,
        bbox=dict(
            boxstyle="round,pad=0.6,rounding_size=0.3",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.0,
        ),
    )

    png_path = out_dir / f"intra_prompt_diversity_coronal_sagittal_{mod_id}.png"
    pdf_path = out_dir / f"intra_prompt_diversity_coronal_sagittal_{mod_id}.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {info['title']} Coronal/Sagittal -> {png_path} and {pdf_path}")


def main():
    cache_dir = Path("output/eval_memorization/diversity_cache")
    out_dir = Path("output/eval_memorization")

    for mod_id, info in MODALITIES_INFO.items():
        coronals = []
        sagittals = []
        for s in SEEDS:
            vol_file = cache_dir / f"vol_{info['key']}_real_seed{s}.npy"
            if not vol_file.exists():
                raise FileNotFoundError(f"Missing volume: {vol_file}")
            vol = np.load(vol_file)
            cor, sag = extract_coronal_and_sagittal(vol)
            coronals.append(cor)
            sagittals.append(sag)

        plot_coronal_sagittal_modality(mod_id, info, coronals, sagittals, SEEDS, out_dir)

    print("All coronal & sagittal diversity figures successfully generated!")


if __name__ == "__main__":
    main()
