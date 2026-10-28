#!/usr/bin/env python3
"""
generate_triplanar_figures.py
=============================
Generates publication figures showing complete tri-planar (Axial, Coronal, Sagittal)
multi-planar reconstructions for 5 diverse samples per modality:
  1. T1-Weighted (T1w)
  2. T2-Weighted (T2w)
  3. Proton Density Weighted (PDw)
  4. T2-FLAIR

Layout:
  - 5 samples across columns: Seed 101, 202, 303, 404, 505
  - 3 rows:
      Row 1: Axial
      Row 2: Coronal
      Row 3: Sagittal
  - Left row badges: "Axial", "Coronal", "Sagittal" in modality theme color
  - 3-line clinical MIDI prompt card below
  - Exports both 300 DPI PNG and vector PDF
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

from scripts.generate_10seeds_separate_figures import MODALITIES_INFO, extract_mid_axial_slice
from scripts.generate_coronal_sagittal_figures import extract_coronal_and_sagittal

SEEDS = [101, 202, 303, 404, 505]


def plot_triplanar_modality(
    mod_id: str,
    info: dict,
    axials: list[np.ndarray],
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

    fig = plt.figure(figsize=(15.5, 11.2), facecolor="#ffffff")

    # Grid for 3 rows x 5 columns (symmetrically centered with generous space below)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        left=0.035,
        right=0.965,
        top=0.96,
        bottom=0.15,
        wspace=0.05,
        hspace=0.12,
    )

    rows_data = [
        ("Axial", axials),
        ("Coronal", coronals),
        ("Sagittal", sagittals),
    ]

    for r, (plane_name, plane_imgs) in enumerate(rows_data):
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(plane_imgs[c], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor("#cbd5e1")
                spine.set_linewidth(1.1)

            if r == 0:
                ax.set_title(
                    f"Sample {c+1:02d}  (Seed {seeds[c]})",
                    fontsize=10.0,
                    fontweight="bold",
                    color="#1e293b",
                    pad=6,
                )

    # Prompt card below images: centralized, clean, no 1/2/3 numbering
    lines = info["lines"]
    formatted_prompt = (
        f"{lines[0]}\n"
        f"{lines[1]}\n"
        f"{lines[2]}"
    )

    fig.text(
        0.5,
        0.082,
        formatted_prompt,
        fontsize=10.5,
        fontfamily="monospace",
        fontweight="medium",
        color="#1e293b",
        ha="center",
        va="center",
        linespacing=1.45,
        bbox=dict(
            boxstyle="round,pad=0.75,rounding_size=0.35",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.1,
        ),
    )

    # Invisible spacer at bottom to guarantee generous breathing room between prompt and end of image
    fig.text(0.5, 0.015, " ", fontsize=4, color="none")

    # Save to triplanar filenames
    png_path_tri = out_dir / f"intra_prompt_diversity_triplanar_{mod_id}.png"
    pdf_path_tri = out_dir / f"intra_prompt_diversity_triplanar_{mod_id}.pdf"
    plt.savefig(png_path_tri, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.savefig(pdf_path_tri, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)

    # Also update coronal_sagittal filenames so user's active viewer tab refreshes
    png_path_cs = out_dir / f"intra_prompt_diversity_coronal_sagittal_{mod_id}.png"
    pdf_path_cs = out_dir / f"intra_prompt_diversity_coronal_sagittal_{mod_id}.pdf"
    plt.savefig(png_path_cs, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.savefig(pdf_path_cs, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)

    plt.close(fig)
    print(f"Saved {info['title']} Triplanar -> {png_path_tri} and {pdf_path_tri}")


def main():
    cache_dir = Path("output/eval_memorization/diversity_cache")
    out_dir = Path("output/eval_memorization")

    for mod_id, info in MODALITIES_INFO.items():
        axials = []
        coronals = []
        sagittals = []
        for s in SEEDS:
            vol_file = cache_dir / f"vol_{info['key']}_real_seed{s}.npy"
            if not vol_file.exists():
                raise FileNotFoundError(f"Missing volume: {vol_file}")
            vol = np.load(vol_file)

            ax = extract_mid_axial_slice(vol)
            cor, sag = extract_coronal_and_sagittal(vol)

            axials.append(ax)
            coronals.append(cor)
            sagittals.append(sag)

        plot_triplanar_modality(mod_id, info, axials, coronals, sagittals, SEEDS, out_dir)

    print("All tri-planar diversity figures successfully generated!")


if __name__ == "__main__":
    main()
