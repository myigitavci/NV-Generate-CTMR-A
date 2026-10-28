"""
scripts/plot_fid_publication.py
================================
Standalone script to generate publication-quality FID figures from
output/eval_fid_vs_csv/fid_summary.json.

Displays 2D Inception FID grouped by modality with an elegant pastel plane palette.

Usage:
  python scripts/plot_fid_publication.py \
    --json output/eval_fid_vs_csv/fid_summary.json \
    --output-dir output/eval_fid_vs_csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Elegant Pastel Palette for Anatomical Planes
# Soft, harmonious, publication-grade pastels
# ---------------------------------------------------------------------------
PLANE_CONFIG = {
    "axial": {
        "color": "#76A5D6",       # Soft Pastel Blue / Periwinkle
        "edge": "#4A7CB2",        # Refined Slate Blue border
        "label": "Axial",
    },
    "coronal": {
        "color": "#74C496",       # Soft Pastel Mint / Sage Green
        "edge": "#48986B",        # Refined Forest Sage border
        "label": "Coronal",
    },
    "sagittal": {
        "color": "#F39E75",       # Soft Pastel Peach / Apricot
        "edge": "#C86E45",        # Refined Terracotta border
        "label": "Sagittal",
    },
}

RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "mathtext.fontset":   "dejavusans",
    "font.size":          9.5,
    "axes.titlesize":     11.5,
    "axes.labelsize":     10.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    9.5,
    "axes.linewidth":     0.9,
    "axes.edgecolor":     "#475569",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.linewidth":     0.6,
    "grid.alpha":         0.40,
    "grid.color":         "#e2e8f0",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
}


def make_publication_figure(data: dict, out_dir: Path):
    plt.rcParams.update(RC)

    modalities = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in data]
    if not modalities:
        modalities = list(data.keys())
    n_mods = len(modalities)

    planes = ["axial", "coronal", "sagittal"]
    n_planes = len(planes)

    # -----------------------------------------------------------------------
    # Figure Layout: Clean, Focused, High-Impact Panel
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=300, facecolor="#ffffff")

    cluster_step = 1.05          # Distance between modality centers
    bar_step = 0.225             # Center-to-center distance between adjacent plane bars
    bar_width = 0.165            # Width of each bar (leaves distinct, clean ~0.06 gap between bars)
    group_centers = np.arange(n_mods) * cluster_step
    offsets = [-bar_step, 0.0, bar_step]

    max_val = 0.0
    for m in modalities:
        for pl in planes:
            if pl in data[m]:
                v = data[m][pl]["raw_fid"]
                if v > max_val:
                    max_val = v

    # Background subtle modality column shading for visual grouping
    for mi, gc in enumerate(group_centers):
        if mi % 2 == 1:
            ax.axvspan(
                gc - cluster_step / 2,
                gc + cluster_step / 2,
                color="#f8fafc",
                zorder=0,
            )

    # Plot Bars
    for pi, pl in enumerate(planes):
        cfg = PLANE_CONFIG[pl]
        for mi, mod in enumerate(modalities):
            if pl not in data[mod]:
                continue
            val = data[mod][pl]["raw_fid"]
            xpos = group_centers[mi] + offsets[pi]

            # Main pastel bar with clean edge and breathing space
            ax.bar(
                xpos,
                val,
                width=bar_width,
                color=cfg["color"],
                edgecolor=cfg["edge"],
                linewidth=0.9,
                zorder=3,
            )

            # Value label cleanly positioned above bar
            ax.text(
                xpos,
                val + 3.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#334155",
                fontweight="semibold",
                zorder=4,
            )

    # Styling Axes
    ax.set_ylabel("Fréchet Distance", fontsize=11, fontweight="bold", labelpad=8, color="#1e293b")
    ax.set_title("2D Inception FID by Plane", fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.50, color="#cbd5e1", zorder=1)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    # Modality Labels on X-axis (clean, no plane abbreviations)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(modalities, fontsize=10.5, fontweight="bold", color="#1e293b")
    ax.tick_params(axis="x", pad=8, length=0)
    ax.tick_params(axis="y", colors="#334155", labelsize=9.5)

    ylim_top = 300.0 if max_val <= 270 else max_val * 1.16
    ax.set_ylim(0, ylim_top)
    ax.set_xlim(
        group_centers[0] - cluster_step / 2,
        group_centers[-1] + cluster_step / 2,
    )

    # Dedicated Plane Legend (Top-Right) without abbreviations
    legend_handles = [
        mpatches.Patch(
            facecolor=PLANE_CONFIG[pl]["color"],
            edgecolor=PLANE_CONFIG[pl]["edge"],
            label=PLANE_CONFIG[pl]["label"],
            linewidth=1.0,
        )
        for pl in planes
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9.0,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.96,
        borderpad=0.6,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.0,
        title="Plane",
        title_fontproperties={"weight": "bold", "size": 9.5},
    )

    # -----------------------------------------------------------------------
    # Save Figures in Both PDF (Vector) and PNG (300 DPI)
    # -----------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = out_dir / f"fid_publication.{ext}"
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0.06)
        print(f"Saved: {path}")

    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="output/eval_fid_vs_csv/fid_summary.json")
    p.add_argument("--output-dir", default="output/eval_fid_vs_csv")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.json) as f:
        data = json.load(f)
    make_publication_figure(data, Path(args.output_dir))


if __name__ == "__main__":
    main()
