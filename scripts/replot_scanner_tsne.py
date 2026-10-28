#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
replot_scanner_tsne.py
======================
Generates publication-quality t-SNE and PCA manifold visualizations for scanner clustering:
  1. Modality-specific marker shapes (T1W: circle, T2W: square, FLAIR: triangle, PDW: diamond, T2STARW: plus).
  2. Vendor-clustered color families (Siemens: Blues, GE: Reds/Oranges, Philips: Greens).
  3. Harmonized scanner labels (GE Signa HDx and HDxt merged as 'GE Signa HDx/t', underscores stripped from 'Discovery MR XXX').
  4. Dual legends for Scanner Hardware and MRI Modality.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def clean_scanner_label(scanner: str) -> str:
    """
    Sanitizes and harmonizes scanner labels:
      - Removes underscores (e.g. Discovery_MR750 -> Discovery MR750).
      - Unifies GE Signa HDX and HDxt into 'GE Signa HDx/t' idempotently.
      - Normalizes casing (DISCOVERY -> Discovery, SIGNA -> Signa, MR750W -> MR750w).
      - Removes redundant whitespace.
    """
    s = scanner.replace("_", " ")
    s = re.sub(r"\bDISCOVERY\b", "Discovery", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSIGNA\b", "Signa", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR750W\b", "MR750w", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR750\b", "MR750", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR450\b", "MR450", s, flags=re.IGNORECASE)
    # Robust and idempotent regex: cleanly condenses any Signa HDX / HDxt / HDx/t / HDx/t/t...
    s = re.sub(r"\bGE\s+Signa\s+HD[^\(\d]+", "GE Signa HDx/t ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSigna\s+HD[^\(\d]+", "Signa HDx/t ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_vendor_color_map(scanners: list[str]) -> dict[str, str]:
    """
    Assigns visually coherent, vendor-clustered and field-strength aware colors:
      - GE 3.0T Discovery MR750: Deep Crimson / Ruby Red (#b2182b)
      - GE 3.0T Discovery MR750w: Rich Coral Crimson (#d6604d)
      - GE 1.5T Discovery MR450: Rich Amber / Burnt Orange (#e6550d)
      - GE 1.5T Signa HDx/t: Warm Golden Amber (#feb24c)
      - GE 1.5T Signa other: Tangerine Orange (#fd8d3c)
      - Siemens 3.0T (MAGNETOM Vida, Trio): Midnight Navy Blue (#08306b)
      - Siemens 1.5T (Symphony): Deep Royal / Cobalt Blue (#08519c)
      - Siemens 1.5T (Aera): Vivid Ocean Blue (#2171b5)
      - Siemens 1.5T (Avanto): Crisp Sky Blue (#9ecae1)
      - Siemens 1.5T (MAGNETOM Sola): Steel Blue (#4292c6)
      - Philips 3.0T (Achieva dStream, Achieva): Deep Forest Green (#005a32, #006d2c)
      - Philips 1.5T (Ingenia, Intera): Fresh Mint / Sea Greens (#238b45, #74c476)
      - Others: Purples / Grays
    """
    color_map = {}
    for sc in sorted(scanners):
        sc_upper = sc.upper()
        is_3t = "3.0T" in sc_upper or "3T" in sc_upper

        if "GE" in sc_upper:
            # 3.0T GE Scanners: Crimson family
            if is_3t or "MR750" in sc_upper:
                if "MR750W" in sc_upper:
                    color_map[sc] = "#d6604d"  # Coral Crimson for 3.0T MR750w
                else:
                    color_map[sc] = "#b2182b"  # Deep Crimson / Ruby Red for 3.0T MR750
            # 1.5T Discovery MR450: Distinct Rich Amber-Orange
            elif "MR450" in sc_upper:
                color_map[sc] = "#e6550d"  # Rich Burnt Amber-Orange for 1.5T MR450
            # 1.5T Signa HDx/t: Warm Golden Amber
            elif "HDX" in sc_upper or "SIGNA" in sc_upper:
                color_map[sc] = "#feb24c"  # Warm Golden Amber for 1.5T Signa HDx/t
            else:
                color_map[sc] = "#fd8d3c"
        elif "SIEMENS" in sc_upper:
            if is_3t or "VIDA" in sc_upper or "TRIO" in sc_upper:
                color_map[sc] = "#081c3c"  # Midnight Prussian Navy for 3.0T
            elif "SYMPHONY" in sc_upper:
                color_map[sc] = "#0d47a1"  # Deep Royal / Indigo Cobalt Blue (Dark)
            elif "AERA" in sc_upper:
                color_map[sc] = "#0284c7"  # Vivid Electric Cerulean / Azure Blue (Medium)
            elif "AVANTO" in sc_upper:
                color_map[sc] = "#7dd3fc"  # Crisp Light Sky / Ice Blue (Light)
            elif "SOLA" in sc_upper:
                color_map[sc] = "#475569"  # Steel Slate Blue
            else:
                color_map[sc] = "#0284c7"
        elif "PHILIPS" in sc_upper:
            if is_3t or "DSTREAM" in sc_upper or "ACHIEVA" in sc_upper:
                color_map[sc] = "#005a32"  # Deep Emerald for 3.0T
            elif "INGENIA" in sc_upper:
                color_map[sc] = "#238b45"  # Medium Green for 1.5T
            elif "INTERA" in sc_upper:
                color_map[sc] = "#74c476"  # Mint Green for 1.5T
            else:
                color_map[sc] = "#2ca02c"
        elif "CANON" in sc_upper or "TOSHIBA" in sc_upper:
            color_map[sc] = "#54278f" if is_3t else "#756bb1"
        else:
            color_map[sc] = "#636363"
    return color_map


def render_projections(
    all_embeddings: list[dict],
    out_dir: Path,
    title_suffix: str = "Across All Modalities",
    filename_suffix: str = "_all_modalities",
):
    synth_items = [
        item for item in all_embeddings
        if item.get("type", "Synthetic") == "Synthetic" and item.get("modality") != "T2STARW"
    ]
    if len(synth_items) < 4:
        print(f"[WARN] Not enough synthetic items ({len(synth_items)}) to plot.")
        return

    X = np.array([item["feat"] for item in synth_items])
    scanners = [clean_scanner_label(item["scanner"]) for item in synth_items]
    modalities = [item.get("modality", "MRI") for item in synth_items]

    unique_scanners = sorted(list(set(scanners)))
    unique_modalities = sorted(list(set(modalities)))

    scanner_color_map = get_vendor_color_map(unique_scanners)
    modality_markers = {
        "T1W": "o",       # Circle
        "T2W": "s",       # Square
        "FLAIR": "^",     # Triangle Up
        "PDW": "D",       # Diamond
    }
    default_marker = "v"

    scanner_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scanner_color_map[sc],
                   markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                   label=sc)
        for sc in unique_scanners
    ]
    modality_handles = [
        plt.Line2D([0], [0], marker=modality_markers.get(mod, default_marker), color="w",
                   markerfacecolor="#4d4d4d", markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                   label=mod)
        for mod in unique_modalities
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PCA Projection (Generous width to ensure all scanner names fit comfortably)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16.0, 9.0),
        dpi=220,
        gridspec_kw={"width_ratios": [3.8, 1.3]}
    )
    ax_leg.axis("off")

    for sc in unique_scanners:
        for mod in unique_modalities:
            idx = [i for i in range(len(synth_items)) if scanners[i] == sc and modalities[i] == mod]
            if idx:
                marker = modality_markers.get(mod, default_marker)
                ax.scatter(
                    X_pca[idx, 0], X_pca[idx, 1],
                    color=scanner_color_map[sc],
                    marker=marker,
                    s=85,
                    alpha=0.92,
                    edgecolors="black",
                    linewidth=0.6,
                )

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.025
    except Exception:
        leg2_top = max(0.05, 1.0 - (len(unique_scanners) * 0.040 + 0.08))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.0, leg2_top),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )

    pca_path = out_dir / f"scanner_pca{filename_suffix}.png"
    fig.savefig(str(pca_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {pca_path}")

    # 2. t-SNE Projection
    perplexity = min(30, max(3, len(X) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16.0, 9.0),
        dpi=220,
        gridspec_kw={"width_ratios": [3.8, 1.3]}
    )
    ax_leg.axis("off")

    for sc in unique_scanners:
        for mod in unique_modalities:
            idx = [i for i in range(len(synth_items)) if scanners[i] == sc and modalities[i] == mod]
            if idx:
                marker = modality_markers.get(mod, default_marker)
                ax.scatter(
                    X_tsne[idx, 0], X_tsne[idx, 1],
                    color=scanner_color_map[sc],
                    marker=marker,
                    s=85,
                    alpha=0.92,
                    edgecolors="black",
                    linewidth=0.6,
                )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.025
    except Exception:
        leg2_top = max(0.05, 1.0 - (len(unique_scanners) * 0.040 + 0.08))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.0, leg2_top),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )

    tsne_path = out_dir / f"scanner_tsne{filename_suffix}.png"
    fig.savefig(str(tsne_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {tsne_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-render t-SNE / PCA scanner clustering figures with new styling.")
    parser.add_argument(
        "--embeddings-pkl",
        type=str,
        default="output/validation_experiments/exp4_scanner_clustering/scanner_embeddings.pkl",
        help="Path to saved embeddings pickle",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/validation_experiments/exp4_scanner_clustering",
        help="Directory to save output figures",
    )
    args = parser.parse_args()

    pkl_path = Path(args.embeddings_pkl)
    if not pkl_path.exists():
        print(f"[ERROR] Embeddings pickle not found at {pkl_path}.")
        print("To generate new embeddings and plots, run Experiment 4 via scripts.run_mrclip_validation_experiments:")
        print("  CUDA_VISIBLE_DEVICES=1 python3 -m scripts.run_mrclip_validation_experiments -e ... -c ... -t ... --experiments 4")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        all_embeddings = pickle.load(f)

    print(f"Loaded {len(all_embeddings)} embedding items from {pkl_path}.")
    out_dir = Path(args.out_dir)
    render_projections(all_embeddings, out_dir, title_suffix="Across All Modalities", filename_suffix="_all_modalities")
    print("Done!")


if __name__ == "__main__":
    main()
