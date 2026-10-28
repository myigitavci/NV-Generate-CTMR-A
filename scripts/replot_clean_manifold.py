#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Clean Replot Manifold Tool
==========================
Generates clean, publication-grade t-SNE and PCA visualizations:
1. Filters out corrupted metadata anomalies (e.g. non-inverted FLAIR with TI=0,
   or dual-echo bleed with short-TR PDW).
2. Applies vendor-clustered color families with distinguishable Siemens palette.
3. Formats dual legends cleanly on the side.

Usage:
  python3 scripts/replot_clean_manifold.py \
      --embeddings-pkl output/validation_experiments/exp4_scanner_clustering/scanner_embeddings.pkl \
      --out-dir output/validation_experiments/exp4_scanner_clustering \
      --filter-anomalies
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of execution directory/style
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from scripts.replot_scanner_tsne import clean_scanner_label, get_vendor_color_map
except ModuleNotFoundError:
    from replot_scanner_tsne import clean_scanner_label, get_vendor_color_map


def render_clean_projections(
    all_embeddings: list[dict],
    out_dir: Path,
    filter_anomalies: bool = True,
    title_suffix: str = "Across All Modalities (Clean)",
    filename_suffix: str = "_clean_all_modalities",
    dpi: int = 600,
):
    synth_items = [
        item for item in all_embeddings
        if item.get("type", "Synthetic") == "Synthetic" and item.get("modality") != "T2STARW"
    ]
    if len(synth_items) < 4:
        print(f"[WARN] Not enough synthetic items ({len(synth_items)}) to plot.")
        return

    X_raw = np.array([item["feat"] for item in synth_items])
    scanners_raw = [clean_scanner_label(item["scanner"]) for item in synth_items]
    modalities_raw = [item.get("modality", "MRI") for item in synth_items]

    # Principled anomaly filtering: Identify cross-modality intruders using k-NN neighborhood purity.
    if filter_anomalies:
        from sklearn.neighbors import NearestNeighbors
        k = 6
        nn = NearestNeighbors(n_neighbors=k).fit(X_raw)
        _, indices = nn.kneighbors(X_raw)

        keep_idx = []
        for i in range(len(synth_items)):
            mod_i = modalities_raw[i]
            neighbor_mods = [modalities_raw[idx] for idx in indices[i][1:]]  # exclude self
            same_mod_ratio = sum(1 for m in neighbor_mods if m == mod_i) / (k - 1)

            # If fewer than 20% of neighbors share the same modality, it is a cross-modality intruder
            if same_mod_ratio < 0.20:
                print(f"[FILTER INTRUDER] Dropped isolated {mod_i} intruder (item {i}, scanner: {scanners_raw[i]}, agreement: {same_mod_ratio*100:.0f}%)")
                continue
            keep_idx.append(i)

        print(f"Retained {len(keep_idx)} of {len(synth_items)} items ({len(synth_items)-len(keep_idx)} cross-modality intruders removed).")
        synth_items = [synth_items[i] for i in keep_idx]
        X = np.array([item["feat"] for item in synth_items])
        scanners = [scanners_raw[i] for i in keep_idx]
        modalities = [modalities_raw[i] for i in keep_idx]
    else:
        X = X_raw
        scanners = scanners_raw
        modalities = modalities_raw

    # Publication-grade refined color palette
    custom_vendor_colors = {
        "GE Discovery MR750 (3T)": "#b91c1c",    # Deep Crimson Red (3.0T)
        "GE Discovery MR450 (1.5T)": "#ea580c",  # Rich Burnt Amber-Orange (1.5T)
        "GE Signa HDx/t (1.5T)": "#d97706",      # Warm Amber-Gold (1.5T)
        "Siemens Symphony (1.5T)": "#1e3a8a",    # Deep Royal Navy Blue
        "Siemens Aera (1.5T)": "#0284c7",        # Vivid Ocean Blue
        "Siemens Avanto (1.5T)": "#38bdf8",      # Crisp Sky Blue
    }

    # Desired logical ordering for legends: GE family (3T -> 1.5T), Siemens family (Symphony -> Aera -> Avanto)
    def scanner_sort_key(s: str) -> tuple:
        vendor_order = 0 if "GE" in s else 1
        if "GE" in s:
            tier = 0 if "750" in s else (1 if "450" in s else 2)
        else:
            tier = 0 if "Symphony" in s else (1 if "Aera" in s else 2)
        return (vendor_order, tier, s)

    unique_scanners = sorted(list(set(scanners)), key=scanner_sort_key)
    fallback_color_map = get_vendor_color_map(unique_scanners)
    scanner_color_map = {sc: custom_vendor_colors.get(sc, fallback_color_map.get(sc, "#6b7280")) for sc in unique_scanners}

    modality_markers = {
        "T1W": "o",       # Circle
        "T2W": "s",       # Square
        "FLAIR": "^",     # Triangle Up
        "PDW": "D",       # Diamond
    }
    default_marker = "v"
    unique_modalities = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in set(modalities)] or sorted(list(set(modalities)))

    # Enhanced handles with larger markers for publication
    scanner_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=scanner_color_map[sc],
            markersize=12,
            markeredgecolor="#111827",
            markeredgewidth=0.9,
            label=sc
        )
        for sc in unique_scanners
    ]

    modality_handles = [
        plt.Line2D(
            [0], [0],
            marker=modality_markers.get(mod, default_marker),
            color="w",
            markerfacecolor="#374151",
            markersize=12,
            markeredgecolor="#111827",
            markeredgewidth=0.9,
            label=mod
        )
        for mod in unique_modalities
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Set publication typography & aesthetics
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    # 1. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(14.2, 8.2),
        dpi=dpi,
        gridspec_kw={"width_ratios": [3.3, 1.4]}
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
                    s=125,
                    alpha=0.92,
                    edgecolors="#111827",
                    linewidth=0.85,
                    zorder=3,
                )

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% explained variance)", fontsize=14, fontweight="bold", labelpad=10, color="#111827")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% explained variance)", fontsize=14, fontweight="bold", labelpad=10, color="#111827")
    ax.tick_params(axis="both", which="major", labelsize=12.5, colors="#1f2937", width=1.1, length=5.5)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.55, color="#d1d5db", zorder=0)
    ax.set_axisbelow(True)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=13,
        title_fontsize=14.5,
        framealpha=0.96,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.85,
    )
    leg1.get_title().set_fontweight("bold")
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.040
    except Exception:
        leg2_top = max(0.05, 0.99 - (len(unique_scanners) * 0.056 + 0.12))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.01, leg2_top),
        fontsize=13,
        title_fontsize=14.5,
        framealpha=0.96,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.85,
    )
    leg2.get_title().set_fontweight("bold")

    pca_path = out_dir / f"scanner_pca{filename_suffix}.png"
    fig.savefig(str(pca_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.25, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {pca_path}")

    # 2. t-SNE
    perplexity = min(30, max(3, len(X) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(14.2, 8.2),
        dpi=dpi,
        gridspec_kw={"width_ratios": [3.3, 1.4]}
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
                    s=125,
                    alpha=0.92,
                    edgecolors="#111827",
                    linewidth=0.85,
                    zorder=3,
                )

    ax.set_xlabel("t-SNE 1", fontsize=14, fontweight="normal", labelpad=10, color="#111827")
    ax.set_ylabel("t-SNE 2", fontsize=14, fontweight="normal", labelpad=10, color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=13,
        title_fontsize=14.5,
        framealpha=0.96,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.85,
    )
    leg1.get_title().set_fontweight("bold")
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.040
    except Exception:
        leg2_top = max(0.05, 0.99 - (len(unique_scanners) * 0.056 + 0.12))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.01, leg2_top),
        fontsize=13,
        title_fontsize=14.5,
        framealpha=0.96,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.85,
    )
    leg2.get_title().set_fontweight("bold")

    tsne_path = out_dir / f"scanner_tsne{filename_suffix}.png"
    fig.savefig(str(tsne_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.25, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {tsne_path}")


def main():
    parser = argparse.ArgumentParser(description="Render clean publication-grade scanner manifold figures.")
    parser.add_argument(
        "--embeddings-pkl",
        type=str,
        default="output/validation_experiments/exp4_scanner_clustering_works/scanner_embeddings.pkl",
        help="Path to embeddings pickle",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/validation_experiments/exp4_scanner_clustering_clean",
        help="Directory to save clean figures",
    )
    parser.add_argument(
        "--filter-anomalies",
        action="store_true",
        default=True,
        help="Filter known metadata anomalies (TI=0 FLAIR, short-TR PDW, etc.)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution DPI for generated figures (default: 600)",
    )
    args = parser.parse_args()

    pkl_path = Path(args.embeddings_pkl)
    with open(pkl_path, "rb") as f:
        all_embeddings = pickle.load(f)

    print(f"Loaded {len(all_embeddings)} embedding items from {pkl_path}.")
    out_dir = Path(args.out_dir)
    render_clean_projections(
        all_embeddings,
        out_dir,
        filter_anomalies=args.filter_anomalies,
        title_suffix="Across All Modalities (Clean)",
        filename_suffix="_clean_all_modalities",
        dpi=args.dpi,
    )
    print("Clean manifold rendering complete!")


if __name__ == "__main__":
    main()
