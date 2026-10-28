#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Standalone Replot Tool for Systematic Evaluation Latent Space (t-SNE & PCA) & Metrics
=====================================================================================
Loads precomputed intermediate embeddings (systematic_embeddings.pkl) and/or
metrics_per_sample.csv to generate publication-grade:
  1. t-SNE & PCA feature maps (tsne_embedding_space.png, pca_embedding_space.png)
     with dual legends: Data Type (Synthetic, Real, Metadata) and MRI Modality (T1W, T2W, FLAIR, PDW).
  2. CFG Alignment Curves (cosine_similarity_vs_cfg.png) with bottom-right legends.

Usage:
  # Replot Latent Space
  python3 scripts/replot_systematic_tsne.py \
      --embeddings-pkl output/systematic_evaluation_30cases/systematic_embeddings.pkl \
      --out-dir output/systematic_evaluation_30cases \
      --exclude-modalities T2STARW \
      --no-axis-numbers

  # Replot Metrics vs CFG
  python3 scripts/replot_systematic_tsne.py \
      --metrics-csv output/systematic_evaluation_30cases/metrics_per_sample.csv \
      --out-dir output/systematic_evaluation_30cases
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def render_metrics_summary(df_metrics: pd.DataFrame, out_path: Path, dpi: int = 300):
    """Plot Mean Cosine Similarity vs CFG scale with Real-Metadata baseline in publication quality."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=dpi)

    grouped = df_metrics.groupby("cfg_scale")
    cfg_vals = sorted(df_metrics["cfg_scale"].unique())

    mean_text = [grouped.get_group(c)["sim_text"].mean() for c in cfg_vals]
    std_text = [grouped.get_group(c)["sim_text"].std() for c in cfg_vals]

    mean_real = [grouped.get_group(c)["sim_real"].mean() for c in cfg_vals]
    std_real = [grouped.get_group(c)["sim_real"].std() for c in cfg_vals]

    baseline_real_text = df_metrics["sim_real_text"].mean() if "sim_real_text" in df_metrics.columns else None

    # Plot 1: Synthetic <-> Metadata Alignment
    ax[0].errorbar(
        cfg_vals, mean_text, yerr=std_text,
        marker="o", markersize=7, color="#1d4ed8", capsize=5, elinewidth=1.6, capthick=1.6, lw=2.2,
        label="Synthetic ↔ Metadata", zorder=3
    )
    if baseline_real_text is not None:
        ax[0].axhline(
            baseline_real_text, color="#dc2626", linestyle="--", lw=2.0,
            label=f"Real ↔ Metadata Baseline ({baseline_real_text:.3f})", zorder=2
        )
    ax[0].set_title("MR-CLIP Metadata ↔ Image Alignment", fontsize=13.5, fontweight="bold", pad=10, color="#111827")
    ax[0].set_xlabel("CFG Guidance Scale", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[0].set_ylabel("Cosine Similarity", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[0].tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)
    for spine in ax[0].spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)
    ax[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#d1d5db", zorder=0)
    ax[0].set_axisbelow(True)
    ax[0].legend(
        loc="lower right", fontsize=11, framealpha=0.95, facecolor="#f8fafc", edgecolor="#cbd5e1", borderpad=0.7, labelspacing=0.5
    )

    # Plot 2: Synthetic <-> Real Image Alignment
    ax[1].errorbar(
        cfg_vals, mean_real, yerr=std_real,
        marker="s", markersize=7, color="#15803d", capsize=5, elinewidth=1.6, capthick=1.6, lw=2.2,
        label="Synthetic ↔ Real Image", zorder=3
    )
    ax[1].set_title("Synthetic ↔ Real Image Alignment", fontsize=13.5, fontweight="bold", pad=10, color="#111827")
    ax[1].set_xlabel("CFG Guidance Scale", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[1].set_ylabel("Cosine Similarity", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[1].tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)
    for spine in ax[1].spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)
    ax[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#d1d5db", zorder=0)
    ax[1].set_axisbelow(True)
    ax[1].legend(
        loc="lower right", fontsize=11, framealpha=0.95, facecolor="#f8fafc", edgecolor="#cbd5e1", borderpad=0.7, labelspacing=0.5
    )

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


def render_systematic_projections(
    all_embeddings: list[dict],
    out_dir: Path,
    exclude_modalities: list[str] = None,
    no_axis_numbers: bool = True,
    dpi: int = 300,
):
    if exclude_modalities is None:
        exclude_modalities = ["T2STARW"]

    exclude_set = {m.upper() for m in exclude_modalities}
    filtered_feats = [
        item for item in all_embeddings
        if item.get("modality", "").upper() not in exclude_set
    ]

    if len(filtered_feats) < 4:
        print(f"[WARN] Not enough feature embeddings ({len(filtered_feats)}) to plot after excluding {exclude_set}.")
        return

    print(f"Loaded {len(all_embeddings)} total items; plotting {len(filtered_feats)} items (excluded: {exclude_set}).")

    X = np.array([item["feat"] for item in filtered_feats])  # (N, 512)
    types = [item["type"] for item in filtered_feats]
    modalities = [item["modality"].upper() for item in filtered_feats]

    # Harmonized high-contrast color palette per modality
    modality_colors = {
        "T1W": "#8c564b",    # Rich Sienna Brown
        "T2W": "#0284c7",    # Ocean Cyan / Azure Blue
        "FLAIR": "#1e3a8a",  # Deep Royal Navy Blue
        "PDW": "#16a34a",    # Forest Emerald Green
    }
    unique_mods = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in set(modalities)] or sorted(list(set(modalities)))
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_mods), 1)))
    mod_color_map = {m: modality_colors.get(m, fallback_colors[i]) for i, m in enumerate(unique_mods)}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure publication typography
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    # 1. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. t-SNE
    perplexity = min(30, max(5, len(X) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    for proj_name, coords in [("t-SNE", X_tsne), ("PCA", X_pca)]:
        fig, (ax, ax_leg) = plt.subplots(
            1, 2,
            figsize=(13.8, 8.2),
            dpi=dpi,
            gridspec_kw={"width_ratios": [3.3, 1.2]}
        )
        ax_leg.axis("off")

        # Plot synthetic samples (bottom layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]

            if t == "synthetic":
                cfg = item.get("cfg_scale", 0.0)
                alpha = 0.35 if cfg == 0 else (0.65 if cfg <= 2.0 else 0.90)
                ax.scatter(x, y, color=col, marker="o", s=70, alpha=alpha, edgecolors="white", linewidth=0.5, zorder=3)

        # Plot real scans (middle layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]
            if t == "real":
                ax.scatter(x, y, color=col, marker="s", s=115, edgecolors="#111827", linewidth=1.2, zorder=5)

        # Plot metadata prompts (top layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]
            if t in ["text", "metadata"]:
                ax.scatter(x, y, color=col, marker="*", s=210, edgecolors="#111827", linewidth=1.1, zorder=6)

        # Dual Legends: 1) Data Type, 2) MRI Modality
        type_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#374151", markersize=10, markeredgecolor="#ffffff", markeredgewidth=0.6, label="Synthetic"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#374151", markersize=10.5, markeredgecolor="#111827", markeredgewidth=0.9, label="Real"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#374151", markersize=13, markeredgecolor="#111827", markeredgewidth=0.9, label="Metadata"),
        ]

        modality_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=mod_color_map[mod], markersize=11, markeredgecolor="#111827", markeredgewidth=0.8, label=mod)
            for mod in unique_mods
        ]

        if proj_name == "t-SNE":
            ax.set_xlabel("t-SNE 1", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
            ax.set_ylabel("t-SNE 2", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
        else:
            ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% explained variance)", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
            ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% explained variance)", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")

        if no_axis_numbers:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)

        for spine in ax.spines.values():
            spine.set_color("#1f2937")
            spine.set_linewidth(1.2)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#d1d5db", zorder=0)
        ax.set_axisbelow(True)

        leg1 = ax_leg.legend(
            handles=type_handles,
            title="Data Type",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=12,
            title_fontsize=13.5,
            framealpha=0.96,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            borderpad=0.9,
            labelspacing=0.75,
            handletextpad=0.8,
        )
        leg1.get_title().set_fontweight("bold")
        ax_leg.add_artist(leg1)

        fig.tight_layout()
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
            leg2_top = bbox1.y0 - 0.050
        except Exception:
            leg2_top = 0.52

        leg2 = ax_leg.legend(
            handles=modality_handles,
            title="MRI Modality",
            loc="upper left",
            bbox_to_anchor=(0.02, leg2_top),
            fontsize=12,
            title_fontsize=13.5,
            framealpha=0.96,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            borderpad=0.9,
            labelspacing=0.75,
            handletextpad=0.8,
        )
        leg2.get_title().set_fontweight("bold")

        save_name = f"{proj_name.lower().replace('-', '')}_embedding_space.png"
        save_path = out_dir / save_name
        fig.savefig(str(save_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Replot publication-ready systematic evaluation latent space maps & metrics.")
    parser.add_argument(
        "--embeddings-pkl",
        type=str,
        default=None,
        help="Path to saved systematic_embeddings.pkl (e.g. output/systematic_evaluation_30cases/systematic_embeddings.pkl)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to metrics_per_sample.csv to replot cosine_similarity_vs_cfg.png",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/systematic_evaluation_30cases",
        help="Directory to save replotted figures",
    )
    parser.add_argument(
        "--exclude-modalities",
        nargs="+",
        default=["T2STARW"],
        help="Modalities to exclude from plotting (default: T2STARW)",
    )
    parser.add_argument(
        "--no-axis-numbers",
        action="store_true",
        default=True,
        help="Omit numeric ticks on axes for clean minimalist styling (default: True)",
    )
    parser.add_argument(
        "--show-axis-numbers",
        action="store_false",
        dest="no_axis_numbers",
        help="Show numeric ticks on axes",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Replot metrics if provided or present
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else out_dir / "metrics_per_sample.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        metrics_png_path = out_dir / "cosine_similarity_vs_cfg.png"
        render_metrics_summary(df_metrics, metrics_png_path, dpi=args.dpi)

    # 2. Replot embeddings if provided or present
    pkl_path = Path(args.embeddings_pkl) if args.embeddings_pkl else out_dir / "systematic_embeddings.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            all_embeddings = pickle.load(f)

        render_systematic_projections(
            all_embeddings=all_embeddings,
            out_dir=out_dir,
            exclude_modalities=args.exclude_modalities,
            no_axis_numbers=args.no_axis_numbers,
            dpi=args.dpi,
        )
    elif not metrics_path.exists():
        print(f"[ERROR] Neither {pkl_path} nor {metrics_path} found.")
        sys.exit(1)

    print("Replot complete!")


if __name__ == "__main__":
    main()
