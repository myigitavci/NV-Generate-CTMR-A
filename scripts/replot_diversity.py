#!/usr/bin/env python3
"""
replot_diversity.py
===================
Quickly loads the 10 already-synthesized 3D volumes (with real MIDI prompt)
from diversity_cache, extracts canonical mid-ventricular axial slices,
and renders the publication-quality panel.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


def extract_mid_ventricular_axial(vol_3d: np.ndarray) -> np.ndarray:
    """
    Extracts the canonical mid-ventricular axial slice from 3D volume (256, 256, 128).
    Detects the plane containing lateral ventricles and centers the brain cleanly.
    """
    bg_thresh = 0.05
    z_areas = np.array([np.sum(vol_3d[:, :, z] > bg_thresh) for z in range(vol_3d.shape[2])])
    fg_z = np.where(z_areas > 3000)[0]
    if len(fg_z) < 10:
        raw_slice = np.rot90(vol_3d[:, :, vol_3d.shape[2] // 2])
    else:
        z_start, z_end = fg_z[0], fg_z[-1]
        cand_start = int(z_start + 0.48 * (z_end - z_start))
        cand_end = int(z_start + 0.62 * (z_end - z_start))
        cand_end = max(cand_end, cand_start + 1)
        
        fg_vox = vol_3d[vol_3d > bg_thresh]
        csf_thresh = float(np.percentile(fg_vox, 82.0))
        
        best_z = int(z_start + 0.54 * (z_end - z_start))
        max_score = -1.0
        for z in range(cand_start, cand_end):
            center = vol_3d[70:186, 70:186, z]
            csf_count = np.sum(center > csf_thresh)
            score = float(csf_count)
            if score > max_score:
                max_score = score
                best_z = z
        raw_slice = np.rot90(vol_3d[:, :, best_z])

    # Find bounding box of brain on this slice and center with consistent margin
    fg_pts = np.where(raw_slice > 0.04)
    if len(fg_pts[0]) > 200:
        y0, y1 = int(np.min(fg_pts[0])), int(np.max(fg_pts[0]))
        x0, x1 = int(np.min(fg_pts[1])), int(np.max(fg_pts[1]))
        h, w = y1 - y0, x1 - x0
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half_dim = int(max(h, w) * 0.55)  # ~10% margin around brain

        y_start = max(0, cy - half_dim)
        y_end = min(raw_slice.shape[0], cy + half_dim)
        x_start = max(0, cx - half_dim)
        x_end = min(raw_slice.shape[1], cx + half_dim)

        cropped = raw_slice[y_start:y_end, x_start:x_end]
        from PIL import Image
        pil_im = Image.fromarray((cropped * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
        norm = np.array(pil_im).astype(np.float32) / 255.0
    else:
        norm = raw_slice

    p1, p99 = np.percentile(norm[norm > 0.02], [1.0, 99.5]) if np.any(norm > 0.02) else (0, 1)
    norm = np.clip((norm - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return norm


def main():
    cache_dir = Path("output/eval_memorization/diversity_cache")
    out_dir = Path("output/eval_memorization")
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    
    print(f"Loading {len(seeds)} cached volumes...")
    vols = []
    for s in seeds:
        vf = cache_dir / f"vol_T2W_seed{s}.npy"
        if not vf.exists():
            print(f"Missing {vf}")
            return
        vols.append(np.load(vf))

    print(f"Computing pairwise 3D SSIM across {len(vols)*(len(vols)-1)//2} pairs...")
    pairwise_ssims = []
    for i in range(len(vols)):
        for j in range(i + 1, len(vols)):
            pairwise_ssims.append(float(structural_similarity(vols[i], vols[j], data_range=1.0)))

    mean_ssim = float(np.mean(pairwise_ssims))
    std_ssim = float(np.std(pairwise_ssims))
    diversity_score = 1.0 - mean_ssim
    min_ssim = float(np.min(pairwise_ssims))
    max_ssim = float(np.max(pairwise_ssims))

    print(f"Mean SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"Diversity Score (1 - SSIM): {diversity_score:.4f}")

    # Extract slices
    std_slices = [extract_mid_ventricular_axial(v) for v in vols]

    # Style configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    nrows = 2
    ncols = 5
    fig = plt.figure(figsize=(15.0, 7.2), facecolor="#ffffff")
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        top=0.82,
        bottom=0.06,
        left=0.03,
        right=0.97,
        wspace=0.07,
        hspace=0.22,
    )

    # Title & Subtitle Banner
    fig.text(
        0.03,
        0.955,
        "Intra-Prompt Anatomical Diversity: Multi-Seed Whole-Brain Synthesis (T2W)",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
        ha="left",
    )

    subtitle = (
        "Identical Real MIDI Prompt: \"A brain MRI, plane axial, Scanner (Siemens, MAGNETOM_Vida, 3.0), Acquisition (TRA, SE, SK), TE=0.095, TR=4.5, FA=90\"\n"
        "10 Independent Latent Realizations  |  Fixed Conditioning Vector  |  Resolution: 256×256×128 (0.94×0.94×1.36 mm³)"
    )
    fig.text(0.03, 0.900, subtitle, fontsize=8.6, color="#475569", ha="left", linespacing=1.35)

    badge_text = (
        f"Pairwise 3D SSIM: {mean_ssim:.3f} ± {std_ssim:.3f} (Range: {min_ssim:.3f}–{max_ssim:.3f})\n"
        f"Intra-Prompt Diversity (1 - SSIM): {diversity_score:.3f}  |  N = 10 Volumes ({len(pairwise_ssims)} Pairs)"
    )
    fig.text(
        0.97,
        0.925,
        badge_text,
        fontsize=9.0,
        fontweight="bold",
        color="#0369a1",
        ha="right",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.35",
            facecolor="#f0f9ff",
            edgecolor="#0284c7",
            linewidth=1.2,
        ),
        linespacing=1.35,
    )

    for idx in range(len(vols)):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])

        slc = std_slices[idx]
        ax.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)

        ax.set_title(
            f"Sample {idx+1:02d}  (Seed {seeds[idx]})",
            fontsize=9.0,
            fontweight="bold",
            color="#1e293b",
            pad=6,
        )

    png_path = out_dir / "intra_prompt_diversity_panel.png"
    pdf_path = out_dir / "intra_prompt_diversity_panel.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png_path} and {pdf_path}")

    # Update CSV metrics
    csv_out = out_dir / "memorization_audit_metrics.csv"
    df = pd.DataFrame([{
        "mean_pairwise_ssim": mean_ssim,
        "std_pairwise_ssim": std_ssim,
        "intra_prompt_diversity": diversity_score,
    }])
    df.to_csv(csv_out, index=False)
    print(f"Saved metrics -> {csv_out}")


if __name__ == "__main__":
    main()
