#!/usr/bin/env python3
"""
generate_separate_diversity_figures.py
======================================
Renders separate, standalone publication figures for each of the 4 clinical MRI modalities:
  1. T1-Weighted (T1w): intra_prompt_diversity_t1w.png / .pdf
  2. T2-Weighted (T2w): intra_prompt_diversity_t2w.png / .pdf
  3. Proton Density Weighted (PDw): intra_prompt_diversity_pdw.png / .pdf
  4. T2-FLAIR: intra_prompt_diversity_flair.png / .pdf

For each modality:
  - Displays 5 independent anatomical realizations across seeds [101, 202, 303, 404, 505].
  - Extracts the canonical mid-cerebral axial slice (plane with maximum tissue area).
  - Shows the real clinical MIDI prompt directly below the images.
  - No SSIM, no pairwise calculations, no metrics cards.
"""

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

from scripts.run_mrclip_validation_experiments import normalize_for_display


MODALITIES_INFO = {
    "t1w": {
        "key": "T1W",
        "title": "T1-Weighted (T1w)",
        "badge_color": "#8c564b",
        "bg_badge": "#fdf2f0",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax T1, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, 0.4, NONE, 90)",
    },
    "t2w": {
        "key": "T2W",
        "title": "T2-Weighted (T2w)",
        "badge_color": "#0284c7",
        "bg_badge": "#f0f9ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 3.68, NONE, 150)",
    },
    "pdw": {
        "key": "PDW",
        "title": "Proton Density-Weighted (PDw)",
        "badge_color": "#16a34a",
        "bg_badge": "#f0fdf4",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 3.68, NONE, 150)",
    },
    "flair": {
        "key": "FLAIR",
        "title": "T2-FLAIR",
        "badge_color": "#1e3a8a",
        "bg_badge": "#eff6ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fluid_320, SE_IR, SK_SP_MP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 5.04, 1.8118, 150)",
    },
}


def extract_max_parenchyma_axial(vol: np.ndarray) -> np.ndarray:
    norm_vol = normalize_for_display(vol)
    bg_thresh = 0.08
    z_areas = [np.sum(norm_vol[:, :, z] > bg_thresh) for z in range(norm_vol.shape[2])]
    best_z = int(np.argmax(z_areas))
    raw_slice = np.rot90(norm_vol[:, :, best_z])

    # Tight square center crop
    fg_pts = np.where(raw_slice > 0.04)
    if len(fg_pts[0]) > 200:
        y0, y1 = int(np.min(fg_pts[0])), int(np.max(fg_pts[0]))
        x0, x1 = int(np.min(fg_pts[1])), int(np.max(fg_pts[1]))
        h, w = y1 - y0, x1 - x0
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half_dim = int(max(h, w) * 0.55)

        y_start = max(0, cy - half_dim)
        y_end = min(raw_slice.shape[0], cy + half_dim)
        x_start = max(0, cx - half_dim)
        x_end = min(raw_slice.shape[1], cx + half_dim)

        cropped = raw_slice[y_start:y_end, x_start:x_end]
        pil_im = Image.fromarray((cropped * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
        norm = np.array(pil_im).astype(np.float32) / 255.0
    else:
        norm = raw_slice

    p1, p99 = np.percentile(norm[norm > 0.02], [0.5, 99.5]) if np.any(norm > 0.02) else (0, 1)
    norm = np.clip((norm - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return norm


def plot_single_modality(
    mod_id: str,
    info: dict,
    slices: list[np.ndarray],
    seeds: list[int],
    out_dir: Path,
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    ncols = len(seeds)
    fig = plt.figure(figsize=(15.5, 4.6), facecolor="#ffffff")

    # Title & Subtitle Banner
    fig.text(
        0.03,
        0.935,
        f"Intra-Prompt Anatomical Diversity: Multi-Seed Synthesis ({info['title']})",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
        ha="left",
    )
    subtitle = (
        f"5 independent latent realizations synthesized from identical real clinical conditioning | Free latent space (unconstrained anatomy)\n"
        f"Voxel Resolution: 256×256×128 (0.94×0.94×1.36 mm³)  |  Modality: {info['title']}"
    )
    fig.text(0.03, 0.855, subtitle, fontsize=8.8, color="#475569", ha="left", linespacing=1.3)

    # Grid for the 5 image tiles
    gs = fig.add_gridspec(
        nrows=1,
        ncols=ncols,
        left=0.03,
        right=0.97,
        top=0.79,
        bottom=0.20,
        wspace=0.06,
    )

    for c in range(ncols):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(slices[c], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)

        ax.set_title(
            f"Sample {c+1:02d}  (Seed {seeds[c]})",
            fontsize=8.8,
            fontweight="bold",
            color="#1e293b",
            pad=5,
        )

    # Prompt card below images
    prompt_str = f"Real Clinical MIDI Prompt:  \"{info['prompt']}\""
    fig.text(
        0.03,
        0.085,
        prompt_str,
        fontsize=8.2,
        fontfamily="monospace",
        fontweight="medium",
        color="#1e293b",
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.3",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=0.9,
        ),
    )

    png_path = out_dir / f"intra_prompt_diversity_{mod_id}.png"
    pdf_path = out_dir / f"intra_prompt_diversity_{mod_id}.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {info['title']} -> {png_path} and {pdf_path}")


def main():
    cache_dir = Path("output/eval_memorization/diversity_cache")
    out_dir = Path("output/eval_memorization")
    seeds = [101, 202, 303, 404, 505]

    for mod_id, info in MODALITIES_INFO.items():
        slices = []
        for s in seeds:
            vol_file = cache_dir / f"vol_{info['key']}_real_seed{s}.npy"
            if not vol_file.exists():
                raise FileNotFoundError(f"Missing volume: {vol_file}")
            vol = np.load(vol_file)
            slc = extract_max_parenchyma_axial(vol)
            np.save(cache_dir / f"slice_{info['key']}_real_seed{s}.npy", slc)
            slices.append(slc)

        plot_single_modality(mod_id, info, slices, seeds, out_dir)

    # Also update intra_prompt_diversity_panel.png and .pdf with T2w
    import shutil
    shutil.copyfile(out_dir / "intra_prompt_diversity_t2w.png", out_dir / "intra_prompt_diversity_panel.png")
    shutil.copyfile(out_dir / "intra_prompt_diversity_t2w.pdf", out_dir / "intra_prompt_diversity_panel.pdf")
    print("All separate figures successfully updated!")


if __name__ == "__main__":
    main()
