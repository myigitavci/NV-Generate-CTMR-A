#!/usr/bin/env python3
"""
generate_intra_prompt_diversity_publication.py
==============================================
Generates a publication-grade multi-modality Intra-Prompt Anatomical Diversity figure
from the trained 3D MR-CLIP flow matching diffusion model for:
  - T1-Weighted (T1w)
  - T2-Weighted (T2w)
  - Proton Density-Weighted (PDw / pd2)
  - T2-FLAIR

Features:
  - Uses 100% genuine real clinical MIDI prompts (not artificial templates).
  - Shows 5 diverse independent anatomical realizations across seeds [101, 202, 303, 404, 505]
    for each modality.
  - Displays the clean real prompt text directly below each row of images.
  - Completely omits SSIM metrics / calculations as requested ("just show images and below them, the prompt").
  - Exports both 300 DPI PNG and vector PDF to output/eval_memorization/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    ensure_axial_prompt,
    generate_complete_brain,
    get_middle_slices,
)
from scripts.utils import define_instance


REAL_PROMPTS = {
    "T1W": {
        "title": "T1-Weighted (T1w)",
        "badge_color": "#8c564b",
        "bg_badge": "#fdf2f0",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax T1, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, 0.4, NONE, 90)",
    },
    "T2W": {
        "title": "T2-Weighted (T2w)",
        "badge_color": "#0284c7",
        "bg_badge": "#f0f9ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 3.68, NONE, 150)",
    },
    "PDW": {
        "title": "Proton Density Weighted (PDw)",
        "badge_color": "#16a34a",
        "bg_badge": "#f0fdf4",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 3.68, NONE, 150)",
    },
    "FLAIR": {
        "title": "T2-FLAIR",
        "badge_color": "#1e3a8a",
        "bg_badge": "#eff6ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fluid_320, SE_IR, SK_SP_MP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 5.04, 1.8118, 150)",
    },
}


def extract_mid_ventricular_axial(vol_3d: np.ndarray) -> np.ndarray:
    """
    Extracts canonical mid-ventricular axial slice from 3D volume (256, 256, 128).
    Detects plane containing lateral ventricles and centers brain cleanly.
    """
    bg_thresh = 0.05
    fg_vox = vol_3d[vol_3d > bg_thresh]
    if len(fg_vox) < 1000:
        raw_slice = np.rot90(vol_3d[:, :, vol_3d.shape[2] // 2])
    else:
        csf_thresh = float(np.percentile(fg_vox, 82.0))
        best_z = vol_3d.shape[2] // 2
        max_score = -1.0
        for z in range(35, min(95, vol_3d.shape[2] - 10)):
            center = vol_3d[60:196, 60:196, z]
            csf_count = np.sum(center > csf_thresh)
            tissue_count = np.sum(vol_3d[:, :, z] > bg_thresh)
            score = float(csf_count) * 2.5 + (tissue_count > 7000) * 800.0
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

    p1, p99 = np.percentile(norm[norm > 0.02], [1.0, 99.5]) if np.any(norm > 0.02) else (0, 1)
    norm = np.clip((norm - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return norm


def plot_multimodality_diversity_panel(
    data_by_modality: dict[str, dict],
    seeds: list[int],
    out_dir: Path,
    logger: logging.Logger | None = None,
):
    """
    Renders publication figure:
      - 4 rows (T1w, T2w, PDw, FLAIR)
      - 5 columns of diverse images per row
      - Prompt callout card placed directly below each row
      - No SSIM or quantitative metrics
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    modalities = ["T1W", "T2W", "PDW", "FLAIR"]
    ncols = len(seeds)
    nrows = len(modalities)

    # Height: 4 rows * ~3.2 inches + banner space = ~14.8 inches total
    fig = plt.figure(figsize=(15.5, 14.8), facecolor="#ffffff")

    # Title & Subtitle Banner at the top
    fig.text(
        0.03,
        0.982,
        "Intra-Prompt Anatomical Diversity: Multi-Seed Whole-Brain Synthesis",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        ha="left",
    )
    subtitle = (
        "Independent latent realizations synthesized from identical real clinical MIDI text prompts across 4 core modalities.\n"
        "Conditioning vector fixed per row  |  Free latent space (unconstrained anatomy)  |  Resolution: 256×256×128 (0.94×0.94×1.36 mm³)"
    )
    fig.text(0.03, 0.957, subtitle, fontsize=9.2, color="#475569", ha="left", linespacing=1.35)

    # We divide the canvas into 4 vertical sections, one per modality
    # Y-bounds for the 4 modality sections (top to bottom)
    row_bounds = [
        (0.725, 0.935),  # T1w
        (0.490, 0.700),  # T2w
        (0.255, 0.465),  # PDw
        (0.020, 0.230),  # FLAIR
    ]

    for m_idx, mod in enumerate(modalities):
        mod_info = REAL_PROMPTS[mod]
        mod_data = data_by_modality[mod]
        slices = mod_data["slices"]
        prompt_text = mod_info["prompt"]

        y_bot, y_top = row_bounds[m_idx]

        # Section Header Badge (Left)
        badge_y = y_top + 0.008
        fig.text(
            0.03,
            badge_y,
            f"{mod_info['title']}",
            fontsize=11.5,
            fontweight="bold",
            color=mod_info["badge_color"],
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35,rounding_size=0.25",
                facecolor=mod_info["bg_badge"],
                edgecolor=mod_info["badge_color"],
                linewidth=1.0,
            ),
        )

        # Gridspec for the 5 image tiles in this row
        gs = fig.add_gridspec(
            nrows=1,
            ncols=ncols,
            left=0.03,
            right=0.97,
            bottom=y_bot + 0.052,
            top=y_top,
            wspace=0.06,
        )

        for col_idx in range(ncols):
            ax = fig.add_subplot(gs[0, col_idx])
            slc = slices[col_idx]
            ax.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor("#cbd5e1")
                spine.set_linewidth(1.1)

            ax.set_title(
                f"Sample {col_idx+1:02d}  (Seed {seeds[col_idx]})",
                fontsize=8.5,
                fontweight="bold",
                color="#1e293b",
                pad=5,
            )

        # Prompt card below this row of images
        prompt_display = f"Real Clinical MIDI Prompt:  \"{prompt_text}\""
        fig.text(
            0.03,
            y_bot + 0.024,
            prompt_display,
            fontsize=8.0,
            fontfamily="monospace",
            fontweight="medium",
            color="#334155",
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5,rounding_size=0.3",
                facecolor="#f8fafc",
                edgecolor="#cbd5e1",
                linewidth=0.8,
            ),
        )

    # Save both PNG (300 DPI) and vector PDF
    png_path = out_dir / "intra_prompt_diversity_panel.png"
    pdf_path = out_dir / "intra_prompt_diversity_panel.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved publication-quality diversity panel -> {png_path} and {pdf_path}")
    print(f"Successfully saved -> {png_path} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 4-modality real-prompt intra-diversity figure.")
    parser.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    parser.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    parser.add_argument(
        "--clip-checkpoint",
        default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt",
    )
    parser.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    parser.add_argument("--clip-context-length", type=int, default=98)
    parser.add_argument("--output-dir", default="output/eval_memorization")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--force", action="store_true", help="Force regeneration of volumes ignoring cache")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "diversity_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("intra_prompt_diversity")
    logger.info("=" * 70)
    logger.info("GENERATING 4-MODALITY REAL-PROMPT INTRA-PROMPT DIVERSITY FIGURE")
    logger.info("=" * 70)

    device = torch.device(args_cli.device)
    logger.info(f"Using compute device: {device}")

    # Canonical seeds across all modalities for direct cross-comparison
    candidate_seeds = [101, 202, 303, 404, 505][:args_cli.num_seeds]

    # Check cache for each modality
    data_by_modality = {}
    missing_tasks = []  # list of (modality, seed)

    for mod in ["T1W", "T2W", "PDW", "FLAIR"]:
        mod_slices = []
        mod_vols = []
        data_by_modality[mod] = {"slices": mod_slices, "vols": mod_vols}
        for s in candidate_seeds:
            vol_file = cache_dir / f"vol_{mod}_real_seed{s}.npy"
            slc_file = cache_dir / f"slice_{mod}_real_seed{s}.npy"
            if not args_cli.force and vol_file.exists() and slc_file.exists():
                mod_vols.append(np.load(vol_file))
                mod_slices.append(np.load(slc_file))
            else:
                missing_tasks.append((mod, s))

    if missing_tasks:
        logger.info(f"Need to synthesize {len(missing_tasks)} volume(s) across missing modalities/seeds.")
        args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
        target_dim = tuple(args.diffusion_unet_inference.get("dim", args.diffusion_unet_inference.get("output_size", [256, 256, 128])))

        logger.info("Loading MAISI Autoencoder...")
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        autoencoder.eval()

        logger.info("Loading Diffusion UNet...")
        diff_unet = define_instance(args, "diffusion_unet_def").to(device)
        model = MRCLIPConditionedUNet(diff_unet).to(device)
        unet_ckpt = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
        state = unet_ckpt.get("model_state_dict", unet_ckpt.get("unet_state_dict", unet_ckpt.get("state_dict", unet_ckpt)))
        cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys()):
            model.load_state_dict(cleaned_state, strict=False)
        else:
            model.unet.load_state_dict(cleaned_state, strict=False)
        model.eval()
        scale_factor = float(unet_ckpt.get("scale_factor", 1.0422))

        logger.info(f"Loading MR-CLIP text encoder from {args_cli.clip_checkpoint}...")
        clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))
        clip_model.eval()

        for mod in ["T1W", "T2W", "PDW", "FLAIR"]:
            prompt = ensure_axial_prompt(REAL_PROMPTS[mod]["prompt"])
            logger.info(f"\nConditioning for {mod}: {prompt}")
            for s in candidate_seeds:
                vol_file = cache_dir / f"vol_{mod}_real_seed{s}.npy"
                slc_file = cache_dir / f"slice_{mod}_real_seed{s}.npy"
                if not args_cli.force and vol_file.exists() and slc_file.exists():
                    continue

                logger.info(f"Synthesizing {mod} (Seed {s})...")
                cur_seed = s
                valid_vol = None
                for attempt in range(4):
                    v, _, _ = generate_complete_brain(
                        args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                        target_dim, args_cli.cfg_scale, prompt=prompt, initial_seed=cur_seed,
                        logger=logger, max_retries=3, fallback_prompt=prompt
                    )
                    if v is not None:
                        valid_vol = v
                        break
                    cur_seed += 113

                if valid_vol is not None:
                    vol_norm = (valid_vol / 1000.0).astype(np.float32)
                    mid_slice = extract_mid_ventricular_axial(vol_norm)
                    np.save(vol_file, vol_norm)
                    np.save(slc_file, mid_slice)
                    data_by_modality[mod]["vols"].append(vol_norm)
                    data_by_modality[mod]["slices"].append(mid_slice)
                else:
                    logger.error(f"Failed generating volume for {mod} seed {s}")

    # Verify all slices are populated
    for mod in ["T1W", "T2W", "PDW", "FLAIR"]:
        if len(data_by_modality[mod]["slices"]) != len(candidate_seeds):
            # Reload from disk if populated
            data_by_modality[mod]["slices"] = [
                np.load(cache_dir / f"slice_{mod}_real_seed{s}.npy") for s in candidate_seeds
            ]

    # Generate publication figure
    plot_multimodality_diversity_panel(
        data_by_modality=data_by_modality,
        seeds=candidate_seeds,
        out_dir=out_dir,
        logger=logger,
    )

    logger.info("Done! Multi-modality intra-prompt diversity panel successfully generated.")


if __name__ == "__main__":
    main()
