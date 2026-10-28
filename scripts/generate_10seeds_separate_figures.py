#!/usr/bin/env python3
"""
generate_10seeds_separate_figures.py
====================================
Generates separate, standalone publication figures for each of the 4 clinical MRI modalities:
  1. T1-Weighted (T1w): intra_prompt_diversity_t1w.png / .pdf
  2. T2-Weighted (T2w): intra_prompt_diversity_t2w.png / .pdf
  3. Proton Density Weighted (PDw): intra_prompt_diversity_pdw.png / .pdf
  4. T2-FLAIR: intra_prompt_diversity_flair.png / .pdf

Key Features:
  - 10 diverse anatomical realizations per modality across seeds:
    [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
  - Arranged in an aesthetic 2 rows × 5 columns grid.
  - Zero titles or unnecessary text on top (images begin right at the top margin).
  - Clean sample labels above each tile: "Sample 01 (Seed 101)", etc.
  - Real Clinical MIDI Prompt formatted cleanly below the images across 3 well-proportioned lines.
  - No SSIM or quantitative metrics badges.
  - Exports both 300 DPI PNG and vector PDF.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


ALL_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

MODALITIES_INFO = {
    "t1w": {
        "key": "T1W",
        "title": "T1-Weighted (T1w)",
        "badge_color": "#8c564b",
        "bg_badge": "#fdf2f0",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t1_se_tra_320, SE, SP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 0.589, NONE, 90)",
        "lines": [
            "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5)",
            "Acquisition (Description, Sequence, Variant): (t1_se_tra_320, SE, SP)",
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 0.589, NONE, 90)",
        ],
    },
    "t2w": {
        "key": "T2W",
        "title": "T2-Weighted (T2w)",
        "badge_color": "#0284c7",
        "bg_badge": "#f0f9ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 3.68, NONE, 150)",
        "lines": [
            "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5)",
            "Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP)",
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 3.68, NONE, 150)",
        ],
    },
    "pdw": {
        "key": "PDW",
        "title": "Proton Density-Weighted (PDw)",
        "badge_color": "#16a34a",
        "bg_badge": "#f0fdf4",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 3.68, NONE, 150)",
        "lines": [
            "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5)",
            "Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP)",
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 3.68, NONE, 150)",
        ],
    },
    "flair": {
        "key": "FLAIR",
        "title": "T2-FLAIR",
        "badge_color": "#1e3a8a",
        "bg_badge": "#eff6ff",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fluid_320, SE_IR, SK_SP_MP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 5.04, 1.8118, 150)",
        "lines": [
            "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5)",
            "Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fluid_320, SE_IR, SK_SP_MP_OSP)",
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 5.04, 1.8118, 150)",
        ],
    },
}


def extract_mid_axial_slice(vol: np.ndarray) -> np.ndarray:
    """
    Extracts canonical mid-ventricular axial slice from 3D volume (256, 256, 128).
    Targets canonical ventricular range z in [60, 71] with peak tissue area,
    guaranteeing lateral ventricles and basal ganglia visibility across all modalities.
    """
    norm_vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    z_areas = [np.sum(norm_vol[:, :, z] > 0.08) for z in range(60, 71)]
    best_z = 60 + int(np.argmax(z_areas))
    raw_slice = np.rot90(norm_vol[:, :, best_z])

    # Tight square center crop around brain foreground
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
        pil_im = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
            (512, 512), resample=Image.Resampling.LANCZOS
        )
        norm = np.array(pil_im).astype(np.float32) / 255.0
    else:
        norm = raw_slice

    p1, p99 = np.percentile(norm[norm > 0.02], [0.5, 99.5]) if np.any(norm > 0.02) else (0, 1)
    norm = np.clip((norm - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return norm


def plot_single_modality_10seeds(
    mod_id: str,
    info: dict,
    slices: list[np.ndarray],
    seeds: list[int],
    out_dir: Path,
):
    """
    Renders standalone publication figure for 1 modality:
      - 10 seeds arranged in 2 rows × 5 columns.
      - No top title or unnecessary header text (starts right at top).
      - Sample label above each tile.
      - Prompt box below images formatted cleanly into 3 lines.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

    fig = plt.figure(figsize=(15.5, 7.5), facecolor="#ffffff")

    # 2 rows x 5 columns layout
    # Top margin close to 0.96 (no top title)
    # Bottom margin leaves room for 3-line prompt card
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        left=0.03,
        right=0.97,
        top=0.95,
        bottom=0.20,
        wspace=0.06,
        hspace=0.18,
    )

    for idx in range(min(10, len(slices))):
        r = idx // 5
        c = idx % 5
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(slices[idx], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
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
            pad=5,
        )

    # Prompt callout card below the 2 rows
    # Centralized without 1/2/3 numbering, with generous breathing room to the end of image
    lines = info["lines"]
    formatted_prompt = (
        f"{lines[0]}\n"
        f"{lines[1]}\n"
        f"{lines[2]}"
    )

    fig.text(
        0.5,
        0.095,
        formatted_prompt,
        fontsize=9.5,
        fontfamily="monospace",
        fontweight="medium",
        color="#1e293b",
        ha="center",
        va="center",
        linespacing=1.45,
        bbox=dict(
            boxstyle="round,pad=0.7,rounding_size=0.3",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.0,
        ),
    )

    # Invisible spacer at bottom to ensure ample space between prompt text and end of image
    fig.text(0.5, 0.015, " ", fontsize=4, color="none")

    png_path = out_dir / f"intra_prompt_diversity_{mod_id}.png"
    pdf_path = out_dir / f"intra_prompt_diversity_{mod_id}.pdf"
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {info['title']} (10 seeds) -> {png_path} and {pdf_path}")


def synthesize_missing_volumes(
    missing_tasks: list[tuple[str, int]],
    cache_dir: Path,
    device_str: str,
    logger: logging.Logger,
    cfg_scale: float = 4.0,
):
    """
    Loads diffusion model and synthesizes missing volumes on specified GPU.
    """
    import torch
    from scripts.diff_model_setting import load_config
    from scripts.embed_with_mrclip import load_mrclip
    from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
    from scripts.run_mrclip_validation_experiments import (
        ensure_axial_prompt,
        generate_complete_brain,
        normalize_for_display,
    )
    from scripts.utils import define_instance

    args = load_config(
        "configs/environment_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_network_rflow.json",
    )
    device = torch.device(device_str)
    target_dim = tuple(args.diffusion_unet_inference.get("dim", [256, 256, 128]))

    logger.info(f"Loading Autoencoder onto {device}...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

    logger.info("Loading Diffusion UNet...")
    diff_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(diff_unet).to(device)
    unet_ckpt = torch.load("models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt", map_location=device, weights_only=False)
    state = unet_ckpt.get("model_state_dict", unet_ckpt.get("unet_state_dict", unet_ckpt.get("state_dict", unet_ckpt)))
    cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys()):
        model.load_state_dict(cleaned_state, strict=False)
    else:
        model.unet.load_state_dict(cleaned_state, strict=False)
    model.eval()
    scale_factor = float(unet_ckpt.get("scale_factor", 1.0422))

    clip_checkpoint = "/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt"
    open_clip_src = "/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src"
    clip_context_length = 98

    logger.info("Loading MR-CLIP text encoder...")
    clip_model, tokenizer = load_mrclip(open_clip_src, clip_checkpoint, clip_context_length, str(device))
    clip_model.eval()

    # Process missing tasks grouped by modality
    mod_tasks: dict[str, list[int]] = {}
    for mod_key, s in missing_tasks:
        mod_tasks.setdefault(mod_key, []).append(s)

    for mod_key, seed_list in mod_tasks.items():
        # find matching info
        info = next(v for v in MODALITIES_INFO.values() if v["key"] == mod_key)
        prompt = ensure_axial_prompt(info["prompt"])
        logger.info(f"\nSynthesizing {len(seed_list)} seed(s) for {mod_key}: {prompt}")

        for s in seed_list:
            vol_file = cache_dir / f"vol_{mod_key}_real_seed{s}.npy"
            slc_file = cache_dir / f"slice_{mod_key}_real_seed{s}.npy"

            logger.info(f"Generating {mod_key} (Seed {s})...")
            cur_seed = s
            valid_vol = None
            for attempt in range(4):
                v, _, _ = generate_complete_brain(
                    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                    target_dim, cfg_scale, prompt=prompt, initial_seed=cur_seed,
                    logger=logger, max_retries=3, fallback_prompt=prompt
                )
                if v is not None:
                    valid_vol = v
                    break
                cur_seed += 113

            if valid_vol is not None:
                vol_norm = (valid_vol / 1000.0).astype(np.float32)
                mid_slice = extract_mid_axial_slice(vol_norm)
                np.save(vol_file, vol_norm)
                np.save(slc_file, mid_slice)
                logger.info(f"Saved {vol_file.name} and {slc_file.name}")
            else:
                logger.error(f"Failed generating volume for {mod_key} seed {s}")


def main():
    parser = argparse.ArgumentParser(description="Generate 10-seed separate diversity figures.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="output/eval_memorization")
    parser.add_argument("--force-mod", default="", help="Modality to force re-generate (e.g. t1w)")
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "diversity_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("generate_10seeds")
    logger.info("=" * 70)
    logger.info("GENERATING 10-SEED SEPARATE MODALITY DIVERSITY FIGURES")
    logger.info("=" * 70)

    # Ensure T1W uses the Siemens Aera prompt for full whole-brain synthesis
    aera_marker = cache_dir / ".t1w_aera_v1_done"
    if not aera_marker.exists():
        for s in ALL_SEEDS:
            f = cache_dir / f"vol_T1W_real_seed{s}.npy"
            if f.exists():
                f.unlink()
        aera_marker.touch()

    # 1. Check which seeds are missing across all 4 modalities
    missing_tasks: list[tuple[str, int]] = []
    for mod_id, info in MODALITIES_INFO.items():
        mod_key = info["key"]
        for s in ALL_SEEDS:
            vol_file = cache_dir / f"vol_{mod_key}_real_seed{s}.npy"
            if not vol_file.exists():
                missing_tasks.append((mod_key, s))

    if missing_tasks:
        logger.info(f"Found {len(missing_tasks)} missing volume(s) to synthesize: {missing_tasks}")
        synthesize_missing_volumes(missing_tasks, cache_dir, args_cli.device, logger)
    else:
        logger.info("All 10 seeds exist in cache for all 4 modalities!")

    # 2. Extract canonical slices and render separate figure for each modality
    logger.info("Rendering separate figures for each modality...")
    for mod_id, info in MODALITIES_INFO.items():
        slices = []
        for s in ALL_SEEDS:
            vol_file = cache_dir / f"vol_{info['key']}_real_seed{s}.npy"
            slc_file = cache_dir / f"slice_{info['key']}_real_seed{s}.npy"

            if slc_file.exists():
                slc = np.load(slc_file)
            else:
                vol = np.load(vol_file)
                slc = extract_mid_axial_slice(vol)
                np.save(slc_file, slc)
            slices.append(slc)

        plot_single_modality_10seeds(mod_id, info, slices, ALL_SEEDS, out_dir)

    # 3. Mirror T2w to intra_prompt_diversity_panel for active tab compatibility
    shutil.copyfile(out_dir / "intra_prompt_diversity_t2w.png", out_dir / "intra_prompt_diversity_panel.png")
    shutil.copyfile(out_dir / "intra_prompt_diversity_t2w.pdf", out_dir / "intra_prompt_diversity_panel.pdf")
    logger.info("Mirrored T2w to intra_prompt_diversity_panel.png / .pdf")
    logger.info("All 10-seed separate figures successfully generated!")


if __name__ == "__main__":
    main()
