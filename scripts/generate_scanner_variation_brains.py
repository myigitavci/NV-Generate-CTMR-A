#!/usr/bin/env python3
"""
generate_scanner_variation_brains.py
====================================
Generates 3D brain MRIs across diverse MRI scanner manufacturers and models:
  - Siemens (MAGNETOM Vida 3.0T, Avanto 1.5T, Aera 1.5T, MAGNETOM Sola 1.5T, Skyra 3.0T, TrioTim 3.0T)
  - GE Healthcare (Discovery MR750 3.0T, Signa HDxt 1.5T, SIGNA Premier 3.0T, Optima MR450w 1.5T)
  - Philips Healthcare (Achieva dStream 3.0T, Ingenia 1.5T, Intera 1.5T, Ingenia Elition 3.0T)

Across clinical contrasts:
  - T1-Weighted (T1w)
  - T2-Weighted (T2w)
  - T2-FLAIR (FLAIR)

Saves:
  1. Full 3D NIfTI volumes (.nii.gz)
  2. Standardized 512x512 tight-cropped Axial, Coronal, Sagittal PNG slices
  3. Publication-quality multi-scanner comparison panels per modality
  4. Comprehensive metadata manifest and metrics CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from monai.utils import set_determinism

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
    replace_scanner,
)
from scripts.utils import define_instance


# ---------------------------------------------------------------------------
# Visual & Crop Helpers
# ---------------------------------------------------------------------------

def normalize_volume(vol: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.2) -> np.ndarray:
    """Robust percentile windowing to [0, 1]."""
    v = vol.astype(np.float32)
    bg_thresh = np.percentile(v, 8.0)
    fg = v[v > bg_thresh]
    if len(fg) > 100:
        p_low = np.percentile(fg, lower_pct)
        p_high = np.percentile(fg, upper_pct)
    else:
        p_low = np.percentile(v, lower_pct)
        p_high = np.percentile(v, upper_pct)

    if p_high > p_low:
        v_norm = np.clip((v - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        v_norm = np.clip(v, 0.0, 1.0)
    return v_norm


def crop_brain_tight(slice_2d: np.ndarray, pad_ratio: float = 0.04) -> np.ndarray:
    """Tight cranial bounding box centering."""
    gray = slice_2d.astype(np.float32)
    thresh = max(0.06, float(np.percentile(gray[gray > 0], 6.0)) if (gray > 0).any() else 0.06)
    pts = np.where(gray > thresh)

    if len(pts[0]) > 50:
        y_min, y_max = int(np.min(pts[0])), int(np.max(pts[0]))
        x_min, x_max = int(np.min(pts[1])), int(np.max(pts[1]))

        h = y_max - y_min
        w = x_max - x_min
        side = max(h, w)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        y0 = cy - side_padded // 2
        y1 = y0 + side_padded
        x0 = cx - side_padded // 2
        x1 = x0 + side_padded

        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - slice_2d.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - slice_2d.shape[1])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            slice_2d = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left

        return slice_2d[y0:y1, x0:x1]
    return slice_2d


def extract_standard_slices(vol: np.ndarray, modality: str = "T1W") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts aspect-ratio corrected and tight-cropped Axial, Coronal, Sagittal slices at optimal anatomical levels."""
    norm_vol = normalize_volume(vol)
    sx, sy, sz = norm_vol.shape

    # Find bounding box and maximum parenchyma slices
    bg_thresh = max(0.06, float(np.percentile(norm_vol, 15.0)))
    brain_pts = np.where(norm_vol > bg_thresh)

    if len(brain_pts[0]) > 500:
        x_min, x_max = int(np.percentile(brain_pts[0], 2.0)), int(np.percentile(brain_pts[0], 98.0))
        y_min, y_max = int(np.percentile(brain_pts[1], 2.0)), int(np.percentile(brain_pts[1], 98.0))
        z_min, z_max = int(np.percentile(brain_pts[2], 2.0)), int(np.percentile(brain_pts[2], 98.0))

        # Axial: find maximum brain area in the mid 30%-70% range
        z_low = int(z_min + (z_max - z_min) * 0.30)
        z_high = int(z_min + (z_max - z_min) * 0.75)
        if z_high > z_low:
            z_slice_vox = [np.sum(norm_vol[:, :, z] > bg_thresh) for z in range(z_low, z_high + 1)]
            z_idx = z_low + int(np.argmax(z_slice_vox))
        else:
            z_idx = int(z_min + (z_max - z_min) * 0.52)

        # Coronal: find maximum area in mid 40%-60%
        y_low = int(y_min + (y_max - y_min) * 0.35)
        y_high = int(y_min + (y_max - y_min) * 0.65)
        if y_high > y_low:
            y_slice_vox = [np.sum(norm_vol[:, y, :] > bg_thresh) for y in range(y_low, y_high + 1)]
            y_idx = y_low + int(np.argmax(y_slice_vox))
        else:
            y_idx = sy // 2

        # Sagittal: mid-sagittal plane
        x_idx = int(x_min + (x_max - x_min) * 0.50)
    else:
        z_idx = sz // 2
        y_idx = sy // 2
        x_idx = sx // 2

    ax = np.rot90(norm_vol[:, :, z_idx])
    cor = np.rot90(norm_vol[:, y_idx, :])
    sag = np.rot90(norm_vol[x_idx, :, :])

    # Anisotropic z aspect-ratio correction
    z_aspect = 1.36 / 0.94
    im_cor = Image.fromarray((np.clip(cor, 0, 1) * 255).astype(np.uint8))
    cor = np.array(im_cor.resize((cor.shape[1], int(round(cor.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

    im_sag = Image.fromarray((np.clip(sag, 0, 1) * 255).astype(np.uint8))
    sag = np.array(im_sag.resize((sag.shape[1], int(round(sag.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

    return (
        crop_brain_tight(ax),
        crop_brain_tight(cor),
        crop_brain_tight(sag),
    )


def save_512_slice(slice_arr: np.ndarray, out_path: Path):
    """Saves slice as standard 512x512 PNG."""
    im = Image.fromarray((np.clip(slice_arr, 0.0, 1.0) * 255).astype(np.uint8))
    im_512 = im.resize((512, 512), resample=Image.Resampling.LANCZOS)
    im_512.save(out_path)


# ---------------------------------------------------------------------------
# Scanner Presets
# ---------------------------------------------------------------------------

SCANNER_CONFIGS = [
    # Siemens
    {"vendor": "Siemens", "model": "MAGNETOM_Vida", "field_strength": 3.0, "display_name": "Siemens MAGNETOM Vida 3.0T"},
    {"vendor": "Siemens", "model": "Avanto", "field_strength": 1.5, "display_name": "Siemens Avanto 1.5T"},
    {"vendor": "Siemens", "model": "Aera", "field_strength": 1.5, "display_name": "Siemens Aera 1.5T"},
    {"vendor": "Siemens", "model": "MAGNETOM_Sola", "field_strength": 1.5, "display_name": "Siemens MAGNETOM Sola 1.5T"},
    {"vendor": "Siemens", "model": "Skyra", "field_strength": 3.0, "display_name": "Siemens Skyra 3.0T"},
    {"vendor": "Siemens", "model": "TrioTim", "field_strength": 3.0, "display_name": "Siemens TrioTim 3.0T"},
    # GE
    {"vendor": "GE", "model": "DISCOVERY_MR750", "field_strength": 3.0, "display_name": "GE Discovery MR750 3.0T"},
    {"vendor": "GE", "model": "Signa_HDxt", "field_strength": 1.5, "display_name": "GE Signa HDxt 1.5T"},
    {"vendor": "GE", "model": "SIGNA_Premier", "field_strength": 3.0, "display_name": "GE SIGNA Premier 3.0T"},
    {"vendor": "GE", "model": "Optima_MR450w", "field_strength": 1.5, "display_name": "GE Optima MR450w 1.5T"},
    # Philips
    {"vendor": "Philips", "model": "Achieva_dStream", "field_strength": 3.0, "display_name": "Philips Achieva dStream 3.0T"},
    {"vendor": "Philips", "model": "Ingenia", "field_strength": 1.5, "display_name": "Philips Ingenia 1.5T"},
    {"vendor": "Philips", "model": "Intera", "field_strength": 1.5, "display_name": "Philips Intera 1.5T"},
    {"vendor": "Philips", "model": "Ingenia_Elition", "field_strength": 3.0, "display_name": "Philips Ingenia Elition 3.0T"},
]


# ---------------------------------------------------------------------------
# Main Generation Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate brains with different scanner vendors and models.")
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
    parser.add_argument("--output-dir", default="output/eval_hardware/scanner_variation_brains")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=6256)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    parser.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "FLAIR"])
    args_cli = parser.parse_args()

    out_base = Path(args_cli.output_dir)
    nii_dir = out_base / "volumes"
    slice_dir = out_base / "slices_512x512"
    panel_dir = out_base / "panels"
    for d in [nii_dir, slice_dir, panel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("generate_scanner_variation")
    logger.info("=" * 75)
    logger.info("GENERATING BRAIN MRIs ACROSS SCANNER VENDORS AND MODELS")
    logger.info("=" * 75)

    device = torch.device(args_cli.device)
    logger.info(f"Using compute device: {device}")

    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    target_dim = tuple(args.diffusion_unet_inference["dim"])

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

    manifest_records = []
    
    # Run across modalities and scanners
    for modality in args_cli.modalities:
        mod_key = modality.upper()
        base_prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T1W"]))
        logger.info(f"\n==================== MODALITY: {mod_key} ====================")
        
        mod_slices = []

        for sc in SCANNER_CONFIGS:
            vendor = sc["vendor"]
            sc_model = sc["model"]
            field = sc["field_strength"]
            display_name = sc["display_name"]
            
            sc_prompt = replace_scanner(base_prompt, vendor=vendor, model=sc_model, field_strength=field)
            prompt_axial = ensure_axial_prompt(sc_prompt)
            
            logger.info(f"Synthesizing [{mod_key}] {display_name}...")
            vol_np, final_prompt, final_seed = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, args_cli.cfg_scale, prompt=prompt_axial, initial_seed=args_cli.seed,
                logger=logger, max_retries=3, fallback_prompt=base_prompt
            )
            
            if vol_np is None:
                logger.error(f"Failed to generate valid volume for {display_name} ({mod_key})")
                continue

            # Save 3D NIfTI
            file_stem = f"{mod_key.lower()}_{vendor.lower()}_{sc_model.lower()}_{field}T"
            nii_path = nii_dir / f"{file_stem}.nii.gz"
            nib.save(nib.Nifti1Image(vol_np.astype(np.float32), affine=np.eye(4)), nii_path)

            # Extract 512x512 slices
            ax_slc, cor_slc, sag_slc = extract_standard_slices(vol_np, modality=mod_key)
            
            ax_path = slice_dir / f"{file_stem}_axial.png"
            cor_path = slice_dir / f"{file_stem}_coronal.png"
            sag_path = slice_dir / f"{file_stem}_sagittal.png"
            
            save_512_slice(ax_slc, ax_path)
            save_512_slice(cor_slc, cor_path)
            save_512_slice(sag_slc, sag_path)

            mod_slices.append({
                "scanner_info": sc,
                "axial": ax_slc,
                "coronal": cor_slc,
                "sagittal": sag_slc,
                "paths": {"axial": str(ax_path), "coronal": str(cor_path), "sagittal": str(sag_path), "nii": str(nii_path)},
                "prompt": prompt_axial,
            })

            manifest_records.append({
                "modality": mod_key,
                "vendor": vendor,
                "model": sc_model,
                "field_strength": field,
                "display_name": display_name,
                "prompt": prompt_axial,
                "nifti_path": str(nii_path),
                "slice_axial_512": str(ax_path),
                "slice_coronal_512": str(cor_path),
                "slice_sagittal_512": str(sag_path),
            })

        # Generate elegant multi-scanner comparison grid for this modality
        if mod_slices:
            n_scanners = len(mod_slices)
            fig, axes = plt.subplots(n_scanners, 3, figsize=(12, 3.4 * n_scanners), dpi=180)
            if n_scanners == 1:
                axes = np.expand_dims(axes, 0)

            for row_idx, item in enumerate(mod_slices):
                disp_name = item["scanner_info"]["display_name"]
                
                # Axial
                axes[row_idx, 0].imshow(item["axial"], cmap="gray", vmin=0.0, vmax=1.0)
                axes[row_idx, 0].set_title(f"{disp_name}\nAxial", fontsize=10, fontweight="bold", pad=4)
                axes[row_idx, 0].axis("off")

                # Coronal
                axes[row_idx, 1].imshow(item["coronal"], cmap="gray", vmin=0.0, vmax=1.0)
                axes[row_idx, 1].set_title(f"{disp_name}\nCoronal", fontsize=10, fontweight="bold", pad=4)
                axes[row_idx, 1].axis("off")

                # Sagittal
                axes[row_idx, 2].imshow(item["sagittal"], cmap="gray", vmin=0.0, vmax=1.0)
                axes[row_idx, 2].set_title(f"{disp_name}\nSagittal", fontsize=10, fontweight="bold", pad=4)
                axes[row_idx, 2].axis("off")

            fig.suptitle(f"Multi-Scanner Brain MRI Synthesis: {mod_key} (Fixed Latent Seed: {args_cli.seed})", fontsize=14, fontweight="bold", y=0.995)
            plt.tight_layout()
            mod_panel_path = panel_dir / f"scanner_comparison_panel_{mod_key.lower()}.png"
            plt.savefig(mod_panel_path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved modality scanner comparison panel -> {mod_panel_path}")

    # Save manifest and CSV
    manifest_path = out_base / "scanner_variation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2)

    df_manifest = pd.DataFrame(manifest_records)
    csv_path = out_base / "scanner_variation_manifest.csv"
    df_manifest.to_csv(csv_path, index=False)
    
    logger.info("=" * 75)
    logger.info(f"ALL SCANNER VARIATION BRAINS GENERATED SUCCESSFULLY!")
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info(f"CSV saved to: {csv_path}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
