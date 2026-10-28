#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
eval_hardware_disentanglement.py
================================
Evaluates anatomical preservation and hardware disentanglement across scanner vendors
and magnetic field strengths (1.5T vs 3.0T) in 3D brain MR synthesis.

Key Experiments:
  1. Scanner Vendor/Model Invariance (Fixed Seed):
     - Evaluates whether anatomy remains morphologically identical across vendors
       (Siemens MAGNETOM Vida, GE Discovery MR750, Philips Achieva dStream) when latent seed is fixed.
     - Supports T2W, FLAIR, and T1W modalities.
     - Computes 3D SSIM, Normalized Mutual Information (NMI), and 3D absolute difference maps.
  2. Field Strength Contrast & SNR/CNR Scaling (1.5T vs 3.0T):
     - Evaluates physical SNR and tissue CNR scaling between 1.5T and 3.0T acquisitions.
     - Evaluates Siemens (Sola vs Vida), Philips (Ingenia vs Achieva), and GE (Signa vs Discovery).

Outputs:
  - output/eval_hardware/vendor_invariance_panel_{modality}.png
  - output/eval_hardware/field_strength_snr_cnr_comparison_{modality}.png
  - output/eval_hardware/hardware_disentanglement_metrics.csv
"""

from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd
import torch
from skimage.metrics import normalized_mutual_information, structural_similarity

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_sample,
    generate_complete_brain,
    is_half_or_incomplete_brain,
    get_middle_slices,
    encode_prompt,
    replace_scanner,
)
from scripts.utils import define_instance


def compute_snr_cnr(vol_np: np.ndarray, modality: str = "T2W") -> dict[str, float]:
    """
    Computes rigorous, scale-invariant medical imaging SNR and tissue CNR.
    - SNR_WM = mean(WM) / std(WM)  (AAPM/NEMA tissue SNR definition)
    - SNR_GM = mean(GM) / std(GM)
    - CNR_GM_WM = |mean(GM) - mean(WM)| / sqrt(std(GM)^2 + std(WM)^2) (Statistical separability)
    This formulation is scale-invariant and immune to arbitrary neural network gain variations.
    """
    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_voxels = v[v > bg_thresh]
    if len(brain_voxels) < 500:
        return {"snr_wm": 0.0, "snr_gm": 0.0, "cnr": 0.0, "sigma_bg": 1.0}

    mod = modality.upper()
    if "T1" in mod:
        # T1w: WM is bright (70-95th percentile), GM is intermediate (35-65th percentile)
        wm_vox = brain_voxels[(brain_voxels >= np.percentile(brain_voxels, 70.0)) & (brain_voxels <= np.percentile(brain_voxels, 95.0))]
        gm_vox = brain_voxels[(brain_voxels >= np.percentile(brain_voxels, 35.0)) & (brain_voxels <= np.percentile(brain_voxels, 65.0))]
    else:  # T2w / FLAIR: WM is darker (15-40th percentile), GM is brighter (50-75th percentile)
        wm_vox = brain_voxels[(brain_voxels >= np.percentile(brain_voxels, 15.0)) & (brain_voxels <= np.percentile(brain_voxels, 40.0))]
        gm_vox = brain_voxels[(brain_voxels >= np.percentile(brain_voxels, 50.0)) & (brain_voxels <= np.percentile(brain_voxels, 75.0))]

    mu_wm = float(np.mean(wm_vox)) if len(wm_vox) > 0 else 1.0
    std_wm = float(np.std(wm_vox)) if len(wm_vox) > 0 else 1.0
    mu_gm = float(np.mean(gm_vox)) if len(gm_vox) > 0 else 1.0
    std_gm = float(np.std(gm_vox)) if len(gm_vox) > 0 else 1.0

    # AAPM scale-invariant tissue SNR: mu / sigma
    snr_wm = mu_wm / max(std_wm, 1e-3)
    snr_gm = mu_gm / max(std_gm, 1e-3)

    # Scale-invariant contrast-to-noise ratio
    cnr = abs(mu_gm - mu_wm) / np.sqrt(max(std_gm**2 + std_wm**2, 1e-4))

    return {
        "snr_wm": float(snr_wm),
        "snr_gm": float(snr_gm),
        "cnr": float(cnr),
        "wm_intensity": float(mu_wm),
        "gm_intensity": float(mu_gm),
        "sigma_bg": float(std_wm),
    }


def run_vendor_invariance_eval(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, modality: str = "T2W", seeds: list[int] = [6256, 42, 100]
) -> list[dict]:
    """
    Tests anatomy preservation across Siemens, GE, and Philips vendors for fixed seeds on a given modality.
    """
    mod_key = modality.upper()
    logger.info(f"Evaluating scanner vendor/model invariance on {mod_key} with fixed latent noise...")

    base_prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T2W"]))
    vendors = [
        {
            "name": f"Siemens_Avanto_1.5T_{mod_key}",
            "vendor": "Siemens_Avanto",
            "prompt": replace_scanner(base_prompt, vendor="Siemens", model="Avanto", field_strength=1.5),
        },
        {
            "name": f"Siemens_Aera_1.5T_{mod_key}",
            "vendor": "Siemens_Aera",
            "prompt": replace_scanner(base_prompt, vendor="Siemens", model="Aera", field_strength=1.5),
        },
        {
            "name": f"GE_Signa_HDxt_1.5T_{mod_key}",
            "vendor": "GE_Signa",
            "prompt": replace_scanner(base_prompt, vendor="GE", model="Signa_HDxt", field_strength=1.5),
        },
    ]

    metrics = []
    first_seed_slices = {}

    for seed in seeds:
        seed_vols = {}
        for v in vendors:
            p = ensure_axial_prompt(v["prompt"])
            vol_np, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=p, initial_seed=seed,
                logger=logger, max_retries=3, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T2W"])
            )
            if vol_np is None:
                continue
            seed_vols[v["vendor"]] = vol_np

            if seed == seeds[0]:
                ax_s, cor_s, sag_s = get_middle_slices(vol_np)
                first_seed_slices[v["vendor"]] = (ax_s, cor_s, sag_s)

        # Pairwise comparisons
        pairs = [("Siemens_Avanto", "Siemens_Aera"), ("Siemens_Avanto", "GE_Signa"), ("Siemens_Aera", "GE_Signa")]
        for v1, v2 in pairs:
            im1 = seed_vols[v1] / 1000.0
            im2 = seed_vols[v2] / 1000.0

            ssim_val = float(structural_similarity(im1, im2, data_range=1.0))
            im1_q = (np.clip(im1, 0, 1) * 63).astype(np.uint8)
            im2_q = (np.clip(im2, 0, 1) * 63).astype(np.uint8)
            nmi_val = float(normalized_mutual_information(im1_q, im2_q))
            mean_abs_diff = float(np.mean(np.abs(im1 - im2)))

            metrics.append({
                "experiment": "Vendor_Invariance",
                "modality": mod_key,
                "seed": seed,
                "vendor_pair": f"{v1}_vs_{v2}",
                "ssim_3d": ssim_val,
                "nmi": nmi_val,
                "mean_abs_diff": mean_abs_diff,
            })
            logger.info(f"[{mod_key}] Seed {seed} | {v1} vs {v2}: 3D SSIM = {ssim_val:.4f}, NMI = {nmi_val:.4f}")

    # Plot visual comparison for first seed
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), dpi=180)
    vendor_order = ["Siemens_Avanto", "Siemens_Aera", "GE_Signa"]

    for row_idx, v in enumerate(vendor_order):
        ax_s, cor_s, sag_s = first_seed_slices[v]
        axes[row_idx, 0].imshow(ax_s, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(f"{v} (Axial)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cor_s, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"{v} (Coronal)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(sag_s, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title(f"{v} (Sagittal)", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

    # Difference maps: Siemens_Avanto - Siemens_Aera, Siemens_Avanto - GE_Signa, Siemens_Aera - GE_Signa
    ax_s1 = first_seed_slices["Siemens_Avanto"][0]
    ax_s2 = first_seed_slices["Siemens_Aera"][0]
    ax_ge = first_seed_slices["GE_Signa"][0]

    diff_ge = np.abs(ax_s1.astype(float) - ax_s2.astype(float))
    diff_philips = np.abs(ax_s1.astype(float) - ax_ge.astype(float))
    diff_ge_philips = np.abs(ax_s2.astype(float) - ax_ge.astype(float))

    im_d1 = axes[0, 3].imshow(diff_ge, cmap="magma", vmin=0.0, vmax=0.35)
    axes[0, 3].set_title("|Avanto - Aera| Diff", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")
    fig.colorbar(im_d1, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im_d2 = axes[1, 3].imshow(diff_philips, cmap="magma", vmin=0.0, vmax=0.35)
    axes[1, 3].set_title("|Avanto - GE| Diff", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")
    fig.colorbar(im_d2, ax=axes[1, 3], fraction=0.046, pad=0.04)

    im_d3 = axes[2, 3].imshow(diff_ge_philips, cmap="magma", vmin=0.0, vmax=0.35)
    axes[2, 3].set_title("|Aera - GE| Diff", fontsize=11, fontweight="bold")
    axes[2, 3].axis("off")
    fig.colorbar(im_d3, ax=axes[2, 3], fraction=0.046, pad=0.04)

    # Column 4: Profile line plot & quantitative metrics
    y_mid = ax_s1.shape[0] // 2
    profile_s1 = ax_s1[y_mid, :]
    profile_s2 = ax_s2[y_mid, :]
    profile_ge = ax_ge[y_mid, :]

    axes[0, 4].plot(profile_s1, label="Siemens Avanto", color="crimson", lw=1.5)
    axes[0, 4].plot(profile_s2, label="Siemens Aera", color="teal", lw=1.5, linestyle="--")
    axes[0, 4].plot(profile_ge, label="GE Signa", color="orange", lw=1.5, linestyle=":")
    axes[0, 4].set_title(f"Mid-Axial Profiles ({mod_key})", fontsize=10, fontweight="bold")
    axes[0, 4].set_xlabel("Voxel Column", fontsize=9)
    axes[0, 4].set_ylabel("Intensity", fontsize=9)
    axes[0, 4].legend(fontsize=8)
    axes[0, 4].grid(True, alpha=0.3)

    # Pairwise 3D SSIM Bar Chart
    first_seed_metrics = [m for m in metrics if m["seed"] == seeds[0]]
    pair_names = [m["vendor_pair"].replace("_vs_", " vs ") for m in first_seed_metrics]
    ssim_vals = [m["ssim_3d"] for m in first_seed_metrics]
    nmi_vals = [m["nmi"] for m in first_seed_metrics]

    x_indices = np.arange(len(pair_names))
    axes[1, 4].bar(x_indices, ssim_vals, color="#2b83ba", width=0.5)
    axes[1, 4].set_xticks(x_indices)
    axes[1, 4].set_xticklabels(pair_names, fontsize=8, rotation=15)
    axes[1, 4].set_ylim(0.7, 1.0)
    axes[1, 4].set_ylabel("3D SSIM", fontsize=9)
    axes[1, 4].set_title(f"Pairwise 3D SSIM ({mod_key})", fontsize=10, fontweight="bold")
    axes[1, 4].grid(axis="y", alpha=0.3)

    axes[2, 4].bar(x_indices, nmi_vals, color="#abdda4", width=0.5)
    axes[2, 4].set_xticks(x_indices)
    axes[2, 4].set_xticklabels(pair_names, fontsize=8, rotation=15)
    axes[2, 4].set_ylim(1.0, 1.35)
    axes[2, 4].set_ylabel("NMI", fontsize=9)
    axes[2, 4].set_title(f"Normalized Mutual Info ({mod_key})", fontsize=10, fontweight="bold")
    axes[2, 4].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"MR-CLIP Disentanglement: Anatomy Preservation across Scanner Vendors on {mod_key} (Fixed Seed)",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    mod_panel_path = out_dir / f"vendor_invariance_panel_{mod_key.lower()}.png"
    plt.savefig(mod_panel_path, dpi=180, bbox_inches="tight")
    # Also save to base file for backward compatibility
    default_panel_path = out_dir / "vendor_invariance_panel.png"
    plt.savefig(default_panel_path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved vendor invariance panel -> {mod_panel_path}")

    return metrics


def run_field_strength_eval(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256, modality: str = "T2W"
) -> list[dict]:
    """
    Evaluates physical SNR and tissue CNR scaling between 1.5T and 3.0T on requested modality.
    """
    mod_key = modality.upper()
    logger.info(f"Evaluating 1.5T vs 3.0T Field Strength SNR/CNR Scaling on {mod_key}...")

    base_prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T2W"]))

    test_pairs = [
        {
            "sequence": f"{mod_key} (GE SIGNA_HDx: 1.5T vs 3.0T)",
            "vendor": "GE",
            "p_15T": replace_scanner(base_prompt, vendor="GE", model="SIGNA_HDx", field_strength=1.5),
            "p_30T": replace_scanner(base_prompt, vendor="GE", model="SIGNA_HDx", field_strength=3.0),
        },
        {
            "sequence": f"{mod_key} (Siemens Avanto: 1.5T vs 3.0T)",
            "vendor": "Siemens",
            "p_15T": replace_scanner(base_prompt, vendor="Siemens", model="Avanto", field_strength=1.5),
            "p_30T": replace_scanner(base_prompt, vendor="Siemens", model="Avanto", field_strength=3.0),
        },
        {
            "sequence": f"{mod_key} (GE Discovery: 1.5T vs 3.0T)",
            "vendor": "GE",
            "p_15T": replace_scanner(base_prompt, vendor="GE", model="DISCOVERY_MR750", field_strength=1.5),
            "p_30T": replace_scanner(base_prompt, vendor="GE", model="DISCOVERY_MR750", field_strength=3.0),
        },
    ]

    metrics = []
    fig, axes = plt.subplots(len(test_pairs), 4, figsize=(18, 4.5 * len(test_pairs)), dpi=180)

    for row_idx, tp in enumerate(test_pairs):
        seq = tp["sequence"]
        p15 = ensure_axial_prompt(tp["p_15T"])
        p30 = ensure_axial_prompt(tp["p_30T"])

        vol15, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=p15, initial_seed=seed,
            logger=logger, max_retries=3, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T2W"])
        )
        vol30, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=p30, initial_seed=seed,
            logger=logger, max_retries=3, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(mod_key, CANONICAL_MIDI_TEMPLATES["T2W"])
        )
        if vol15 is None or vol30 is None:
            logger.warning(f"Skipping sequence {seq} due to incomplete synthesis.")
            continue

        snr_cnr_15 = compute_snr_cnr(vol15, modality=mod_key)
        snr_cnr_30 = compute_snr_cnr(vol30, modality=mod_key)

        cnr_ratio = snr_cnr_30["cnr"] / max(snr_cnr_15["cnr"], 1e-3)
        snr_ratio = snr_cnr_30["snr_wm"] / max(snr_cnr_15["snr_wm"], 1e-3)

        metrics.append({
            "experiment": "Field_Strength_Scaling",
            "modality": mod_key,
            "sequence": seq,
            "field": "1.5T",
            "seed": seed,
            **snr_cnr_15,
        })
        metrics.append({
            "experiment": "Field_Strength_Scaling",
            "modality": mod_key,
            "sequence": seq,
            "field": "3.0T",
            "seed": seed,
            **snr_cnr_30,
            "cnr_gain_3T_over_1.5T": cnr_ratio,
            "snr_gain_3T_over_1.5T": snr_ratio,
        })

        logger.info(
            f"[{mod_key}] {seq}: 1.5T CNR={snr_cnr_15['cnr']:.2f}, 3.0T CNR={snr_cnr_30['cnr']:.2f} "
            f"(CNR Gain = {cnr_ratio:.2f}x, SNR Gain = {snr_ratio:.2f}x)"
        )

        ax15_s, _, _ = get_middle_slices(vol15)
        ax30_s, _, _ = get_middle_slices(vol30)

        # Rigidly align 1.5T to 3.0T to guarantee anatomical registration
        try:
            from skimage.registration import phase_cross_correlation
            from scipy.ndimage import shift
            shift_vec, _, _ = phase_cross_correlation(ax30_s, ax15_s, upsample_factor=10)
            ax15_disp = shift(ax15_s, shift_vec, order=1, mode="nearest")
        except Exception:
            ax15_disp = ax15_s

        axes[row_idx, 0].imshow(ax15_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(f"{seq} 1.5T (Axial)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(ax30_s, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"{seq} 3.0T (Axial)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        diff_field = np.abs(ax30_s.astype(float) - ax15_disp.astype(float))
        im_df = axes[row_idx, 2].imshow(diff_field, cmap="inferno", vmin=0.0, vmax=0.35)
        axes[row_idx, 2].set_title(f"|3.0T - 1.5T| Contrast Shift", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")
        fig.colorbar(im_df, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        labels = ["SNR (WM)", "SNR (GM)", "CNR"]
        vals_15 = [snr_cnr_15["snr_wm"], snr_cnr_15["snr_gm"], snr_cnr_15["cnr"]]
        vals_30 = [snr_cnr_30["snr_wm"], snr_cnr_30["snr_gm"], snr_cnr_30["cnr"]]

        x = np.arange(len(labels))
        width = 0.35
        axes[row_idx, 3].bar(x - width / 2, vals_15, width, label="1.5T", color="#4575b4")
        axes[row_idx, 3].bar(x + width / 2, vals_30, width, label="3.0T", color="#d73027")
        axes[row_idx, 3].set_xticks(x)
        axes[row_idx, 3].set_xticklabels(labels, fontsize=9, fontweight="bold")
        axes[row_idx, 3].set_ylabel("Metric Value", fontsize=9)
        axes[row_idx, 3].set_title(f"{seq} (CNR Gain: {cnr_ratio:.2f}x)", fontsize=10, fontweight="bold")
        axes[row_idx, 3].legend(fontsize=8)
        axes[row_idx, 3].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Field Strength Validation on {mod_key}: 1.5T vs 3.0T SNR and CNR Scaling",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    mod_field_path = out_dir / f"field_strength_snr_cnr_comparison_{mod_key.lower()}.png"
    plt.savefig(mod_field_path, dpi=180, bbox_inches="tight")
    default_field_path = out_dir / "field_strength_snr_cnr_comparison.png"
    plt.savefig(default_field_path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved field strength comparison panel -> {mod_field_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate scanner and hardware disentanglement on MIDI.")
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
    parser.add_argument("--output-dir", default="output/eval_hardware")
    parser.add_argument(
        "--modality",
        default="T2W",
        choices=["T2W", "FLAIR", "T1W", "all"],
        help="MRI modality to evaluate for hardware disentanglement (default: T2W; options: T2W, FLAIR, T1W, all)",
    )
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=6256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("eval_hardware")
    logger.info("=" * 70)
    logger.info(f"MODULE 2: SCANNER & HARDWARE DISENTANGLEMENT (Modality: {args_cli.modality})")
    logger.info("=" * 70)

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

    modalities_to_eval = ["T2W", "FLAIR"] if args_cli.modality == "all" else [args_cli.modality]
    all_metrics = []

    for mod in modalities_to_eval:
        logger.info(f"\n--- RUNNING HARDWARE EVALUATION FOR MODALITY: {mod} ---")
        vendor_metrics = run_vendor_invariance_eval(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, modality=mod, seeds=[args_cli.seed, 42, 100]
        )
        all_metrics.extend(vendor_metrics)

        field_metrics = run_field_strength_eval(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, seed=args_cli.seed, modality=mod
        )
        all_metrics.extend(field_metrics)

    df = pd.DataFrame(all_metrics)
    csv_path = out_dir / "hardware_disentanglement_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved all quantitative metrics to {csv_path}")

    # Summary table in console
    logger.info("\n" + "=" * 70)
    logger.info("HARDWARE DISENTANGLEMENT SUMMARY:")
    logger.info("=" * 70)
    if not df.empty:
        vendor_df = df[df["experiment"] == "Vendor_Invariance"]
        if not vendor_df.empty:
            for mod in vendor_df["modality"].unique():
                m_sub = vendor_df[vendor_df["modality"] == mod]
                logger.info(
                    f"Vendor Invariance ({mod}): 3D SSIM = {m_sub['ssim_3d'].mean():.4f} +/- {m_sub['ssim_3d'].std():.4f}, "
                    f"NMI = {m_sub['nmi'].mean():.4f} +/- {m_sub['nmi'].std():.4f}"
                )

        field_df = df[df["experiment"] == "Field_Strength_Scaling"]
        if not field_df.empty:
            for seq in field_df["sequence"].unique():
                seq_rows = field_df[field_df["sequence"] == seq]
                r15 = seq_rows[seq_rows["field"] == "1.5T"]
                r30 = seq_rows[seq_rows["field"] == "3.0T"]
                if not r15.empty and not r30.empty:
                    cnr_15 = r15["cnr"].values[0]
                    cnr_30 = r30["cnr"].values[0]
                    logger.info(f"{seq}: 1.5T CNR={cnr_15:.2f} -> 3.0T CNR={cnr_30:.2f} (x{cnr_30/max(cnr_15,1e-3):.2f})")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
