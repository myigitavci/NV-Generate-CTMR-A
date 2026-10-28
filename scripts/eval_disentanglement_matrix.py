"""
scripts/eval_disentanglement_matrix.py
======================================
Pillar 2: Same Latent = Anatomy / Acquisition Disentanglement

Proves that MR-CLIP conditioning achieves genuine physical disentanglement:
For 100 fixed latent seeds z:
    I_A = G(z, m_A),   I_B = G(z, m_B)
where m_A, m_B differ only in acquisition (Vendor, Field Strength, Modality).

Measures:
- Anatomy:
    * SynthSeg 2.0 multi-structure Dice between I_A and I_B (WM, GM, Ventricles, Subcortical nuclei)
- Acquisition:
    * Frozen acquisition classifier accuracy
    * MR-CLIP visual embedding distance (1 - cos(v_A, v_B))
    * Tissue-level contrast difference (|CNR_A - CNR_B|)

Demonstrates:
    [HIGH ANATOMY SIMILARITY + HIGH ACQUISITION DISCRIMINATION]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.frozen_acquisition_predictors import (
    MRCLIPZeroShotPredictor,
    encode_volume_mrclip,
)
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
    replace_scanner,
)
from scripts.utils import define_instance


def run_synthseg(vol_np: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray | None:
    """Run FreeSurfer SynthSeg 2.0 to extract anatomical segmentation."""
    synthseg_bin = "/usr/local/freesurfer/8.1.0/bin/mri_synthseg"
    if not Path(synthseg_bin).exists():
        synthseg_bin = "mri_synthseg"

    env = os.environ.copy()
    env["FREESURFER_HOME"] = "/usr/local/freesurfer/8.1.0"

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "input.nii.gz"
        out_path = Path(tmp_dir) / "synthseg_out.nii.gz"

        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        nib.save(nib.Nifti1Image(vol_np.astype(np.float32), affine), str(in_path))

        cmd = [synthseg_bin, "--i", str(in_path), "--o", str(out_path), "--fast", "--keepgeom", "--threads", "4"]
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and out_path.exists():
                seg_nii = nib.load(str(out_path))
                seg_np = np.asanyarray(seg_nii.dataobj, dtype=np.int32)
                if seg_np.shape != vol_np.shape:
                    from scipy.ndimage import zoom
                    factors = [v / s for v, s in zip(vol_np.shape, seg_np.shape)]
                    seg_np = zoom(seg_np, factors, order=0).astype(np.int32)
                return seg_np
            elif logger:
                logger.warning(f"SynthSeg error {res.returncode}: {res.stderr[:200]}")
        except Exception as e:
            if logger:
                logger.warning(f"SynthSeg execution exception: {e}")

    return None


def compute_structure_dice(seg1: np.ndarray, seg2: np.ndarray) -> dict[str, float]:
    """Compute Dice scores across FreeSurfer SynthSeg anatomical structures."""
    structures = {
        "White_Matter": [2, 41],
        "Cortex_GM": [3, 42],
        "Lateral_Ventricles": [4, 43],
        "Thalamus": [10, 49],
        "Caudate": [11, 50],
        "Putamen": [12, 51],
        "Hippocampus": [17, 53],
        "Brainstem": [16],
    }
    dice_scores = {}
    all_dices = []
    for s_name, label_list in structures.items():
        m1 = np.isin(seg1, label_list)
        m2 = np.isin(seg2, label_list)
        inter = 2.0 * np.sum(m1 & m2)
        total = np.sum(m1) + np.sum(m2)
        if total > 0:
            d = float(inter / total)
            dice_scores[s_name] = d
            all_dices.append(d)
        else:
            dice_scores[s_name] = np.nan

    dice_scores["Mean_Dice"] = float(np.nanmean(all_dices)) if all_dices else np.nan
    return dice_scores


def compute_tissue_cnr(vol_np: np.ndarray, seg_np: np.ndarray | None) -> float:
    """Compute GM-WM Contrast-to-Noise Ratio."""
    if seg_np is not None:
        wm_mask = np.isin(seg_np, [2, 41])
        gm_mask = np.isin(seg_np, [3, 42])
        if wm_mask.sum() > 100 and gm_mask.sum() > 100:
            mu_wm, sig_wm = float(np.mean(vol_np[wm_mask])), float(np.std(vol_np[wm_mask]))
            mu_gm, sig_gm = float(np.mean(vol_np[gm_mask])), float(np.std(vol_np[gm_mask]))
            denom = np.sqrt(0.5 * (sig_wm ** 2 + sig_gm ** 2))
            return float(abs(mu_wm - mu_gm) / max(denom, 1e-4))
    return float(np.nan)


def encode_volume_mrclip(vol_np: np.ndarray, clip_model: Any, device: str) -> np.ndarray:
    """Extract normalized 512-dim MR-CLIP visual feature vector."""
    vol_th = torch.from_numpy(vol_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    v_min, v_max = float(np.percentile(vol_np, 1.0)), float(np.percentile(vol_np, 99.0))
    if v_max > v_min:
        vol_th = (vol_th - v_min) / (v_max - v_min)
        vol_th = torch.clamp(vol_th, 0.0, 1.0)
    if vol_th.shape[2:] != (128, 128, 128):
        vol_th = F.interpolate(vol_th, size=(128, 128, 128), mode="trilinear", align_corners=False)
    vol_th = vol_th.to(device)
    with torch.no_grad():
        if hasattr(clip_model, "encode_image"):
            feat = clip_model.encode_image(vol_th)
        else:
            feat = clip_model.visual(vol_th)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


def run_disentanglement_evaluation(
    args,
    device: str,
    autoencoder: Any,
    model: Any,
    scale_factor: float,
    clip_model: Any,
    tokenizer: Any,
    target_dim: tuple[int, int, int],
    cfg_scale: float,
    predictor: MRCLIPZeroShotPredictor,
    out_dir: Path,
    num_seeds: int = 100,
    logger: logging.Logger | None = None,
):
    if logger is None:
        logger = logging.getLogger("eval_disentanglement")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"STARTING ANATOMY/ACQUISITION DISENTANGLEMENT EVALUATION (Seeds: {num_seeds})")
    logger.info("=" * 70)

    # Pairs of acquisition conditions differing strictly in acquisition metadata
    pairs_config = {
        "Field_Strength": {
            "name": "1.5T vs 3.0T (GE SIGNA_HDx)",
            "cat_key": "field_strength",
            "prompt_A": ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "GE", "SIGNA_HDx", 1.5)),
            "prompt_B": ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "GE", "SIGNA_HDx", 3.0)),
            "val_A": "1.5",
            "val_B": "3.0",
        },
        "Vendor": {
            "name": "Siemens Vida vs GE MR750 (3.0T)",
            "cat_key": "vendor",
            "prompt_A": ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "Siemens", "MAGNETOM_Vida", 3.0)),
            "prompt_B": ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "GE", "DISCOVERY_MR750", 3.0)),
            "val_A": "Siemens",
            "val_B": "GE",
        },
        "Modality": {
            "name": "T1w vs T2w (Siemens Vida 3.0T)",
            "cat_key": "modality",
            "prompt_A": ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES["T1W"]),
            "prompt_B": ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES["T2W"]),
            "val_A": "T1W",
            "val_B": "T2W",
        },
    }

    base_seed_offset = 7890
    seeds = [base_seed_offset + i * 17 for i in range(num_seeds)]

    all_records = []
    rep_visuals = {}

    for p_key, p_info in pairs_config.items():
        logger.info(f"\n--- Testing Disentanglement Condition: {p_info['name']} ---")
        prompt_A = p_info["prompt_A"]
        prompt_B = p_info["prompt_B"]
        cat_key = p_info["cat_key"]
        target_A, target_B = p_info["val_A"], p_info["val_B"]

        for s_idx, seed in enumerate(seeds):
            # Generate I_A = G(z, m_A)
            vol_A, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt_A, initial_seed=seed,
                logger=logger, max_retries=2
            )
            # Generate I_B = G(z, m_B) with the exact same seed z
            vol_B, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt_B, initial_seed=seed,
                logger=logger, max_retries=2
            )

            if vol_A is None or vol_B is None:
                continue

            # Run SynthSeg segmentation
            seg_A = run_synthseg(vol_A, logger=logger)
            seg_B = run_synthseg(vol_B, logger=logger)

            if seg_A is not None and seg_B is not None:
                dices = compute_structure_dice(seg_A, seg_B)
            else:
                dices = {"Mean_Dice": np.nan}

            # Acquisition metrics: zero-shot discrimination between target_A and target_B
            pair_prompts = {target_A: prompt_A, target_B: prompt_B}
            v_A = encode_volume_mrclip(vol_A, clip_model, device)
            v_B = encode_volume_mrclip(vol_B, clip_model, device)
            cos_dist = float(1.0 - np.dot(v_A, v_B))

            pred_cat_A, _ = predictor.predict_categorical(v_A, pair_prompts)
            pred_cat_B, _ = predictor.predict_categorical(v_B, pair_prompts)

            discriminated = (pred_cat_A == target_A) and (pred_cat_B == target_B)

            cnr_A = compute_tissue_cnr(vol_A, seg_A)
            cnr_B = compute_tissue_cnr(vol_B, seg_B)
            cnr_diff = abs(cnr_A - cnr_B) if not (np.isnan(cnr_A) or np.isnan(cnr_B)) else np.nan

            rec = {
                "condition": p_key,
                "pair_name": p_info["name"],
                "seed": seed,
                "dice_mean": dices.get("Mean_Dice", np.nan),
                "dice_wm": dices.get("White_Matter", np.nan),
                "dice_gm": dices.get("Cortex_GM", np.nan),
                "dice_ventricles": dices.get("Lateral_Ventricles", np.nan),
                "mrclip_distance": cos_dist,
                "acquisition_discriminated": int(discriminated),
                "cnr_diff": cnr_diff,
            }
            all_records.append(rec)

            if p_key not in rep_visuals and seg_A is not None and seg_B is not None:
                # Save middle slice for publication figure
                z_mid = vol_A.shape[2] // 2
                rep_visuals[p_key] = {
                    "slice_A": vol_A[:, :, z_mid],
                    "slice_B": vol_B[:, :, z_mid],
                    "seg_A": seg_A[:, :, z_mid],
                    "seg_B": seg_B[:, :, z_mid],
                }

            if (s_idx + 1) % 25 == 0 or (s_idx + 1) == len(seeds):
                logger.info(f"  [{p_key}] Done seed {s_idx + 1}/{len(seeds)} | Dice: {dices.get('Mean_Dice', np.nan):.3f} | MR-CLIP Dist: {cos_dist:.3f}")

    df_pairs = pd.DataFrame(all_records)
    df_pairs.to_csv(out_dir / "disentanglement_per_seed.csv", index=False)
    logger.info(f"Saved per-seed records to: {out_dir / 'disentanglement_per_seed.csv'}")

    # Publication Summary Figure: Anatomy Invariance vs Acquisition Discrimination
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Panel 1: Scatter plot demonstrating high anatomy similarity + high acquisition discrimination
    ax1 = axes[0]
    palette = {"Field_Strength": "#1f77b4", "Vendor": "#2ca02c", "Modality": "#d62728"}
    labels = {"Field_Strength": "1.5T vs 3.0T", "Vendor": "Siemens vs GE", "Modality": "T1w vs T2w"}

    for cond in ["Field_Strength", "Vendor", "Modality"]:
        sub = df_pairs[df_pairs["condition"] == cond]
        if len(sub) > 0:
            ax1.scatter(
                sub["dice_mean"],
                sub["mrclip_distance"],
                color=palette[cond],
                alpha=0.6,
                s=40,
                edgecolors="none",
                label=labels[cond],
            )

    ax1.axvline(0.85, color="gray", linestyle="--", lw=1.5, alpha=0.7, label="Anatomy Preservation Threshold (0.85)")
    ax1.set_xlabel("Anatomical SynthSeg Dice (Same Latent z)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Acquisition MR-CLIP Distance (1 - cos)", fontsize=11, fontweight="bold")
    ax1.set_title("Anatomy Invariance vs Acquisition Divergence\n[Target: High Dice + High Acquisition Discrimination]", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Panel 2: Bar chart of Mean Dice vs Acquisition Discrimination Accuracy
    ax2 = axes[1]
    conditions = ["Field_Strength", "Vendor", "Modality"]
    cond_labels = ["1.5T vs 3.0T", "Siemens vs GE", "T1w vs T2w"]
    mean_dices = [df_pairs[df_pairs["condition"] == c]["dice_mean"].mean() for c in conditions]
    discrim_accs = [df_pairs[df_pairs["condition"] == c]["acquisition_discriminated"].mean() for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35
    b1 = ax2.bar(x - width / 2, mean_dices, width, label="SynthSeg Mean Dice (Anatomy)", color="#2ca02c", edgecolor="black")
    b2 = ax2.bar(x + width / 2, discrim_accs, width, label="Classifier Accuracy (Acquisition)", color="#1f77b4", edgecolor="black")

    ax2.set_ylabel("Score [0 to 1]", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Quantified Disentanglement Metrics", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    for bars in [b1, b2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "disentanglement_evaluation.png"
    fig.savefig(str(plot_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved disentanglement figure to: {plot_path}")

    # Summary JSON
    summary_dict = {}
    for c in conditions:
        sub = df_pairs[df_pairs["condition"] == c]
        summary_dict[c] = {
            "mean_dice": float(sub["dice_mean"].mean()),
            "std_dice": float(sub["dice_mean"].std()),
            "acquisition_accuracy": float(sub["acquisition_discriminated"].mean()),
            "mean_mrclip_dist": float(sub["mrclip_distance"].mean()),
            "n_seeds": int(len(sub)),
        }

    with open(out_dir / "disentanglement_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved disentanglement summary to: {out_dir / 'disentanglement_summary.json'}")

    return summary_dict


def parse_args():
    p = argparse.ArgumentParser(description="Pillar 2: Anatomy/Acquisition Disentanglement Suite")
    p.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    p.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    p.add_argument(
        "--clip-checkpoint",
        default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt",
    )
    p.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--num-seeds", type=int, default=100)
    p.add_argument("--output-dir", default="output/eval_disentanglement")
    p.add_argument("--predictors-path", default="models/frozen_param_predictors.joblib")
    p.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main():
    args_cli = parse_args()
    logger = setup_logging("eval_disentanglement")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args_cli.device)
    logger.info(f"Target compute device: {device}")

    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))
    predictor = MRCLIPZeroShotPredictor(clip_model, tokenizer, str(device))

    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    autoencoder.eval()

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(base_unet, clip_embed_dim=512).to(device)
    diff_ckpt = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    sd = diff_ckpt.get("model_state_dict", diff_ckpt.get("state_dict", diff_ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    scale_factor = args.scale_factor if hasattr(args, "scale_factor") else 1.0
    target_dim = tuple(args.diffusion_unet_inference.get("dim", args.diffusion_unet_inference.get("output_size", [256, 256, 128])))

    run_disentanglement_evaluation(
        args=args,
        device=str(device),
        autoencoder=autoencoder,
        model=model,
        scale_factor=scale_factor,
        clip_model=clip_model,
        tokenizer=tokenizer,
        target_dim=target_dim,
        cfg_scale=args_cli.cfg_scale,
        predictor=predictor,
        out_dir=out_dir,
        num_seeds=args_cli.num_seeds,
        logger=logger,
    )


if __name__ == "__main__":
    main()
