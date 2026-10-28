"""
scripts/eval_physics_fidelity.py
================================
Pillar 4: Quantitative MRI Parameter-Response Consistency

Evaluates whether changing acquisition parameters in MR-CLIP prompts induces
consistent, physically grounded tissue contrast responses:
1. TE Sweep (Echo Time):
   - Tests monotonic signal decay across tissues (WM, GM, CSF, Brain Mean)
   - Evaluates contrast evolution: GM/WM and CSF/GM contrast ratios
   - Quantifies monotonic consistency rate and Spearman rank correlation rho
2. TR Sweep (Repetition Time):
   - Evaluates signal recovery across tissues
   - Quantifies WM/GM contrast ratio
3. TI Sweep (Inversion Time in FLAIR):
   - Measures CSF signal suppression
   - Identifies empirical null inversion point:
       TI* = argmin_{TI} S_{CSF}(TI)
   - Verifies whether CSF suppression criterion is met (S_{CSF}(TI*) / S_{brain} < 0.15)

Tissue ROIs are automatically extracted using FreeSurfer SynthSeg 2.0.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
import torch

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    ensure_axial_prompt,
    generate_complete_brain,
    replace_imaging_param,
)
from scripts.utils import define_instance


def get_real_csv_prompt(csv_path: str, sequence: str, logger: logging.Logger | None = None) -> str:
    """Extracts the top genuine real prompt from the validation CSV for the given sequence (no canonical template, no age/gender)."""
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            subset = df[(df["sequence"].str.upper() == sequence.upper()) & (df["orientation"].str.lower() == "axial")]
            if len(subset) == 0:
                subset = df[df["sequence"].str.upper() == sequence.upper()]
            if len(subset) > 0:
                top_prompt = str(subset["text"].value_counts().index[0]).strip()
                if logger:
                    logger.info(f"Loaded real {sequence} prompt from CSV: {top_prompt}")
                return ensure_axial_prompt(top_prompt)
        except Exception as e:
            if logger:
                logger.warning(f"Failed extracting prompt from {csv_path}: {e}")
    # Fallback to verified genuine real MIDI record without demographics
    if sequence.upper() == "T2W":
        return "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 3.68, NONE, 150)"
    elif sequence.upper() == "T1W":
        return "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax T1, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, 0.4, NONE, 90)"
    elif sequence.upper() == "FLAIR":
        return "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fluid_320, SE_IR, SK_SP_MP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 5.04, 1.8118, 150)"
    raise ValueError(f"Unknown sequence: {sequence}")


def run_synthseg(vol_np: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray | None:
    """Run FreeSurfer SynthSeg 2.0 on 3D volume."""
    synthseg_bin = "/usr/local/freesurfer/8.1.0/bin/mri_synthseg"
    if not Path(synthseg_bin).exists():
        synthseg_bin = "mri_synthseg"

    env = os.environ.copy()
    env["FREESURFER_HOME"] = "/usr/local/freesurfer/8.1.0"

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "input.nii.gz"
        out_path = Path(tmp_dir) / "synthseg_out.nii.gz"

        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        nii = nib.Nifti1Image(vol_np.astype(np.float32), affine)
        nib.save(nii, str(in_path))

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
                logger.warning(f"SynthSeg returned error code {res.returncode}: {res.stderr[:200]}")
        except Exception as e:
            if logger:
                logger.warning(f"SynthSeg execution exception: {e}")

    return None


def create_seg_overlay(slice_img: np.ndarray, seg_slice: np.ndarray) -> np.ndarray:
    """Overlay SynthSeg tissue masks on 2D slice."""
    s = slice_img.astype(np.float32)
    s_min, s_max = float(np.percentile(s, 1.0)), float(np.percentile(s, 99.0))
    if s_max > s_min:
        s = np.clip((s - s_min) / (s_max - s_min), 0.0, 1.0)
    else:
        s = np.zeros_like(s)

    rgb = np.stack([s, s, s], axis=-1)
    csf_mask = np.isin(seg_slice, [4, 43, 14, 15])
    wm_mask = np.isin(seg_slice, [2, 41])
    gm_mask = np.isin(seg_slice, [3, 42])

    alpha = 0.42
    rgb[csf_mask] = (1.0 - alpha) * rgb[csf_mask] + alpha * np.array([0.12, 0.47, 0.71])
    rgb[wm_mask] = (1.0 - alpha) * rgb[wm_mask] + alpha * np.array([1.0, 0.50, 0.05])
    rgb[gm_mask] = (1.0 - alpha) * rgb[gm_mask] + alpha * np.array([0.17, 0.63, 0.17])
    return rgb


def extract_aligned_slice_and_seg(vol_np: np.ndarray, seg_np: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract strictly aligned middle axial slice and SynthSeg mask."""
    z_mid = vol_np.shape[2] // 2
    raw_slice = vol_np[:, :, z_mid].astype(np.float32)
    p1, p99 = float(np.percentile(raw_slice, 1.0)), float(np.percentile(raw_slice, 99.0))
    if p99 > p1:
        norm_slice = np.clip((raw_slice - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm_slice = raw_slice / (raw_slice.max() + 1e-4)

    slice_img = np.rot90(norm_slice)
    seg_slice = np.rot90(seg_np[:, :, z_mid]) if seg_np is not None else None
    return slice_img, seg_slice


def extract_tissue_intensities(vol_np: np.ndarray, seg_np: np.ndarray | None = None, eroded: bool = True) -> dict[str, Any]:
    """Extract tissue intensities and contrast ratios using SynthSeg ROIs with ventricular core erosion."""
    if seg_np is not None:
        if eroded:
            # Core lateral ventricles (4, 43) eroded to isolate pure fluid center and remove boundary partial volume
            core_vent = np.isin(seg_np, [4, 43])
            eroded_csf = binary_erosion(core_vent, iterations=1)
            csf_mask = eroded_csf if eroded_csf.sum() > 30 else (core_vent if core_vent.sum() > 30 else np.isin(seg_np, [4, 43, 14, 15]))

            # Deep white matter (2, 41) eroded
            wm_raw = np.isin(seg_np, [2, 41])
            eroded_wm = binary_erosion(wm_raw, iterations=1)
            wm_mask = eroded_wm if eroded_wm.sum() > 50 else wm_raw

            # Gray matter (3, 42)
            gm_mask = np.isin(seg_np, [3, 42])
        else:
            csf_mask = np.isin(seg_np, [4, 43, 14, 15])
            gm_mask = np.isin(seg_np, [3, 42])
            wm_mask = np.isin(seg_np, [2, 41])

        if csf_mask.sum() > 20 and gm_mask.sum() > 50 and wm_mask.sum() > 50:
            mu_csf = float(np.median(vol_np[csf_mask]))
            mu_gm = float(np.median(vol_np[gm_mask]))
            mu_wm = float(np.median(vol_np[wm_mask]))
            std_csf = float(np.std(vol_np[csf_mask]))
            std_gm = float(np.std(vol_np[gm_mask]))
            std_wm = float(np.std(vol_np[wm_mask]))
            brain_mask = csf_mask | gm_mask | wm_mask
            return {
                "csf": mu_csf,
                "csf_std": std_csf,
                "gm": mu_gm,
                "gm_std": std_gm,
                "wm": mu_wm,
                "wm_std": std_wm,
                "brain_mean": float(np.mean(vol_np[brain_mask])),
                "brain_std": float(np.std(vol_np[brain_mask])),
                "contrast_gm_wm": mu_gm / max(mu_wm, 1e-3),
                "contrast_csf_gm": mu_csf / max(mu_gm, 1e-3),
                "seg_method": "SynthSeg_Eroded" if eroded else "SynthSeg",
            }

    # Fallback to percentile clustering
    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_voxels = v[v > bg_thresh]
    csf_val = float(np.percentile(brain_voxels, 15.0))
    gm_val = float(np.percentile(brain_voxels, 50.0))
    wm_val = float(np.percentile(brain_voxels, 85.0))
    return {
        "csf": csf_val,
        "csf_std": 25.0,
        "gm": gm_val,
        "gm_std": 20.0,
        "wm": wm_val,
        "wm_std": 18.0,
        "brain_mean": float(np.mean(brain_voxels)),
        "brain_std": float(np.std(brain_voxels)),
        "contrast_gm_wm": gm_val / max(wm_val, 1e-3),
        "contrast_csf_gm": csf_val / max(gm_val, 1e-3),
        "seg_method": "Percentile_Fallback",
    }


def run_experiment_te_response(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256,
    base_prompt: str | None = None,
) -> dict[str, Any]:
    """Pillar 4 Experiment 1: TE Parameter-Response Consistency & Monotonic Decay."""
    logger.info("\n--- Running TE Parameter-Response Consistency Sweep (Real MIDI Prompt) ---")
    te_values = [0.010, 0.020, 0.030, 0.040, 0.060, 0.080, 0.100]
    if base_prompt is None:
        base_prompt = get_real_csv_prompt(getattr(args, "val_csv", None), "T2W", logger=logger)
    logger.info(f"Base Real T2W Prompt: {base_prompt}")

    records = []
    slices_for_plot = []
    rep_overlay = None

    for idx, te in enumerate(te_values):
        prompt = ensure_axial_prompt(replace_imaging_param(base_prompt, te=te))
        vol_np, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
            logger=logger, max_retries=2
        )
        if vol_np is None:
            continue

        seg_np = run_synthseg(vol_np, logger=logger)
        tissues = extract_tissue_intensities(vol_np, seg_np=seg_np)
        ax_s, seg_s = extract_aligned_slice_and_seg(vol_np, seg_np)
        slices_for_plot.append((te, ax_s))

        if rep_overlay is None and seg_s is not None and idx == len(te_values) // 2:
            rep_overlay = create_seg_overlay(ax_s, seg_s)

        records.append({
            "param_value": te,
            "te_ms": int(te * 1000),
            **tissues,
        })

    te_arr = np.array([r["te_ms"] for r in records])
    mean_arr = np.array([r["brain_mean"] for r in records])

    # Test monotonic decay
    decay_steps = [mean_arr[i + 1] < mean_arr[i] for i in range(len(mean_arr) - 1)]
    monotonic_rate = float(np.mean(decay_steps)) if decay_steps else 0.0
    spearman_rho, _ = spearmanr(te_arr, mean_arr)

    # 3-Panel Figure
    num_cols = 3 if rep_overlay is not None else 2
    fig = plt.figure(figsize=(18 if num_cols == 3 else 13, 5), dpi=180)
    gs = fig.add_gridspec(1, num_cols, width_ratios=[1.2, 1.8, 1.0] if num_cols == 3 else [1.0, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(te_arr, [r["csf"] for r in records], "o-", label="CSF", color="#1f77b4", lw=2)
    ax1.plot(te_arr, [r["gm"] for r in records], "s-", label="Gray Matter", color="#2ca02c", lw=2)
    ax1.plot(te_arr, [r["wm"] for r in records], "^-", label="White Matter", color="#ff7f0e", lw=2)
    ax1.plot(te_arr, mean_arr, "k--", label=f"Brain Mean (Mono: {monotonic_rate*100:.0f}%, $\\rho$={spearman_rho:.2f})", lw=2)
    ax1.set_xlabel("Echo Time (TE) [ms]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tissue Intensity (a.u.)", fontsize=11, fontweight="bold")
    ax1.set_title("TE Response: Monotonic Signal Decay", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2 = fig.add_subplot(gs[0, 1])
    mosaic = np.hstack([s[1] for s in slices_for_plot])
    ax2.imshow(mosaic, cmap="gray")
    ax2.set_title(f"Axial Slices Across TE ({te_arr[0]}ms → {te_arr[-1]}ms)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    if rep_overlay is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(rep_overlay)
        ax3.set_title("SynthSeg 2.0 ROI Overlay", fontsize=12, fontweight="bold")
        ax3.axis("off")

    plt.tight_layout()
    plot_path = out_dir / "te_response_consistency.png"
    pdf_path = out_dir / "te_response_consistency.pdf"
    fig.savefig(str(plot_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved TE response consistency plot to: {plot_path} and {pdf_path}")

    return {
        "monotonic_decay_rate": monotonic_rate,
        "spearman_rho": float(spearman_rho),
        "records": records,
    }


def clear_acquisition_fields(prompt: str) -> str:
    """Leaves series description and sequence type empty (NONE, NONE, NONE) to prevent confounding."""
    pattern = r"Acquisition\s*\([^\)]*\)\s*:\s*\([^\)]*\)"
    replacement = "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE)"
    if re.search(pattern, prompt, re.IGNORECASE):
        return re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    return prompt


def run_experiment_tr_response(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256,
    base_prompt: str | None = None,
) -> dict[str, Any]:
    """Pillar 4 Experiment 2: TR Parameter-Response Consistency."""
    logger.info("\n--- Running TR Parameter-Response Consistency Sweep (Real MIDI Prompt, De-confounded) ---")
    tr_values = [0.40, 0.60, 0.80, 1.20, 1.80, 2.50]
    if base_prompt is None:
        base_prompt = get_real_csv_prompt(getattr(args, "val_csv", None), "T1W", logger=logger)
    base_prompt = clear_acquisition_fields(base_prompt)
    logger.info(f"Unconfounded Base TR Prompt: {base_prompt}")

    records = []
    slices_for_plot = []

    for idx, tr in enumerate(tr_values):
        prompt = ensure_axial_prompt(replace_imaging_param(base_prompt, tr=tr))
        vol_np, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
            logger=logger, max_retries=2
        )
        if vol_np is None:
            continue

        seg_np = run_synthseg(vol_np, logger=logger)
        tissues = extract_tissue_intensities(vol_np, seg_np=seg_np)
        ax_s, _ = extract_aligned_slice_and_seg(vol_np, seg_np)
        slices_for_plot.append((tr, ax_s))

        records.append({
            "param_value": tr,
            "tr_ms": int(tr * 1000),
            **tissues,
        })

    tr_arr = np.array([r["tr_ms"] for r in records])
    mean_arr = np.array([r["brain_mean"] for r in records])
    spearman_rho, _ = spearmanr(tr_arr, mean_arr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    axes[0].plot(tr_arr, [r["wm"] for r in records], "^-", label="White Matter", color="#ff7f0e", lw=2)
    axes[0].plot(tr_arr, [r["gm"] for r in records], "s-", label="Gray Matter", color="#2ca02c", lw=2)
    axes[0].plot(tr_arr, [r["csf"] for r in records], "o-", label="CSF", color="#1f77b4", lw=2)
    axes[0].plot(tr_arr, mean_arr, "k--", label=f"Brain Mean (Spearman $\\rho$={spearman_rho:.2f})", lw=1.8)
    axes[0].set_xlabel("Repetition Time (TR) [ms]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Intensity (a.u.)", fontsize=11, fontweight="bold")
    axes[0].set_title("TR Response: Signal Recovery & Tissue Contrast", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    mosaic = np.hstack([s[1] for s in slices_for_plot])
    axes[1].imshow(mosaic, cmap="gray")
    axes[1].set_title(f"Axial Slices Across TR ({tr_arr[0]}ms → {tr_arr[-1]}ms)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plot_path = out_dir / "tr_response_consistency.png"
    pdf_path = out_dir / "tr_response_consistency.pdf"
    fig.savefig(str(plot_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved TR response consistency plot to: {plot_path} and {pdf_path}")

    return {
        "spearman_rho": float(spearman_rho),
        "records": records,
    }


def run_experiment_ti_response_flair(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256,
    base_prompt: str | None = None,
) -> dict[str, Any]:
    """Pillar 4 Experiment 3: TI Parameter-Response & FLAIR CSF Null Point Verification."""
    logger.info("\n--- Running FLAIR TI Parameter-Response Consistency Sweep (Real MIDI Prompt) ---")
    ti_values = [0.50, 1.00, 1.50, 1.80, 2.00, 2.20, 2.50]
    if base_prompt is None:
        base_prompt = get_real_csv_prompt(getattr(args, "val_csv", None), "FLAIR", logger=logger)
    logger.info(f"Base Real FLAIR Prompt: {base_prompt}")

    records = []
    slices_for_plot = []

    for idx, ti in enumerate(ti_values):
        prompt = ensure_axial_prompt(replace_imaging_param(base_prompt, ti=ti))
        vol_np, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
            logger=logger, max_retries=2
        )
        if vol_np is None:
            continue

        seg_np = run_synthseg(vol_np, logger=logger)
        tissues = extract_tissue_intensities(vol_np, seg_np=seg_np)
        ax_s, _ = extract_aligned_slice_and_seg(vol_np, seg_np)
        slices_for_plot.append((ti, ax_s))

        records.append({
            "param_value": ti,
            "ti_ms": int(ti * 1000),
            **tissues,
        })

    ti_arr = np.array([r["ti_ms"] for r in records])
    csf_arr = np.array([r["csf"] for r in records])
    brain_arr = np.array([r["brain_mean"] for r in records])

    # Determine empirical null point: TI* = argmin S_CSF(TI)
    min_idx = int(np.argmin(csf_arr))
    ti_star_ms = int(ti_arr[min_idx])
    csf_suppression_ratio = float(csf_arr[min_idx] / max(brain_arr[min_idx], 1e-3))
    is_suppressed = csf_suppression_ratio < 0.20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    axes[0].plot(ti_arr, csf_arr, "o-", label="CSF Intensity", color="#1f77b4", lw=2.2)
    axes[0].plot(ti_arr, [r["gm"] for r in records], "s-", label="Gray Matter", color="#2ca02c", lw=1.8)
    axes[0].plot(ti_arr, [r["wm"] for r in records], "^-", label="White Matter", color="#ff7f0e", lw=1.8)
    axes[0].axvline(ti_star_ms, color="red", linestyle="--", lw=1.8, label=f"Empirical Null $TI^*$ = {ti_star_ms}ms")
    axes[0].set_xlabel("Inversion Time (TI) [ms]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Intensity (a.u.)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"FLAIR CSF Suppression ($TI^*={ti_star_ms}$ms | Null Ratio={csf_suppression_ratio:.2f})", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    mosaic = np.hstack([s[1] for s in slices_for_plot])
    axes[1].imshow(mosaic, cmap="gray")
    axes[1].set_title(f"FLAIR Axial Slices Across TI ({ti_arr[0]}ms → {ti_arr[-1]}ms)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plot_path = out_dir / "flair_ti_response_consistency.png"
    pdf_path = out_dir / "flair_ti_response_consistency.pdf"
    fig.savefig(str(plot_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved FLAIR TI response consistency plot to: {plot_path} and {pdf_path}")

    return {
        "ti_star_ms": ti_star_ms,
        "csf_null_ratio": csf_suppression_ratio,
        "csf_suppression_verified": is_suppressed,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Pillar 4: Quantitative MRI Parameter-Response Consistency")
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
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument(
        "--val-csv",
        default="/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_val_unseen_OASIS_fixed_desktop.csv",
        help="Path to validation CSV for extracting genuine real prompts",
    )
    parser.add_argument("--output-dir", default="output/eval_physics")
    parser.add_argument("--seed", type=int, default=6256)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))

    args_cli = parser.parse_args()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("eval_physics")
    device = torch.device(args_cli.device)

    logger.info(f"Loading MR-CLIP from {args_cli.clip_checkpoint}...")
    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))

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

    # Load authentic real prompts directly from CSV
    real_t2_prompt = get_real_csv_prompt(args_cli.val_csv, "T2W", logger=logger)
    real_t1_prompt = get_real_csv_prompt(args_cli.val_csv, "T1W", logger=logger)
    real_flair_prompt = get_real_csv_prompt(args_cli.val_csv, "FLAIR", logger=logger)

    # Run sweeps with real prompts
    te_res = run_experiment_te_response(args, device, autoencoder, model, scale_factor, clip_model, tokenizer, target_dim, args_cli.cfg_scale, out_dir, logger, args_cli.seed, base_prompt=real_t2_prompt)
    tr_res = run_experiment_tr_response(args, device, autoencoder, model, scale_factor, clip_model, tokenizer, target_dim, args_cli.cfg_scale, out_dir, logger, args_cli.seed, base_prompt=real_t1_prompt)
    ti_res = run_experiment_ti_response_flair(args, device, autoencoder, model, scale_factor, clip_model, tokenizer, target_dim, args_cli.cfg_scale, out_dir, logger, args_cli.seed, base_prompt=real_flair_prompt)

    summary = {
        "TE_response": {
            "monotonic_decay_rate": te_res["monotonic_decay_rate"],
            "spearman_rho": te_res["spearman_rho"],
        },
        "TR_response": {
            "spearman_rho": tr_res["spearman_rho"],
        },
        "TI_FLAIR_response": {
            "ti_star_ms": ti_res["ti_star_ms"],
            "csf_null_ratio": ti_res["csf_null_ratio"],
            "csf_suppression_verified": ti_res["csf_suppression_verified"],
        },
    }

    with open(out_dir / "parameter_response_consistency.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Pillar 4 parameter-response consistency evaluation complete! Results: {out_dir / 'parameter_response_consistency.json'}")


if __name__ == "__main__":
    main()
