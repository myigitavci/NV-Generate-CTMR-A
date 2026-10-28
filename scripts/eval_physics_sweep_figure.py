#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
scripts/eval_physics_sweep_figure.py
======================================
Unified Publication-Grade Figure for MRI Physics Parameter Sweeps:
  1. Echo Time (TE) Sweep:       0 -> 200 ms  (T2-decay & contrast evolution)
  2. Repetition Time (TR) Sweep: 0 -> 10000 ms (T1-recovery & saturation)
  3. Inversion Time (TI) Sweep:  0 -> 3000 ms (FLAIR CSF nulling & tissue suppression)

For each acquisition parameter, the figure presents:
  - Column 1: Theoretical Expected Physics (Bloch Simulation at 3T: CSF, GM, WM)
  - Column 2: Diffusion Model Empirical Response (SynthSeg 2.0 ROI Intensities)
  - Column 3: Synthesized Axial Slice Progression (Visual Anatomical Contrast Evolution)

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_physics_sweep_figure.py \
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \
      -t configs/config_network_rflow.json \
      --mrclip-ckpt models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt \
      --clip-checkpoint /home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt \
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \
      --output-dir output/eval_physics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
    replace_imaging_param,
)
from scripts.utils import define_instance


# ---------------------------------------------------------------------------
# 3T MRI Tissue Relaxation Parameters (Stanisz et al. 2005, Lu et al. 2005)
# ---------------------------------------------------------------------------
TISSUE_PROPERTIES_3T = {
    "CSF":          {"T1": 4300.0, "T2": 1500.0, "rho": 1.00, "color": "#0284c7", "marker": "o", "label": "CSF (Ventricles)"},
    "Gray Matter":  {"T1": 1330.0, "T2": 90.0,   "rho": 0.85, "color": "#16a34a", "marker": "s", "label": "Gray Matter (Cortex)"},
    "White Matter": {"T1": 850.0,  "T2": 80.0,   "rho": 0.70, "color": "#ea580c", "marker": "^", "label": "White Matter"},
}


# ---------------------------------------------------------------------------
# Analytical Bloch Equations
# ---------------------------------------------------------------------------
def bloch_spin_echo(te_ms: float | np.ndarray, tr_ms: float, t1: float, t2: float, rho: float) -> np.ndarray:
    """Spin Echo Signal S(TE, TR) = rho * (1 - exp(-TR/T1)) * exp(-TE/T2)"""
    return rho * (1.0 - np.exp(-tr_ms / t1)) * np.exp(-te_ms / t2)


def bloch_inversion_recovery(ti_ms: float | np.ndarray, tr_ms: float, te_ms: float, t1: float, t2: float, rho: float) -> np.ndarray:
    """Inversion Recovery Magnitude Signal S(TI, TR, TE) = rho * |1 - 2*exp(-TI/T1) + exp(-TR/T1)| * exp(-TE/T2)"""
    longitudinal = np.abs(1.0 - 2.0 * np.exp(-ti_ms / t1) + np.exp(-tr_ms / t1))
    return rho * longitudinal * np.exp(-te_ms / t2)


# ---------------------------------------------------------------------------
# SynthSeg & ROI Extraction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SynthSeg & ROI Extraction with Morphological Erosion
# ---------------------------------------------------------------------------
from scipy.ndimage import binary_erosion

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


def extract_tissue_intensities(vol_np: np.ndarray, seg_np: np.ndarray | None = None) -> dict[str, float]:
    """Extract tissue median and mean intensities using SynthSeg ROIs with core erosion."""
    if seg_np is not None:
        # Lateral ventricles + 3rd/4th ventricles
        raw_csf = np.isin(seg_np, [4, 43, 14, 15])
        # Core lateral ventricles only to eliminate edge partial volume effects
        core_ventricles = np.isin(seg_np, [4, 43])
        eroded_csf = binary_erosion(core_ventricles, iterations=1)
        csf_mask = eroded_csf if eroded_csf.sum() > 30 else raw_csf

        gm_mask = binary_erosion(np.isin(seg_np, [3, 42]), iterations=1)
        if gm_mask.sum() < 50:
            gm_mask = np.isin(seg_np, [3, 42])

        wm_mask = binary_erosion(np.isin(seg_np, [2, 41]), iterations=1)
        if wm_mask.sum() < 50:
            wm_mask = np.isin(seg_np, [2, 41])

        if csf_mask.sum() > 20 and gm_mask.sum() > 50 and wm_mask.sum() > 50:
            return {
                "csf": float(np.median(vol_np[csf_mask])),
                "gm": float(np.median(vol_np[gm_mask])),
                "wm": float(np.median(vol_np[wm_mask])),
                "brain_mean": float(np.mean(vol_np[csf_mask | gm_mask | wm_mask])),
                "seg_method": "SynthSeg_Eroded",
            }

    # Fallback to robust percentile clustering
    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_voxels = v[v > bg_thresh]
    return {
        "csf": float(np.percentile(brain_voxels, 15.0)),
        "gm": float(np.percentile(brain_voxels, 50.0)),
        "wm": float(np.percentile(brain_voxels, 85.0)),
        "brain_mean": float(np.mean(brain_voxels)),
        "seg_method": "Percentile_Fallback",
    }


# ---------------------------------------------------------------------------
# Publication-Quality Unified Figure Generator
# ---------------------------------------------------------------------------
def compute_reference_crop_box(slice_2d: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    """Calculate fixed bounding box coordinates from reference slice."""
    bg_thresh = max(0.04, float(np.percentile(slice_2d, 15.0)))
    brain_pts = np.where(slice_2d > bg_thresh)
    if len(brain_pts[0]) > 50:
        y_min = max(0, int(brain_pts[0].min() - pad))
        y_max = min(slice_2d.shape[0], int(brain_pts[0].max() + pad + 1))
        x_min = max(0, int(brain_pts[1].min() - pad))
        x_max = min(slice_2d.shape[1], int(brain_pts[1].max() + pad + 1))
        return y_min, y_max, x_min, x_max
    return 0, slice_2d.shape[0], 0, slice_2d.shape[1]


def plot_unified_physics_sweeps(
    te_data: dict[str, Any],
    tr_data: dict[str, Any],
    ti_data: dict[str, Any],
    out_path: str | Path,
):
    """
    Renders a publication-grade 3-row figure comparing Bloch theoretical physics,
    diffusion model measured response, and individual synthesized anatomical slices.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    fig = plt.figure(figsize=(22.5, 17.0), dpi=300)
    
    # Outer Grid: 3 Rows (A: TE, B: TR, C: TI)
    outer_gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.46,
        left=0.045,
        right=0.985,
        top=0.920,
        bottom=0.045,
    )

    fig.suptitle(
        "Quantitative MRI Physics Fidelity & Continuous Parameter Controllability",
        fontsize=19,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    row_configs = [
        {
            "title": "A. Echo Time (TE) Sweep [0 → 200 ms] — T2 Transverse Relaxation",
            "theory_title": "Theoretical Bloch Decay (3T)",
            "model_title": "Synthesized Response (SynthSeg ROIs)",
            "slices_title": "Synthesized Axial Slices Across TE (0 → 200 ms)",
            "xlabel": "Echo Time (TE) [ms]",
            "xlim": (-5, 205),
            "data": te_data,
            "theory_fn": lambda te_arr: {
                t: bloch_spin_echo(te_arr, tr_ms=4500.0, t1=p["T1"], t2=p["T2"], rho=p["rho"])
                for t, p in TISSUE_PROPERTIES_3T.items()
            },
            "callout": "• CSF decay: slow ($T_2=1500$ ms)\n• WM/GM decay: fast ($T_2=80-90$ ms)\n• Strong $T_2$ contrast at TE $\\geq$ 80 ms",
            "unit": "ms",
            "legend_loc": "upper right",
        },
        {
            "title": "B. Repetition Time (TR) Sweep [800 → 6000 ms] — T1 Longitudinal Recovery",
            "theory_title": "Theoretical Bloch Recovery (3T)",
            "model_title": "Synthesized Response (SynthSeg ROIs)",
            "slices_title": "Synthesized Axial Slices Across TR (800 → 6000 ms)",
            "xlabel": "Repetition Time (TR) [ms]",
            "xlim": (600, 6250),
            "data": tr_data,
            "theory_fn": lambda tr_arr: {
                t: bloch_spin_echo(te_ms=12.0, tr_ms=tr_arr, t1=p["T1"], t2=p["T2"], rho=p["rho"])
                for t, p in TISSUE_PROPERTIES_3T.items()
            },
            "callout": "• Short TR (800 ms): Strong $T_1$w (WM > GM > CSF)\n• Intermediate TR (2200 ms): Contrast transition\n• Long TR (6000 ms): $T_1$ saturation plateau",
            "unit": "ms",
            "legend_loc": "lower right",
        },
        {
            "title": "C. Inversion Time (TI) Sweep [0 → 3000 ms] — FLAIR Inversion Nulling",
            "theory_title": "Theoretical Inversion Recovery (3T)",
            "model_title": "Synthesized Response (SynthSeg ROIs)",
            "slices_title": "Synthesized Axial Slices Across TI (0 → 3000 ms)",
            "xlabel": "Inversion Time (TI) [ms]",
            "xlim": (-50, 3050),
            "data": ti_data,
            "theory_fn": lambda ti_arr: {
                t: bloch_inversion_recovery(ti_arr, tr_ms=4800.0, te_ms=340.0, t1=p["T1"], t2=p["T2"], rho=p["rho"])
                for t, p in TISSUE_PROPERTIES_3T.items()
            },
            "callout": "• Non-monotonic V-shaped inversion recovery\n• CSF zeroes at $TI^* = T_{1,\\mathrm{CSF}} \\ln 2 \\approx 2000$ ms\n• Fluid-attenuated contrast at null point",
            "unit": "ms",
            "legend_loc": "upper right",
            "null_line": 2000,
        },
    ]

    for row_idx, cfg in enumerate(row_configs):
        row_gs = outer_gs[row_idx].subgridspec(
            1, 3,
            width_ratios=[1.0, 1.05, 2.3],
            wspace=0.22,
        )

        # -------------------------------------------------------------------
        # 1. Theoretical Bloch Physics Plot
        # -------------------------------------------------------------------
        ax_theory = fig.add_subplot(row_gs[0, 0])
        x_fine = np.linspace(cfg["xlim"][0] if cfg["xlim"][0] >= 0 else 0, cfg["xlim"][1], 300)
        theory_curves = cfg["theory_fn"](x_fine)

        for tissue, props in TISSUE_PROPERTIES_3T.items():
            ax_theory.plot(x_fine, theory_curves[tissue], color=props["color"], lw=2.4, label=props["label"])

        if "null_line" in cfg:
            ax_theory.axvline(cfg["null_line"], color="#dc2626", linestyle=":", lw=1.8, label="CSF Null ($TI^* \\approx 2200$ ms)")

        ax_theory.set_title(cfg["theory_title"], fontsize=12.5, fontweight="bold", pad=9, color="#1e293b")
        ax_theory.set_xlabel(cfg["xlabel"], fontsize=11, fontweight="bold", color="#334155")
        ax_theory.set_ylabel("Expected Signal (a.u.)", fontsize=11, fontweight="bold", color="#334155")
        ax_theory.set_xlim(cfg["xlim"])
        ax_theory.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax_theory.legend(loc=cfg["legend_loc"], fontsize=8.5, framealpha=0.93, edgecolor="#cbd5e1")
        ax_theory.annotate(
            cfg["callout"],
            xy=(0.04, 0.10 if cfg["legend_loc"] == "upper right" else 0.45), xycoords="axes fraction",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#94a3b8", alpha=0.92),
        )

        # -------------------------------------------------------------------
        # 2. Diffusion Model Measured Response Plot
        # -------------------------------------------------------------------
        ax_model = fig.add_subplot(row_gs[0, 1])
        data_dict = cfg["data"]
        param_vals = data_dict["param_vals"]
        records = data_dict["records"]

        for tissue, props in TISSUE_PROPERTIES_3T.items():
            key = "csf" if "CSF" in tissue else ("gm" if "Gray" in tissue else "wm")
            ax_model.plot(
                param_vals, [r[key] for r in records],
                marker=props["marker"], color=props["color"], lw=2.2, markersize=6.8, label=props["label"],
            )

        if "brain_mean" in records[0]:
            rho_val = data_dict.get("spearman_rho", None)
            rho_str = f" ($\\rho$={rho_val:.2f})" if rho_val is not None else ""
            ax_model.plot(
                param_vals, [r["brain_mean"] for r in records],
                "k--", lw=1.8, label=f"Brain Mean{rho_str}",
            )

        if "null_line" in cfg:
            csf_vals = [r["csf"] for r in records]
            min_ti_idx = int(np.argmin(csf_vals))
            ti_null_emp = param_vals[min_ti_idx]
            ax_model.axvline(ti_null_emp, color="#dc2626", linestyle=":", lw=1.8, label=f"Empirical Null ($TI^* = {int(ti_null_emp)}$ ms)")

        ax_model.set_title(cfg["model_title"], fontsize=12.5, fontweight="bold", pad=9, color="#1e293b")
        ax_model.set_xlabel(cfg["xlabel"], fontsize=11, fontweight="bold", color="#334155")
        ax_model.set_ylabel("Measured Intensity", fontsize=11, fontweight="bold", color="#334155")
        ax_model.set_xlim(cfg["xlim"])
        ax_model.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax_model.legend(loc=cfg["legend_loc"], fontsize=8.5, framealpha=0.93, edgecolor="#cbd5e1")

        # -------------------------------------------------------------------
        # 3. Individual Slices Gallery Subplot Grid
        # -------------------------------------------------------------------
        slices_list = data_dict["slices"]
        n_slices = len(slices_list)
        slices_gs = row_gs[0, 2].subgridspec(1, n_slices, wspace=0.08)

        # Container title for slice gallery
        ax_header = fig.add_subplot(row_gs[0, 2])
        ax_header.set_title(cfg["slices_title"], fontsize=12.5, fontweight="bold", pad=9, color="#1e293b")
        ax_header.axis("off")

        for s_idx, (p_val, slc) in enumerate(slices_list):
            ax_slc = fig.add_subplot(slices_gs[0, s_idx])
            ax_slc.imshow(slc, cmap="gray")
            ax_slc.set_xticks([])
            ax_slc.set_yticks([])

            # Highlight null point in TI
            is_null = False
            if "null_line" in cfg:
                csf_vals = [r["csf"] for r in records]
                if s_idx == int(np.argmin(csf_vals)):
                    is_null = True

            border_color = "#dc2626" if is_null else "#64748b"
            for spine in ax_slc.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.0 if is_null else 0.8)

            tag = f"\n(Null)" if is_null else ""
            txt_color = "#dc2626" if is_null else "#0f172a"
            ax_slc.set_xlabel(
                f"{int(p_val)} {cfg['unit']}{tag}",
                fontsize=9.5, fontweight="bold" if is_null else "normal",
                color=txt_color, labelpad=6,
            )

        # Row Header Banner
        fig.text(0.045, 0.942 - row_idx * 0.322, cfg["title"], fontsize=13.5, fontweight="bold", color="#0f172a")

    fig.savefig(str(out_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved refined publication-grade physics figure to: {out_path}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Unified Publication-Grade Figure for MRI Physics Sweeps")
    p.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    p.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    p.add_argument("--clip-checkpoint", default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt")
    p.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--output-dir", default="output/eval_physics")
    p.add_argument("--seed", type=int, default=6256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-generation", action="store_true", help="Skip diffusion generation if cache pickle exists")
    args_cli = p.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "physics_sweep_cache.pkl"

    logger = setup_logging("eval_physics_sweeps")
    logger.info("Initializing Unified MRI Physics Evaluation Suite...")

    # Define exact parameter sweeps requested by user
    te_sweep_ms = [10, 30, 60, 90, 130, 170, 200]           # 0 -> 200 ms (T2 decay)
    tr_sweep_ms = [800, 1400, 2200, 3200, 4500, 6000]       # 800 -> 6000 ms (T1 recovery)
    ti_sweep_ms = [400, 900, 1500, 1800, 2200, 2600, 3000]  # 0 -> 3000 ms (FLAIR nulling)

    if args_cli.skip_generation and cache_path.exists():
        logger.info(f"Loading cached sweep results from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        te_data = cache["te_data"]
        tr_data = cache["tr_data"]
        ti_data = cache["ti_data"]
    else:
        # Load configs and models
        logger.info("Loading environment and network configs...")
        env_dict = json.load(open(args_cli.env_config))
        model_dict = json.load(open(args_cli.model_config))
        net_dict = json.load(open(args_cli.model_def))
        args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)

        device = torch.device(args_cli.device)
        logger.info(f"Using device: {device}")

        # Autoencoder
        ae_path = env_dict.get("trained_autoencoder_path", "models/autoencoder_v1.pt")
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        ckpt_ae = torch.load(ae_path, map_location=device, weights_only=False)
        autoencoder.load_state_dict(ckpt_ae["state_dict"] if "state_dict" in ckpt_ae else ckpt_ae)
        autoencoder.eval()
        scale_factor = float(ckpt_ae.get("scale_factor", 1.0422))

        # MR-CLIP Conditioned UNet
        base_unet = define_instance(args, "diffusion_unet_def").to(device)
        model = MRCLIPConditionedUNet(unet=base_unet).to(device)
        ckpt_diff = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_diff["model_state_dict"] if "model_state_dict" in ckpt_diff else ckpt_diff)
        model.eval()

        # MR-CLIP Text & Vision Encoder
        clip_model, tokenizer = load_mrclip(
            checkpoint_path=args_cli.clip_checkpoint,
            open_clip_src=args_cli.open_clip_src,
            context_length=args_cli.clip_context_length,
            device=device,
        )
        clip_model.eval()

        target_dim = tuple(model_dict["diffusion_unet_inference"]["dim"])

        # -------------------------------------------------------------------
        # Helper: Generate with strictly locked anatomical trajectory
        # -------------------------------------------------------------------
        from monai.networks.schedulers import RFlowScheduler
        from scripts.utils_infer import ReconModel
        from scripts.utils import dynamic_infer
        from monai.inferers import SlidingWindowInferer
        from monai.utils import set_determinism
        from scripts.run_mrclip_validation_experiments import encode_prompt

        @torch.no_grad()
        def generate_anatomically_locked(
            prompt_str: str,
            anchor_latent: torch.Tensor | None = None,
            split_step: int = 14,
        ) -> tuple[np.ndarray, list[torch.Tensor] | None]:
            set_determinism(seed=args_cli.seed)
            divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
            latent_shape = (
                1,
                args.latent_channels,
                target_dim[0] // divisor,
                target_dim[1] // divisor,
                target_dim[2] // divisor,
            )

            spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
            spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

            noise_scheduler = define_instance(args, "noise_scheduler")
            num_steps = args.diffusion_unet_inference["num_inference_steps"]
            dummy_noise = torch.empty(latent_shape, device=device)
            if isinstance(noise_scheduler, RFlowScheduler):
                noise_scheduler.set_timesteps(
                    num_inference_steps=num_steps,
                    input_img_size_numel=torch.prod(torch.tensor(dummy_noise.shape[2:])),
                )
            else:
                noise_scheduler.set_timesteps(num_inference_steps=num_steps)

            recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
            null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)
            cond_emb = encode_prompt(prompt_str, clip_model, tokenizer, device).to(device)

            all_t = noise_scheduler.timesteps
            all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

            if anchor_latent is None:
                noise = torch.randn(latent_shape, device=device)
                image = noise.clone()
                start_idx = 0
                saved_latents = [image.detach().clone()]
            else:
                image = anchor_latent.clone()
                start_idx = split_step
                saved_latents = None

            with torch.amp.autocast("cuda", enabled=True):
                for idx in range(start_idx, len(all_t)):
                    t = all_t[idx]
                    next_t = all_next_t[idx]

                    if args_cli.cfg_scale == 0.0:
                        model_output = model(
                            x=image, timesteps=torch.tensor([t], device=device),
                            clip_embedding=null_emb, spacing_tensor=spacing_tensor,
                        )
                    elif args_cli.cfg_scale == 1.0:
                        model_output = model(
                            x=image, timesteps=torch.tensor([t], device=device),
                            clip_embedding=cond_emb, spacing_tensor=spacing_tensor,
                        )
                    else:
                        x2 = torch.cat([image, image], dim=0)
                        t2 = torch.tensor([t, t], device=device)
                        emb2 = torch.cat([cond_emb, null_emb], dim=0)
                        sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                        out2 = model(
                            x=x2, timesteps=t2,
                            clip_embedding=emb2, spacing_tensor=sp2,
                        )
                        model_cond, model_uncond = out2.chunk(2)
                        model_output = model_uncond + args_cli.cfg_scale * (model_cond - model_uncond)

                    if isinstance(noise_scheduler, RFlowScheduler):
                        image, _ = noise_scheduler.step(model_output, t, image, next_t)
                    else:
                        image, _ = noise_scheduler.step(model_output, t, image)

                    if saved_latents is not None:
                        saved_latents.append(image.detach().clone())

                inferer = SlidingWindowInferer(
                    roi_size=[80, 80, 80],
                    sw_batch_size=1,
                    progress=False,
                    mode="gaussian",
                    overlap=0.4,
                    sw_device=device,
                    device=device,
                )
                image = image.float()
                synthetic = dynamic_infer(inferer, recon_model, image)
                data = synthetic.squeeze().cpu().detach().numpy()
                a_min, a_max, b_min, b_max = 0, 1000, 0, 1
                data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
                vol_np = np.clip(data, a_min, a_max).astype(np.float32)

            torch.cuda.empty_cache()
            return vol_np, saved_latents

        base_t2_prompt = CANONICAL_MIDI_TEMPLATES["T2W"]
        base_t1_prompt = CANONICAL_MIDI_TEMPLATES["T1W"]
        base_flair_prompt = CANONICAL_MIDI_TEMPLATES["FLAIR"]
        anchor_step = 8

        # -------------------------------------------------------------------
        # 1. TE SWEEP (0 -> 200 ms) on T2W with Locked T2 Anatomy
        # -------------------------------------------------------------------
        logger.info(f"\n--- Running TE Sweep ({te_sweep_ms} ms) with Locked T2 Anatomy ---")
        ref_te_prompt = replace_imaging_param(base_t2_prompt, te=0.095, tr=4.5, ti=None)
        ref_vol_te, ref_latents_te = generate_anatomically_locked(ensure_axial_prompt(ref_te_prompt))
        ref_seg_te = run_synthseg(ref_vol_te, logger=logger) if ref_vol_te is not None else None
        anchor_latent_te = ref_latents_te[anchor_step]

        # Determine reference ventricular slice index & reference crop box for T2
        v_ref_te = ref_vol_te.astype(np.float32)
        bg_th = max(10.0, float(np.percentile(v_ref_te[v_ref_te > 0], 12.0)))
        b_pts = np.where(v_ref_te > bg_th)
        z_mid_te = int(np.percentile(b_pts[2], 2.0) + (np.percentile(b_pts[2], 98.0) - np.percentile(b_pts[2], 2.0)) * 0.54) if len(b_pts[2]) > 500 else int(ref_vol_te.shape[2] * 0.54)
        raw_ref_s_te = np.rot90(ref_vol_te[:, :, z_mid_te].astype(np.float32))
        crop_y0_te, crop_y1_te, crop_x0_te, crop_x1_te = compute_reference_crop_box(raw_ref_s_te)

        def extract_slice_te(vol: np.ndarray) -> np.ndarray:
            raw_s = vol[:, :, z_mid_te].astype(np.float32)
            p1, p99 = float(np.percentile(raw_s, 1.0)), float(np.percentile(raw_s, 99.0))
            norm_s = np.clip((raw_s - p1) / (max(p99 - p1, 1e-4)), 0.0, 1.0)
            ax_s = np.rot90(norm_s)
            return ax_s[crop_y0_te:crop_y1_te, crop_x0_te:crop_x1_te]

        te_records = []
        te_slices = []

        for te_ms in te_sweep_ms:
            te_sec = te_ms / 1000.0
            prompt = ensure_axial_prompt(replace_imaging_param(base_t2_prompt, te=te_sec, tr=4.5, ti=None))
            vol_np, _ = generate_anatomically_locked(prompt, anchor_latent=anchor_latent_te, split_step=anchor_step)
            if vol_np is None:
                continue

            tissues = extract_tissue_intensities(vol_np, seg_np=ref_seg_te)
            ax_s = extract_slice_te(vol_np)

            te_slices.append((te_ms, ax_s))
            te_records.append({"param_ms": te_ms, **tissues})
            logger.info(f"  TE={te_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}")

        te_arr = np.array([r["param_ms"] for r in te_records])
        te_mean = np.array([r["brain_mean"] for r in te_records])
        spearman_te, _ = spearmanr(te_arr, te_mean)
        te_data = {"param_vals": te_arr, "records": te_records, "slices": te_slices, "spearman_rho": float(spearman_te)}

        # -------------------------------------------------------------------
        # 2. TR SWEEP (800 -> 6000 ms) on T1W with Locked T1 Anatomy
        # -------------------------------------------------------------------
        logger.info(f"\n--- Running TR Sweep ({tr_sweep_ms} ms) with Locked T1 Anatomy ---")
        ref_tr_prompt = replace_imaging_param(base_t1_prompt, te=0.012, tr=2.2, ti=None)
        ref_vol_tr, ref_latents_tr = generate_anatomically_locked(ensure_axial_prompt(ref_tr_prompt))
        ref_seg_tr = run_synthseg(ref_vol_tr, logger=logger) if ref_vol_tr is not None else None
        anchor_latent_tr = ref_latents_tr[anchor_step]

        v_ref_tr = ref_vol_tr.astype(np.float32)
        bg_th_tr = max(10.0, float(np.percentile(v_ref_tr[v_ref_tr > 0], 12.0)))
        b_pts_tr = np.where(v_ref_tr > bg_th_tr)
        z_mid_tr = int(np.percentile(b_pts_tr[2], 2.0) + (np.percentile(b_pts_tr[2], 98.0) - np.percentile(b_pts_tr[2], 2.0)) * 0.54) if len(b_pts_tr[2]) > 500 else int(ref_vol_tr.shape[2] * 0.54)
        raw_ref_s_tr = np.rot90(ref_vol_tr[:, :, z_mid_tr].astype(np.float32))
        crop_y0_tr, crop_y1_tr, crop_x0_tr, crop_x1_tr = compute_reference_crop_box(raw_ref_s_tr)

        def extract_slice_tr(vol: np.ndarray) -> np.ndarray:
            raw_s = vol[:, :, z_mid_tr].astype(np.float32)
            p1, p99 = float(np.percentile(raw_s, 1.0)), float(np.percentile(raw_s, 99.0))
            norm_s = np.clip((raw_s - p1) / (max(p99 - p1, 1e-4)), 0.0, 1.0)
            ax_s = np.rot90(norm_s)
            return ax_s[crop_y0_tr:crop_y1_tr, crop_x0_tr:crop_x1_tr]

        tr_records = []
        tr_slices = []

        for tr_ms in tr_sweep_ms:
            tr_sec = tr_ms / 1000.0
            prompt = ensure_axial_prompt(replace_imaging_param(base_t1_prompt, te=0.012, tr=tr_sec, ti=None))
            vol_np, _ = generate_anatomically_locked(prompt, anchor_latent=anchor_latent_tr, split_step=anchor_step)
            if vol_np is None:
                continue

            tissues = extract_tissue_intensities(vol_np, seg_np=ref_seg_tr)
            ax_s = extract_slice_tr(vol_np)

            tr_slices.append((tr_ms, ax_s))
            tr_records.append({"param_ms": tr_ms, **tissues})
            logger.info(f"  TR={tr_ms}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

        tr_arr = np.array([r["param_ms"] for r in tr_records])
        tr_mean = np.array([r["brain_mean"] for r in tr_records])
        spearman_tr, _ = spearmanr(tr_arr, tr_mean)
        tr_data = {"param_vals": tr_arr, "records": tr_records, "slices": tr_slices, "spearman_rho": float(spearman_tr)}

        # -------------------------------------------------------------------
        # 3. TI SWEEP (0 -> 3000 ms) on FLAIR with Locked FLAIR Anatomy
        # -------------------------------------------------------------------
        logger.info(f"\n--- Running TI Sweep ({ti_sweep_ms} ms) with Locked FLAIR Anatomy ---")
        ref_ti_prompt = replace_imaging_param(base_flair_prompt, ti=2.2)
        ref_vol_ti, ref_latents_ti = generate_anatomically_locked(ensure_axial_prompt(ref_ti_prompt))
        ref_seg_ti = run_synthseg(ref_vol_ti, logger=logger) if ref_vol_ti is not None else None
        anchor_latent_ti = ref_latents_ti[anchor_step]

        v_ref_ti = ref_vol_ti.astype(np.float32)
        bg_th_ti = max(10.0, float(np.percentile(v_ref_ti[v_ref_ti > 0], 12.0)))
        b_pts_ti = np.where(v_ref_ti > bg_th_ti)
        z_mid_ti = int(np.percentile(b_pts_ti[2], 2.0) + (np.percentile(b_pts_ti[2], 98.0) - np.percentile(b_pts_ti[2], 2.0)) * 0.54) if len(b_pts_ti[2]) > 500 else int(ref_vol_ti.shape[2] * 0.54)
        raw_ref_s_ti = np.rot90(ref_vol_ti[:, :, z_mid_ti].astype(np.float32))
        crop_y0_ti, crop_y1_ti, crop_x0_ti, crop_x1_ti = compute_reference_crop_box(raw_ref_s_ti)

        def extract_slice_ti(vol: np.ndarray) -> np.ndarray:
            raw_s = vol[:, :, z_mid_ti].astype(np.float32)
            p1, p99 = float(np.percentile(raw_s, 1.0)), float(np.percentile(raw_s, 99.0))
            norm_s = np.clip((raw_s - p1) / (max(p99 - p1, 1e-4)), 0.0, 1.0)
            ax_s = np.rot90(norm_s)
            return ax_s[crop_y0_ti:crop_y1_ti, crop_x0_ti:crop_x1_ti]

        ti_records = []
        ti_slices = []

        for ti_ms in ti_sweep_ms:
            ti_sec = ti_ms / 1000.0
            prompt = ensure_axial_prompt(replace_imaging_param(base_flair_prompt, ti=ti_sec))
            vol_np, _ = generate_anatomically_locked(prompt, anchor_latent=anchor_latent_ti, split_step=anchor_step)
            if vol_np is None:
                continue

            tissues = extract_tissue_intensities(vol_np, seg_np=ref_seg_ti)
            ax_s = extract_slice_ti(vol_np)

            ti_slices.append((ti_ms, ax_s))
            ti_records.append({"param_ms": ti_ms, **tissues})
            logger.info(f"  TI={ti_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}")

        ti_arr = np.array([r["param_ms"] for r in ti_records])
        ti_data = {"param_vals": ti_arr, "records": ti_records, "slices": ti_slices}

        # Cache results
        cache = {"te_data": te_data, "tr_data": tr_data, "ti_data": ti_data}
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"Saved sweep cache to: {cache_path}")

    # Render Unified Figure
    plot_unified_physics_sweeps(
        te_data=te_data,
        tr_data=tr_data,
        ti_data=ti_data,
        out_path=out_dir / "physics_parameter_sweeps_unified.png",
    )


if __name__ == "__main__":
    main()
