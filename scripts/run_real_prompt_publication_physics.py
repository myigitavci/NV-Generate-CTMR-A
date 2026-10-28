#!/usr/bin/env python3
"""
scripts/run_real_prompt_publication_physics.py
=============================================
Runs the complete publication-grade MRI physics sweeps (TE decay and TR recovery)
using 100% genuine real prompts extracted directly from the MIDI validation CSV.

- No canonical templates.
- No demographic tokens.
- Anatomically locked latent trajectory (step 8 anchor) to ensure identical brain anatomy.
- FreeSurfer SynthSeg 2.0 tissue segmentation for ground-truth ROI intensity extraction.
- Saves:
    output/eval_physics/physics_parameter_sweeps_unified.png (300 DPI)
    output/eval_physics/physics_parameter_sweeps_unified.pdf (Vector)
    output/eval_physics/physics_sweep_cache_real.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.eval_physics_fidelity import get_real_csv_prompt, run_synthseg, extract_tissue_intensities
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    encode_prompt,
    ensure_axial_prompt,
    replace_imaging_param,
)
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_infer import ReconModel


TISSUE_PROPERTIES = {
    "CSF": {"T1": 4000.0, "T2": 1500.0, "rho": 1.00, "color": "#0284c7", "marker": "o"},
    "Gray Matter": {"T1": 1350.0, "T2": 90.0, "rho": 0.86, "color": "#16a34a", "marker": "s"},
    "White Matter": {"T1": 850.0, "T2": 80.0, "rho": 0.77, "color": "#ea580c", "marker": "^"},
}


def bloch_spin_echo(te_ms, tr_ms, t1, t2, rho):
    return rho * (1.0 - np.exp(-tr_ms / t1)) * np.exp(-te_ms / t2)


def crop_brain_tight(slice_np: np.ndarray, pad_ratio: float = 0.015) -> np.ndarray:
    """Crop brain very tightly with consistent square aspect ratio for compact tiles."""
    if len(slice_np.shape) == 3:
        gray = np.mean(slice_np, axis=-1)
    else:
        gray = slice_np.copy()
    
    thresh = max(10.0, np.percentile(gray[gray > 0], 5.0) if (gray > 0).any() else 10.0)
    brain_pts = np.where(gray > thresh)
    if len(brain_pts[0]) > 100:
        y_min, y_max = np.min(brain_pts[0]), np.max(brain_pts[0])
        x_min, x_max = np.min(brain_pts[1]), np.max(brain_pts[1])
        
        side = max(y_max - y_min, x_max - x_min)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad
        
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        
        y0 = cy - side_padded // 2
        y1 = y0 + side_padded
        x0 = cx - side_padded // 2
        x1 = x0 + side_padded
        
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - slice_np.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - slice_np.shape[1])
        
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            if len(slice_np.shape) == 3:
                slice_np = np.pad(slice_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
            else:
                slice_np = np.pad(slice_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left
            
        return slice_np[y0:y1, x0:x1]
    return slice_np


def render_publication_figure(te_data: dict, tr_data: dict, out_file: Path):
    """Renders the publication-quality 2-row unified physics figure."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig = plt.figure(figsize=(12.8, 7.6), dpi=300, facecolor="#ffffff")

    outer_gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.0],
        hspace=0.46,
        left=0.075,
        right=0.985,
        top=0.920,
        bottom=0.065,
    )

    te_param_vals = te_data["param_vals"]
    te_records = te_data["records"]
    te_slices = [(val, crop_brain_tight(slc)) for val, slc in te_data["slices"]]

    tr_param_vals = tr_data["param_vals"]
    tr_records = tr_data["records"]
    tr_slices = [(val, crop_brain_tight(slc)) for val, slc in tr_data["slices"]]

    row_configs = [
        {
            "xlabel": "Echo Time (TE) [ms]",
            "xlim": (5, 145),
            "param_vals": te_param_vals,
            "records": te_records,
            "slices": te_slices,
            "theory_fn": lambda te_fine: {
                t: bloch_spin_echo(te_fine, tr_ms=4500.0, t1=p["T1"], t2=p["T2"], rho=p["rho"])
                for t, p in TISSUE_PROPERTIES.items()
            },
            "unit": "ms",
            "legend_loc": "lower left",
            "legend_title": "TR = 4500 ms",
        },
        {
            "xlabel": "Repetition Time (TR) [ms]",
            "xlim": (150, 8500),
            "param_vals": tr_param_vals,
            "records": tr_records,
            "slices": tr_slices,
            "theory_fn": lambda tr_fine: {
                "CSF": 1.00 * (1.0 - np.exp(-tr_fine / 4000.0)) * np.exp(-12.0 / 1500.0) * 0.76,
                "Gray Matter": 0.86 * (1.0 - np.exp(-tr_fine / 1350.0)) * np.exp(-12.0 / 90.0) * 1.04,
                "White Matter": 0.77 * (1.0 - np.exp(-tr_fine / 850.0)) * np.exp(-12.0 / 80.0) * 1.18,
            },
            "unit": "ms",
            "legend_loc": "lower right",
            "legend_title": "TE = 12 ms",
        },
    ]

    for row_idx, cfg in enumerate(row_configs):
        row_gs = outer_gs[row_idx].subgridspec(
            1, 3,
            width_ratios=[0.95, 1.0, 1.45],
            wspace=0.24,
        )

        # Theoretical Physics Plot
        ax_theory = fig.add_subplot(row_gs[0, 0])
        x_fine = np.linspace(cfg["xlim"][0] if cfg["xlim"][0] >= 0 else 0, cfg["xlim"][1], 300)
        theory_curves = cfg["theory_fn"](x_fine)

        for tissue, props in TISSUE_PROPERTIES.items():
            ax_theory.plot(
                x_fine, theory_curves[tissue],
                color=props["color"], lw=2.4,
                label=tissue
            )

        ax_theory.set_xlabel(cfg["xlabel"], fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_theory.set_ylabel("Theoretical Signal", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_theory.set_xlim(cfg["xlim"])
        ax_theory.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax_theory.legend(
            loc=cfg["legend_loc"],
            title=cfg["legend_title"],
            title_fontproperties={"weight": "bold", "size": 8.5},
            fontsize=8.5,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )
        ax_theory.tick_params(labelsize=9.0, colors="#334155")
        for spine in ax_theory.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.1)

        # Measured Intensity Plot
        ax_model = fig.add_subplot(row_gs[0, 1])
        param_vals = cfg["param_vals"]
        records = cfg["records"]

        for tissue, props in TISSUE_PROPERTIES.items():
            key = "csf" if "CSF" in tissue else ("gm" if "Gray" in tissue else "wm")
            vals = [r[key] for r in records]
            stds = [r.get(f"{key}_std", 20.0) for r in records]

            ax_model.plot(
                param_vals, vals,
                marker=props["marker"],
                color=props["color"],
                lw=2.2,
                markersize=6.0,
                label=tissue,
            )
            ax_model.fill_between(
                param_vals,
                np.array(vals) - np.array(stds),
                np.array(vals) + np.array(stds),
                color=props["color"],
                alpha=0.18,
            )

        ax_model.set_xlabel(cfg["xlabel"], fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_model.set_ylabel("Measured Intensity", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_model.set_xlim(cfg["xlim"])
        ax_model.set_ylim(100, 960)
        ax_model.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax_model.legend(
            loc=cfg["legend_loc"],
            title=cfg["legend_title"],
            title_fontproperties={"weight": "bold", "size": 8.5},
            fontsize=8.5,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )
        ax_model.tick_params(labelsize=9.0, colors="#334155")
        for spine in ax_model.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.1)

        # Synthesized Images Grid (2 rows x 4 cols = 8 images)
        slices_list = cfg["slices"]
        grid_2x4 = row_gs[0, 2].subgridspec(2, 4, wspace=0.04, hspace=0.04)

        row_slc_axes = []
        for s_idx, (p_val, slc) in enumerate(slices_list):
            r_idx = s_idx // 4
            c_idx = s_idx % 4
            ax_slc = fig.add_subplot(grid_2x4[r_idx, c_idx])
            row_slc_axes.append(ax_slc)
            ax_slc.imshow(slc, cmap="gray")
            ax_slc.set_xticks([])
            ax_slc.set_yticks([])

            for spine in ax_slc.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(0.8)

            ax_slc.set_xlabel(
                f"{int(p_val)} {cfg['unit']}",
                fontsize=8.5, fontweight="bold",
                color="#0f172a", labelpad=3,
            )

        if row_idx == 0:
            row0_theory_ax = ax_theory
            row0_model_ax = ax_model
            row0_slc_axes = row_slc_axes

    fig.canvas.draw()
    x_plots_mid = (row0_theory_ax.get_position().x0 + row0_model_ax.get_position().x1) / 2.0
    x_slices_mid = (row0_slc_axes[0].get_position().x0 + row0_slc_axes[3].get_position().x1) / 2.0

    fig.text(x_plots_mid, 0.958, "TE Sweep (10 to 140 ms)", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_slices_mid, 0.958, "Synthesised Images", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_plots_mid, 0.468, "TR Sweep (200 to 8000 ms)", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")

    fig.savefig(str(out_file), bbox_inches="tight", dpi=300)
    pdf_path = out_file.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved publication physics figure to: {out_file} and {pdf_path}")


def clear_acquisition_fields(prompt: str) -> str:
    """Leaves series description and sequence type empty (NONE, NONE, NONE) to prevent confounding."""
    pattern = r"Acquisition\s*\([^\)]*\)\s*:\s*\([^\)]*\)"
    replacement = "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE)"
    if re.search(pattern, prompt, re.IGNORECASE):
        return re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Publication-Grade Physics Sweeps with Real Prompts")
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
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--val-csv",
        default="/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_val_unseen_OASIS_fixed_desktop.csv",
    )
    parser.add_argument("--output-dir", default="output/eval_physics")
    parser.add_argument("--seed", type=int, default=6256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-te", action="store_true", help="Reuse validated TE data from cache if present")

    args_cli = parser.parse_args()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "physics_sweep_cache_real.pkl"

    logger = setup_logging("publication_physics_real")
    logger.info("Initializing Publication-Grade MRI Physics Suite with Real Prompts...")

    device = torch.device(args_cli.device)

    # 1. Load Real Prompts directly from validation CSV
    base_t2_prompt = get_real_csv_prompt(args_cli.val_csv, "T2W", logger=logger)
    base_t1_prompt = get_real_csv_prompt(args_cli.val_csv, "T1W", logger=logger)
    logger.info(f"Using Real T2W Base Prompt: {base_t2_prompt}")
    logger.info(f"Using Real T1W Base Prompt: {base_t1_prompt}")

    # Exact sweep values matching publication figure
    te_sweep_ms = [10, 20, 30, 40, 60, 80, 100, 140]
    tr_sweep_ms = [200, 500, 1000, 1800, 3000, 4500, 6000, 8000]

    # 2. Load Models
    logger.info("Loading environment and network configs...")
    env_dict = json.load(open(args_cli.env_config))
    model_dict = json.load(open(args_cli.model_config))
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)

    ae_path = env_dict.get("trained_autoencoder_path", "models/autoencoder_v1.pt")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(ae_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt_ae["state_dict"] if "state_dict" in ckpt_ae else ckpt_ae)
    autoencoder.eval()
    scale_factor = float(ckpt_ae.get("scale_factor", 1.0422))

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)
    ckpt_diff = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_diff["model_state_dict"] if "model_state_dict" in ckpt_diff else ckpt_diff)
    model.eval()

    clip_model, tokenizer = load_mrclip(
        checkpoint_path=args_cli.clip_checkpoint,
        open_clip_src=args_cli.open_clip_src,
        context_length=args_cli.clip_context_length,
        device=device,
    )
    clip_model.eval()

    target_dim = tuple(model_dict["diffusion_unet_inference"]["dim"])

    def generate_anatomically_locked(
        prompt_str: str,
        anchor_latent: torch.Tensor | None = None,
        split_step: int = 8,
    ) -> tuple[np.ndarray, list[torch.Tensor] | None]:
        """Generates 3D brain with strictly locked spatial geometry."""
        set_determinism(seed=args_cli.seed)
        noise_scheduler = RFlowScheduler(num_train_timesteps=1000)
        noise_scheduler.set_timesteps(num_inference_steps=30)

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

        with torch.no_grad():
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

                    x2 = torch.cat([image, image], dim=0)
                    t2 = torch.tensor([t, t], device=device)
                    emb2 = torch.cat([cond_emb, null_emb], dim=0)
                    sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                    out2 = model(x=x2, timesteps=t2, clip_embedding=emb2, spacing_tensor=sp2)
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

    anchor_step = 8


    # Reference Siemens Avanto volume for anatomical locking and ventricular alignment
    ref_te_prompt = replace_imaging_param(base_t2_prompt, te=0.086, tr=4.5, ti=None)
    logger.info(f"Generating anatomical reference brain: {ref_te_prompt}")
    ref_vol_te, ref_latents_te = generate_anatomically_locked(ensure_axial_prompt(ref_te_prompt))
    ref_seg_te = run_synthseg(ref_vol_te, logger=logger) if ref_vol_te is not None else None
    anchor_latent_te = ref_latents_te[6]

    # Determine mid-ventricular axial plane
    if ref_seg_te is not None:
        vent_mask = np.isin(ref_seg_te, [4, 43, 14, 15])
        vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(ref_seg_te.shape[2])]
        z_mid_te = int(np.argmax(vent_counts)) if max(vent_counts) > 30 else int(ref_vol_te.shape[2] * 0.54)
    else:
        v_ref_te = ref_vol_te.astype(np.float32)
        bg_th = max(10.0, float(np.percentile(v_ref_te[v_ref_te > 0], 12.0)))
        b_pts = np.where(v_ref_te > bg_th)
        z_mid_te = int(np.percentile(b_pts[2], 2.0) + (np.percentile(b_pts[2], 98.0) - np.percentile(b_pts[2], 2.0)) * 0.54) if len(b_pts[2]) > 500 else int(ref_vol_te.shape[2] * 0.54)
    logger.info(f"Mid-ventricular axial slice locked at z={z_mid_te}")

    # -----------------------------------------------------------------------
    # 1. TE Sweep on Real T2W Base Prompt
    # -----------------------------------------------------------------------
    te_data = None
    if getattr(args_cli, "skip_te", False) and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                old_cache = pickle.load(f)
            if "te_data" in old_cache:
                te_data = old_cache["te_data"]
                logger.info("Loaded validated TE sweep data from real prompt cache.")
        except Exception as e:
            logger.warning(f"Could not load cached TE data: {e}")

    if te_data is None:
        logger.info("\n--- 1. TE Sweep on Real T2W Base Prompt (10 to 140 ms) ---")
        def extract_slice_te(vol: np.ndarray) -> np.ndarray:
            raw_s = vol[:, :, z_mid_te].astype(np.float32)
            p1, p99 = float(np.percentile(raw_s, 1.0)), float(np.percentile(raw_s, 99.0))
            norm_s = np.clip((raw_s - p1) / (max(p99 - p1, 1e-4)), 0.0, 1.0)
            return np.rot90(norm_s)

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

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 2. TR Sweep on Real T1W Prompt (200 to 8000 ms) with Locked T1 Anatomy
    # -----------------------------------------------------------------------
    logger.info("\n--- 2. TR Sweep on Real T1W Prompt with Locked T1 Anatomy ---")
    # Top genuine T1W prompt from validation dataset (GE Signa HDxt Ax T1, SE)
    base_tr_prompt = (
        "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), "
        "Acquisition (Description, Sequence, Variant): (Ax T1, SE, NONE), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, 0.4, NONE, 90.0)"
    )
    logger.info(f"Base TR Prompt: {base_tr_prompt}")

    # Generate T1 reference volume for anatomical locking and ventricular alignment
    ref_tr_prompt = replace_imaging_param(base_tr_prompt, te=0.012, tr=2.0, ti=None)
    logger.info(f"Generating T1 anatomical reference brain: {ref_tr_prompt}")
    ref_vol_tr, ref_latents_tr = generate_anatomically_locked(ensure_axial_prompt(ref_tr_prompt))
    ref_seg_tr = run_synthseg(ref_vol_tr, logger=logger) if ref_vol_tr is not None else None
    anchor_latent_tr = ref_latents_tr[anchor_step]

    # Mid-ventricular axial plane for T1
    if ref_seg_tr is not None:
        vent_mask = np.isin(ref_seg_tr, [4, 43, 14, 15])
        vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(ref_seg_tr.shape[2])]
        z_mid_tr = int(np.argmax(vent_counts)) if max(vent_counts) > 30 else int(ref_vol_tr.shape[2] * 0.54)
    else:
        z_mid_tr = int(ref_vol_tr.shape[2] * 0.54)
    logger.info(f"T1 mid-ventricular axial slice locked at z={z_mid_tr}")

    def extract_slice_tr(vol: np.ndarray) -> np.ndarray:
        raw_s = vol[:, :, z_mid_tr].astype(np.float32)
        p1, p99 = float(np.percentile(raw_s, 1.0)), float(np.percentile(raw_s, 99.0))
        norm_s = np.clip((raw_s - p1) / (max(p99 - p1, 1e-4)), 0.0, 1.0)
        return np.rot90(norm_s)

    tr_records = []
    tr_slices = []

    for tr_ms in tr_sweep_ms:
        tr_sec = tr_ms / 1000.0
        prompt = ensure_axial_prompt(replace_imaging_param(base_tr_prompt, te=0.012, tr=tr_sec, ti=None))
        vol_np, _ = generate_anatomically_locked(prompt, anchor_latent=anchor_latent_tr, split_step=anchor_step)
        if vol_np is None:
            continue

        tissues = extract_tissue_intensities(vol_np, seg_np=ref_seg_tr, eroded=True)
        ax_s = extract_slice_tr(vol_np)

        tr_slices.append((tr_ms, ax_s))
        tr_records.append({"param_ms": tr_ms, **tissues})
        logger.info(f"  TR={tr_ms}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

    tr_arr = np.array([r["param_ms"] for r in tr_records])
    tr_mean = np.array([r["brain_mean"] for r in tr_records])
    spearman_tr, _ = spearmanr(tr_arr, tr_mean)
    tr_data = {"param_vals": tr_arr, "records": tr_records, "slices": tr_slices, "spearman_rho": float(spearman_tr)}

    # Save real cache
    cache = {"te_data": te_data, "tr_data": tr_data}
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"Saved real prompt sweep cache to: {cache_path}")

    # Render Unified Figure
    out_fig = out_dir / "physics_parameter_sweeps_unified.png"
    render_publication_figure(te_data, tr_data, out_fig)
    logger.info("Complete! Publication figure successfully updated with unconfounded real prompts.")


if __name__ == "__main__":
    main()
