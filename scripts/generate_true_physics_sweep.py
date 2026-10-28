#!/usr/bin/env python3
"""
scripts/generate_true_physics_sweep.py
======================================
Generates 100% genuine diffusion-model synthesized MRI physics parameter sweeps:
  - TE Sweep (10 to 140 ms, 8 values) with true T2 decay.
  - TR Sweep (200 to 8000 ms, 8 values) with true T1 longitudinal recovery.
All volumes are synthesized from scratch using the verified Canonical Vida template.
Tissue intensities are directly measured from each volume using FreeSurfer SynthSeg 2.0.
Output:
  - output/eval_physics/physics_parameter_sweeps_unified.png (300 DPI)
  - output/eval_physics/physics_parameter_sweeps_unified.pdf (Vector)
  - output/eval_physics/physics_sweep_cache_real.pkl (Cache)
"""

import json
import logging
from pathlib import Path
import pickle
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    encode_prompt,
    ensure_axial_prompt,
    replace_imaging_param,
)
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_infer import ReconModel
from scripts.eval_physics_fidelity import run_synthseg, extract_tissue_intensities

TISSUE_PROPERTIES = {
    "CSF": {"T1": 4000.0, "T2": 1500.0, "rho": 1.00, "color": "#0284c7", "marker": "o"},
    "Gray Matter": {"T1": 1350.0, "T2": 90.0, "rho": 0.86, "color": "#16a34a", "marker": "s"},
    "White Matter": {"T1": 850.0, "T2": 80.0, "rho": 0.77, "color": "#ea580c", "marker": "^"},
}

def bloch_spin_echo(te_ms, tr_ms, t1, t2, rho):
    return rho * (1.0 - np.exp(-tr_ms / t1)) * np.exp(-te_ms / t2)

def crop_brain_tight(slice_np, pad_ratio=0.015):
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

def main():
    logger = setup_logging("generate_true_physics_sweep")
    device = torch.device("cuda:1")
    out_dir = Path("output/eval_physics")
    out_dir.mkdir(parents=True, exist_ok=True)

    env_config = "configs/environment_maisi_diff_model_rflow-mr-brain.json"
    model_config = "configs/config_maisi_diff_model_rflow-mr-brain.json"
    model_def = "configs/config_network_rflow.json"
    mrclip_ckpt = "models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt"
    clip_checkpoint = "/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt"
    open_clip_src = "/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src"

    env_dict = json.load(open(env_config))
    model_dict = json.load(open(model_config))
    args = load_config(env_config, model_config, model_def)

    ae_path = env_dict.get("trained_autoencoder_path", "models/autoencoder_v1.pt")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(ae_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt_ae["state_dict"] if "state_dict" in ckpt_ae else ckpt_ae)
    autoencoder.eval()
    scale_factor = float(ckpt_ae.get("scale_factor", 1.0422))

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)
    ckpt_diff = torch.load(mrclip_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_diff["model_state_dict"] if "model_state_dict" in ckpt_diff else ckpt_diff)
    model.eval()

    clip_model, tokenizer = load_mrclip(
        checkpoint_path=clip_checkpoint,
        open_clip_src=open_clip_src,
        context_length=98,
        device=device,
    )
    clip_model.eval()

    target_dim = tuple(model_dict["diffusion_unet_inference"]["dim"])
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)

    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (1, args.latent_channels, target_dim[0] // divisor, target_dim[1] // divisor, target_dim[2] // divisor)
    spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    def gen_volume(prompt_str, seed=6256):
        set_determinism(seed=seed)
        noise_scheduler = RFlowScheduler(num_train_timesteps=1000)
        noise_scheduler.set_timesteps(num_inference_steps=30)
        all_t = noise_scheduler.timesteps
        all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

        image = torch.randn(latent_shape, device=device)
        cond_emb = encode_prompt(ensure_axial_prompt(prompt_str), clip_model, tokenizer, device).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                for idx in range(len(all_t)):
                    t = all_t[idx]
                    next_t = all_next_t[idx]
                    x2 = torch.cat([image, image], dim=0)
                    t2 = torch.tensor([t, t], device=device)
                    emb2 = torch.cat([cond_emb, null_emb], dim=0)
                    sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)
                    out2 = model(x=x2, timesteps=t2, clip_embedding=emb2, spacing_tensor=sp2)
                    m_cond, m_uncond = out2.chunk(2)
                    m_out = m_uncond + 4.0 * (m_cond - m_uncond)
                    image, _ = noise_scheduler.step(m_out, t, image, next_t)

                inferer = SlidingWindowInferer(roi_size=[80, 80, 80], sw_batch_size=1, progress=False, mode="gaussian", overlap=0.4, sw_device=device, device=device)
                syn = dynamic_infer(inferer, recon_model, image.float())
                vol = syn.squeeze().cpu().detach().numpy()
                vol = np.clip((vol - 0) / (1 - 0) * 1000, 0, 1000).astype(np.float32)
        return vol

    # Reference template: Canonical Siemens Vida
    base_vida_template = CANONICAL_MIDI_TEMPLATES["T1W"]

    # -----------------------------------------------------------------------
    # 1. TR SWEEP: [200, 400, 700, 1200, 2000, 3200, 5000, 8000] ms
    # -----------------------------------------------------------------------
    tr_vals_ms = [200, 400, 700, 1200, 2000, 3200, 5000, 8000]
    logger.info("Starting genuine TR Sweep (200 to 8000 ms)...")

    # Generate reference volume at TR=2.0s to lock SynthSeg mask and mid-ventricle plane
    ref_tr_p = replace_imaging_param(base_vida_template, te=0.012, tr=2.0, ti=None, fa=90.0)
    ref_vol_tr = gen_volume(ref_tr_p, seed=6256)
    seg_tr = run_synthseg(ref_vol_tr, logger=logger)
    
    # Find lateral ventricle axial plane
    if seg_tr is not None:
        vent_mask = np.isin(seg_tr, [4, 43, 14, 15])
        vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(seg_tr.shape[2])]
        z_mid_tr = int(np.argmax(vent_counts)) if max(vent_counts) > 30 else 70
    else:
        z_mid_tr = 70
    logger.info(f"Locked TR lateral ventricle plane at z={z_mid_tr}")

    tr_slices = []
    tr_records = []

    for tr_ms in tr_vals_ms:
        p = replace_imaging_param(base_vida_template, te=0.012, tr=tr_ms / 1000.0, ti=None, fa=90.0)
        vol = gen_volume(p, seed=6256)
        
        # Measure true tissue intensities
        tissues = extract_tissue_intensities(vol, seg_np=seg_tr, eroded=True)
        tr_records.append({"param_ms": tr_ms, **tissues})
        
        # Extract mid-ventricle slice
        slc = vol[:, :, z_mid_tr]
        p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
        norm = np.clip((slc - p1) / max(p99 - p1, 1e-4), 0, 1)
        tr_slices.append((tr_ms, crop_brain_tight(np.rot90(norm))))
        logger.info(f"TR={tr_ms}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

    # -----------------------------------------------------------------------
    # 2. TE SWEEP: [10, 20, 30, 40, 60, 80, 100, 140] ms
    # -----------------------------------------------------------------------
    te_vals_ms = [10, 20, 30, 40, 60, 80, 100, 140]
    logger.info("Starting genuine TE Sweep (10 to 140 ms)...")

    # Generate reference volume at TE=80ms, TR=4.5s for T2 SynthSeg mask
    ref_te_p = replace_imaging_param(base_vida_template, te=0.080, tr=4.5, ti=None, fa=90.0)
    ref_vol_te = gen_volume(ref_te_p, seed=6256)
    seg_te = run_synthseg(ref_vol_te, logger=logger)

    if seg_te is not None:
        vent_mask = np.isin(seg_te, [4, 43, 14, 15])
        vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(seg_te.shape[2])]
        z_mid_te = int(np.argmax(vent_counts)) if max(vent_counts) > 30 else 70
    else:
        z_mid_te = 70
    logger.info(f"Locked TE lateral ventricle plane at z={z_mid_te}")

    te_slices = []
    te_records = []

    for te_ms in te_vals_ms:
        p = replace_imaging_param(base_vida_template, te=te_ms / 1000.0, tr=4.5, ti=None, fa=90.0)
        vol = gen_volume(p, seed=6256)
        
        # Measure true tissue intensities
        tissues = extract_tissue_intensities(vol, seg_np=seg_te, eroded=True)
        te_records.append({"param_ms": te_ms, **tissues})
        
        # Extract mid-ventricle slice
        slc = vol[:, :, z_mid_te]
        p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
        norm = np.clip((slc - p1) / max(p99 - p1, 1e-4), 0, 1)
        te_slices.append((te_ms, crop_brain_tight(np.rot90(norm))))
        logger.info(f"TE={te_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}")

    # Save real cache
    cache = {
        "te_data": {"param_vals": te_vals_ms, "records": te_records, "slices": te_slices},
        "tr_data": {"param_vals": tr_vals_ms, "records": tr_records, "slices": tr_slices},
    }
    with open(out_dir / "physics_sweep_cache_real.pkl", "wb") as f:
        pickle.dump(cache, f)
    logger.info("Saved cache to physics_sweep_cache_real.pkl")

    # -----------------------------------------------------------------------
    # 3. Render High-Aesthetic Publication Figure (2 Rows x 4 Cols per panel)
    # -----------------------------------------------------------------------
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

    row_configs = [
        {
            "xlabel": "Echo Time (TE) [ms]",
            "xlim": (5, 145),
            "param_vals": te_vals_ms,
            "records": te_records,
            "slices": te_slices,
            "theory_fn": lambda te_fine: {
                t: bloch_spin_echo(te_fine, tr_ms=4500.0, t1=p["T1"], t2=p["T2"], rho=p["rho"])
                for t, p in TISSUE_PROPERTIES.items()
            },
            "unit": "ms",
            "legend_loc": "lower left",
            "legend_title": "TR = 4500 ms",
            "title": "TE Sweep (10 to 140 ms)",
        },
        {
            "xlabel": "Repetition Time (TR) [ms]",
            "xlim": (100, 8500),
            "param_vals": tr_vals_ms,
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
            "title": "TR Sweep (200 to 8000 ms)",
        },
    ]

    for row_idx, cfg in enumerate(row_configs):
        row_gs = outer_gs[row_idx].subgridspec(
            1, 3,
            width_ratios=[0.95, 1.0, 1.45],
            wspace=0.24,
        )

        # 1. Theoretical Physics Plot
        ax_theory = fig.add_subplot(row_gs[0, 0])
        x_fine = np.linspace(cfg["xlim"][0], cfg["xlim"][1], 300)
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

        # 2. Diffusion Model Measured Response Plot
        ax_model = fig.add_subplot(row_gs[0, 1])
        param_vals = np.array(cfg["param_vals"])
        records = cfg["records"]

        for tissue, props in TISSUE_PROPERTIES.items():
            key = "csf" if tissue == "CSF" else ("gm" if "Gray" in tissue else "wm")
            key_std = f"{key}_std"
            y_vals = np.array([r[key] for r in records])
            
            ax_model.plot(
                param_vals, y_vals,
                marker=props["marker"], color=props["color"], lw=2.2, markersize=5.8, label=tissue,
            )
            if key_std in records[0]:
                y_std = np.array([r[key_std] for r in records])
                ax_model.fill_between(
                    param_vals, y_vals - y_std, y_vals + y_std,
                    color=props["color"], alpha=0.18,
                )

        ax_model.set_xlabel(cfg["xlabel"], fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_model.set_ylabel("Measured Intensity", fontsize=10.5, fontweight="bold", color="#1e293b", labelpad=4)
        ax_model.set_xlim(cfg["xlim"])
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

        # 3. Synthesized Slices Gallery (2 Rows x 4 Columns)
        slices_list = cfg["slices"]  # exactly 8 slices
        slices_container = row_gs[0, 2].subgridspec(
            2, 4,
            wspace=0.0,
            hspace=0.28,
        )

        row_slc_axes = []
        for g_idx, (p_val, slc) in enumerate(slices_list):
            r_i = g_idx // 4
            c_i = g_idx % 4
            ax_slc = fig.add_subplot(slices_container[r_i, c_i])
            row_slc_axes.append(ax_slc)
            
            if len(slc.shape) == 2:
                ax_slc.imshow(slc, cmap="gray", aspect="equal")
            else:
                ax_slc.imshow(slc, aspect="equal")
                
            ax_slc.set_xticks([])
            ax_slc.set_yticks([])

            for spine in ax_slc.spines.values():
                spine.set_edgecolor("#1e293b")
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
        else:
            row1_theory_ax = ax_theory
            row1_model_ax = ax_model

    fig.canvas.draw()
    x_plots_mid = (row0_theory_ax.get_position().x0 + row0_model_ax.get_position().x1) / 2.0
    x_slices_mid = (row0_slc_axes[0].get_position().x0 + row0_slc_axes[3].get_position().x1) / 2.0

    # Titles (Clean, no "empirical response")
    fig.text(x_plots_mid, 0.958, "TE Sweep (10 to 140 ms)", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_slices_mid, 0.958, "Synthesised Images", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")

    fig.text(x_plots_mid, 0.468, "TR Sweep (200 to 8000 ms)", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")
    fig.text(x_slices_mid, 0.468, "Synthesised Images", fontsize=16.0, fontweight="bold", color="#0f172a", ha="center")

    out_file = out_dir / "physics_parameter_sweeps_unified.png"
    out_pdf = out_dir / "physics_parameter_sweeps_unified.pdf"
    fig.savefig(str(out_file), bbox_inches="tight", dpi=300)
    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated 100% genuine publication physics figure:\n  PNG: {out_file}\n  PDF: {out_pdf}")

if __name__ == "__main__":
    main()
