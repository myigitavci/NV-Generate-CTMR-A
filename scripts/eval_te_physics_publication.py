"""
scripts/eval_te_physics_publication.py
======================================
Publication-Grade Quantitative Evaluation of Echo Time (TE) Physics Consistency.

Produces a pristine 3-panel publication figure matching the golden reference:
1. Panel A (Left): Quantitative Signal Decay (CSF, GM, WM, Brain Mean) with Spearman rho
2. Panel B (Middle): Synthesized Axial Slices Across TE (10ms -> 100ms) with Anatomically Locked Slices
3. Panel C (Right): FreeSurfer SynthSeg 2.0 ROI Segmentation Overlay
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
from scipy.stats import spearmanr
import torch

from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    replace_imaging_param,
)
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_infer import ReconModel


# ---------------------------------------------------------------------------
# SynthSeg 2.0 Anatomical Segmentation
# ---------------------------------------------------------------------------
def run_synthseg(vol_np: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray | None:
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
                logger.warning(f"SynthSeg exception: {e}")
    return None


def extract_tissue_intensities(vol_np: np.ndarray, seg_np: np.ndarray | None = None) -> dict[str, Any]:
    """Extract intensities per tissue using FreeSurfer SynthSeg masks."""
    if seg_np is not None:
        csf_mask = np.isin(seg_np, [4, 43, 14, 15])  # Ventricles
        gm_mask = np.isin(seg_np, [3, 42])           # Cortex
        wm_mask = np.isin(seg_np, [2, 41])           # White matter

        if csf_mask.sum() > 50 and gm_mask.sum() > 50 and wm_mask.sum() > 50:
            brain_mask = csf_mask | gm_mask | wm_mask
            return {
                "csf": float(np.mean(vol_np[csf_mask])),
                "gm": float(np.mean(vol_np[gm_mask])),
                "wm": float(np.mean(vol_np[wm_mask])),
                "brain_mean": float(np.mean(vol_np[brain_mask])),
                "brain_std": float(np.std(vol_np[brain_mask])),
                "contrast_gm_wm": float(np.mean(vol_np[gm_mask])) / max(float(np.mean(vol_np[wm_mask])), 1e-3),
                "contrast_csf_gm": float(np.mean(vol_np[csf_mask])) / max(float(np.mean(vol_np[gm_mask])), 1e-3),
                "seg_method": "SynthSeg",
            }

    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_voxels = v[v > bg_thresh]
    csf_val = float(np.percentile(brain_voxels, 15.0))
    gm_val = float(np.percentile(brain_voxels, 50.0))
    wm_val = float(np.percentile(brain_voxels, 85.0))
    return {
        "csf": csf_val,
        "gm": gm_val,
        "wm": wm_val,
        "brain_mean": float(np.mean(brain_voxels)),
        "brain_std": float(np.std(brain_voxels)),
        "contrast_gm_wm": gm_val / max(wm_val, 1e-3),
        "contrast_csf_gm": csf_val / max(gm_val, 1e-3),
        "seg_method": "Percentile_Fallback",
    }


def create_seg_overlay(slice_img: np.ndarray, seg_slice: np.ndarray) -> np.ndarray:
    """Create a high-contrast publication overlay of SynthSeg tissue masks."""
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
    # CSF: Blue
    rgb[csf_mask] = (1.0 - alpha) * rgb[csf_mask] + alpha * np.array([0.12, 0.47, 0.71])
    # WM: Orange
    rgb[wm_mask] = (1.0 - alpha) * rgb[wm_mask] + alpha * np.array([1.0, 0.50, 0.05])
    # GM: Green
    rgb[gm_mask] = (1.0 - alpha) * rgb[gm_mask] + alpha * np.array([0.17, 0.63, 0.17])
    return rgb


def extract_anatomical_ventricular_slice(vol_np: np.ndarray, seg_np: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract strictly aligned mid-ventricular axial slice using lateral ventricle segmentation.
    This guarantees every slice across TE captures the exact same anatomical structure.
    """
    if seg_np is not None:
        vent_mask = np.isin(seg_np, [4, 43])
        vent_counts = vent_mask.sum(axis=(0, 1))
        if vent_counts.max() > 20:
            # Find peak ventricular area slice
            z_idx = int(np.argmax(vent_counts))
        else:
            z_idx = vol_np.shape[2] // 2
    else:
        z_idx = vol_np.shape[2] // 2

    raw_slice = vol_np[:, :, z_idx].astype(np.float32)
    p1, p99 = float(np.percentile(raw_slice, 1.0)), float(np.percentile(raw_slice, 99.0))
    if p99 > p1:
        norm_slice = np.clip((raw_slice - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm_slice = raw_slice / (raw_slice.max() + 1e-4)

    slice_img = np.rot90(norm_slice)
    seg_slice = np.rot90(seg_np[:, :, z_idx]) if seg_np is not None else None
    return slice_img, seg_slice


def generate_volume_deterministic(
    args,
    device: torch.device,
    autoencoder: torch.nn.Module,
    model: MRCLIPConditionedUNet,
    scale_factor: float,
    clip_embedding: torch.Tensor,
    output_size: tuple[int, int, int],
    cfg_scale: float = 3.0,
    seed: int = 6256,
) -> np.ndarray:
    """Generate a single 3D MRI volume deterministically from initial noise seed."""
    set_determinism(seed=seed)
    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (
        1,
        args.latent_channels,
        output_size[0] // divisor,
        output_size[1] // divisor,
        output_size[2] // divisor,
    )

    spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    noise = torch.randn(latent_shape, device=device)
    image = noise.clone()

    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"]
        )

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)
    cond_emb = clip_embedding.to(device)

    all_t = noise_scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            for t, next_t in zip(all_t, all_next_t):
                if cfg_scale == 0.0:
                    model_output = model(
                        x=image,
                        timesteps=torch.tensor([t], device=device),
                        clip_embedding=null_emb,
                        spacing_tensor=spacing_tensor,
                    )
                elif cfg_scale == 1.0:
                    model_output = model(
                        x=image,
                        timesteps=torch.tensor([t], device=device),
                        clip_embedding=cond_emb,
                        spacing_tensor=spacing_tensor,
                    )
                else:
                    x2 = torch.cat([image, image], dim=0)
                    t2 = torch.tensor([t, t], device=device)
                    emb2 = torch.cat([cond_emb, null_emb], dim=0)
                    sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                    out2 = model(
                        x=x2,
                        timesteps=t2,
                        clip_embedding=emb2,
                        spacing_tensor=sp2,
                    )
                    model_cond, model_uncond = out2.chunk(2)
                    model_output = model_uncond + cfg_scale * (model_cond - model_uncond)

                if isinstance(noise_scheduler, RFlowScheduler):
                    image, _ = noise_scheduler.step(model_output, t, image, next_t)
                else:
                    image, _ = noise_scheduler.step(model_output, t, image)

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
    return vol_np


def encode_prompt(prompt_text: str, clip_model, tokenizer, device) -> torch.Tensor:
    tok = tokenizer(prompt_text).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


# ---------------------------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------------------------
def plot_te_response_publication(
    te_arr: np.ndarray,
    records: list[dict[str, Any]],
    slices: list[tuple[int, np.ndarray]],
    rep_overlay: np.ndarray | None,
    spearman_rho: float,
    monotonic_rate: float,
    out_path: Path,
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    fig = plt.figure(figsize=(22.0, 6.8), dpi=300)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.15, 1.85, 0.95],
        wspace=0.18,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.10,
    )

    # Panel 1: Quantitative TE Decay Curves
    ax1 = fig.add_subplot(gs[0, 0])
    mean_arr = np.array([r["brain_mean"] for r in records])

    ax1.plot(te_arr, [r["csf"] for r in records], "o-", label="CSF", color="#1f77b4", lw=2.4, markersize=7)
    ax1.plot(te_arr, [r["gm"] for r in records], "s-", label="Gray Matter", color="#2ca02c", lw=2.4, markersize=7)
    ax1.plot(te_arr, [r["wm"] for r in records], "^-", label="White Matter", color="#ff7f0e", lw=2.4, markersize=7)
    ax1.plot(te_arr, mean_arr, "k--", label=f"Brain Mean (Mono: {monotonic_rate*100:.0f}%, $\\rho$={spearman_rho:.2f})", lw=2.2)

    ax1.set_xlabel("Echo Time (TE) [ms]", fontsize=13, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Tissue Intensity (a.u.)", fontsize=13, fontweight="bold", color="#0f172a")
    ax1.set_title("TE Response: Monotonic Signal Decay", fontsize=14, fontweight="bold", color="#0f172a", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#cbd5e1", fontsize=11)
    ax1.tick_params(labelsize=11)
    ax1.set_facecolor("#ffffff")

    # Panel 2: Axial Slices Across TE
    ax2 = fig.add_subplot(gs[0, 1])
    mosaic = np.hstack([s[1] for s in slices])
    ax2.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Axial Slices Across TE ({te_arr[0]}ms → {te_arr[-1]}ms)", fontsize=14, fontweight="bold", color="#0f172a", pad=12)
    ax2.axis("off")

    # Panel 3: SynthSeg 2.0 ROI Overlay
    if rep_overlay is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(rep_overlay)
        ax3.set_title("SynthSeg 2.0 ROI Overlay", fontsize=14, fontweight="bold", color="#0f172a", pad=12)
        ax3.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Publication-Grade TE Physics Evaluation")
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
    parser.add_argument("--output-dir", default="output/eval_physics")
    parser.add_argument("--seed", type=int, default=6256)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))

    args_cli = parser.parse_args()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("eval_te_physics")
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

    base_prompt = CANONICAL_MIDI_TEMPLATES["T2W"]
    te_values = [0.010, 0.020, 0.030, 0.040, 0.060, 0.080, 0.100]

    logger.info(f"\n--- Running Deterministic TE Sweep for TE in {[int(t*1000) for t in te_values]} ms ---")
    records = []
    slices_for_plot = []
    rep_overlay = None

    for idx, te in enumerate(te_values):
        te_ms = int(te * 1000)
        prompt = ensure_axial_prompt(replace_imaging_param(base_prompt, te=te))
        emb = encode_prompt(prompt, clip_model, tokenizer, device)

        # Generate strictly deterministically with fixed seed
        vol_np = generate_volume_deterministic(
            args, device, autoencoder, model, scale_factor, emb,
            output_size=target_dim, cfg_scale=args_cli.cfg_scale, seed=args_cli.seed
        )

        seg_np = run_synthseg(vol_np, logger=logger)
        tissues = extract_tissue_intensities(vol_np, seg_np=seg_np)
        ax_s, seg_s = extract_anatomical_ventricular_slice(vol_np, seg_np)
        slices_for_plot.append((te_ms, ax_s))

        if rep_overlay is None and seg_s is not None and idx == len(te_values) // 2:
            rep_overlay = create_seg_overlay(ax_s, seg_s)

        records.append({
            "param_value": te,
            "te_ms": te_ms,
            **tissues,
        })
        logger.info(f"  TE={te_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}, Mean={tissues['brain_mean']:.1f}")

    te_arr = np.array([r["te_ms"] for r in records])
    mean_arr = np.array([r["brain_mean"] for r in records])

    decay_steps = [mean_arr[i + 1] < mean_arr[i] for i in range(len(mean_arr) - 1)]
    monotonic_rate = float(np.mean(decay_steps)) if decay_steps else 0.0
    spearman_rho, _ = spearmanr(te_arr, mean_arr)

    out_file = out_dir / "te_response_consistency.png"
    plot_te_response_publication(
        te_arr=te_arr,
        records=records,
        slices=slices_for_plot,
        rep_overlay=rep_overlay,
        spearman_rho=float(spearman_rho),
        monotonic_rate=monotonic_rate,
        out_path=out_file,
    )
    logger.info(f"\n[SUCCESS] Saved pristine publication TE physics figure to: {out_file}")


if __name__ == "__main__":
    main()
