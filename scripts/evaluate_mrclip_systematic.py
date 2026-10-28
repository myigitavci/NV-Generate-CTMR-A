"""
evaluate_mrclip_systematic.py
==============================
Systematic evaluation of MR-CLIP conditioned diffusion model:
  1. Multi-CFG sampling: [0.0, 2.0, 5.0, 10.0] with 3 samples per CFG across N test cases.
  2. Visual montage: Side-by-side central slices (Axial, Coronal, Sagittal) for Real vs Synthetic.
  3. Quantitative metrics: Cosine similarity in MR-CLIP embedding space (Syn vs Text, Syn vs Real).
  4. t-SNE & PCA mapping: 2D feature space projection verifying modality clustering and contrast preservation.

Usage example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_mrclip_systematic \\
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \\
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \\
      -t configs/config_network_rflow.json \\
      --mrclip-ckpt models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt \\
      --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \\
      --data-root /data/mr_niftis \\
      --clip-checkpoint /home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt \\
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \\
      --n-cases 10 \\
      --num-samples 3 \\
      --cfg-scales 0.0 2.0 5.0 10.0 \\
      --output-dir output/systematic_evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
)
from monai.utils import set_determinism
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.sample import ReconModel
from scripts.utils import define_instance, dynamic_infer


# ---------------------------------------------------------------------------
# Slice extraction & Vision Encoding helper
# ---------------------------------------------------------------------------

def extract_triplane_slices(vol: torch.Tensor, in_channels: int = 3) -> torch.Tensor:
    """
    Extract 3 orthogonal central slices (axial, coronal, sagittal) from 3D MRI volume (B, 1, D, H, W)
    and resize each to (224, 224) for MR-CLIP vision encoder.
    """
    B, C, D, H, W = vol.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    # axial (B, 1, H, W), coronal (B, 1, D, W), sagittal (B, 1, D, H)
    axial = vol[:, :, mid_d, :, :]
    coronal = vol[:, :, :, mid_h, :]
    sagittal = vol[:, :, :, :, mid_w]

    axial_res = F.interpolate(axial, size=(224, 224), mode="bilinear", align_corners=False)
    coronal_res = F.interpolate(coronal, size=(224, 224), mode="bilinear", align_corners=False)
    sagittal_res = F.interpolate(sagittal, size=(224, 224), mode="bilinear", align_corners=False)

    triplanes = torch.cat([axial_res, coronal_res, sagittal_res], dim=0)  # (3*B, 1, 224, 224)

    if in_channels == 3 and triplanes.shape[1] == 1:
        triplanes = triplanes.repeat(1, 3, 1, 1)

    return triplanes


@torch.no_grad()
def encode_volume_with_clip(vol_tensor: torch.Tensor, clip_model, device: torch.device) -> torch.Tensor:
    """
    Given a volume tensor (1, 1, D, H, W) in range [0, 1],
    encode with 3D MR-CLIP vision encoder (resized to 128x128x128) or 2D triplane fallback.
    """
    clip_model.eval()
    vol_norm = torch.clamp(vol_tensor.to(device).float(), 0.0, 1.0)

    # Check if 3D vision encoder
    is_3d = False
    in_ch = 1
    if hasattr(clip_model, "visual"):
        for m in clip_model.visual.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.modules.conv.Conv3d)):
                is_3d = True
                in_ch = m.in_channels
                break

    if is_3d or getattr(clip_model, "vis_3d", False):
        vol_128 = F.interpolate(vol_norm, size=(128, 128, 128), mode="trilinear", align_corners=False)
        if in_ch == 3 and vol_128.shape[1] == 1:
            vol_128 = vol_128.repeat(1, 3, 1, 1, 1)
        elif in_ch == 1 and vol_128.shape[1] == 3:
            vol_128 = vol_128.mean(dim=1, keepdim=True)
        img_feats = clip_model.encode_image(vol_128)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
        return img_feats
    else:
        # Fallback to 2D triplane slices
        in_ch = 3
        if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "conv1"):
            in_ch = getattr(clip_model.visual.conv1, "in_channels", 3)
        slices = extract_triplane_slices(vol_norm, in_channels=in_ch)
        img_feats = clip_model.encode_image(slices)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
        vol_feat = img_feats.mean(dim=0, keepdim=True)
        vol_feat = vol_feat / (vol_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return vol_feat


@torch.no_grad()
def encode_text_with_clip(text: str, clip_model, tokenizer, device: torch.device) -> torch.Tensor:
    """Given a prompt string, return normalized (1, 512) text embedding."""
    clip_model.eval()
    tokens = tokenizer([text]).to(device)
    text_feat = clip_model.encode_text(tokens)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
    return text_feat


# ---------------------------------------------------------------------------
# Filepath resolution helper
# ---------------------------------------------------------------------------

def resolve_filepath(raw_path: str, data_root: str) -> str | None:
    if os.path.exists(raw_path):
        return raw_path
    candidate = os.path.join(data_root, raw_path.lstrip("/"))
    if os.path.exists(candidate):
        return candidate
    candidate2 = data_root + raw_path
    if os.path.exists(candidate2):
        return candidate2
    return None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_evaluation_models(args, device: torch.device, logger: logging.Logger, mrclip_ckpt_path: str):
    logger.info("Loading Autoencoder...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

    logger.info(f"Loading UNet and wrapping from {mrclip_ckpt_path}...")
    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)

    ckpt = torch.load(mrclip_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
    cleaned_state = {}
    for k, v in state.items():
        cleaned_k = k[len("module."):] if k.startswith("module.") else k
        cleaned_state[cleaned_k] = v

    has_wrapper_keys = any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys())
    if has_wrapper_keys:
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    else:
        missing, unexpected = model.unet.load_state_dict(cleaned_state, strict=False)

    scale_factor = ckpt.get("scale_factor", 1.039)
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.item()
    scale_factor = float(scale_factor)

    model.eval()
    logger.info(f"Models loaded successfully. scale_factor = {scale_factor:.4f}")
    return autoencoder, model, scale_factor


# ---------------------------------------------------------------------------
# Single-Volume Diffusion Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(
    args,
    device: torch.device,
    autoencoder: torch.nn.Module,
    model: MRCLIPConditionedUNet,
    scale_factor: float,
    clip_embedding: torch.Tensor,
    output_size: tuple[int, int, int],
    cfg_scale: float,
    seed: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Generate a 3D MRI volume given a text CLIP embedding and CFG scale.
    Returns:
      volume_np: (D, H, W) numpy array scaled [0, 1000]
      volume_tensor: (1, 1, D, H, W) torch tensor in [0, 1] range for vision encoder
    """
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
    image = noise

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

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in zip(all_t, all_next_t):
            if cfg_scale == 0.0:
                # Pure unconditional generation (no text conditioning)
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=null_emb,
                    spacing_tensor=spacing_tensor,
                )
            elif cfg_scale == 1.0:
                # Standard conditional generation (single-pass)
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=cond_emb,
                    spacing_tensor=spacing_tensor,
                )
            else:
                # Guided generation: uncond + s * (cond - uncond)
                x2 = torch.cat([image, image], dim=0)
                t2 = torch.tensor([t, t], device=device)
                emb2 = torch.cat([cond_emb, null_emb], dim=0)
                sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                out2 = model(
                    x=x2, timesteps=t2,
                    clip_embedding=emb2, spacing_tensor=sp2,
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
        synthetic = dynamic_infer(inferer, recon_model, image)  # (1, 1, D, H, W)
        vol_tensor = torch.clamp(synthetic, 0.0, 1.0)

        data = synthetic.squeeze().cpu().detach().numpy()
        a_min, a_max, b_min, b_max = 0, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        vol_np = np.clip(data, a_min, a_max).astype(np.float32)

        return vol_np, vol_tensor


# ---------------------------------------------------------------------------
# Preprocessing for Real Scans
# ---------------------------------------------------------------------------

def load_and_preprocess_real(filepath: str, target_dim: tuple[int, int, int]) -> tuple[np.ndarray, torch.Tensor]:
    """Load real NIfTI, orient to RAS, resize to target_dim, and return [0, 1000] array + [0, 1] tensor."""
    transforms = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image", dtype=torch.float32),
        Resized(keys="image", spatial_size=target_dim, mode="trilinear"),
    ])
    d = transforms({"image": filepath})
    tensor_01 = d["image"].unsqueeze(0)  # (1, 1, D, H, W)
    np_1000 = (tensor_01.squeeze().cpu().numpy() * 1000.0).astype(np.float32)
    return np_1000, tensor_01


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def normalize_for_display(vol: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.5) -> np.ndarray:
    """
    Standardize MRI intensity range to [0, 1] using robust percentile clipping.
    This guarantees identical contrast windowing and black background across all real and synthetic scans.
    """
    v = vol.astype(np.float32)
    # Exclude near-zero background for robust brain tissue percentile calculation
    bg_thresh = np.percentile(v, 5.0)
    non_bg = v[v > bg_thresh]
    if len(non_bg) > 100:
        p_low = np.percentile(non_bg, lower_pct)
        p_high = np.percentile(non_bg, upper_pct)
    else:
        p_low = np.percentile(v, lower_pct)
        p_high = np.percentile(v, upper_pct)

    if p_high > p_low:
        v_norm = np.clip((v - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        v_norm = np.clip(v, 0.0, 1.0)
    return v_norm


def get_middle_slices(vol: np.ndarray, axial_height_ratio: float = 0.57) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return normalized central axial, coronal, and sagittal 2D slices in [0, 1].
    Uses adaptive brain bounding-box detection to ensure slices cut through 
    the mid-ventricular brain level rather than lower neck/cerebellum.
    """
    norm_vol = normalize_for_display(vol)
    sx, sy, sz = norm_vol.shape

    # Find brain tissue extent (exclude background air)
    bg_thresh = max(0.08, float(np.percentile(norm_vol, 20.0)))
    brain_pts = np.where(norm_vol > bg_thresh)

    if len(brain_pts[0]) > 500:
        x_min, x_max = int(brain_pts[0].min()), int(brain_pts[0].max())
        y_min, y_max = int(brain_pts[1].min()), int(brain_pts[1].max())
        z_min, z_max = int(brain_pts[2].min()), int(brain_pts[2].max())

        # Mid-sagittal (50% of brain width)
        x_mid = int(x_min + (x_max - x_min) * 0.50)
        # Mid-coronal (50% of brain depth)
        y_mid = int(y_min + (y_max - y_min) * 0.50)
        # Mid-axial (57% of brain height for lateral ventricles & basal ganglia)
        z_mid = int(z_min + (z_max - z_min) * axial_height_ratio)
    else:
        x_mid = sx // 2
        y_mid = sy // 2
        z_mid = int(sz * 0.58)

    # Clamp indices to volume boundaries
    x_mid = int(np.clip(x_mid, 0, sx - 1))
    y_mid = int(np.clip(y_mid, 0, sy - 1))
    z_mid = int(np.clip(z_mid, 0, sz - 1))

    # Axial: slice across Z (axis 2)
    axial = np.rot90(norm_vol[:, :, z_mid])
    # Coronal: slice across Y (axis 1)
    coronal = np.rot90(norm_vol[:, y_mid, :])
    # Sagittal: slice across X (axis 0)
    sagittal = np.rot90(norm_vol[x_mid, :, :])

    return axial, coronal, sagittal


def plot_case_grid(
    case_idx: int,
    modality_str: str,
    text_prompt: str,
    real_vol: np.ndarray,
    results_by_cfg: dict,  # cfg -> list of dicts {sample_idx, vol, sim_text, sim_real}
    out_path: str,
):
    """
    Create a comprehensive montage figure for a single test case:
    Top Row: Ground Truth (Axial, Coronal, Sagittal)
    Rows below: CFG scales (0, 2, 5, 10), each with Samples 1..3
    All slices are normalized with identical robust percentile windowing [0, 1].
    """
    cfg_list = sorted(results_by_cfg.keys())
    num_cfg = len(cfg_list)
    num_samples = len(results_by_cfg[cfg_list[0]])

    # 1 Real row + num_cfg * num_samples rows
    total_rows = 1 + num_cfg * num_samples
    fig, axes = plt.subplots(total_rows, 3, figsize=(12, 3.2 * total_rows), dpi=150)

    # Clean styling
    plt.subplots_adjust(top=0.96, bottom=0.02, left=0.04, right=0.96, hspace=0.35, wspace=0.1)

    # Ground Truth Row
    r_ax, r_cor, r_sag = get_middle_slices(real_vol)
    axes[0, 0].imshow(r_ax, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"Real Ground Truth ({modality_str})\nAxial", fontsize=10, fontweight="bold")
    axes[0, 1].imshow(r_cor, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(f"Ground Truth\nCoronal", fontsize=10, fontweight="bold")
    axes[0, 2].imshow(r_sag, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(f"Ground Truth\nSagittal", fontsize=10, fontweight="bold")

    for col in range(3):
        axes[0, col].axis("off")

    row_idx = 1
    for cfg in cfg_list:
        for s_idx, s_info in enumerate(results_by_cfg[cfg]):
            s_ax, s_cor, s_sag = get_middle_slices(s_info["vol"])
            cos_t = s_info["sim_text"]
            cos_r = s_info["sim_real"]

            axes[row_idx, 0].imshow(s_ax, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 0].set_ylabel(f"CFG {cfg}\nSample {s_idx+1}", fontsize=10, fontweight="bold", labelpad=10)
            axes[row_idx, 0].set_title(f"Axial (Sim-Text: {cos_t:.3f} | Sim-Real: {cos_r:.3f})", fontsize=8)

            axes[row_idx, 1].imshow(s_cor, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 1].set_title(f"Coronal", fontsize=8)

            axes[row_idx, 2].imshow(s_sag, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 2].set_title(f"Sagittal", fontsize=8)

            for col in range(3):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

            row_idx += 1

    prompt_short = text_prompt[:140] + ("..." if len(text_prompt) > 140 else "")
    fig.suptitle(f"Case #{case_idx} [{modality_str}]\nPrompt: \"{prompt_short}\"", fontsize=11, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(df_metrics: pd.DataFrame, out_path: str):
    """Plot Mean Cosine Similarity vs CFG scale with Real-Metadata baseline in publication quality."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    grouped = df_metrics.groupby("cfg_scale")
    cfg_vals = sorted(df_metrics["cfg_scale"].unique())

    mean_text = [grouped.get_group(c)["sim_text"].mean() for c in cfg_vals]
    std_text = [grouped.get_group(c)["sim_text"].std() for c in cfg_vals]

    mean_real = [grouped.get_group(c)["sim_real"].mean() for c in cfg_vals]
    std_real = [grouped.get_group(c)["sim_real"].std() for c in cfg_vals]

    baseline_real_text = df_metrics["sim_real_text"].mean() if "sim_real_text" in df_metrics.columns else None

    # Plot 1: Synthetic <-> Metadata Alignment
    ax[0].errorbar(
        cfg_vals, mean_text, yerr=std_text,
        marker="o", markersize=7, color="#1d4ed8", capsize=5, elinewidth=1.6, capthick=1.6, lw=2.2,
        label="Synthetic ↔ Metadata", zorder=3
    )
    if baseline_real_text is not None:
        ax[0].axhline(
            baseline_real_text, color="#dc2626", linestyle="--", lw=2.0,
            label=f"Real ↔ Metadata Baseline ({baseline_real_text:.3f})", zorder=2
        )
    ax[0].set_title("Metadata ↔ Image Alignment", fontsize=13.5, fontweight="bold", pad=10, color="#111827")
    ax[0].set_xlabel("CFG Guidance Scale", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[0].set_ylabel("Cosine Similarity", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[0].tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)
    for spine in ax[0].spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)
    ax[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#d1d5db", zorder=0)
    ax[0].set_axisbelow(True)
    ax[0].legend(
        loc="lower right", fontsize=11, framealpha=0.95, facecolor="#f8fafc", edgecolor="#cbd5e1", borderpad=0.7, labelspacing=0.5
    )

    # Plot 2: Synthetic <-> Real Image Alignment
    ax[1].errorbar(
        cfg_vals, mean_real, yerr=std_real,
        marker="s", markersize=7, color="#15803d", capsize=5, elinewidth=1.6, capthick=1.6, lw=2.2,
        label="Synthetic ↔ Real Scan", zorder=3
    )
    ax[1].set_title("Synthetic ↔ Real Scan Cosine Similarity", fontsize=13.5, fontweight="bold", pad=10, color="#111827")
    ax[1].set_xlabel("CFG Guidance Scale", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[1].set_ylabel("Cosine Similarity", fontsize=12, fontweight="bold", labelpad=8, color="#111827")
    ax[1].tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)
    for spine in ax[1].spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.2)
    ax[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#d1d5db", zorder=0)
    ax[1].set_axisbelow(True)
    ax[1].legend(
        loc="lower right", fontsize=11, framealpha=0.95, facecolor="#f8fafc", edgecolor="#cbd5e1", borderpad=0.7, labelspacing=0.5
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_feature_embeddings(
    all_feats: list[dict],
    out_dir: str,
    exclude_modalities: list[str] = None,
    no_axis_numbers: bool = True,
):
    """
    Compute 2D t-SNE and PCA maps on all accumulated MR-CLIP embeddings with dual legends (Data Type & Modality).
    """
    if exclude_modalities is None:
        exclude_modalities = ["T2STARW"]

    exclude_set = {m.upper() for m in exclude_modalities}
    filtered_feats = [item for item in all_feats if item.get("modality", "").upper() not in exclude_set]

    if len(filtered_feats) < 4:
        print(f"[WARN] Not enough feature embeddings ({len(filtered_feats)}) to plot.")
        return

    X = np.array([item["feat"] for item in filtered_feats])  # (M, 512)
    types = [item["type"] for item in filtered_feats]
    modalities = [item["modality"].upper() for item in filtered_feats]

    # Harmonized high-contrast color palette per modality
    modality_colors = {
        "T1W": "#8c564b",    # Rich Sienna Brown
        "T2W": "#0284c7",    # Ocean Cyan / Azure Blue
        "FLAIR": "#1e3a8a",  # Deep Royal Navy Blue
        "PDW": "#16a34a",    # Forest Emerald Green
    }
    unique_mods = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in set(modalities)] or sorted(list(set(modalities)))
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_mods), 1)))
    mod_color_map = {m: modality_colors.get(m, fallback_colors[i]) for i, m in enumerate(unique_mods)}

    # Projections
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    perplexity = min(30, max(5, len(X) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

    for proj_name, coords in [("t-SNE", X_tsne), ("PCA", X_pca)]:
        fig, (ax, ax_leg) = plt.subplots(
            1, 2,
            figsize=(13.8, 8.2),
            dpi=300,
            gridspec_kw={"width_ratios": [3.3, 1.2]}
        )
        ax_leg.axis("off")

        # Plot synthetic samples first (background layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]

            if t == "synthetic":
                cfg = item.get("cfg_scale", 0.0)
                alpha = 0.35 if cfg == 0 else (0.65 if cfg <= 2.0 else 0.90)
                ax.scatter(x, y, color=col, marker="o", s=70, alpha=alpha, edgecolors="white", linewidth=0.5, zorder=3)

        # Plot real scans (middle layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]
            if t == "real":
                ax.scatter(x, y, color=col, marker="s", s=115, edgecolors="#111827", linewidth=1.2, zorder=5)

        # Plot metadata prompts (top layer)
        for i, item in enumerate(filtered_feats):
            t = item["type"]
            mod = item["modality"].upper()
            col = mod_color_map.get(mod, "#6b7280")
            x, y = coords[i, 0], coords[i, 1]
            if t in ["text", "metadata"]:
                ax.scatter(x, y, color=col, marker="*", s=210, edgecolors="#111827", linewidth=1.1, zorder=6)

        # Dual Legends: 1) Data Type, 2) MRI Modality
        type_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#374151", markersize=10, markeredgecolor="#ffffff", markeredgewidth=0.6, label="Synthetic"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#374151", markersize=10.5, markeredgecolor="#111827", markeredgewidth=0.9, label="Real"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#374151", markersize=13, markeredgecolor="#111827", markeredgewidth=0.9, label="Metadata"),
        ]

        modality_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=mod_color_map[mod], markersize=11, markeredgecolor="#111827", markeredgewidth=0.8, label=mod)
            for mod in unique_mods
        ]

        if proj_name == "t-SNE":
            ax.set_xlabel("t-SNE 1", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
            ax.set_ylabel("t-SNE 2", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
        else:
            ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% explained variance)", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")
            ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% explained variance)", fontsize=13.5, fontweight="normal", labelpad=10, color="#111827")

        if no_axis_numbers:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis="both", which="major", labelsize=11.5, colors="#1f2937", width=1.1, length=5)

        for spine in ax.spines.values():
            spine.set_color("#1f2937")
            spine.set_linewidth(1.2)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#d1d5db", zorder=0)
        ax.set_axisbelow(True)

        leg1 = ax_leg.legend(
            handles=type_handles,
            title="Data Type",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=12,
            title_fontsize=13.5,
            framealpha=0.96,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            borderpad=0.9,
            labelspacing=0.75,
            handletextpad=0.8,
        )
        leg1.get_title().set_fontweight("bold")
        ax_leg.add_artist(leg1)

        fig.tight_layout()
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
            leg2_top = bbox1.y0 - 0.050
        except Exception:
            leg2_top = 0.52

        leg2 = ax_leg.legend(
            handles=modality_handles,
            title="MRI Modality",
            loc="upper left",
            bbox_to_anchor=(0.02, leg2_top),
            fontsize=12,
            title_fontsize=13.5,
            framealpha=0.96,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            borderpad=0.9,
            labelspacing=0.75,
            handletextpad=0.8,
        )
        leg2.get_title().set_fontweight("bold")

        save_path = os.path.join(out_dir, f"{proj_name.lower().replace('-', '')}_embedding_space.png")
        fig.savefig(save_path, bbox_extra_artists=(leg1, leg2), bbox_inches="tight", dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Systematic MR-CLIP Diffusion Model Evaluation")
    p.add_argument("-e", "--env-config", required=True, help="Environment config JSON")
    p.add_argument("-c", "--model-config", required=True, help="Model config JSON")
    p.add_argument("-t", "--model-def", required=True, help="Network definition JSON")
    p.add_argument("--mrclip-ckpt", required=True, help="Path to trained MR-CLIP UNet checkpoint")
    p.add_argument("--csv", required=True, help="Metadata CSV path with filepath and text columns")
    p.add_argument("--data-root", default="/data/mr_niftis", help="Root where NIfTI images live")
    p.add_argument("--clip-checkpoint", required=True, help="Path to MR-CLIP ViT-B checkpoint")
    p.add_argument("--open-clip-src", required=True, help="Path to open_clip src directory")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--n-cases", type=int, default=10, help="Number of distinct test cases to evaluate")
    p.add_argument("--num-samples", type=int, default=3, help="Number of samples per CFG scale")
    p.add_argument("--cfg-scales", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    p.add_argument("--output-dir", default="output/systematic_evaluation", help="Directory to save evaluation results")
    p.add_argument("--save-nifti", action="store_true", help="Save generated 3D NIfTI volumes to disk")
    p.add_argument("--seed", type=int, default=42, help="Master random seed")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--filter-axial", action="store_true", default=True,
                   help="Filter dataset for axial plane brain scans to align with the 3D generator orientation (default: True)")
    p.add_argument("--exclude-modalities", nargs="+", default=["T2STARW"],
                   help="Modalities to exclude from evaluation and feature maps (e.g. --exclude-modalities T2STARW)")
    p.add_argument("--no-axis-numbers", action="store_true", default=False,
                   help="Omit numeric ticks on t-SNE and PCA axes for minimalist publication style")
    args_cli = p.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("mrclip_eval")
    fh = logging.FileHandler(os.path.join(out_dir, "evaluation.log"))
    fh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logger.info(f"Starting Systematic Evaluation | Cases: {args_cli.n_cases}, CFGs: {args_cli.cfg_scales}, Samples/CFG: {args_cli.num_samples}, Excluded: {args_cli.exclude_modalities}")

    # Load configs
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    device = torch.device(args_cli.device)

    # 1. Load Models
    autoencoder, model, scale_factor = load_evaluation_models(args, device, logger, args_cli.mrclip_ckpt)
    logger.info("Loading MR-CLIP text and vision encoder...")
    clip_model, tokenizer = load_mrclip(
        args_cli.open_clip_src,
        args_cli.clip_checkpoint,
        args_cli.clip_context_length,
        str(device),
    )
    clip_model.eval()

    # 2. Load and Filter Test CSV Cases
    df = pd.read_csv(args_cli.csv)
    logger.info(f"Loaded test CSV with {len(df)} rows.")

    # Filter for axial plane brain scans to match the 3D diffusion generator orientation
    if args_cli.filter_axial:
        axial_mask = df["text"].astype(str).str.contains(r"plane axial", case=False, na=False) | df["filepath"].astype(str).str.contains(r"-axi_", case=False, na=False)
        df_axial = df[axial_mask].reset_index(drop=True)
        if len(df_axial) > 0:
            logger.info(f"Filtered for axial brain scans: {len(df_axial)}/{len(df)} rows matched 'plane axial'.")
            df = df_axial
        else:
            logger.warning("No rows matched 'plane axial'; proceeding with unfiltered dataset.")

    # Exclude unwanted modalities from dataset
    exclude_set = {m.upper() for m in (args_cli.exclude_modalities or [])}
    if exclude_set:
        if "sequence" in df.columns:
            seq_col = df["sequence"].astype(str).str.upper()
            df = df[~seq_col.isin(exclude_set)].reset_index(drop=True)
            logger.info(f"Filtered out excluded modalities {exclude_set}: {len(df)} rows remaining.")

    # Find valid rows with existing NIfTI files
    valid_rows = []
    for idx, row in df.iterrows():
        resolved = resolve_filepath(str(row["filepath"]), args_cli.data_root)
        if resolved is not None:
            valid_rows.append({**row.to_dict(), "resolved_path": resolved, "csv_idx": idx})

    logger.info(f"Found {len(valid_rows)} accessible NIfTI files on disk.")
    if len(valid_rows) == 0:
        logger.error(f"No NIfTI files found under {args_cli.data_root}. Check --data-root path!")
        return

    random.seed(args_cli.seed)
    # Stratified selection across sequences if available
    selected_cases = []
    if "sequence" in valid_rows[0]:
        seq_groups = {}
        for r in valid_rows:
            seq_groups.setdefault(str(r["sequence"]).upper(), []).append(r)
        # Round-robin select from each sequence
        while len(selected_cases) < min(args_cli.n_cases, len(valid_rows)):
            for seq, items in seq_groups.items():
                if items and len(selected_cases) < args_cli.n_cases:
                    selected_cases.append(items.pop(random.randrange(len(items))))
    else:
        selected_cases = random.sample(valid_rows, min(args_cli.n_cases, len(valid_rows)))

    logger.info(f"Selected {len(selected_cases)} cases for systematic evaluation.")

    # Target volume dimensions
    target_dim = tuple(args.diffusion_unet_inference["dim"])  # (256, 256, 128)

    # Containers for metrics and embeddings
    metrics_records = []
    all_embeddings = []  # list of dicts: {feat, type: 'real'|'text'|'synthetic', modality, cfg_scale}
    emb_pkl_path = out_dir / "systematic_embeddings.pkl"

    # 3. Main Evaluation Loop
    for c_i, case in enumerate(tqdm(selected_cases, desc="Evaluating cases")):
        case_idx = case.get("csv_idx", c_i)
        modality_str = str(case.get("sequence", case.get("label", "MRI"))).upper()
        text_prompt = str(case["text"])
        real_path = case["resolved_path"]

        logger.info(f"\n--- Case {c_i+1}/{len(selected_cases)} (Row {case_idx}) | Modality: {modality_str} ---")

        # A. Encode Text Prompt
        emb_text = encode_text_with_clip(text_prompt, clip_model, tokenizer, device)
        all_embeddings.append({
            "feat": emb_text.squeeze().cpu().numpy(),
            "type": "text",
            "modality": modality_str,
            "case_idx": case_idx,
            "text_prompt": text_prompt,
        })

        # B. Load Real Image & Encode with Vision CLIP
        real_np, real_tensor = load_and_preprocess_real(real_path, target_dim)
        emb_real = encode_volume_with_clip(real_tensor, clip_model, device)
        cos_sim_real_text = float(torch.sum(emb_real * emb_text).item())
        logger.info(f"Baseline Real ↔ Text Cosine Sim: {cos_sim_real_text:.4f}")
        all_embeddings.append({
            "feat": emb_real.squeeze().cpu().numpy(),
            "type": "real",
            "modality": modality_str,
            "case_idx": case_idx,
            "real_path": real_path,
        })

        results_by_cfg = {}

        # C. Generate across CFG scales & Samples
        for cfg in args_cli.cfg_scales:
            results_by_cfg[cfg] = []
            for s in range(args_cli.num_samples):
                sample_seed = args_cli.seed + case_idx * 100 + s * 13 + int(cfg * 7)
                syn_np, syn_tensor = generate_sample(
                    args=args,
                    device=device,
                    autoencoder=autoencoder,
                    model=model,
                    scale_factor=scale_factor,
                    clip_embedding=emb_text,
                    output_size=target_dim,
                    cfg_scale=cfg,
                    seed=sample_seed,
                )

                # Encode synthetic volume with Vision CLIP
                emb_syn = encode_volume_with_clip(syn_tensor, clip_model, device)

                # Compute Cosine Similarities
                cos_sim_text = float(torch.sum(emb_syn * emb_text).item())
                cos_sim_real = float(torch.sum(emb_syn * emb_real).item())

                results_by_cfg[cfg].append({
                    "sample_idx": s + 1,
                    "vol": syn_np,
                    "sim_text": cos_sim_text,
                    "sim_real": cos_sim_real,
                })

                all_embeddings.append({
                    "feat": emb_syn.squeeze().cpu().numpy(),
                    "type": "synthetic",
                    "modality": modality_str,
                    "cfg_scale": cfg,
                    "case_idx": case_idx,
                    "sample_idx": s + 1,
                })

                metrics_records.append({
                    "case_idx": case_idx,
                    "modality": modality_str,
                    "cfg_scale": cfg,
                    "sample_idx": s + 1,
                    "seed": sample_seed,
                    "sim_text": cos_sim_text,
                    "sim_real": cos_sim_real,
                    "sim_real_text": cos_sim_real_text,
                })

                if args_cli.save_nifti:
                    nii_dir = out_dir / "nifti_volumes" / f"case_{case_idx}_{modality_str}"
                    nii_dir.mkdir(parents=True, exist_ok=True)
                    nii_path = nii_dir / f"syn_cfg{cfg}_sample{s+1}.nii.gz"
                    nib.save(nib.Nifti1Image(syn_np, np.eye(4)), str(nii_path))

        # Progressively save intermediate embeddings pickle after each case
        with open(emb_pkl_path, "wb") as f:
            pickle.dump(all_embeddings, f)

        # D. Save Per-Case Visual Comparison Montage
        case_fig_path = out_dir / f"case_{c_i+1:02d}_row{case_idx}_{modality_str}_comparison.png"
        plot_case_grid(
            case_idx=case_idx,
            modality_str=modality_str,
            text_prompt=text_prompt,
            real_vol=real_np,
            results_by_cfg=results_by_cfg,
            out_path=str(case_fig_path),
        )
        logger.info(f"Saved visual comparison to {case_fig_path}")

    # 4. Save Quantitative Metrics
    df_metrics = pd.DataFrame(metrics_records)
    metrics_csv_path = out_dir / "metrics_per_sample.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved detailed metrics to {metrics_csv_path}")

    # Final embeddings pickle save
    with open(emb_pkl_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    logger.info(f"Saved {len(all_embeddings)} intermediate embeddings to {emb_pkl_path}")

    # Summary table
    df_summary = df_metrics.groupby(["cfg_scale"]).agg(
        mean_sim_text=("sim_text", "mean"),
        std_sim_text=("sim_text", "std"),
        mean_sim_real=("sim_real", "mean"),
        std_sim_real=("sim_real", "std"),
    ).reset_index()
    summary_csv_path = out_dir / "metrics_summary_by_cfg.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"\n--- Quantitative Summary by CFG Scale ---\n{df_summary.to_string(index=False)}")

    # Plot metrics vs CFG
    plot_metrics_summary(df_metrics, str(out_dir / "cosine_similarity_vs_cfg.png"))

    # 5. Generate t-SNE & PCA Feature Space Maps
    logger.info("Computing t-SNE and PCA feature space maps...")
    plot_feature_embeddings(
        all_embeddings,
        str(out_dir),
        exclude_modalities=args_cli.exclude_modalities,
        no_axis_numbers=args_cli.no_axis_numbers,
    )
    logger.info(f"All evaluation plots and metrics saved to: {out_dir}")


if __name__ == "__main__":
    main()
