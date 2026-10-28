"""
evaluate_mrrate_systematic.py
==============================
Systematic evaluation for MR-RATE MR-CLIP conditioned diffusion model:
  1. Multi-CFG sampling sweep across test prompt cases.
  2. Integrates real physical scans when present on disk (paired evaluation).
  3. Quantitative metrics in MR-CLIP embedding space:
     - Synthetic ↔ Text Cosine Similarity
     - Synthetic ↔ Real Scan Cosine Similarity
     - Real ↔ Text Baseline Alignment
  4. Visual Case Montages:
     - Top Row: Ground Truth Real Scan (Axial, Coronal, Sagittal)
     - Rows below: Synthetic samples across all CFG scales
  5. Global t-SNE & PCA manifold mapping: Real Scans (■), Synthetic Scans (●), Text Prompts (★).

Usage example:
  CUDA_VISIBLE_DEVICES=1 python3 -m scripts.evaluate_mrrate_systematic \\
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \\
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \\
      -t configs/config_network_rflow.json \\
      --mrclip-ckpt /home/mav24/projects/NV-Generate-CTMR-A/models/diff_unet_3d_rflow-mr-brain_v1_mrrate.pt \\
      --csv /home/mav24/projects/NV-Generate-CTMR-A/data/batch00_metadata_processed_ready_age_sex.csv \\
      --data-root /data/mrrate \\
      --clip-checkpoint /home/mav24/projects/NV-Generate-CTMR-A/models/epoch_47.pt \\
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \\
      --n-cases 20 \\
      --num-samples 2 \\
      --cfg-scales 0.0 2.0 4.0 6.0 8.0 \\
      --output-dir output/systematic_evaluation_mrrate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Path Resolution & Real Scan Loading
# ---------------------------------------------------------------------------

def resolve_filepath(raw_path: str, data_root: str | None) -> str | None:
    """Resolve NIfTI file path across local and custom data root locations."""
    if not raw_path or raw_path.lower() in ["nan", "none"]:
        return None

    if os.path.exists(raw_path):
        return raw_path

    candidates = []
    if data_root:
        candidates.append(os.path.join(data_root, raw_path.lstrip("/")))
        candidates.append(os.path.join(data_root, os.path.basename(raw_path)))

        parts = raw_path.strip("/").split("/")
        if len(parts) >= 3:
            candidates.append(os.path.join(data_root, "/".join(parts[-3:])))
        if len(parts) >= 2:
            candidates.append(os.path.join(data_root, "/".join(parts[-2:])))

    common_roots = ["/data/mrrate", "/data/mr_niftis"]
    for cr in common_roots:
        candidates.append(os.path.join(cr, os.path.basename(raw_path)))
        parts = raw_path.strip("/").split("/")
        if len(parts) >= 3:
            candidates.append(os.path.join(cr, "/".join(parts[-3:])))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_and_preprocess_real(filepath: str, target_dim: tuple[int, int, int]) -> tuple[np.ndarray, torch.Tensor]:
    """Load, reorient to RAS, resize to target_dim, and normalize real MRI volume."""
    transforms = Compose([
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=target_dim, mode="trilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1000.0, clip=True
        ),
    ])
    data = transforms({"image": filepath})
    img = data["image"]  # (1, D, H, W) in [0, 1000]
    vol_np = img.squeeze(0).cpu().numpy()
    vol_norm = (img / 1000.0).unsqueeze(0)  # (1, 1, D, H, W) in [0, 1]
    return vol_np, vol_norm


# ---------------------------------------------------------------------------
# Slice Extraction & Vision Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_volume_with_clip(vol_tensor: torch.Tensor, clip_model, device: torch.device) -> torch.Tensor:
    """Encode 3D volume with 3D MR-CLIP vision encoder with 0.0-99.5 percentile normalization."""
    clip_model.eval()
    t = vol_tensor.to(device).float()
    v_min = torch.quantile(t, 0.0)
    v_max = torch.quantile(t, 0.995)
    if v_max > v_min:
        vol_norm = torch.clamp((t - v_min) / (v_max - v_min), 0.0, 1.0)
    else:
        vol_norm = torch.clamp(t, 0.0, 1.0)

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
        # Fallback triplane
        B, C, D, H, W = vol_norm.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
        ax = F.interpolate(vol_norm[:, :, mid_d, :, :], size=(224, 224), mode="bilinear", align_corners=False)
        cor = F.interpolate(vol_norm[:, :, :, mid_h, :], size=(224, 224), mode="bilinear", align_corners=False)
        sag = F.interpolate(vol_norm[:, :, :, :, mid_w], size=(224, 224), mode="bilinear", align_corners=False)
        tri = torch.cat([ax, cor, sag], dim=0)
        if tri.shape[1] == 1:
            tri = tri.repeat(1, 3, 1, 1)
        img_feats = clip_model.encode_image(tri)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
        vol_feat = img_feats.mean(dim=0, keepdim=True)
        vol_feat = vol_feat / (vol_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return vol_feat


@torch.no_grad()
def encode_prompt(prompt: str, clip_model, tokenizer, device: torch.device) -> torch.Tensor:
    """Encode text prompt string into normalized 512-D MR-CLIP text embedding."""
    tokens = tokenizer([prompt]).to(device)
    emb = clip_model.encode_text(tokens)
    return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)


import textwrap

# ---------------------------------------------------------------------------
# Modality & Metadata Parsers
# ---------------------------------------------------------------------------

def is_valid_brain_diagnostic_scan(row: dict | pd.Series) -> bool:
    """Explicit non-brain / non-diagnostic filter:
    - Exclude mapping & calibration scans: b1map, b1_map, mapping, calib, noise, phase
    - Exclude spine protocols: tor1, tor2, torakal, serv, spine, cerv, lumbar, dorsal
    - Exclude scouts & localizers: scout, localizer, survey, loc, navigator
    - Exclude diffusion / non-anatomical: dwi, adc, diff, b0, b1000
    """
    fp = str(row.get("filepath", "")).lower()
    text = str(row.get("text", "")).lower()
    combined = f"{fp} {text}"

    calib_keywords = ["b1map", "b1_map", "mapping", "calib", "noise", "phase"]
    if any(k in combined for k in calib_keywords):
        return False

    spine_keywords = ["tor1", "tor2", "torakal", "serv", "spine", "cerv", "lumbar", "dorsal"]
    if any(k in combined for k in spine_keywords):
        return False

    scout_keywords = ["scout", "localizer", "survey", "loc", "navigator"]
    if any(k in combined for k in scout_keywords):
        return False

    diff_keywords = ["dwi", "adc", "diff", "b0", "b1000"]
    if any(k in combined for k in diff_keywords):
        return False

    return True


def extract_modality(row: dict | pd.Series) -> str:
    """Extract canonical modality identifier (T1W, T2W, FLAIR, PDW, T2STARW) using strict regex matching."""
    raw_fp = str(row.get("filepath", ""))
    fname = os.path.basename(raw_fp).lower()
    raw_text = str(row.get("text", "")).lower()

    # Priority 1: Strict token and regex matching on filename stem
    if re.search(r"_flair-raw-axi", fname) or "flair" in fname:
        return "FLAIR"
    elif re.search(r"_swi-raw-axi|_t2starw-raw-axi", fname) or "swi" in fname or "t2star" in fname:
        return "T2STARW"
    elif re.search(r"_pdw-raw-axi", fname) or "pdw" in fname:
        return "PDW"
    elif re.search(r"_t2w-raw-axi", fname):
        return "T2W"
    elif re.search(r"_t1w-raw-axi", fname):
        return "T1W"

    # Priority 2: Fallback to sequence or text pattern matching
    raw_seq = str(row.get("sequence", row.get("label", ""))).upper().strip()
    if raw_seq in ["FLAIR"] or "flair" in raw_text or "dark-fluid" in raw_text:
        return "FLAIR"
    elif raw_seq in ["T2*", "T2STAR", "SWI", "GRE"] or "swi" in raw_text or "t2*" in raw_text:
        return "T2STARW"
    elif raw_seq in ["PD", "PDW", "PROTON"] or "pdw" in raw_text or "pd_tse" in raw_text:
        return "PDW"
    elif raw_seq in ["T2", "T2W", "T2-W", "T2_WEIGHTED"] or "t2_tse" in raw_text or "t2w" in raw_text:
        return "T2W"
    elif raw_seq in ["T1", "T1W", "T1-W", "T1_WEIGHTED"] or "t1_mprage" in raw_text or "t1w" in raw_text:
        return "T1W"

    return "MRI"


def extract_scanner(row: dict | pd.Series) -> str:
    """Extract scanner name from text or columns."""
    text = str(row.get("text", ""))
    m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", text)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1]} ({parts[2]}T)"
    return "Clinical Scanner"


# ---------------------------------------------------------------------------
# Rectified Flow Diffusion Generator
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
    cfg_scale: float = 4.0,
    seed: int = 6256,
) -> tuple[np.ndarray, torch.Tensor]:
    """Generate a 3D MRI volume given a text CLIP embedding and CFG scale."""
    set_determinism(seed=seed)
    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (
        1,
        args.latent_channels,
        output_size[0] // divisor,
        output_size[1] // divisor,
        output_size[2] // divisor,
    )

    spacing_raw = np.array(args.diffusion_unet_inference.get("spacing", [1.0, 1.0, 1.0])).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    noise = torch.randn(latent_shape, device=device)
    image = noise.clone()

    num_steps = args.diffusion_unet_inference.get(
        "num_inference_steps",
        args.diffusion_unet_inference.get("num_eval_timesteps", 100)
    )

    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=num_steps,
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=num_steps)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)
    cond_emb = clip_embedding.to(device)

    all_t = noise_scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

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
        synthetic = dynamic_infer(inferer, recon_model, image)
        vol_tensor = torch.clamp(synthetic, 0.0, 1.0)

        data = synthetic.squeeze().cpu().detach().numpy()
        a_min, a_max, b_min, b_max = 0, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        vol_np = np.clip(data, a_min, a_max).astype(np.float32)

    return vol_np, vol_tensor


def get_middle_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract normalized central 2D slices: Axial, Coronal, Sagittal."""
    p01, p99 = np.percentile(vol, [1, 99])
    norm_vol = np.clip((vol - p01) / max(p99 - p01, 1e-6), 0.0, 1.0)
    sx, sy, sz = norm_vol.shape

    brain_mask = norm_vol > 0.08
    if brain_mask.sum() > 500:
        x_coords, y_coords, z_coords = np.where(brain_mask)
        x_mid = int(np.median(x_coords))
        y_mid = int(np.median(y_coords))
        z_min, z_max = z_coords.min(), z_coords.max()
        z_mid = int(z_min + (z_max - z_min) * 0.58)
    else:
        x_mid = sx // 2
        y_mid = sy // 2
        z_mid = int(sz * 0.58)

    x_mid = int(np.clip(x_mid, 0, sx - 1))
    y_mid = int(np.clip(y_mid, 0, sy - 1))
    z_mid = int(np.clip(z_mid, 0, sz - 1))

    axial = np.rot90(norm_vol[:, :, z_mid])
    coronal = np.rot90(norm_vol[:, y_mid, :])
    sagittal = np.rot90(norm_vol[x_mid, :, :])

    return axial, coronal, sagittal


# ---------------------------------------------------------------------------
# Visual Montage Plotter for Each Case
# ---------------------------------------------------------------------------

def plot_case_cfg_montage(
    case_idx: int,
    modality_str: str,
    scanner_str: str,
    text_prompt: str,
    real_vol: np.ndarray | None,
    sim_real_text: float | None,
    results_by_cfg: dict,  # cfg -> list of dicts {sample_idx, vol, sim_text, sim_real, seed}
    out_path: Path,
):
    """Plot multi-row visual montage comparing ground truth real scan (if available) with all CFG scales."""
    cfg_list = sorted(results_by_cfg.keys())
    num_cfg = len(cfg_list)
    num_samples = len(results_by_cfg[cfg_list[0]])

    has_real = real_vol is not None
    total_rows = (1 if has_real else 0) + num_cfg * num_samples
    fig, axes = plt.subplots(total_rows, 3, figsize=(11, 2.8 * total_rows), dpi=160)
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    plt.subplots_adjust(top=0.94, bottom=0.03, left=0.14, right=0.96, hspace=0.35, wspace=0.08)

    row_idx = 0

    # Ground Truth Real Scan Row (if available)
    if has_real:
        r_ax, r_cor, r_sag = get_middle_slices(real_vol)
        axes[0, 0].imshow(r_ax, cmap="gray", vmin=0.0, vmax=1.0)
        sim_rt_str = f" (Text-Sim: {sim_real_text:.3f})" if sim_real_text is not None else ""
        axes[0, 0].set_ylabel(f"Ground Truth\nReal Scan{sim_rt_str}", fontsize=9, fontweight="bold", color="#d62728")
        axes[0, 0].set_title(f"Real Axial", fontsize=9, fontweight="bold")

        axes[0, 1].imshow(r_cor, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("Real Coronal", fontsize=9, fontweight="bold")

        axes[0, 2].imshow(r_sag, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("Real Sagittal", fontsize=9, fontweight="bold")

        for col in range(3):
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])

        row_idx += 1

    # Synthetic CFG Rows
    for cfg in cfg_list:
        for s_idx, s_info in enumerate(results_by_cfg[cfg]):
            s_ax, s_cor, s_sag = get_middle_slices(s_info["vol"])
            sim_t = s_info["sim_text"]
            sim_r = s_info.get("sim_real", None)

            axes[row_idx, 0].imshow(s_ax, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 0].set_ylabel(f"CFG {cfg:g}\n(Seed #{s_info['seed']})", fontsize=9, fontweight="bold")

            title_str = f"Axial (Sim-Text: {sim_t:.3f}"
            if sim_r is not None:
                title_str += f" | Sim-Real: {sim_r:.3f}"
            title_str += ")"
            axes[row_idx, 0].set_title(title_str, fontsize=8)

            axes[row_idx, 1].imshow(s_cor, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 1].set_title("Coronal", fontsize=8)

            axes[row_idx, 2].imshow(s_sag, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 2].set_title("Sagittal", fontsize=8)

            for col in range(3):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

            row_idx += 1

    # Format metadata prompt into 2 or 3 lines to fit nicely across the image
    wrapped_lines = textwrap.wrap(text_prompt, width=105)
    if len(wrapped_lines) > 3:
        wrapped_prompt = "\n".join(wrapped_lines[:3]) + "..."
    else:
        wrapped_prompt = "\n".join(wrapped_lines)

    top_margin = 0.965 if total_rows >= 8 else 0.935
    plt.subplots_adjust(top=top_margin, bottom=0.03, left=0.14, right=0.96, hspace=0.35, wspace=0.08)

    title_text = f"MR-RATE Systematic Case #{case_idx} — [{modality_str} | {scanner_str}]\nPrompt:\n\"{wrapped_prompt}\""
    fig.suptitle(title_text, fontsize=9.5, fontweight="bold", y=0.995 if total_rows >= 8 else 0.985)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Global Summary Plotters
# ---------------------------------------------------------------------------

def plot_metrics_summary(df_metrics: pd.DataFrame, out_dir: Path, logger):
    """Plot Mean Cosine Similarity vs CFG scale (Synthetic ↔ Text and Synthetic ↔ Real)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)

    cfg_vals = sorted(df_metrics["cfg_scale"].unique())
    grouped_cfg = df_metrics.groupby("cfg_scale")

    mean_text = [grouped_cfg.get_group(c)["sim_text"].mean() for c in cfg_vals]
    std_text = [grouped_cfg.get_group(c)["sim_text"].std() for c in cfg_vals]

    # Plot 1: Synthetic ↔ Text Alignment vs CFG
    ax[0].errorbar(cfg_vals, mean_text, yerr=std_text, marker="o", color="#1f77b4", capsize=5, lw=2.2, label="Synthetic ↔ Text")
    if "sim_real_text" in df_metrics.columns and df_metrics["sim_real_text"].notna().any():
        base_rt = df_metrics["sim_real_text"].dropna().mean()
        ax[0].axhline(base_rt, color="#d62728", linestyle="--", lw=2, label=f"Real ↔ Text Baseline ({base_rt:.3f})")

    ax[0].set_title("MR-CLIP Text ↔ Image Alignment vs CFG Scale", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("Classifier-Free Guidance (CFG) Scale", fontsize=10)
    ax[0].set_ylabel("Cosine Similarity", fontsize=10)
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[0].legend(loc="best", fontsize=9)

    # Plot 2: Synthetic ↔ Real Scan Cosine Similarity (if real scans evaluated) or Per-Modality Alignment
    has_real_metrics = "sim_real" in df_metrics.columns and df_metrics["sim_real"].notna().any()
    if has_real_metrics:
        df_real_subset = df_metrics.dropna(subset=["sim_real"])
        grouped_real = df_real_subset.groupby("cfg_scale")
        mean_real = [grouped_real.get_group(c)["sim_real"].mean() if c in grouped_real.groups else np.nan for c in cfg_vals]
        std_real = [grouped_real.get_group(c)["sim_real"].std() if c in grouped_real.groups else 0.0 for c in cfg_vals]

        ax[1].errorbar(cfg_vals, mean_real, yerr=std_real, marker="s", color="#2ca02c", capsize=5, lw=2.2, label="Synthetic ↔ Real Scan")
        ax[1].set_title("Visual Fidelity to Physical Scans vs CFG", fontsize=11, fontweight="bold")
        ax[1].set_xlabel("Classifier-Free Guidance (CFG) Scale", fontsize=10)
        ax[1].set_ylabel("Cosine Similarity", fontsize=10)
        ax[1].grid(True, linestyle="--", alpha=0.5)
        ax[1].legend(loc="best", fontsize=9)
    else:
        modalities = sorted(df_metrics["modality"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(modalities), 1)))
        for i, mod in enumerate(modalities):
            mod_df = df_metrics[df_metrics["modality"] == mod]
            mod_grp = mod_df.groupby("cfg_scale")
            m_vals = [mod_grp.get_group(c)["sim_text"].mean() if c in mod_grp.groups else np.nan for c in cfg_vals]
            ax[1].plot(cfg_vals, m_vals, marker="s", color=colors[i], lw=2.0, label=mod)

        ax[1].set_title("Alignment vs CFG (Per Modality)", fontsize=11, fontweight="bold")
        ax[1].set_xlabel("Classifier-Free Guidance (CFG) Scale", fontsize=10)
        ax[1].set_ylabel("Cosine Similarity", fontsize=10)
        ax[1].grid(True, linestyle="--", alpha=0.5)
        ax[1].legend(loc="best", fontsize=9)

    plt.tight_layout()
    summary_plot = out_dir / "cfg_vs_similarity.png"
    fig.savefig(str(summary_plot), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {summary_plot}")


def plot_latent_projections(all_embeddings: list[dict], out_dir: Path, logger):
    """Plot PCA and t-SNE of Real scans (■), Synthetic scans (●), and Text prompts (★)."""
    X = np.array([item["feat"] for item in all_embeddings])
    modalities = [item["modality"] for item in all_embeddings]
    types = [item["type"] for item in all_embeddings]

    unique_mods = sorted(list(set(modalities)))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_mods), 1)))
    color_map = {m: colors[i] for i, m in enumerate(unique_mods)}

    marker_map = {"Real": "s", "Synthetic": "o", "Text": "*"}
    size_map = {"Real": 60, "Synthetic": 55, "Text": 100}
    alpha_map = {"Real": 0.85, "Synthetic": 0.85, "Text": 0.95}

    # 1. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=200)
    for mod in unique_mods:
        for tp in ["Real", "Synthetic", "Text"]:
            idx = [i for i in range(len(all_embeddings)) if modalities[i] == mod and types[i] == tp]
            if idx:
                ax.scatter(
                    X_pca[idx, 0], X_pca[idx, 1],
                    color=color_map[mod],
                    marker=marker_map[tp],
                    s=size_map[tp],
                    alpha=alpha_map[tp],
                    edgecolors="black" if tp != "Text" else "none",
                    linewidth=0.5,
                    label=f"{mod} ({tp})"
                )

    ax.set_title("MR-RATE Latent Space PCA (Real vs Synthetic vs Text)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    pca_path = out_dir / "pca_embedding_space.png"
    fig.savefig(str(pca_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {pca_path}")

    # 2. t-SNE
    perplexity = min(30, max(5, len(X) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=200)
    for mod in unique_mods:
        for tp in ["Real", "Synthetic", "Text"]:
            idx = [i for i in range(len(all_embeddings)) if modalities[i] == mod and types[i] == tp]
            if idx:
                ax.scatter(
                    X_tsne[idx, 0], X_tsne[idx, 1],
                    color=color_map[mod],
                    marker=marker_map[tp],
                    s=size_map[tp],
                    alpha=alpha_map[tp],
                    edgecolors="black" if tp != "Text" else "none",
                    linewidth=0.5,
                    label=f"{mod} ({tp})"
                )

    ax.set_title("MR-RATE Latent Space t-SNE (Real vs Synthetic vs Text)", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    tsne_path = out_dir / "tsne_embedding_space.png"
    fig.savefig(str(tsne_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {tsne_path}")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Systematic Evaluation Suite for MR-RATE Diffusion Model")
    parser.add_argument("-e", "--env-config", required=True, help="Environment config JSON")
    parser.add_argument("-c", "--model-config", required=True, help="Model training config JSON")
    parser.add_argument("-t", "--model-def", required=True, help="Network definition config JSON")
    parser.add_argument("--mrclip-ckpt", required=True, help="Path to trained MR-RATE MR-CLIP UNet checkpoint")
    parser.add_argument("--clip-checkpoint", required=True, help="Path to MR-CLIP vision/text checkpoint (epoch_47.pt)")
    parser.add_argument("--open-clip-src", required=True, help="Path to open_clip src directory")
    parser.add_argument("--csv", required=True, help="Path to MR-RATE metadata CSV")
    parser.add_argument("--data-root", default="/data/mrrate", help="Path to downloaded MR-RATE NIfTI root (optional)")
    parser.add_argument("--n-cases", type=int, default=20, help="Total number of balanced prompt cases to evaluate")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of random seeds per CFG scale")
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0, 8.0],
                        help="List of CFG scales to evaluate")
    parser.add_argument("--output-dir", default="output/systematic_evaluation_mrrate", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--clip-context-length", type=int, default=98)
    parser.add_argument("--axial-only", action="store_true", default=True,
                        help="Filter dataset for axial plane brain scans to align with the 3D generator orientation (default: True)")
    parser.add_argument("--no-axial-only", dest="axial_only", action="store_false",
                        help="Disable axial-only filtering")

    args_cli = parser.parse_args()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("mrrate_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running MR-RATE Systematic Evaluation on device: {device}")
    logger.info(f"CFG Scales: {args_cli.cfg_scales} | Num Samples per CFG: {args_cli.num_samples} | Total Cases: {args_cli.n_cases}")

    # Load configs
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    target_dim = tuple(args.diffusion_unet_inference["dim"])

    # Load Autoencoder & UNet
    logger.info("Loading Autoencoder & MR-RATE UNet...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)

    ckpt = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
    cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys()):
        model.load_state_dict(cleaned_state, strict=False)
    else:
        model.unet.load_state_dict(cleaned_state, strict=False)

    scale_factor = float(ckpt.get("scale_factor", 1.0422))
    model.eval()

    # Load MR-CLIP Text & Vision Encoders
    logger.info("Loading MR-CLIP Encoders...")
    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))
    clip_model.eval()

    # Read and parse CSV
    df = pd.read_csv(args_cli.csv)
    logger.info(f"Loaded {len(df)} metadata rows from {args_cli.csv}")

    # Explicit non-brain / non-diagnostic filter
    valid_mask = df.apply(is_valid_brain_diagnostic_scan, axis=1)
    df_valid = df[valid_mask].copy()
    logger.info(f"Filtered out non-brain/calibration/scout scans: {len(df_valid)}/{len(df)} rows retained.")
    df = df_valid

    df["modality"] = df.apply(extract_modality, axis=1)
    df["scanner"] = df.apply(extract_scanner, axis=1)

    # Filter for axial plane brain scans to match the 3D diffusion generator orientation
    if args_cli.axial_only:
        axial_mask = df["text"].astype(str).str.contains(r"plane axial", case=False, na=False) | df["filepath"].astype(str).str.contains(r"-axi_", case=False, na=False)
        df_axial = df[axial_mask].copy()
        if len(df_axial) > 0:
            logger.info(f"Filtered for axial brain scans: {len(df_axial)}/{len(df)} rows matched 'plane axial'.")
            df = df_axial
        else:
            logger.warning("No rows matched 'plane axial'; proceeding with unfiltered dataset.")

    # Check for real files on disk
    df["resolved_path"] = df["filepath"].apply(lambda p: resolve_filepath(str(p), args_cli.data_root))
    real_available_count = df["resolved_path"].notna().sum()
    logger.info(f"Found {real_available_count} real scans matching on disk in {args_cli.data_root or 'default paths'}")

    # Prioritize cases with real scans on disk when available
    modalities = [m for m in ["T1W", "T2W", "FLAIR", "T2STARW", "PDW"] if m in df["modality"].values]
    if not modalities:
        modalities = list(df["modality"].value_counts().index[:4])

    per_mod = max(1, args_cli.n_cases // len(modalities))
    selected_rows = []

    for mod in modalities:
        m_df = df[df["modality"] == mod]
        m_with_real = m_df[m_df["resolved_path"].notna()]
        m_without_real = m_df[m_df["resolved_path"].isna()]

        picked = []
        if len(m_with_real) > 0:
            sample_real = min(len(m_with_real), per_mod)
            picked.append(m_with_real.sample(n=sample_real, random_state=args_cli.seed))

        needed = per_mod - sum(len(p) for p in picked)
        if needed > 0 and len(m_without_real) > 0:
            sample_synth = min(len(m_without_real), needed)
            picked.append(m_without_real.sample(n=sample_synth, random_state=args_cli.seed))

        if picked:
            selected_rows.append(pd.concat(picked))

    df_selected = pd.concat(selected_rows).reset_index(drop=True)
    real_in_cohort = df_selected["resolved_path"].notna().sum()
    logger.info(f"Selected {len(df_selected)} prompt cases ({real_in_cohort} with paired real scans) across modalities: {dict(df_selected['modality'].value_counts())}")

    # Export selected cohort cases metadata CSV for audit and inspection
    cohort_csv = out_dir / "test_cohort_cases.csv"
    df_selected.to_csv(cohort_csv, index=False)
    logger.info(f"Saved selected test cohort metadata to {cohort_csv}")

    all_metrics = []
    all_embeddings = []

    # Run generation across all selected cases & CFG scales
    for case_idx, row in df_selected.iterrows():
        prompt = str(row["text"])
        mod = str(row["modality"])
        scanner = str(row["scanner"])
        real_fp = row.get("resolved_path", None)

        logger.info(f"\n[{case_idx+1}/{len(df_selected)}] Evaluating Case #{case_idx} [{mod} | {scanner}] (Real Scan: {'YES' if real_fp else 'NO'})")

        # 1. Encode text prompt
        emb_text = encode_prompt(prompt, clip_model, tokenizer, device)
        all_embeddings.append({
            "feat": emb_text.squeeze().cpu().numpy(),
            "modality": mod,
            "type": "Text",
            "case_idx": case_idx,
        })

        # 2. Encode Real Scan if present
        real_vol_np = None
        emb_real = None
        sim_real_text = None

        if real_fp and os.path.exists(real_fp):
            try:
                real_vol_np, real_tensor = load_and_preprocess_real(real_fp, target_dim)
                emb_real = encode_volume_with_clip(real_tensor, clip_model, device)
                sim_real_text = float(torch.sum(emb_real * emb_text).item())
                all_embeddings.append({
                    "feat": emb_real.squeeze().cpu().numpy(),
                    "modality": mod,
                    "type": "Real",
                    "case_idx": case_idx,
                })
            except Exception as e:
                logger.warning(f"Failed loading real scan {real_fp}: {e}")
                real_vol_np = None
                emb_real = None

        results_by_cfg = {}

        # 3. Multi-CFG generation sweep
        for cfg in args_cli.cfg_scales:
            results_by_cfg[cfg] = []
            for s_idx in range(args_cli.num_samples):
                seed = args_cli.seed + case_idx * 100 + int(cfg * 10) + s_idx * 7
                vol_np, vol_tensor = generate_sample(
                    args, device, autoencoder, model, scale_factor, emb_text, target_dim, cfg_scale=cfg, seed=seed
                )
                
                # Pass reconstructed volume directly to MR-CLIP vision encoder (matching pretraining [0, 1] range)
                emb_syn = encode_volume_with_clip(vol_tensor, clip_model, device)

                sim_text = float(torch.sum(emb_syn * emb_text).item())
                sim_real = float(torch.sum(emb_syn * emb_real).item()) if emb_real is not None else None

                results_by_cfg[cfg].append({
                    "sample_idx": s_idx,
                    "vol": vol_np,
                    "sim_text": sim_text,
                    "sim_real": sim_real,
                    "seed": seed,
                })

                all_metrics.append({
                    "case_idx": case_idx,
                    "modality": mod,
                    "scanner": scanner,
                    "cfg_scale": cfg,
                    "sample_idx": s_idx,
                    "seed": seed,
                    "sim_text": sim_text,
                    "sim_real": sim_real,
                    "sim_real_text": sim_real_text,
                    "vol_mean": float(np.mean(vol_np)),
                    "vol_std": float(np.std(vol_np)),
                })

                # Record synthetic embedding for manifold plots (middle CFG)
                target_cfg = 4.0 if 4.0 in args_cli.cfg_scales else (args_cli.cfg_scales[-2] if len(args_cli.cfg_scales) > 1 else args_cli.cfg_scales[0])
                if cfg == target_cfg and s_idx == 0:
                    all_embeddings.append({
                        "feat": emb_syn.squeeze().cpu().numpy(),
                        "modality": mod,
                        "type": "Synthetic",
                        "case_idx": case_idx,
                    })

        # Save Visual Montage for this Case
        montage_path = out_dir / f"case_{case_idx:03d}_{mod}_cfg_montage.png"
        plot_case_cfg_montage(case_idx, mod, scanner, prompt, real_vol_np, sim_real_text, results_by_cfg, montage_path)
        logger.info(f"  Saved montage: {montage_path}")

    # Save quantitative CSV
    df_metrics = pd.DataFrame(all_metrics)
    csv_out = out_dir / "metrics_per_sample.csv"
    df_metrics.to_csv(csv_out, index=False)
    logger.info(f"\nSaved metrics: {csv_out}")

    # Summary table by CFG scale
    summary_cols = ["sim_text"]
    if df_metrics["sim_real"].notna().any():
        summary_cols.append("sim_real")

    agg_dict = {c: ["mean", "std"] for c in summary_cols}
    agg_dict["vol_mean"] = "mean"
    agg_dict["vol_std"] = "mean"

    summary_df = df_metrics.groupby("cfg_scale").agg(agg_dict).reset_index()
    summary_csv = out_dir / "metrics_summary_by_cfg.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved CFG summary table: {summary_csv}")

    # Save intermediate embeddings pickle for fast replotting
    emb_pkl_path = out_dir / "systematic_embeddings.pkl"
    with open(emb_pkl_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    logger.info(f"Saved {len(all_embeddings)} intermediate embeddings to {emb_pkl_path}")

    # Global Plots
    plot_metrics_summary(df_metrics, out_dir, logger)
    if len(all_embeddings) >= 6:
        plot_latent_projections(all_embeddings, out_dir, logger)

    logger.info("\n" + "=" * 60)
    logger.info(f"MR-RATE Systematic Evaluation Complete! Output directory: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
