#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
MR-CLIP Diffusion Model Validation & Verification Suite

Implements 4 Core Clinical, Physical, & Hardware Validation Experiments:
1. Continuous Physics Sweep (TE, TR, TI, Flip Angle interpolation on fixed anatomy)
2. Multi-Modal Registered Contrast Panel (T1, T2, FLAIR, PD, T2* on identical anatomy)
3. Hardware & Field Strength Transfer (1.5T vs 3.0T across GE, Siemens, Philips)
4. Scanner Clustering & Manifold Analysis (t-SNE/PCA of 20 samples per scanner for Real vs Synthetic vs Text)

Usage:
  python -m scripts.run_mrclip_validation_experiments \
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \
      -t configs/config_network_rflow.json \
      --mrclip-ckpt models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt \
      --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \
      --data-root /data/mr_niftis \
      --clip-checkpoint /home/mav24/projects/MR-Styler/model/.../epoch_125.pt \
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \
      --experiments 1 2 3 4 \
      --n-per-scanner 20 \
      --cfg-scale 4.0 \
      --output-dir output/validation_experiments
"""

from __future__ import annotations

import argparse
import logging
import os
import re
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
from monai.inferers import SlidingWindowInferer
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

from .diff_model_infer_mrclip import ReconModel
from .diff_model_setting import load_config, setup_logging
from .embed_with_mrclip import load_mrclip
from .mrclip_conditioned_unet import MRCLIPConditionedUNet
from .utils import define_instance, dynamic_infer
from .utils_infer import RFlowScheduler


# ---------------------------------------------------------------------------
# Visualization & Preprocessing Helpers
# ---------------------------------------------------------------------------

def normalize_for_display(vol: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.5) -> np.ndarray:
    """Standardize MRI intensity range to [0, 1] using robust percentile clipping."""
    v = vol.astype(np.float32)
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


def get_middle_slices(vol: np.ndarray, axial_height_ratio: float = 0.55) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized central axial, coronal, and sagittal 2D slices in [0, 1].
    Uses robust percentile-based brain bounding box and selects the axial slice
    with the maximum brain parenchyma within the mid-cerebral region to guarantee
    zero black or truncated cerebellar slices.
    """
    norm_vol = normalize_for_display(vol)
    sx, sy, sz = norm_vol.shape

    bg_thresh = max(0.08, float(np.percentile(norm_vol, 25.0)))
    brain_pts = np.where(norm_vol > bg_thresh)

    if len(brain_pts[0]) > 500:
        x_min, x_max = int(np.percentile(brain_pts[0], 2.0)), int(np.percentile(brain_pts[0], 98.0))
        y_min, y_max = int(np.percentile(brain_pts[1], 2.0)), int(np.percentile(brain_pts[1], 98.0))
        z_min, z_max = int(np.percentile(brain_pts[2], 2.0)), int(np.percentile(brain_pts[2], 98.0))

        x_mid = int(x_min + (x_max - x_min) * 0.50)
        y_mid = int(y_min + (y_max - y_min) * 0.50)

        # Select axial slice with maximum brain tissue within middle 40%-70% of brain extent
        z_low = int(z_min + (z_max - z_min) * 0.40)
        z_high = int(z_min + (z_max - z_min) * 0.70)
        if z_high > z_low:
            z_slice_vox = [np.sum(norm_vol[:, :, z] > bg_thresh) for z in range(z_low, z_high + 1)]
            z_mid = z_low + int(np.argmax(z_slice_vox))
        else:
            z_mid = int(z_min + (z_max - z_min) * axial_height_ratio)
    else:
        x_mid = sx // 2
        y_mid = sy // 2
        z_mid = int(sz * axial_height_ratio)

    x_mid = int(np.clip(x_mid, 0, sx - 1))
    y_mid = int(np.clip(y_mid, 0, sy - 1))
    z_mid = int(np.clip(z_mid, 0, sz - 1))

    axial = np.rot90(norm_vol[:, :, z_mid])
    coronal = np.rot90(norm_vol[:, y_mid, :])
    sagittal = np.rot90(norm_vol[x_mid, :, :])

    return axial, coronal, sagittal


def resolve_filepath(raw_path: str, data_root: str | None) -> str | None:
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
    vol_norm = (img / 1000.0).unsqueeze(0)  # (1, 1, D, H, W)
    return vol_np, vol_norm


# ---------------------------------------------------------------------------
# Sampling Engine
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


def encode_prompt(prompt_text: str, clip_model, tokenizer, device: torch.device) -> torch.Tensor:
    """Encode text string with MR-CLIP text encoder."""
    tokens = tokenizer([prompt_text]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    return feat


@torch.no_grad()
def encode_volume_with_clip(vol_tensor: torch.Tensor, clip_model, device: torch.device) -> torch.Tensor:
    """Encode 3D volume with 3D MR-CLIP vision encoder."""
    clip_model.eval()
    vol_norm = torch.clamp(vol_tensor.to(device).float(), 0.0, 1.0)

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


# ---------------------------------------------------------------------------
# Prompt Extraction / Canonical Template Helpers
# ---------------------------------------------------------------------------

def clean_scanner_label(scanner: str) -> str:
    """
    Sanitizes and harmonizes scanner labels:
      - Removes underscores (e.g. Discovery_MR750 -> Discovery MR750).
      - Unifies GE Signa HDX and HDxt into 'GE Signa HDx/t' idempotently.
      - Normalizes casing (DISCOVERY -> Discovery, SIGNA -> Signa, MR750W -> MR750w).
      - Removes redundant whitespace.
    """
    s = scanner.replace("_", " ")
    s = re.sub(r"\bDISCOVERY\b", "Discovery", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSIGNA\b", "Signa", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR750W\b", "MR750w", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR750\b", "MR750", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMR450\b", "MR450", s, flags=re.IGNORECASE)
    # Robust and idempotent regex: cleanly condenses any Signa HDX / HDxt / HDx/t / HDx/t/t...
    s = re.sub(r"\bGE\s+Signa\s+HD[^\(\d]+", "GE Signa HDx/t ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSigna\s+HD[^\(\d]+", "Signa HDx/t ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_scanner(row: dict | pd.Series) -> str:
    """Extract canonical scanner identifier from prompt text or CSV columns."""
    text = str(row.get("text", ""))
    m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", text)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            raw_s = f"{parts[0]} {parts[1]} ({parts[2]}T)"
            return clean_scanner_label(raw_s)
    v = str(row.get("Manufacturer", row.get("manufacturer", "Unknown")))
    m_name = str(row.get("ManufacturerModelName", row.get("model", "")))
    f_val = str(row.get("MagneticFieldStrength", row.get("field_strength", "")))
    raw_s = f"{v} {m_name} ({f_val}T)".strip()
    return clean_scanner_label(raw_s)


def extract_modality(row: dict | pd.Series) -> str:
    """Extract canonical modality identifier (T1W, T2W, FLAIR, PDW, T2STARW) from row."""
    raw_seq = str(row.get("sequence", row.get("label", ""))).upper().strip()
    raw_fp = str(row.get("filepath", "")).lower()
    raw_text = str(row.get("text", "")).upper()

    # 1. Authoritative sequence column directly from CSV
    if raw_seq in ["T1W", "T1", "T1-W", "T1_WEIGHTED"]:
        return "T1W"
    if raw_seq in ["T2W", "T2", "T2-W", "T2_WEIGHTED"]:
        return "T2W"
    if raw_seq in ["PDW", "PD", "PROTON"]:
        return "PDW"
    if raw_seq in ["FLAIR"]:
        return "FLAIR"
    if raw_seq in ["T2STARW", "T2*", "T2STAR", "SWI", "GRE"]:
        return "T2STARW"

    # 2. Fallback to regex checks if sequence column is missing or numeric
    # FLAIR
    if "flair" in raw_fp or "FLAIR" in raw_seq or "FLAIR" in raw_text:
        return "FLAIR"

    # T2* / SWI
    if (any(k in raw_fp for k in ["swi", "t2*", "t2star", "gre"]) or
        any(k in raw_seq for k in ["SWI", "T2*", "T2STAR", "GRE"]) or
        any(k in raw_text for k in ["SWI", "T2*", "T2STAR"])):
        return "T2STARW"

    # PDW (strict word-boundary or explicit tag)
    if ("pdw" in raw_fp or re.search(r"(^|[_\-/])pd([_\-/]|$)", raw_fp) or
        any(k in raw_seq for k in ["PD", "PDW", "PROTON"]) or
        any(k in raw_text for k in [" AX PD", "PDW", "PROTON", "PROTON DENSITY"])):
        return "PDW"

    # T1W (Strict word boundary to avoid false matching 'patient1', 'part1', etc.)
    if (re.search(r"(^|[_\-/])t1w?([_\-/]|$)", raw_fp) or
        any(k in raw_seq for k in ["T1", "T1W", "T1-W", "T1_WEIGHTED", "MPRAGE", "SPGR"]) or
        re.search(r"\b(T1|T1W|MPRAGE|SPGR|IR-FSPGR|3D_T1)\b", raw_text) or
        "(TRA, GR, MP)" in raw_text):
        return "T1W"

    # T2W
    if (re.search(r"(^|[_\-/])t2w?([_\-/]|$)", raw_fp) or
        any(k in raw_seq for k in ["T2", "T2W", "T2-W", "T2_WEIGHTED"]) or
        re.search(r"\b(T2|T2W)\b", raw_text)):
        return "T2W"

    if raw_seq and not raw_seq.isdigit():
        return raw_seq
    return "MRI"


def ensure_axial_prompt(prompt: str) -> str:
    """
    Ensure the prompt strictly specifies 'plane axial' and strips non-axial orientations.
    Prevents the diffusion model from generating 2D coronal/sagittal slabs or partial brains.
    """
    if not prompt:
        return prompt

    # Replace any non-axial plane specification (e.g. plane coronal, plane sagittal, plane oblique) with plane axial
    if re.search(r"\bplane\s+[a-zA-Z0-9_\-]+", prompt, re.IGNORECASE):
        prompt = re.sub(r"\bplane\s+[a-zA-Z0-9_\-]+", "plane axial", prompt, flags=re.IGNORECASE)
    elif "plane axial" not in prompt.lower():
        # Insert 'plane axial, ' right after 'A brain MRI, '
        if "a brain mri," in prompt.lower():
            idx = prompt.lower().find("a brain mri,") + len("a brain mri,")
            prompt = prompt[:idx] + " plane axial," + prompt[idx:]
        else:
            prompt = "A brain MRI, plane axial, " + prompt

    # Replace coronal/sagittal descriptors in acquisition description with axial (TRA)
    prompt = re.sub(r"\b(COR|SAG|cor|sag)\b", "TRA", prompt)
    prompt = re.sub(r"\b(coronal|sagittal)\b", "axial", prompt, flags=re.IGNORECASE)
    return prompt


def is_half_or_incomplete_brain(vol_np: np.ndarray) -> tuple[bool, str]:
    """
    Validates that a synthesized 3D volume represents a complete, intact brain
    and not a partial FOV, half-brain, or severely clipped slab.
    Uses translation-invariant bounding box extent and voxel volume checks.
    """
    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_mask = vol_np > bg_thresh
    total_vox = int(np.sum(brain_mask))

    # 1. Reject empty or severely truncated scans
    if total_vox < 45000:
        return True, f"insufficient brain voxels ({total_vox} < 45000)"

    coords = np.argwhere(brain_mask)
    if len(coords) < 1000:
        return True, "sparse brain voxels"

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    x_extent = x_max - x_min  # Left-Right
    y_extent = y_max - y_min  # Anterior-Posterior
    z_extent = z_max - z_min  # Inferior-Superior

    # A complete adult/pediatric brain in 256x256x128 space spans:
    # LR >= 65 voxels, AP >= 75 voxels, IS >= 28 voxels
    if x_extent < 65:
        return True, f"truncated left-right coverage ({x_extent} < 65 voxels)"
    if y_extent < 75:
        return True, f"truncated anterior-posterior coverage ({y_extent} < 75 voxels)"
    if z_extent < 28:
        return True, f"truncated axial coverage ({z_extent} < 28 slices)"

    # 2. Check for severe hemispheric truncation (one hemisphere completely missing)
    x_mid = (x_min + x_max) // 2
    left_vox = int(np.sum(brain_mask[:x_mid, :, :]))
    right_vox = int(np.sum(brain_mask[x_mid:, :, :]))
    lr_ratio = min(left_vox, right_vox) / max(left_vox, right_vox, 1)
    if lr_ratio < 0.20:
        return True, f"severe hemispheric amputation (LR ratio {lr_ratio:.2f} < 0.20)"

    return False, "ok"


def generate_complete_brain(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, prompt: str, initial_seed: int,
    logger: logging.Logger | None = None, max_retries: int = 3,
    fallback_prompt: str | None = None
) -> tuple[np.ndarray | None, str, int]:
    """
    Generates a 3D brain volume with automatic completeness verification.
    If a half-brain or partial slab is detected, retries with another prompt/seed up to max_retries.
    If after max_retries the volume is still incomplete, returns (None, prompt, seed) to discard it.
    """
    current_prompt = prompt
    current_seed = initial_seed

    for attempt in range(1, max_retries + 1):
        emb = encode_prompt(current_prompt, clip_model, tokenizer, device)
        vol_np, _ = generate_sample(
            args, device, autoencoder, model, scale_factor, emb,
            target_dim, cfg_scale, seed=current_seed
        )

        is_bad, reason = is_half_or_incomplete_brain(vol_np)
        if not is_bad:
            return vol_np, current_prompt, current_seed

        if logger:
            logger.warning(
                f"[Attempt {attempt}/{max_retries}] Incomplete brain detected ({reason}). "
                f"Regenerating with alternative prompt/seed..."
            )

        if attempt == 1:
            current_prompt = fallback_prompt or prompt
            current_seed = initial_seed + 107
        elif attempt == 2:
            current_prompt = fallback_prompt or prompt
            current_seed = initial_seed + 251

    if logger:
        logger.error(f"Failed to generate complete brain after {max_retries} attempts. Discarding case.")
    return None, current_prompt, current_seed


CANONICAL_MIDI_TEMPLATES = {
    "T1W": (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
        "Acquisition (Description, Sequence, Variant): (TRA, SE, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, 0.55, NONE, 90.0)"
    ),
    "T2W": (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
        "Acquisition (Description, Sequence, Variant): (TRA, SE, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.095, 4.5, NONE, 90.0)"
    ),
    "FLAIR": (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Philips, Achieva_dStream, 3.0), "
        "Acquisition (Description, Sequence, Variant): (TRA, IR, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.34, 4.8, 1.65, 90.0)"
    ),
    "PDW": (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Sola, 1.5), "
        "Acquisition (Description, Sequence, Variant): (TRA, SE, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.015, 3.2, NONE, 90.0)"
    ),
    "T2STARW": (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
        "Acquisition (Description, Sequence, Variant): (SWI_Images, GR, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, 0.028, NONE, 15.0)"
    ),
}


def get_modality_templates_from_csv(csv_path: str | None) -> dict[str, str]:
    """Extract canonical high-quality brain axial prompt templates per modality for MIDI."""
    templates = {k: ensure_axial_prompt(v) for k, v in CANONICAL_MIDI_TEMPLATES.items()}

    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Find the most frequent canonical brain axial prompt for each modality
            for mod_key in ["T1W", "T2W", "FLAIR", "PDW", "T2STARW"]:
                matching_rows = []
                for _, row in df.iterrows():
                    text = str(row.get("text", "")).strip()
                    if len(text) < 20:
                        continue
                    m = extract_modality(row)
                    if m != mod_key:
                        continue

                    # Strictly filter for axial acquisitions
                    text_lower = text.lower()
                    if "plane axial" not in text_lower:
                        continue
                    # Exclude non-brain, spine, cervical, or thoracic protocols
                    if any(bad in text_lower for bad in ["tor1", "tor2", "torakal", "serv", "spine", "cerv", "coronal", "sagittal"]):
                        continue

                    matching_rows.append(ensure_axial_prompt(text))

                if matching_rows:
                    most_common = pd.Series(matching_rows).value_counts().index[0]
                    templates[mod_key] = ensure_axial_prompt(most_common)
        except Exception as e:
            print(f"[WARN] Failed reading CSV prompt templates: {e}")

    # Ensure all final templates are strictly axial
    for k in templates:
        templates[k] = ensure_axial_prompt(templates[k])

    return templates


def replace_imaging_param(prompt: str, te: float | None = None, tr: float | None = None,
                          ti: float | None = None, fa: float | None = None) -> str:
    m = re.search(r"Imaging Parameters \(Echo Time, Repetition Time, Inversion Time, Flip Angle\):\s*\(([^)]+)\)", prompt)
    if not m:
        return prompt

    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 4:
        return prompt

    old_te, old_tr, old_ti, old_fa = parts
    new_te = f"{te:.6g}" if te is not None else old_te
    new_tr = f"{tr:.6g}" if tr is not None else old_tr
    if ti is not None:
        new_ti = f"{ti:.6g}" if ti > 0 else "NONE"
    else:
        new_ti = old_ti
    new_fa = f"{fa:.6g}" if fa is not None else old_fa

    new_param_str = f"Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): ({new_te}, {new_tr}, {new_ti}, {new_fa})"
    return prompt[:m.start()] + new_param_str + prompt[m.end():]


def replace_scanner(prompt: str, vendor: str, model: str, field_strength: float) -> str:
    m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", prompt)
    if not m:
        return prompt

    new_scanner_str = f"Scanner (Manufacturer, Model, Field Strength): ({vendor}, {model}, {field_strength})"
    return prompt[:m.start()] + new_scanner_str + prompt[m.end():]


# ---------------------------------------------------------------------------
# EXPERIMENT 1: Continuous Physics Sweeps (TE, TR, TI, Flip Angle)
# ---------------------------------------------------------------------------

def run_experiment_1_physics_sweep(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, templates: dict[str, str], seed: int = 6256
):
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: Continuous Physics Parameter Sweeps")
    logger.info("=" * 60)

    exp_dir = out_dir / "exp1_physics_sweeps"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1A. TE Sweep
    base_t2_prompt = ensure_axial_prompt(templates.get("T2W", ""))
    te_values = [0.010, 0.025, 0.050, 0.080, 0.120]
    logger.info(f"Running 1A: TE Sweep ({[int(t*1000) for t in te_values]} ms)...")
    te_results = []
    for te in te_values:
        te_ms = int(te * 1000)
        prompt = ensure_axial_prompt(replace_imaging_param(base_t2_prompt, te=te))
        emb = encode_prompt(prompt, clip_model, tokenizer, device)
        vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, seed)
        ax_s, cor_s, sag_s = get_middle_slices(vol_np)
        te_results.append({"param": f"TE = {te_ms} ms", "axial": ax_s, "coronal": cor_s, "sagittal": sag_s})

    plot_sweep_grid(te_results, "Experiment 1A: Echo Time (TE) Sweep (Spin Echo, Fixed Anatomy)",
                    exp_dir / "1a_te_sweep_spin_echo.png")

    # 1B. TR Sweep
    base_t1_prompt = ensure_axial_prompt(templates.get("T1W", ""))
    tr_values = [0.400, 0.800, 1.500, 3.000, 6.000]
    logger.info(f"Running 1B: TR Sweep ({[int(t*1000) for t in tr_values]} ms)...")
    tr_results = []
    for tr in tr_values:
        tr_ms = int(tr * 1000)
        prompt = ensure_axial_prompt(replace_imaging_param(base_t1_prompt, tr=tr))
        emb = encode_prompt(prompt, clip_model, tokenizer, device)
        vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, seed)
        ax_s, cor_s, sag_s = get_middle_slices(vol_np)
        tr_results.append({"param": f"TR = {tr_ms} ms", "axial": ax_s, "coronal": cor_s, "sagittal": sag_s})

    plot_sweep_grid(tr_results, "Experiment 1B: Repetition Time (TR) Sweep (Fixed Anatomy)",
                    exp_dir / "1b_tr_sweep.png")

    # 1C. TI Sweep for FLAIR
    base_flair_prompt = ensure_axial_prompt(templates.get("FLAIR", ""))
    ti_values = [1.200, 1.600, 2.000, 2.400, 2.800]
    logger.info(f"Running 1C: TI Sweep for FLAIR ({[int(t*1000) for t in ti_values]} ms)...")
    ti_results = []
    for ti in ti_values:
        ti_ms = int(ti * 1000)
        prompt = ensure_axial_prompt(replace_imaging_param(base_flair_prompt, ti=ti))
        emb = encode_prompt(prompt, clip_model, tokenizer, device)
        vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, seed)
        ax_s, cor_s, sag_s = get_middle_slices(vol_np)
        ti_results.append({"param": f"TI = {ti_ms} ms", "axial": ax_s, "coronal": cor_s, "sagittal": sag_s})

    plot_sweep_grid(ti_results, "Experiment 1C: Inversion Time (TI) Sweep (FLAIR CSF Nulling, Fixed Anatomy)",
                    exp_dir / "1c_ti_flair_nulling_sweep.png")

    # 1D. Flip Angle Sweep
    base_gre_prompt = ensure_axial_prompt(templates.get("T2STARW", ""))
    fa_values = [5, 15, 30, 60, 90]
    logger.info(f"Running 1D: Flip Angle Sweep ({fa_values} deg)...")
    fa_results = []
    for fa in fa_values:
        prompt = ensure_axial_prompt(replace_imaging_param(base_gre_prompt, fa=float(fa)))
        emb = encode_prompt(prompt, clip_model, tokenizer, device)
        vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, seed)
        ax_s, cor_s, sag_s = get_middle_slices(vol_np)
        fa_results.append({"param": f"Flip Angle = {fa}°", "axial": ax_s, "coronal": cor_s, "sagittal": sag_s})

    plot_sweep_grid(fa_results, "Experiment 1D: Flip Angle Sweep (Gradient Echo, Fixed Anatomy)",
                    exp_dir / "1d_flip_angle_sweep.png")

    logger.info(f"Experiment 1 complete. Figures saved to: {exp_dir}")


def plot_sweep_grid(results: list[dict], title: str, out_path: Path):
    num_items = len(results)
    fig, axes = plt.subplots(num_items, 3, figsize=(10, 2.7 * num_items), dpi=180)

    for i, res in enumerate(results):
        axes[i, 0].imshow(res["axial"], cmap="gray")
        axes[i, 0].set_ylabel(res["param"], fontsize=10, fontweight="bold")
        axes[i, 1].imshow(res["coronal"], cmap="gray")
        axes[i, 2].imshow(res["sagittal"], cmap="gray")

        if i == 0:
            axes[i, 0].set_title("Axial View", fontsize=11, fontweight="bold")
            axes[i, 1].set_title("Coronal View", fontsize=11, fontweight="bold")
            axes[i, 2].set_title("Sagittal View", fontsize=11, fontweight="bold")

        for c in range(3):
            axes[i, c].set_xticks([])
            axes[i, c].set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXPERIMENT 2: Multi-Modal Registered Contrast Panel on Fixed Anatomy
# ---------------------------------------------------------------------------

def run_experiment_2_multimodal_panel(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, templates: dict[str, str], num_subjects: int = 3
):
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Multi-Modal Registered Contrast Synthesis")
    logger.info("=" * 60)

    exp_dir = out_dir / "exp2_multimodal_panel"
    exp_dir.mkdir(parents=True, exist_ok=True)

    modalities = [
        ("T1-Weighted (T1w)", ensure_axial_prompt(templates.get("T1W", ""))),
        ("T2-Weighted (T2w)", ensure_axial_prompt(templates.get("T2W", ""))),
        ("T2-FLAIR", ensure_axial_prompt(templates.get("FLAIR", ""))),
        ("Proton Density (PDw)", ensure_axial_prompt(templates.get("PDW", ""))),
        ("T2* Gradient Echo", ensure_axial_prompt(templates.get("T2STARW", ""))),
    ]

    verified_seeds = [6256, 99842, 100542, 220042, 38242]

    for subj_idx in range(1, num_subjects + 1):
        subj_seed = verified_seeds[(subj_idx - 1) % len(verified_seeds)]

        logger.info(f"Synthesizing 5-Modality Registered Panel for Subject #{subj_idx} (Seed: {subj_seed})...")
        subj_results = []

        for mod_name, mod_prompt in modalities:
            mod_prompt_axial = ensure_axial_prompt(mod_prompt)
            emb = encode_prompt(mod_prompt_axial, clip_model, tokenizer, device)
            vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, subj_seed)
            ax_s, cor_s, sag_s = get_middle_slices(vol_np)
            subj_results.append({
                "modality": mod_name,
                "axial": ax_s,
                "coronal": cor_s,
                "sagittal": sag_s,
            })

            nii_clean = mod_name.split()[0].replace("-", "_")
            nii_path = exp_dir / f"subject_{subj_idx:02d}_{nii_clean}.nii.gz"
            nib.save(nib.Nifti1Image(vol_np, np.eye(4)), str(nii_path))

        fig, axes = plt.subplots(3, 5, figsize=(15, 8.5), dpi=200)
        for col_idx, res in enumerate(subj_results):
            axes[0, col_idx].imshow(res["axial"], cmap="gray")
            axes[0, col_idx].set_title(res["modality"], fontsize=11, fontweight="bold")
            axes[1, col_idx].imshow(res["coronal"], cmap="gray")
            axes[2, col_idx].imshow(res["sagittal"], cmap="gray")

        axes[0, 0].set_ylabel("Axial View", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Coronal View", fontsize=11, fontweight="bold")
        axes[2, 0].set_ylabel("Sagittal View", fontsize=11, fontweight="bold")

        for r in range(3):
            for c in range(5):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])

        fig.suptitle(f"Subject #{subj_idx}: 5-Modality Registered Contrast Synthesis (Fixed Anatomy Seed: {subj_seed})",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        panel_path = exp_dir / f"subject_{subj_idx:02d}_5modality_registered_panel.png"
        fig.savefig(str(panel_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {panel_path}")

    logger.info(f"Experiment 2 complete. Figures and NIfTIs saved to: {exp_dir}")


def extract_hardware_configs_from_csv(csv_path: str | None) -> list[dict]:
    """Dynamically extract top scanner models & field strengths from dataset CSV."""
    configs = []
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            scanner_counts = {}
            for _, row in df.iterrows():
                text = str(row.get("text", ""))
                m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", text)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    if len(parts) >= 3:
                        vendor, model, field = parts[0], parts[1], parts[2]
                        try:
                            f_val = float(field.replace("T", ""))
                        except ValueError:
                            f_val = 3.0
                        key = (vendor, model, f_val)
                        scanner_counts[key] = scanner_counts.get(key, 0) + 1

            sorted_scanners = sorted(scanner_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for (vendor, model, f_val), _ in sorted_scanners:
                configs.append({
                    "scanner": f"{vendor} {model}",
                    "field": f"{f_val}T",
                    "vendor": vendor,
                    "model": model,
                    "field_val": f_val,
                })
        except Exception as e:
            print(f"[WARN] Failed extracting scanners from CSV: {e}")

    if not configs:
        configs = [
            {"scanner": "GE Signa HDxt", "field": "1.5T", "vendor": "GE", "model": "Signa_HDxt", "field_val": 1.5},
            {"scanner": "Siemens Avanto", "field": "1.5T", "vendor": "Siemens", "model": "Avanto", "field_val": 1.5},
            {"scanner": "Siemens Magnetom Trio", "field": "3.0T", "vendor": "Siemens", "model": "Magnetom_Trio", "field_val": 3.0},
            {"scanner": "Philips Achieva", "field": "3.0T", "vendor": "Philips", "model": "Achieva", "field_val": 3.0},
            {"scanner": "GE Discovery MR750", "field": "3.0T", "vendor": "GE", "model": "Discovery_MR750", "field_val": 3.0},
        ]
    return configs


# ---------------------------------------------------------------------------
# EXPERIMENT 3: Field Strength & Hardware Transfer
# ---------------------------------------------------------------------------

def run_experiment_3_hardware_field_strength(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, templates: dict[str, str], csv_path: str | None = None, seed: int = 6256
):
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Scanner Hardware & Field Strength Transfer")
    logger.info("=" * 60)

    exp_dir = out_dir / "exp3_hardware_field_strength"
    exp_dir.mkdir(parents=True, exist_ok=True)

    field_strength_configs = extract_hardware_configs_from_csv(csv_path)
    logger.info(f"Using Scanner Cohorts: {[f['scanner'] + ' (' + f['field'] + ')' for f in field_strength_configs]}")

    sequences = [
        ("T1w", ensure_axial_prompt(templates.get("T1W", ""))),
        ("T2w", ensure_axial_prompt(templates.get("T2W", ""))),
        ("FLAIR", ensure_axial_prompt(templates.get("FLAIR", ""))),
    ]

    logger.info(f"Generating Hardware & Field Strength Matrix across {len(sequences)} sequences x {len(field_strength_configs)} scanners...")

    for seq_name, base_prompt in sequences:
        results = []
        base_prompt_axial = ensure_axial_prompt(base_prompt)
        for hw in field_strength_configs:
            prompt = ensure_axial_prompt(replace_scanner(base_prompt_axial, vendor=hw["vendor"], model=hw["model"], field_strength=hw["field_val"]))
            emb = encode_prompt(prompt, clip_model, tokenizer, device)
            vol_np, _ = generate_sample(args, device, autoencoder, model, scale_factor, emb, target_dim, cfg_scale, seed)
            ax_s, cor_s, sag_s = get_middle_slices(vol_np)
            results.append({
                "label": f"{hw['scanner']} ({hw['field']})",
                "axial": ax_s,
                "coronal": cor_s,
                "sagittal": sag_s,
            })

        fig, axes = plt.subplots(len(results), 3, figsize=(10, 2.6 * len(results)), dpi=180)
        for i, res in enumerate(results):
            axes[i, 0].imshow(res["axial"], cmap="gray")
            axes[i, 0].set_ylabel(res["label"], fontsize=9, fontweight="bold")
            axes[i, 1].imshow(res["coronal"], cmap="gray")
            axes[i, 2].imshow(res["sagittal"], cmap="gray")

            if i == 0:
                axes[i, 0].set_title("Axial View", fontsize=11, fontweight="bold")
                axes[i, 1].set_title("Coronal View", fontsize=11, fontweight="bold")
                axes[i, 2].set_title("Sagittal View", fontsize=11, fontweight="bold")

            for c in range(3):
                axes[i, c].set_xticks([])
                axes[i, c].set_yticks([])

        fig.suptitle(f"Experiment 3: Hardware & Field Strength Transfer — {seq_name}\n(Identical Anatomy Seed: {seed})",
                     fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        out_fig = exp_dir / f"hardware_transfer_{seq_name}.png"
        fig.savefig(str(out_fig), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_fig}")

    logger.info(f"Experiment 3 complete. Figures saved to: {exp_dir}")


def extract_imaging_params(prompt: str) -> dict[str, float] | None:
    """Parse numerical Echo Time, Repetition Time, Inversion Time, Flip Angle from prompt string."""
    m = re.search(r"Imaging Parameters \(Echo Time, Repetition Time, Inversion Time, Flip Angle\):\s*\(([^)]+)\)", prompt)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 4:
        return None
    try:
        te = float(parts[0]) if parts[0].upper() != "NONE" else 0.05
        tr = float(parts[1]) if parts[1].upper() != "NONE" else 2.0
        ti = float(parts[2]) if parts[2].upper() != "NONE" else 0.0
        fa = float(parts[3]) if parts[3].upper() != "NONE" else 90.0
        return {"te": te, "tr": tr, "ti": ti, "fa": fa}
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# EXPERIMENT 4: Scanner Clustering & Manifold Analysis (Tightly Binned Physics)
# ---------------------------------------------------------------------------

def run_experiment_4_scanner_clustering(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, csv_path: str, data_root: str | None,
    n_per_scanner: int = 20
):
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 4: Scanner Hardware Clustering & Manifold Analysis (Tightly Binned)")
    logger.info("=" * 60)

    if not csv_path or not os.path.exists(csv_path):
        logger.error("Experiment 4 requires --csv path to load real scanner cohorts!")
        return

    exp_dir = out_dir / "exp4_scanner_clustering"
    exp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV: {csv_path}")

    df["scanner_id"] = df.apply(extract_scanner, axis=1)
    df["modality_id"] = df.apply(extract_modality, axis=1)

    # Note: Retain all rows from the CSV (including sagittal/coronal acquisitions like 3D T1w scans)
    # and enforce 'plane axial' dynamically on all prompts via ensure_axial_prompt().
    # This prevents discarding genuine T1w scans from the dataset.

    # Check for valid NIfTI files on disk (optional)
    valid_records = []
    has_real_files = False
    for idx, row in df.iterrows():
        raw_fp = str(row.get("filepath", ""))
        res_fp = resolve_filepath(raw_fp, data_root)
        if res_fp is not None:
            valid_records.append({**row.to_dict(), "resolved_path": res_fp, "csv_idx": idx})
            has_real_files = True
        else:
            valid_records.append({**row.to_dict(), "resolved_path": None, "csv_idx": idx})

    df_valid = pd.DataFrame(valid_records)
    if has_real_files:
        real_count = sum(1 for r in valid_records if r["resolved_path"] is not None)
        logger.info(f"Found {real_count} accessible real scans on disk. Real scan embeddings will be computed.")
    else:
        logger.info(f"No real NIfTI files found on disk (synthetic generation & manifold analysis mode). Using {len(df_valid)} metadata prompt rows.")
    logger.info(f"Modality breakdown in CSV: {dict(df_valid['modality_id'].value_counts())}")

    all_embeddings = []
    summary_records = []

    major_modalities = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in df_valid["modality_id"].values]
    if not major_modalities:
        major_modalities = list(df_valid["modality_id"].value_counts().index[:4])

    for mod in major_modalities:
        mod_df = df_valid[df_valid["modality_id"] == mod]
        if len(mod_df) == 0:
            continue

        if mod == "T1W":
            # Select genuine T1-weighted cases with true T1 relaxation (TR <= 0.75s)
            # to prevent long-TR TSE or mislabeled PDw scans from leaking into the T1W cohort
            t1_mask = mod_df["text"].apply(
                lambda t: (extract_imaging_params(str(t)) or {}).get("tr", 0.5) <= 0.75
            )
            if t1_mask.sum() >= 10:
                mod_df = mod_df[t1_mask]

            # Compute real empirical parameters directly from the CSV's T1W cases (no made-up numbers)
            t1_params = [extract_imaging_params(str(t)) for t in mod_df["text"].dropna()]
            t1_params = [p for p in t1_params if p is not None and p.get("tr", 0) <= 0.75]
            if t1_params:
                real_t1_te = float(np.median([p["te"] for p in t1_params]))
                real_t1_tr = float(np.median([p["tr"] for p in t1_params]))
                real_t1_ti = float(pd.Series([p["ti"] for p in t1_params]).mode().iloc[0])
                real_t1_fa = float(pd.Series([p["fa"] for p in t1_params]).mode().iloc[0])
            else:
                real_t1_te, real_t1_tr, real_t1_ti, real_t1_fa = 0.009, 0.450, 0.0, 90.0

            logger.info(
                f"\n============================================================"
                f"\n--- Modality [T1W]: Empirical CSV Parameters (TE={real_t1_te*1000:.2f}ms, TR={real_t1_tr*1000:.2f}ms, TI={real_t1_ti}, FA={real_t1_fa}°) ---"
                f"\n--- Preserving authentic CSV sequence types (no made-up sequences) ---"
                f"\n============================================================"
            )
        elif mod == "T2W":
            # Select canonical T2-weighted cases with true T2 relaxation (TE >= 0.07s, TR >= 2.5s)
            # to prevent dual-echo, intermediate TE, or gradient-echo variants from splitting the cohort
            t2_mask = mod_df["text"].apply(
                lambda t: (
                    (extract_imaging_params(str(t)) or {}).get("te", 0.1) >= 0.07 and
                    (extract_imaging_params(str(t)) or {}).get("tr", 4.0) >= 2.5
                )
            )
            if t2_mask.sum() >= 10:
                mod_df = mod_df[t2_mask]

            # Compute real empirical parameters directly from the CSV's T2W cases (no made-up numbers)
            t2_params = [extract_imaging_params(str(t)) for t in mod_df["text"].dropna()]
            t2_params = [p for p in t2_params if p is not None and p.get("te", 0) >= 0.07 and p.get("tr", 0) >= 2.5]
            if t2_params:
                real_t2_te = float(np.median([p["te"] for p in t2_params]))
                real_t2_tr = float(np.median([p["tr"] for p in t2_params]))
                real_t2_ti = float(pd.Series([p["ti"] for p in t2_params]).mode().iloc[0])
                real_t2_fa = float(pd.Series([p["fa"] for p in t2_params]).mode().iloc[0])
            else:
                real_t2_te, real_t2_tr, real_t2_ti, real_t2_fa = 0.100, 4.500, 0.0, 90.0

            logger.info(
                f"\n============================================================"
                f"\n--- Modality [T2W]: Empirical CSV Parameters (TE={real_t2_te*1000:.2f}ms, TR={real_t2_tr*1000:.2f}ms, TI={real_t2_ti}, FA={real_t2_fa}°) ---"
                f"\n--- Preserving authentic CSV sequence types (no made-up sequences) ---"
                f"\n============================================================"
            )
        else:
            logger.info(
                f"\n============================================================"
                f"\n--- Modality [{mod}]: Using Authentic CSV Prompts & Parameters Directly ---"
                f"\n============================================================"
            )

        scanner_counts = mod_df["scanner_id"].value_counts()
        top_scanners = [s for s in scanner_counts.index if scanner_counts[s] >= 3][:5]

        if len(top_scanners) < 2:
            top_scanners = list(scanner_counts.index[:2])
            if len(top_scanners) < 2:
                continue

        logger.info(f"--- Top Scanners for [{mod}]: {top_scanners} ---")
        mod_embeddings = []
        mod_visual_samples = {}

        for scanner_name in top_scanners:
            scanner_df = mod_df[mod_df["scanner_id"] == scanner_name].head(n_per_scanner)
            logger.info(f"  [{mod}] Scanner '{scanner_name}' ({len(scanner_df)} cases)...")
            mod_visual_samples[scanner_name] = []

            for c_i, (_, case) in enumerate(scanner_df.iterrows()):
                case_idx = case["csv_idx"]
                raw_prompt = str(case["text"])
                real_path = case.get("resolved_path", None)

                # For T1W and T2W, standardize to the real empirical CSV parameters
                # to prevent protocol splits while preserving the authentic CSV sequence type.
                # For FLAIR and PDW, use each scan's real CSV parameters directly without modification.
                if mod == "T1W":
                    gen_prompt = ensure_axial_prompt(
                        replace_imaging_param(
                            raw_prompt,
                            te=real_t1_te,
                            tr=real_t1_tr,
                            ti=real_t1_ti,
                            fa=real_t1_fa,
                        )
                    )
                elif mod == "T2W":
                    gen_prompt = ensure_axial_prompt(
                        replace_imaging_param(
                            raw_prompt,
                            te=real_t2_te,
                            tr=real_t2_tr,
                            ti=real_t2_ti,
                            fa=real_t2_fa,
                        )
                    )
                else:
                    gen_prompt = ensure_axial_prompt(raw_prompt)

                # 1. Text Prompt Embedding
                emb_text = encode_prompt(gen_prompt, clip_model, tokenizer, device)

                # 2. Synthetic Scan Embedding
                sample_seed = 7000 + case_idx * 11 + c_i * 3
                syn_np, syn_tensor = generate_sample(
                    args, device, autoencoder, model, scale_factor, emb_text, target_dim, cfg_scale, sample_seed
                )
                emb_syn = encode_volume_with_clip(syn_tensor, clip_model, device)

                # 3. Optional Real Scan Embedding (if available on disk)
                if real_path is not None and os.path.exists(real_path):
                    real_np, real_tensor = load_and_preprocess_real(real_path, target_dim)
                    emb_real = encode_volume_with_clip(real_tensor, clip_model, device)
                    sim_real_text = float(torch.sum(emb_real * emb_text).item())
                    sim_syn_real = float(torch.sum(emb_syn * emb_real).item())
                    item_real = {"feat": emb_real.squeeze().cpu().numpy(), "scanner": scanner_name, "type": "Real", "modality": mod}
                    mod_embeddings.append(item_real)
                    all_embeddings.append(item_real)
                else:
                    sim_real_text = None
                    sim_syn_real = None

                sim_syn_text = float(torch.sum(emb_syn * emb_text).item())

                # Save first 2 visual examples per scanner
                if len(mod_visual_samples[scanner_name]) < 2:
                    ax_s, cor_s, sag_s = get_middle_slices(syn_np)
                    mod_visual_samples[scanner_name].append({
                        "axial": ax_s,
                        "coronal": cor_s,
                        "sagittal": sag_s,
                        "seed": sample_seed,
                    })

                summary_records.append({
                    "modality": mod,
                    "scanner": scanner_name,
                    "case_idx": case_idx,
                    "sim_real_text": sim_real_text,
                    "sim_syn_text": sim_syn_text,
                    "sim_syn_real": sim_syn_real,
                })

                item_syn = {"feat": emb_syn.squeeze().cpu().numpy(), "scanner": scanner_name, "type": "Synthetic", "modality": mod}
                item_text = {"feat": emb_text.squeeze().cpu().numpy(), "scanner": scanner_name, "type": "Text", "modality": mod}

                mod_embeddings.extend([item_syn, item_text])
                all_embeddings.extend([item_syn, item_text])

        # Plot 2 visual examples from each scanner in a single grid for this modality
        plot_scanner_visual_panel(mod_visual_samples, mod, exp_dir, logger)

        # Plot Synthetic-Only Per-Modality Projections
        plot_scanner_projections(
            mod_embeddings, exp_dir, logger,
            title_suffix=f"within {mod} (Same Modality)",
            filename_suffix=f"_{mod}"
        )

    # Save quantitative CSV summary
    df_metrics = pd.DataFrame(summary_records)
    csv_out = exp_dir / "scanner_metrics_summary.csv"
    df_metrics.to_csv(csv_out, index=False)
    logger.info(f"Saved scanner metrics summary: {csv_out}")

    # Also save raw embeddings to pkl for fast re-plotting
    emb_pkl = exp_dir / "scanner_embeddings.pkl"
    try:
        import pickle
        with open(emb_pkl, "wb") as f:
            pickle.dump(all_embeddings, f)
        logger.info(f"Saved scanner embeddings to: {emb_pkl}")
    except Exception as e:
        logger.warning(f"Could not save scanner_embeddings.pkl: {e}")

    # Plot Overall Synthetic Projections across all modalities
    if len(all_embeddings) >= 6:
        plot_scanner_projections(
            all_embeddings, exp_dir, logger,
            title_suffix="Across All Modalities",
            filename_suffix="_all_modalities"
        )
    logger.info(f"Experiment 4 complete. Visualizations saved to: {exp_dir}")


def plot_scanner_visual_panel(visual_samples: dict[str, list[dict]], modality: str, exp_dir: Path, logger):
    """Plot 2 generated synthetic examples (Axial, Coronal, Sagittal) per scanner in a single comparison image."""
    scanners = [s for s, samples in visual_samples.items() if len(samples) > 0]
    num_scanners = len(scanners)
    if num_scanners == 0:
        return

    fig, axes = plt.subplots(num_scanners, 6, figsize=(18, 3.0 * num_scanners), dpi=200)
    if num_scanners == 1:
        axes = axes.reshape(1, -1)

    for r, scanner in enumerate(scanners):
        samples = visual_samples[scanner]

        # Sample 1
        if len(samples) >= 1:
            axes[r, 0].imshow(samples[0]["axial"], cmap="gray")
            axes[r, 1].imshow(samples[0]["coronal"], cmap="gray")
            axes[r, 2].imshow(samples[0]["sagittal"], cmap="gray")
        axes[r, 0].set_ylabel(scanner, fontsize=10, fontweight="bold")

        # Sample 2
        if len(samples) >= 2:
            axes[r, 3].imshow(samples[1]["axial"], cmap="gray")
            axes[r, 4].imshow(samples[1]["coronal"], cmap="gray")
            axes[r, 5].imshow(samples[1]["sagittal"], cmap="gray")

        if r == 0:
            axes[0, 0].set_title("Sample #1: Axial", fontsize=11, fontweight="bold")
            axes[0, 1].set_title("Sample #1: Coronal", fontsize=11, fontweight="bold")
            axes[0, 2].set_title("Sample #1: Sagittal", fontsize=11, fontweight="bold")
            axes[0, 3].set_title("Sample #2: Axial", fontsize=11, fontweight="bold")
            axes[0, 4].set_title("Sample #2: Coronal", fontsize=11, fontweight="bold")
            axes[0, 5].set_title("Sample #2: Sagittal", fontsize=11, fontweight="bold")

        for c in range(6):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(f"Scanner Visual Comparison — Modality: {modality} (2 Synthetic Samples per Scanner)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    panel_out = exp_dir / f"scanner_visual_samples_{modality}.png"
    fig.savefig(str(panel_out), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visual sample panel: {panel_out}")


def get_vendor_color_map(scanners: list[str]) -> dict[str, str]:
    """
    Assigns visually coherent, vendor-clustered and field-strength aware colors:
      - GE 3.0T Discovery MR750: Deep Crimson / Ruby Red (#b2182b)
      - GE 3.0T Discovery MR750w: Rich Coral Crimson (#d6604d)
      - GE 1.5T Discovery MR450: Rich Amber / Burnt Orange (#e6550d)
      - GE 1.5T Signa HDx/t: Warm Golden Amber (#feb24c)
      - GE 1.5T Signa other: Tangerine Orange (#fd8d3c)
      - Siemens 3.0T (MAGNETOM Vida, Trio): Midnight Navy Blue (#08306b)
      - Siemens 1.5T (Symphony): Deep Royal / Cobalt Blue (#08519c)
      - Siemens 1.5T (Aera): Vivid Ocean Blue (#2171b5)
      - Siemens 1.5T (Avanto): Crisp Sky Blue (#9ecae1)
      - Siemens 1.5T (MAGNETOM Sola): Steel Blue (#4292c6)
      - Philips 3.0T (Achieva dStream, Achieva): Deep Forest Green (#005a32, #006d2c)
      - Philips 1.5T (Ingenia, Intera): Fresh Mint / Sea Greens (#238b45, #74c476)
      - Others: Purples / Grays
    """
    color_map = {}
    for sc in sorted(scanners):
        sc_upper = sc.upper()
        is_3t = "3.0T" in sc_upper or "3T" in sc_upper

        if "GE" in sc_upper:
            # 3.0T GE Scanners: Crimson family
            if is_3t or "MR750" in sc_upper:
                if "MR750W" in sc_upper:
                    color_map[sc] = "#d6604d"  # Coral Crimson for 3.0T MR750w
                else:
                    color_map[sc] = "#b2182b"  # Deep Crimson / Ruby Red for 3.0T MR750
            # 1.5T Discovery MR450: Distinct Rich Amber-Orange
            elif "MR450" in sc_upper:
                color_map[sc] = "#e6550d"  # Rich Burnt Amber-Orange for 1.5T MR450
            # 1.5T Signa HDx/t: Warm Golden Amber
            elif "HDX" in sc_upper or "SIGNA" in sc_upper:
                color_map[sc] = "#feb24c"  # Warm Golden Amber for 1.5T Signa HDx/t
            else:
                color_map[sc] = "#fd8d3c"
        elif "SIEMENS" in sc_upper:
            if is_3t or "VIDA" in sc_upper or "TRIO" in sc_upper:
                color_map[sc] = "#081c3c"  # Midnight Prussian Navy for 3.0T
            elif "SYMPHONY" in sc_upper:
                color_map[sc] = "#0d47a1"  # Deep Royal / Indigo Cobalt Blue (Dark)
            elif "AERA" in sc_upper:
                color_map[sc] = "#0284c7"  # Vivid Electric Cerulean / Azure Blue (Medium)
            elif "AVANTO" in sc_upper:
                color_map[sc] = "#7dd3fc"  # Crisp Light Sky / Ice Blue (Light)
            elif "SOLA" in sc_upper:
                color_map[sc] = "#475569"  # Steel Slate Blue
            else:
                color_map[sc] = "#0284c7"
        elif "PHILIPS" in sc_upper:
            if is_3t or "DSTREAM" in sc_upper or "ACHIEVA" in sc_upper:
                color_map[sc] = "#005a32"  # Deep Emerald for 3.0T
            elif "INGENIA" in sc_upper:
                color_map[sc] = "#238b45"  # Medium Green for 1.5T
            elif "INTERA" in sc_upper:
                color_map[sc] = "#74c476"  # Mint Green for 1.5T
            else:
                color_map[sc] = "#2ca02c"
        elif "CANON" in sc_upper or "TOSHIBA" in sc_upper:
            color_map[sc] = "#54278f" if is_3t else "#756bb1"
        else:
            color_map[sc] = "#636363"
    return color_map


def plot_scanner_projections(all_embeddings: list[dict], exp_dir: Path, logger, title_suffix: str = "", filename_suffix: str = ""):
    """Plot t-SNE and PCA of Synthetic Embeddings strictly to evaluate scanner clustering."""
    synth_items = [item for item in all_embeddings if item["type"] == "Synthetic" and item.get("modality") != "T2STARW"]
    if len(synth_items) < 4:
        return

    X = np.array([item["feat"] for item in synth_items])
    scanners = [clean_scanner_label(item["scanner"]) for item in synth_items]
    modalities = [item.get("modality", "MRI") for item in synth_items]

    unique_scanners = sorted(list(set(scanners)))
    unique_modalities = sorted(list(set(modalities)))

    scanner_color_map = get_vendor_color_map(unique_scanners)
    modality_markers = {
        "T1W": "o",       # Circle
        "T2W": "s",       # Square
        "FLAIR": "^",     # Triangle Up
        "PDW": "D",       # Diamond
    }
    default_marker = "v"

    # Common Legend Elements
    scanner_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scanner_color_map[sc],
                   markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                   label=sc)
        for sc in unique_scanners
    ]
    modality_handles = [
        plt.Line2D([0], [0], marker=modality_markers.get(mod, default_marker), color="w",
                   markerfacecolor="#4d4d4d", markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                   label=mod)
        for mod in unique_modalities
    ]

    # 1. PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16.0, 9.0),
        dpi=220,
        gridspec_kw={"width_ratios": [3.8, 1.3]}
    )
    ax_leg.axis("off")

    for sc in unique_scanners:
        for mod in unique_modalities:
            idx = [i for i in range(len(synth_items)) if scanners[i] == sc and modalities[i] == mod]
            if idx:
                marker = modality_markers.get(mod, default_marker)
                ax.scatter(
                    X_pca[idx, 0], X_pca[idx, 1],
                    color=scanner_color_map[sc],
                    marker=marker,
                    s=85,
                    alpha=0.92,
                    edgecolors="black",
                    linewidth=0.6,
                )

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.025
    except Exception:
        leg2_top = max(0.05, 1.0 - (len(unique_scanners) * 0.040 + 0.08))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.0, leg2_top),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )

    pca_path = exp_dir / f"scanner_pca{filename_suffix}.png"
    fig.savefig(str(pca_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved: {pca_path}")

    # 2. t-SNE Projection
    perplexity = min(30, max(3, len(X) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16.0, 9.0),
        dpi=220,
        gridspec_kw={"width_ratios": [3.8, 1.3]}
    )
    ax_leg.axis("off")

    for sc in unique_scanners:
        for mod in unique_modalities:
            idx = [i for i in range(len(synth_items)) if scanners[i] == sc and modalities[i] == mod]
            if idx:
                marker = modality_markers.get(mod, default_marker)
                ax.scatter(
                    X_tsne[idx, 0], X_tsne[idx, 1],
                    color=scanner_color_map[sc],
                    marker=marker,
                    s=85,
                    alpha=0.92,
                    edgecolors="black",
                    linewidth=0.6,
                )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    leg1 = ax_leg.legend(
        handles=scanner_handles,
        title="Scanner Hardware",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax_leg.add_artist(leg1)

    fig.tight_layout()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox1 = leg1.get_window_extent(renderer).transformed(ax_leg.transAxes.inverted())
        leg2_top = bbox1.y0 - 0.025
    except Exception:
        leg2_top = max(0.05, 1.0 - (len(unique_scanners) * 0.040 + 0.08))

    leg2 = ax_leg.legend(
        handles=modality_handles,
        title="MRI Modality",
        loc="upper left",
        bbox_to_anchor=(0.0, leg2_top),
        fontsize=9.5,
        title_fontsize=11,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )

    tsne_path = exp_dir / f"scanner_tsne{filename_suffix}.png"
    fig.savefig(str(tsne_path), bbox_extra_artists=(leg1, leg2), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved: {tsne_path}")


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MR-CLIP Diffusion Model Validation Suite (Experiments 1, 2, 3, 4)")
    parser.add_argument("-e", "--env-config", required=True, help="Environment config JSON")
    parser.add_argument("-c", "--model-config", required=True, help="Model training config JSON")
    parser.add_argument("-t", "--model-def", required=True, help="Network definition config JSON")
    parser.add_argument("--mrclip-ckpt", required=True, help="Path to trained MR-CLIP UNet checkpoint (Phase 2)")
    parser.add_argument("--clip-checkpoint", required=True, help="Path to MR-CLIP vision/text checkpoint")
    parser.add_argument("--open-clip-src", required=True, help="Path to open_clip src directory")
    parser.add_argument("--csv", default=None, help="Path to test CSV to pull real scanner cohorts and prompt templates")
    parser.add_argument("--data-root", default=None, help="Root directory containing real NIfTI volumes")
    parser.add_argument("--experiments", nargs="+", default=["1", "2", "3", "4"], choices=["1", "2", "3", "4", "all"],
                        help="Experiments to run (1: Physics, 2: Multi-Modal, 3: Hardware, 4: Scanner t-SNE Clustering)")
    parser.add_argument("--n-per-scanner", type=int, default=20, help="Number of samples per scanner for Experiment 4 (default: 20)")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--output-dir", default="output/validation_experiments", help="Output directory")
    parser.add_argument("--seed", type=int, default=6256, help="Base seed for reproducibility")
    parser.add_argument("--clip-context-length", type=int, default=98)

    args_cli = parser.parse_args()
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("mrclip_val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device} | CFG Scale: {args_cli.cfg_scale}")

    # Load configs
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    target_dim = tuple(args.diffusion_unet_inference["dim"])

    # Load Autoencoder and UNet
    logger.info("Loading Autoencoder & MR-CLIP Diffusion UNet...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_path = args.trained_autoencoder_path
    if not os.path.exists(ae_path):
        local_ae = "models/autoencoder_v1.pt"
        if os.path.exists(local_ae):
            logger.info(f"Config path '{ae_path}' not found; using local '{local_ae}'.")
            ae_path = local_ae
    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=False)
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

    # Load MR-CLIP Text & Vision Encoder
    logger.info("Loading MR-CLIP Text & Vision Encoder...")
    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))
    clip_model.eval()

    # Extract authentic prompt templates
    templates = get_modality_templates_from_csv(args_cli.csv)

    exp_set = set(args_cli.experiments)
    if "all" in exp_set:
        exp_set = {"1", "2", "3", "4"}

    if "1" in exp_set:
        run_experiment_1_physics_sweep(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, templates, seed=args_cli.seed
        )

    if "2" in exp_set:
        run_experiment_2_multimodal_panel(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, templates, num_subjects=3
        )

    if "3" in exp_set:
        run_experiment_3_hardware_field_strength(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, templates, csv_path=args_cli.csv, seed=args_cli.seed
        )

    if "4" in exp_set:
        run_experiment_4_scanner_clustering(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, args_cli.cfg_scale, out_dir, logger, args_cli.csv, args_cli.data_root,
            n_per_scanner=args_cli.n_per_scanner
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"ALL EXPERIMENTS COMPLETE! Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
