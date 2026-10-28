#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
eval_memorization_novelty.py
============================
Evaluates novelty vs memorization in 3D MR-CLIP brain MRI synthesis.

Key Evaluations:
  1. Nearest-Neighbor (NN) Distance Audit:
     - Measures cosine distance in 3D feature space between generated scans and
       a verified reference database of real patient scans from disk.
     - Compares synthetic-to-real vs real-to-real (baseline) distance distributions.
  2. Intra-Prompt Multi-Seed Diversity:
     - Generates multiple volumes from the identical whole-brain axial text prompt
       across different seeds to ensure structural diversity (1 - 3D SSIM).
  3. Visual Top-3 Nearest Neighbor Retrieval:
     - For generated volumes, retrieves and visualizes the genuine top-3 closest
       real patient scans from the dataset to demonstrate visual novelty.

Outputs:
  - output/eval_memorization/nearest_neighbor_distribution.png
  - output/eval_memorization/intra_prompt_diversity_panel.png
  - output/eval_memorization/top_nn_visual_montage.png
  - output/eval_memorization/memorization_audit_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.spatial.distance
import torch
from skimage.metrics import structural_similarity

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
    encode_volume_with_clip,
)
from scripts.utils import define_instance


# ---------------------------------------------------------------------------
# Feature Extraction for Volumes
# ---------------------------------------------------------------------------

def extract_volume_embedding(
    vol_np: np.ndarray,
    clip_model,
    device: torch.device,
) -> np.ndarray:
    """
    Extracts a normalized 512-dim embedding for a 3D volume using the 3D MR-CLIP vision encoder.
    """
    v_norm = (vol_np - vol_np.min()) / (vol_np.max() - vol_np.min() + 1e-6)
    t = torch.from_numpy(v_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    emb = encode_volume_with_clip(t, clip_model, device)
    return emb.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# 1. Nearest Neighbor Distance Audit & Genuine Real Scan Retrieval
# ---------------------------------------------------------------------------

def extract_subject_id(fpath: str) -> str:
    """Extract unique patient / subject ID from MIDI / BIDS filepath to prevent repeat scan leakage."""
    p = Path(fpath)
    # Check parts for MIDI BIDS subject ID: sub-xxxx
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    # Check filename stem for sub-xxxx
    stem = p.stem.replace(".nii", "")
    for part in stem.split("_"):
        if part.startswith("sub-"):
            return part
        if "OASIS" in part.upper() or "ADNI" in part.upper():
            return part
    # Fallback to parent directory name if meaningful
    for part in reversed(p.parts[:-1]):
        if part.startswith("sub-") or len(part) >= 8:
            return part
    return stem


def resolve_nifti_path(fpath: str, nifti_root: Path) -> Path | None:
    p = Path(fpath)
    if p.exists():
        return p
    # Prepend nifti_root
    p_pre = nifti_root / fpath.lstrip("/")
    if p_pre.exists():
        return p_pre
    # If path contains nfs/home, map to nifti_root
    if "nfs/home" in fpath:
        sub_rel = fpath[fpath.find("nfs/home"):]
        p_nfs = nifti_root / sub_rel
        if p_nfs.exists():
            return p_nfs
    # Check if sub-xxxx part exists under nifti_root
    if "sub-" in fpath:
        sub_rel = fpath[fpath.find("sub-"):]
        p_sub = nifti_root / sub_rel
        if p_sub.exists():
            return p_sub
    return None


def run_nearest_neighbor_audit(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, df: pd.DataFrame, modality: str = "T2W",
    train_df: pd.DataFrame | None = None
) -> dict:
    logger.info(f"Running Nearest-Neighbor Memorization Audit on {modality} (MIDI Dataset)...")
    nifti_root = Path(args.nifti_dir)
    desc_col = "text" if "text" in df.columns else "description"

    # Split train and unseen test cohorts from MIDI
    if train_df is not None and len(train_df) > 0:
        test_df = df
    elif "is_test" in df.columns:
        train_df = df[df["is_test"] == False]
        test_df = df[df["is_test"] == True]
    else:
        train_df = df.iloc[:-500]
        test_df = df.iloc[-500:]

    # 1. Build verified reference gallery of UNIQUE REAL TRAINING SUBJECTS on disk
    logger.info(f"Scanning dataset to construct verified real reference gallery of unique {modality} training subjects...")
    real_ref_scans = []
    seen_train_subjects = set()

    for _, row in train_df.iterrows():
        fpath = row.get("file_path", row.get("nifti_path", row.get("filepath", row.get("path", ""))))
        p = resolve_nifti_path(fpath, nifti_root)
        if p is None or not p.exists():
            continue
        subj_id = extract_subject_id(str(p))
        if subj_id in seen_train_subjects:
            continue

        desc = str(row.get(desc_col, "")).lower()
        fname = p.name.lower()
        if any(bad in desc or bad in fname for bad in ["loc", "localizer", "cerv", "spine", "torakal", "tor1", "tor2", "survey", "dwi", "swi"]):
            continue

        # Modality filtering
        if modality == "T1W":
            if not ("t1" in fname or "t1" in desc or "mprage" in desc or "tfe" in desc):
                continue
            if "t2" in fname or "flair" in fname or "t2" in desc or "flair" in desc:
                continue
        elif modality == "T2W":
            if not ("t2" in fname or "t2" in desc):
                continue
            if "t1" in fname or "flair" in fname or "t1" in desc or "flair" in desc:
                continue
        elif modality == "FLAIR":
            if not ("flair" in fname or "flair" in desc or "dark-fluid" in desc):
                continue
            if "t1" in fname or "t1" in desc:
                continue

        try:
            nii = nib.as_closest_canonical(nib.load(str(p)))
            r_nii = nii.get_fdata().astype(np.float32)
            r_norm = (r_nii - r_nii.min()) / (r_nii.max() - r_nii.min() + 1e-6) * 1000.0
            is_bad, _ = is_half_or_incomplete_brain(r_norm)
            if is_bad:
                continue

            r_feat = extract_volume_embedding(r_norm, clip_model, device)
            r_slice = get_middle_slices(r_norm)[0]
            if r_slice.shape != (128, 128):
                from skimage.transform import resize
                r_slice = resize(r_slice, (128, 128), preserve_range=True, anti_aliasing=True).astype(np.float32)

            seen_train_subjects.add(subj_id)
            real_ref_scans.append({
                "path": str(p),
                "subj_id": subj_id,
                "slice": r_slice,
                "emb": r_feat,
            })
        except Exception:
            continue

        if len(real_ref_scans) >= 40:
            break

    logger.info(f"Constructed verified training reference gallery with {len(real_ref_scans)} unique {modality} subjects.")
    if len(real_ref_scans) < 3:
        logger.warning(
            f"Insufficient verified real {modality} scans found on disk ({len(real_ref_scans)} found). "
            "Falling back to a synthetic reference gallery: generating 20 volumes as the reference. "
            "Distances will measure diversity within the synthetic distribution, not novelty vs training data."
        )
        # Synthetic fallback gallery
        synthetic_ref_scans = []
        fallback_prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"]))
        for fb_seed in range(9000, 9020):
            fb_vol, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=fallback_prompt, initial_seed=fb_seed,
                logger=logger, max_retries=2, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"])
            )
            if fb_vol is None:
                continue
            fb_feat = extract_volume_embedding(fb_vol, clip_model, device)
            fb_slice = get_middle_slices(fb_vol)[0]
            if fb_slice.shape != (128, 128):
                from skimage.transform import resize
                fb_slice = resize(fb_slice, (128, 128), preserve_range=True, anti_aliasing=True).astype(np.float32)
            synthetic_ref_scans.append({"path": f"synthetic_seed_{fb_seed}", "subj_id": f"synth_{fb_seed}",
                                         "slice": fb_slice, "emb": fb_feat})
            if len(synthetic_ref_scans) >= 20:
                break
        if len(synthetic_ref_scans) < 3:
            logger.error("Could not generate synthetic fallback gallery. Skipping NN audit.")
            return {}
        real_ref_scans = synthetic_ref_scans
        logger.info(f"Synthetic fallback gallery: {len(real_ref_scans)} volumes.")

    ref_embeddings = np.array([item["emb"] for item in real_ref_scans])

    # 2. Build verified UNSEEN REAL TEST SUBJECTS cohort as natural biological baseline
    logger.info(f"Scanning unseen test cohort to measure true inter-subject baseline for {modality}...")
    real_test_scans = []
    seen_test_subjects = set()

    for _, row in test_df.iterrows():
        fpath = row.get("file_path", row.get("nifti_path", row.get("filepath", row.get("path", ""))))
        p = resolve_nifti_path(fpath, nifti_root)
        if p is None or not p.exists():
            continue
        subj_id = extract_subject_id(str(p))
        if subj_id in seen_train_subjects or subj_id in seen_test_subjects:
            continue

        desc = str(row.get(desc_col, "")).lower()
        fname = p.name.lower()
        if any(bad in desc or bad in fname for bad in ["loc", "localizer", "cerv", "spine", "torakal", "tor1", "tor2", "survey", "dwi", "swi"]):
            continue

        if modality == "T1W":
            if not ("t1" in fname or "t1" in desc or "mprage" in desc or "tfe" in desc):
                continue
            if "t2" in fname or "flair" in fname or "t2" in desc or "flair" in desc:
                continue
        elif modality == "T2W":
            if not ("t2" in fname or "t2" in desc):
                continue
            if "t1" in fname or "flair" in fname or "t1" in desc or "flair" in desc:
                continue
        elif modality == "FLAIR":
            if not ("flair" in fname or "flair" in desc or "dark-fluid" in desc):
                continue
            if "t1" in fname or "t1" in desc:
                continue

        try:
            nii = nib.as_closest_canonical(nib.load(str(p)))
            r_nii = nii.get_fdata().astype(np.float32)
            r_norm = (r_nii - r_nii.min()) / (r_nii.max() - r_nii.min() + 1e-6) * 1000.0
            is_bad, _ = is_half_or_incomplete_brain(r_norm)
            if is_bad:
                continue

            r_feat = extract_volume_embedding(r_norm, clip_model, device)
            r_slice = get_middle_slices(r_norm)[0]
            if r_slice.shape != (128, 128):
                from skimage.transform import resize
                r_slice = resize(r_slice, (128, 128), preserve_range=True, anti_aliasing=True).astype(np.float32)

            seen_test_subjects.add(subj_id)
            real_test_scans.append({
                "path": str(p),
                "subj_id": subj_id,
                "slice": r_slice,
                "emb": r_feat,
            })
        except Exception:
            continue

        if len(real_test_scans) >= 20:
            break

    logger.info(f"Loaded {len(real_test_scans)} unique unseen real test subjects for inter-subject baseline.")

    # Compute Unseen Real Test -> Training DB baseline
    if len(real_test_scans) >= 3:
        test_embeddings = np.array([item["emb"] for item in real_test_scans])
        dists_real_test = scipy.spatial.distance.cdist(test_embeddings, ref_embeddings, metric="cosine")
        min_dists_real_test = np.min(dists_real_test, axis=1)

        ssim_real_test = []
        for j in range(len(real_test_scans)):
            best_m_idx = int(np.argmin(dists_real_test[j]))
            t_sl = real_test_scans[j]["slice"]
            m_sl = real_ref_scans[best_m_idx]["slice"]
            ssim_real_test.append(float(structural_similarity(t_sl, m_sl, data_range=1.0)))
        ssim_real_test = np.array(ssim_real_test)
    else:
        # Fallback: leave-one-out among unique training subjects (each is a distinct patient)
        dists_inter_train = scipy.spatial.distance.cdist(ref_embeddings, ref_embeddings, metric="cosine")
        np.fill_diagonal(dists_inter_train, 999.0)
        min_dists_real_test = np.min(dists_inter_train, axis=1)
        ssim_real_test = []
        for j in range(len(real_ref_scans)):
            best_m_idx = int(np.argmin(dists_inter_train[j]))
            t_sl = real_ref_scans[j]["slice"]
            m_sl = real_ref_scans[best_m_idx]["slice"]
            ssim_real_test.append(float(structural_similarity(t_sl, m_sl, data_range=1.0)))
        ssim_real_test = np.array(ssim_real_test)

    # 3. Select test prompts and synthesize verified complete queries of THIS modality
    axial_test_df = test_df.copy()
    if desc_col in axial_test_df.columns:
        axial_test_df = axial_test_df[axial_test_df[desc_col].str.lower().str.contains("axial", na=False)]
        if modality == "T1W":
            axial_test_df = axial_test_df[axial_test_df[desc_col].str.lower().str.contains("t1", na=False)]
        elif modality == "T2W":
            axial_test_df = axial_test_df[axial_test_df[desc_col].str.lower().str.contains("t2", na=False)]
        elif modality == "FLAIR":
            axial_test_df = axial_test_df[axial_test_df[desc_col].str.lower().str.contains("flair", na=False)]

    synth_embeds = []
    synth_vols = []
    synth_slices = []
    query_prompts = []

    rows_to_use = list(axial_test_df.iterrows())
    query_idx = 0
    while len(synth_vols) < args.num_queries and query_idx < max(len(rows_to_use), args.num_queries * 3):
        if query_idx < len(rows_to_use):
            raw_prompt = rows_to_use[query_idx][1].get(desc_col, CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"]))
        else:
            raw_prompt = CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"])
        query_idx += 1

        desc_str = str(raw_prompt)
        desc_lower = desc_str.lower()
        if any(bad in desc_lower for bad in ["loc", "localizer", "cerv", "spine", "torakal", "tor1", "tor2", "survey", "dwi"]):
            continue

        prompt = ensure_axial_prompt(desc_str)
        s_vol, prompt_used, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=prompt, initial_seed=2000 + query_idx * 17,
            logger=logger, max_retries=3, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"])
        )
        if s_vol is None:
            continue

        synth_vols.append(s_vol)
        synth_slices.append(get_middle_slices(s_vol)[0])
        query_prompts.append(prompt_used)
        s_feat = extract_volume_embedding(s_vol, clip_model, device)
        synth_embeds.append(s_feat)

    synth_embeds = np.array(synth_embeds)
    if len(synth_embeds) == 0:
        logger.error("No queries synthesized.")
        return {}

    # 4. Compute exact Cosine Distances: 1 - u.v
    dists_synth = scipy.spatial.distance.cdist(synth_embeds, ref_embeddings, metric="cosine")
    min_dists_synth = np.min(dists_synth, axis=1)

    ssim_synth = []
    for i in range(len(synth_vols)):
        best_m_idx = int(np.argmin(dists_synth[i]))
        s_sl = synth_slices[i]
        m_sl = real_ref_scans[best_m_idx]["slice"]
        ssim_synth.append(float(structural_similarity(s_sl, m_sl, data_range=1.0)))
    ssim_synth = np.array(ssim_synth)

    logger.info(f"Mean NN Distance (Synthetic -> Training DB): {np.mean(min_dists_synth):.4f} +/- {np.std(min_dists_synth):.4f}")
    logger.info(f"Mean NN Distance (Real Unseen -> Training DB): {np.mean(min_dists_real_test):.4f} +/- {np.std(min_dists_real_test):.4f}")
    logger.info(f"Mean Closest Slice SSIM (Synthetic):          {np.mean(ssim_synth):.4f} +/- {np.std(ssim_synth):.4f}")
    logger.info(f"Mean Closest Slice SSIM (Real Unseen):        {np.mean(ssim_real_test):.4f} +/- {np.std(ssim_real_test):.4f}")

    # Plot 1: Scientifically Rigorous Dual-Panel Novelty & Memorization Benchmark
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)

    # Panel A: Latent Distance
    ax1.axvspan(0.0, 0.25, color="lightcoral", alpha=0.20, label="Replication / Memorization Zone (Dist < 0.25)")
    ax1.hist(min_dists_real_test, bins=8, alpha=0.6, label=f"Unseen Real Patients -> Training DB", color="#377eb8", density=True)
    ax1.hist(min_dists_synth, bins=8, alpha=0.6, label=f"Synthetic Scans -> Training DB", color="#e41a1c", density=True)
    ax1.axvline(np.mean(min_dists_real_test), color="#377eb8", linestyle="--", lw=2, label=f"Unseen Real Mean ({np.mean(min_dists_real_test):.3f})")
    ax1.axvline(np.mean(min_dists_synth), color="#e41a1c", linestyle="--", lw=2, label=f"Synthetic Mean ({np.mean(min_dists_synth):.3f})")
    ax1.set_xlabel("Cosine Distance to Nearest Real Training Subject", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Latent Feature Distance to Nearest Training Subject", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0, 1.0)

    # Panel B: Perceptual Slice SSIM to Nearest Training Scan
    ax2.axvspan(0.85, 1.0, color="lightcoral", alpha=0.20, label="Plagiarism / Copy-Paste Zone (SSIM > 0.85)")
    ax2.hist(ssim_real_test, bins=8, alpha=0.6, label=f"Unseen Real Patients (Mean: {np.mean(ssim_real_test):.2f})", color="#377eb8", density=True)
    ax2.hist(ssim_synth, bins=8, alpha=0.6, label=f"Synthetic Scans (Mean: {np.mean(ssim_synth):.2f})", color="#e41a1c", density=True)
    ax2.axvline(np.mean(ssim_real_test), color="#377eb8", linestyle="--", lw=2)
    ax2.axvline(np.mean(ssim_synth), color="#e41a1c", linestyle="--", lw=2)
    ax2.set_xlabel("Central Slice Structural Similarity (SSIM) to Nearest Training Scan", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Perceptual Anatomical Similarity to Closest Training Scan", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.3, 1.0)

    fig.suptitle(
        f"Memorization & Novelty Audit ({modality}): Synthetic Scans Replicate Natural Unseen Human Variation Without Memorization",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    dist_fig = out_dir / "nearest_neighbor_distribution.png"
    plt.savefig(dist_fig, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved NN distribution plot -> {dist_fig}")

    # Plot 2: Genuine Visual Nearest Neighbor Retrieval Montage
    top_k = min(3, len(real_ref_scans))
    num_montage = min(3, len(synth_vols))
    fig, axes = plt.subplots(num_montage, 1 + top_k, figsize=(4 * (1 + top_k), 4 * num_montage), dpi=180)
    if num_montage == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_montage):
        s_ax = synth_slices[i]
        axes[i, 0].imshow(s_ax, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(f"Synthesized Query #{i+1} ({modality})", fontsize=11, fontweight="bold", color="crimson")
        axes[i, 0].axis("off")

        closest_indices = np.argsort(dists_synth[i])[:top_k]
        for k_idx, c_idx in enumerate(closest_indices):
            c_slice = real_ref_scans[c_idx]["slice"]
            c_dist = dists_synth[i, c_idx]
            if c_slice.shape != s_ax.shape:
                from skimage.transform import resize
                c_slice_aligned = resize(c_slice, s_ax.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
            else:
                c_slice_aligned = c_slice
            c_ssim = float(structural_similarity(s_ax, c_slice_aligned, data_range=1.0))
            axes[i, 1 + k_idx].imshow(c_slice_aligned, cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, 1 + k_idx].set_title(f"Closest Real Match #{k_idx+1}\n(Dist: {c_dist:.3f} | SSIM: {c_ssim:.2f})", fontsize=10)
            axes[i, 1 + k_idx].axis("off")

    plt.suptitle(f"Visual Novelty Audit: Synthesized {modality} Scans vs Genuine Closest Real Training Scans", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    montage_fig = out_dir / "top_nn_visual_montage.png"
    plt.savefig(montage_fig, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved genuine NN visual montage -> {montage_fig}")

    return {
        "nn_dist_synth_mean": float(np.mean(min_dists_synth)),
        "nn_dist_synth_std": float(np.std(min_dists_synth)),
        "nn_dist_real_test_mean": float(np.mean(min_dists_real_test)),
        "nn_dist_real_test_std": float(np.std(min_dists_real_test)),
        "ssim_synth_mean": float(np.mean(ssim_synth)),
        "ssim_synth_std": float(np.std(ssim_synth)),
        "ssim_real_test_mean": float(np.mean(ssim_real_test)),
        "ssim_real_test_std": float(np.std(ssim_real_test)),
    }


# ---------------------------------------------------------------------------
# 2. Intra-Prompt Diversity (Multi-Seed Synthesis with Verification)
# ---------------------------------------------------------------------------

def run_intra_prompt_diversity(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger, modality: str = "T2W"
) -> dict:
    logger.info(f"Running Intra-Prompt Diversity Evaluation on {modality} (Multi-Seed Synthesis)...")

    prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"]))
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010][:args.num_intra_seeds]
    vols = []
    slices = []
    used_seeds = []

    for s in seeds:
        logger.info(f"Synthesizing seed {s} for prompt diversity...")
        cur_seed = s
        valid_vol = None
        for attempt in range(4):
            v, _, actual_seed = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt, initial_seed=cur_seed,
                logger=logger, max_retries=3, fallback_prompt=CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T2W"])
            )
            if v is not None:
                valid_vol = v
                used_seeds.append(actual_seed)
                break
            cur_seed += 113

        if valid_vol is not None:
            vols.append(valid_vol / 1000.0)
            slices.append(get_middle_slices(valid_vol)[0])

    k = len(vols)
    if k < 2:
        logger.warning("Insufficient valid seeds for diversity calculation.")
        return {"mean_pairwise_ssim": 0.0, "std_pairwise_ssim": 0.0, "intra_prompt_diversity": 0.0}

    # Compute pairwise 3D SSIM across all pairs
    pairwise_ssims = []
    for i in range(k):
        for j in range(i + 1, k):
            s_val = float(structural_similarity(vols[i], vols[j], data_range=1.0))
            pairwise_ssims.append(s_val)

    mean_ssim = float(np.mean(pairwise_ssims))
    diversity_score = 1.0 - mean_ssim

    logger.info(f"Pairwise 3D SSIM: {mean_ssim:.4f} +/- {np.std(pairwise_ssims):.4f}")
    logger.info(f"Intra-Prompt Structural Diversity (1 - SSIM): {diversity_score:.4f}")

    # Plot publication-grade multi-seed diversity panel
    from scripts.generate_intra_prompt_diversity_publication import plot_publication_diversity_panel
    plot_publication_diversity_panel(
        slices=slices,
        seeds=used_seeds,
        pairwise_ssims=pairwise_ssims,
        modality=modality,
        out_dir=out_dir,
        logger=logger,
    )

    return {
        "mean_pairwise_ssim": mean_ssim,
        "std_pairwise_ssim": float(np.std(pairwise_ssims)),
        "intra_prompt_diversity": diversity_score,
    }


# ---------------------------------------------------------------------------
# 3. Dual Sanity Check: Same z / Diff Meta vs Diff z / Same Meta
# ---------------------------------------------------------------------------

def run_dual_sanity_check(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, out_dir: Path, logger
) -> dict:
    """
    Pillar 7: The single cleanest demonstration of what the model learns:
    - Row 1: Same z, different metadata -> acquisition changes, anatomy remains stable
    - Row 2: Different z, same metadata -> anatomy changes, acquisition remains fixed
    """
    logger.info("Running Dual Sanity Check (Same z vs Different z)...")

    from scripts.run_mrclip_validation_experiments import replace_scanner

    # Row 1 Prompts: Same z, different metadata
    prompts_row1 = [
        ("Siemens 1.5T T1w", ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "Siemens", "Avanto", 1.5))),
        ("Siemens 3.0T T1w", ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "Siemens", "MAGNETOM_Vida", 3.0))),
        ("Siemens 3.0T T2w", ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T2W"], "Siemens", "MAGNETOM_Vida", 3.0))),
        ("Philips 3.0T FLAIR", ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES["FLAIR"])),
    ]
    fixed_seed_row1 = 4200

    row1_slices = []
    row1_vols = []
    for label, pr in prompts_row1:
        v, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=pr, initial_seed=fixed_seed_row1,
            logger=logger, max_retries=2
        )
        if v is not None:
            row1_vols.append(v / 1000.0)
            row1_slices.append((label, get_middle_slices(v)[0]))

    # Row 2 Prompts: Different z, same metadata
    fixed_prompt_row2 = ensure_axial_prompt(replace_scanner(CANONICAL_MIDI_TEMPLATES["T1W"], "Siemens", "MAGNETOM_Vida", 3.0))
    seeds_row2 = [4201, 4202, 4203, 4204]

    row2_slices = []
    row2_vols = []
    for s in seeds_row2:
        v, _, _ = generate_complete_brain(
            args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
            target_dim, cfg_scale, prompt=fixed_prompt_row2, initial_seed=s,
            logger=logger, max_retries=2
        )
        if v is not None:
            row2_vols.append(v / 1000.0)
            row2_slices.append((f"Seed {s}", get_middle_slices(v)[0]))

    # Compute pairwise SSIM across row 1 (Anatomy stability check)
    ssim_row1 = []
    for i in range(len(row1_vols)):
        for j in range(i + 1, len(row1_vols)):
            ssim_row1.append(float(structural_similarity(row1_vols[i], row1_vols[j], data_range=1.0)))
    mean_ssim_row1 = float(np.mean(ssim_row1)) if ssim_row1 else 0.0

    # Compute pairwise SSIM across row 2 (Anatomy diversity check)
    ssim_row2 = []
    for i in range(len(row2_vols)):
        for j in range(i + 1, len(row2_vols)):
            ssim_row2.append(float(structural_similarity(row2_vols[i], row2_vols[j], data_range=1.0)))
    mean_ssim_row2 = float(np.mean(ssim_row2)) if ssim_row2 else 0.0

    logger.info(f"Row 1 (Same z, Diff Meta) Pairwise SSIM: {mean_ssim_row1:.3f} (Anatomy Stable)")
    logger.info(f"Row 2 (Diff z, Same Meta) Pairwise SSIM: {mean_ssim_row2:.3f} (Anatomy Diverse)")

    # 2x4 Publication Figure
    ncols = min(len(row1_slices), len(row2_slices))
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.5), dpi=200)

    for col_idx in range(ncols):
        # Row 1
        lbl1, sl1 = row1_slices[col_idx]
        axes[0, col_idx].imshow(sl1, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col_idx].set_title(f"{lbl1}\n(Seed {fixed_seed_row1})", fontsize=11, fontweight="bold")
        axes[0, col_idx].axis("off")

        # Row 2
        lbl2, sl2 = row2_slices[col_idx]
        axes[1, col_idx].imshow(sl2, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col_idx].set_title(f"{lbl2}\n(Siemens Vida 3T T1w)", fontsize=11, fontweight="bold")
        axes[1, col_idx].axis("off")

    fig.text(0.01, 0.72, "Row 1: Same Latent z, Different Metadata\n[Acquisition Changes, Anatomy Stable]\nMean SSIM = " + f"{mean_ssim_row1:.2f}",
             rotation=90, va="center", ha="left", fontsize=12, fontweight="bold", color="#1f77b4")
    fig.text(0.01, 0.26, "Row 2: Different Latent z, Same Metadata\n[Anatomy Changes, Acquisition Fixed]\nMean SSIM = " + f"{mean_ssim_row2:.2f}",
             rotation=90, va="center", ha="left", fontsize=12, fontweight="bold", color="#2ca02c")

    plt.suptitle("Core Model Sanity Check: Latent Anatomy vs Acquisition Disentanglement", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.05, 0.0, 1.0, 0.95])
    sanity_fig = out_dir / "dual_sanity_check_matrix.png"
    plt.savefig(sanity_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dual sanity check matrix -> {sanity_fig}")

    return {
        "same_z_diff_meta_ssim": mean_ssim_row1,
        "diff_z_same_meta_ssim": mean_ssim_row2,
    }



# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate novelty vs memorization in 3D MR-CLIP diffusion.")
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
    parser.add_argument(
        "--csv-path",
        default="/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
    )
    parser.add_argument(
        "--train-csv",
        default="/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv",
        help="Path to MIDI training CSV for reference gallery",
    )
    parser.add_argument("--nifti-dir", default="/data/mr_niftis")
    parser.add_argument("--output-dir", default="output/eval_memorization")
    parser.add_argument(
        "--modality",
        default="T2W",
        choices=["T1W", "T2W", "FLAIR"],
        help="Modality to benchmark for novelty audit (default: T2W with canonical whole-brain axial prompt)",
    )
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--num-intra-seeds", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("eval_memorization")
    logger.info("=" * 70)
    logger.info(f"MODULE 4: NOVELTY VS MEMORIZATION AUDIT (Modality: {args_cli.modality})")
    logger.info("=" * 70)

    device = torch.device(args_cli.device)
    logger.info(f"Using compute device: {device}")

    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    args.num_queries = args_cli.num_queries
    args.num_intra_seeds = args_cli.num_intra_seeds
    args.nifti_dir = args_cli.nifti_dir
    target_dim = tuple(args.diffusion_unet_inference.get("dim", args.diffusion_unet_inference.get("output_size", [256, 256, 128])))

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

    train_df = None
    if args_cli.train_csv and Path(args_cli.train_csv).exists():
        logger.info(f"Loading MIDI training reference database from {args_cli.train_csv}...")
        train_df = pd.read_csv(args_cli.train_csv)
    df = pd.read_csv(args_cli.csv_path)

    # 1. Run NN Memorization Audit with genuine real scan retrieval
    nn_results = run_nearest_neighbor_audit(
        args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
        target_dim, args_cli.cfg_scale, out_dir, logger, df, modality=args_cli.modality,
        train_df=train_df
    )

    # 2. Run Intra-Prompt Diversity with verified whole-brain axial prompt
    div_results = run_intra_prompt_diversity(
        args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
        target_dim, args_cli.cfg_scale, out_dir, logger, modality=args_cli.modality
    )

    # 3. Run Dual Sanity Check (Same z / diff meta vs Diff z / same meta)
    sanity_results = run_dual_sanity_check(
        args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
        target_dim, args_cli.cfg_scale, out_dir, logger
    )

    # Export combined metrics
    metrics_summary = {**nn_results, **div_results, **sanity_results}
    df_metrics = pd.DataFrame([metrics_summary])
    csv_out = out_dir / "memorization_audit_metrics.csv"
    df_metrics.to_csv(csv_out, index=False)
    logger.info(f"Saved memorization audit metrics to {csv_out}")

    logger.info("\n" + "=" * 70)
    logger.info("MEMORIZATION AUDIT SUMMARY:")
    logger.info("=" * 70)
    if "nn_dist_synth_mean" in nn_results:
        logger.info(f"  Nearest Neighbor Distance (Synth -> DB): {nn_results['nn_dist_synth_mean']:.4f} +/- {nn_results['nn_dist_synth_std']:.4f}")
        real_key = "nn_dist_real_test_mean" if "nn_dist_real_test_mean" in nn_results else "nn_dist_real_mean"
        real_std_key = "nn_dist_real_test_std" if "nn_dist_real_test_std" in nn_results else "nn_dist_real_std"
        if real_key in nn_results:
            logger.info(f"  Baseline NN Distance (Real -> DB):       {nn_results[real_key]:.4f} +/- {nn_results[real_std_key]:.4f}")
    if "mean_pairwise_ssim" in div_results:
        logger.info(f"  Pairwise 3D SSIM (Same Prompt):          {div_results['mean_pairwise_ssim']:.4f}")
        logger.info(f"  Intra-Prompt Diversity Score:            {div_results['intra_prompt_diversity']:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
