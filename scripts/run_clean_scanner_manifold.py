#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Clean Scanner Manifold & Clustering Suite (Experiment 4 Clean)
=============================================================
Provides a clean, publication-grade scanner manifold analysis with:
1. Standardized clinical physics midpoints per modality (derived from real CSV medians)
   to isolate scanner hardware from physics parameter noise.
2. Robust cohort pre-filtering to prevent mislabeled / hybrid outliers
   (e.g., non-inverted FLAIR with TI=0, or short-TR PDW with TR < 2.0s).
3. Modality-specific marker shapes and vendor-clustered color families.
4. Separate clean output directory preserving all original experiment outputs.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 -m scripts.run_clean_scanner_manifold \
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \
      -t configs/config_network_rflow.json \
      --mrclip-ckpt models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt \
      --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \
      --data-root /data/mr_niftis \
      --clip-checkpoint /home/mav24/projects/MR-Styler/model/contrast_clip/2025-06-03-21-29-21/epoch_125.pt \
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \
      --n-per-scanner 20 \
      --cfg-scale 4.0 \
      --output-dir output/validation_experiments/exp4_scanner_clustering_clean
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of execution method
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism

try:
    from scripts.run_mrclip_validation_experiments import (
        clean_scanner_label,
        ensure_axial_prompt,
        extract_imaging_params,
        extract_modality,
        extract_scanner,
        get_vendor_color_map,
        plot_scanner_projections,
        plot_scanner_visual_panel,
        replace_imaging_param,
        resolve_filepath,
        load_and_preprocess_real,
        generate_sample,
        encode_prompt,
        encode_volume_with_clip,
        get_middle_slices,
    )
except ModuleNotFoundError:
    from run_mrclip_validation_experiments import (
        clean_scanner_label,
        ensure_axial_prompt,
        extract_imaging_params,
        extract_modality,
        extract_scanner,
        get_vendor_color_map,
        plot_scanner_projections,
        plot_scanner_visual_panel,
        replace_imaging_param,
        resolve_filepath,
        load_and_preprocess_real,
        generate_sample,
        encode_prompt,
        encode_volume_with_clip,
        get_middle_slices,
    )

# Standardized clinical physics benchmarks derived from real empirical medians in the dataset
CLEAN_PHYSICS_BENCHMARKS = {
    "T1W": {
        "target": {"te": 0.009, "tr": 0.450, "ti": 0.0, "fa": 90.0},
        "filter": lambda p: p.get("tr", 0.5) <= 0.85,
        "desc": "Empirical T1W (TE=9ms, TR=450ms, TI=0, FA=90°)",
    },
    "T2W": {
        "target": {"te": 0.100, "tr": 4.800, "ti": 0.0, "fa": 90.0},
        "filter": lambda p: p.get("te", 0.1) >= 0.07 and p.get("tr", 4.0) >= 2.5,
        "desc": "Empirical T2W (TE=100ms, TR=4800ms, TI=0, FA=90°)",
    },
    "FLAIR": {
        "target": {"te": 0.120, "tr": 8.500, "ti": 2.200, "fa": 90.0},
        "filter": lambda p: p.get("ti", 2.0) >= 1.2,  # Must have genuine inversion pulse
        "desc": "Empirical FLAIR (TE=120ms, TR=8500ms, TI=2200ms, FA=90°)",
    },
    "PDW": {
        "target": {"te": 0.015, "tr": 3.200, "ti": 0.0, "fa": 90.0},
        "filter": lambda p: p.get("tr", 3.0) >= 2.2 and p.get("te", 0.015) <= 0.040,  # Pure long-TR PD
        "desc": "Empirical PDW (TE=15ms, TR=3200ms, TI=0, FA=90°)",
    },
}


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("clean_manifold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)5s](clean_manifold) - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "clean_manifold.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run_clean_manifold(args):
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    logger.info("=" * 60)
    logger.info("Clean Scanner Manifold & Clustering Suite")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Output Directory: {out_dir}")
    logger.info("=" * 60)

    # 1. Load CLIP Model
    sys.path.insert(0, args.open_clip_src)
    import open_clip

    logger.info(f"Loading MR-CLIP from {args.clip_checkpoint}...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=args.clip_checkpoint,
        device=device,
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 2. Load Diffusion Model & Autoencoder
    from monai.bundle import ConfigParser
    from scripts.utils import define_instance

    logger.info("Loading MAISI Diffusion Model & Autoencoder...")
    env_dict = ConfigParser(config_file=args.environment_file)
    net_dict = ConfigParser(config_file=args.network_config)
    cfg_dict = ConfigParser(config_file=args.config_file)

    autoencoder = define_instance(net_dict, "autoencoder_def").to(device)
    diff_unet = define_instance(net_dict, "diffusion_unet_def").to(device)

    ae_ckpt = env_dict["autoencoder_checkpoint"]
    diff_ckpt = args.mrclip_ckpt or env_dict["diff_model_checkpoint"]

    logger.info(f"  Loading autoencoder weights: {ae_ckpt}")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()

    logger.info(f"  Loading diffusion UNet weights: {diff_ckpt}")
    diff_unet.load_state_dict(torch.load(diff_ckpt, map_location=device))
    diff_unet.eval()

    scale_factor = cfg_dict.get("scale_factor", 1.0)
    target_dim = tuple(cfg_dict.get("spatial_shape", [256, 256, 128]))

    # 3. Load & Pre-filter CSV Cohorts
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} rows from CSV: {args.csv}")

    df["scanner_id"] = df.apply(extract_scanner, axis=1)
    df["modality_id"] = df.apply(extract_modality, axis=1)

    valid_records = []
    has_real_files = False
    for idx, row in df.iterrows():
        raw_fp = str(row.get("filepath", ""))
        res_fp = resolve_filepath(raw_fp, args.data_root)
        if res_fp is not None:
            valid_records.append({**row.to_dict(), "resolved_path": res_fp, "csv_idx": idx})
            has_real_files = True
        else:
            valid_records.append({**row.to_dict(), "resolved_path": None, "csv_idx": idx})

    df_valid = pd.DataFrame(valid_records)
    logger.info(f"Accessible real scans on disk: {sum(1 for r in valid_records if r['resolved_path'] is not None)}")
    logger.info(f"Modality breakdown in CSV: {dict(df_valid['modality_id'].value_counts())}")

    all_embeddings = []
    summary_records = []

    major_modalities = [m for m in ["T1W", "T2W", "FLAIR", "PDW"] if m in df_valid["modality_id"].values]
    n_per_scanner = args.n_per_scanner

    for mod in major_modalities:
        bench = CLEAN_PHYSICS_BENCHMARKS.get(mod)
        if not bench:
            continue

        mod_df = df_valid[df_valid["modality_id"] == mod]
        if len(mod_df) == 0:
            continue

        # Apply clean physics gate to eliminate corrupted / mislabeled / hybrid outliers
        filter_fn = bench["filter"]
        mod_mask = mod_df["text"].apply(lambda t: filter_fn(extract_imaging_params(str(t)) or {}))
        if mod_mask.sum() >= 10:
            mod_df = mod_df[mod_mask]

        target_p = bench["target"]
        logger.info(
            f"\n============================================================"
            f"\n--- Modality [{mod}]: {bench['desc']} ---"
            f"\n--- Target Physics: TE={target_p['te']*1000:.1f}ms, TR={target_p['tr']*1000:.1f}ms, TI={target_p['ti']*1000:.1f}ms, FA={target_p['fa']}° ---"
            f"\n--- Preserving authentic CSV sequence names & hardware labels ---"
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

                # Standardize prompt physics to the clean benchmark while keeping authentic sequence description
                gen_prompt = ensure_axial_prompt(
                    replace_imaging_param(
                        raw_prompt,
                        te=target_p["te"],
                        tr=target_p["tr"],
                        ti=target_p["ti"],
                        fa=target_p["fa"],
                    )
                )

                # 1. Text Prompt Embedding
                emb_text = encode_prompt(gen_prompt, clip_model, tokenizer, device)

                # 2. Synthetic Scan Embedding
                sample_seed = 7000 + case_idx * 11 + c_i * 3
                syn_np, syn_tensor = generate_sample(
                    args, device, autoencoder, diff_unet, scale_factor, emb_text, target_dim, args.cfg_scale, sample_seed
                )
                emb_syn = encode_volume_with_clip(syn_tensor, clip_model, device)

                # 3. Real Scan Embedding (optional if on disk)
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

        # Plot 2 visual examples per scanner
        plot_scanner_visual_panel(mod_visual_samples, mod, out_dir, logger)

        # Plot Per-Modality Projections
        plot_scanner_projections(
            mod_embeddings, out_dir, logger,
            title_suffix=f"within {mod} (Same Modality)",
            filename_suffix=f"_{mod}"
        )

    # Save summary CSV
    df_metrics = pd.DataFrame(summary_records)
    csv_out = out_dir / "scanner_metrics_summary.csv"
    df_metrics.to_csv(csv_out, index=False)
    logger.info(f"Saved scanner metrics summary: {csv_out}")

    # Save clean embeddings pkl
    pkl_out = out_dir / "scanner_embeddings.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(all_embeddings, f)
    logger.info(f"Saved clean embeddings to: {pkl_out}")

    # Plot Overall Clean Manifold Projections
    plot_scanner_projections(
        all_embeddings, out_dir, logger,
        title_suffix="Across All Modalities (Clean)",
        filename_suffix="_all_modalities"
    )

    logger.info(f"\n[DONE] Clean scanner clustering complete! Figures saved to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Scanner Manifold & Clustering Suite")
    parser.add_argument("-e", "--environment-file", required=True, type=str)
    parser.add_argument("-c", "--config-file", required=True, type=str)
    parser.add_argument("-t", "--network-config", required=True, type=str)
    parser.add_argument("--mrclip-ckpt", type=str, default=None)
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--clip-checkpoint", required=True, type=str)
    parser.add_argument("--open-clip-src", required=True, type=str)
    parser.add_argument("--n-per-scanner", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--output-dir", type=str, default="output/validation_experiments/exp4_scanner_clustering_clean")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clean_manifold(args)
