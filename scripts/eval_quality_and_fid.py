"""
scripts/eval_quality_and_fid.py
===============================
Pillar 6: Realism in 3D MR-CLIP Feature Space

Evaluates Fréchet Distance (FID) in 3D MR-CLIP embedding space:
- Uses domain-native 3D MR-CLIP visual encoder (ViT-B-16 3D, 512-dim), NOT 2D ImageNet features
- Minimum sample threshold: >= 100 samples per condition
- Computes bootstrap 95% confidence intervals (B=1000 resamples)
- Sharpness, entropy, and total variation are placed in supplementary diagnostics
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.ndimage import laplace
import torch
import torch.nn.functional as F

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
)
from scripts.utils import define_instance


def compute_mrclip_fid(feats_real: np.ndarray, feats_synth: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet Distance between real and synthetic feature distributions:
    d^2 = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r * Sigma_g)^(1/2))
    """
    mu_r = np.mean(feats_real, axis=0)
    mu_s = np.mean(feats_synth, axis=0)

    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_s = np.cov(feats_synth, rowvar=False)

    diff = mu_r - mu_s
    diff_norm = np.dot(diff, diff)

    covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_s), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_r + offset).dot(sigma_s + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff_norm + np.trace(sigma_r) + np.trace(sigma_s) - 2.0 * tr_covmean)
    return max(0.0, fid)


def bootstrap_fid_ci(
    feats_real: np.ndarray,
    feats_synth: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap mean and 95% confidence interval for MR-CLIP FID."""
    n_r, n_s = len(feats_real), len(feats_synth)
    boot_fids = []

    for _ in range(n_bootstraps):
        idx_r = np.random.choice(n_r, size=n_r, replace=True)
        idx_s = np.random.choice(n_s, size=n_s, replace=True)
        try:
            bf = compute_mrclip_fid(feats_real[idx_r], feats_synth[idx_s])
            boot_fids.append(bf)
        except Exception:
            continue

    if not boot_fids:
        raw_fid = compute_mrclip_fid(feats_real, feats_synth)
        return raw_fid, raw_fid, raw_fid

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_fids, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_fids, 100.0 * (1.0 - alpha)))
    mean_fid = float(np.mean(boot_fids))
    return mean_fid, ci_lower, ci_upper


def extract_3d_mrclip_embedding(vol_np: np.ndarray, clip_model: Any, device: str) -> np.ndarray:
    """Extract 512-dim visual embedding from 3D volume using MR-CLIP."""
    vol_th = torch.from_numpy(vol_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    v_min, v_max = float(np.percentile(vol_np, 1.0)), float(np.percentile(vol_np, 99.0))
    if v_max > v_min:
        vol_th = (vol_th - v_min) / (v_max - v_min)
        vol_th = torch.clamp(vol_th, 0.0, 1.0)
    if vol_th.shape[2:] != (128, 128, 128):
        vol_th = F.interpolate(vol_th, size=(128, 128, 128), mode="trilinear", align_corners=False)
    vol_th = vol_th.to(device)
    with torch.no_grad():
        if hasattr(clip_model, "encode_image"):
            feat = clip_model.encode_image(vol_th)
        else:
            feat = clip_model.visual(vol_th)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


def compute_supplementary_diagnostics(vol_np: np.ndarray) -> dict[str, float]:
    """Supplementary image metrics (Laplacian sharpness, entropy, total variation)."""
    v = vol_np.astype(np.float32)
    p1, p99 = float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))
    if p99 > p1:
        v_norm = (v - p1) / (p99 - p1)
    else:
        v_norm = v / (v.max() + 1e-4)

    # Sharpness: variance of Laplacian on middle axial slice
    mid_s = v_norm[:, :, v_norm.shape[2] // 2]
    lap = laplace(mid_s)
    sharpness = float(np.var(lap))

    # Total variation
    diff_x = np.abs(np.diff(mid_s, axis=0))
    diff_y = np.abs(np.diff(mid_s, axis=1))
    tv = float(np.mean(diff_x) + np.mean(diff_y))

    # Shannon entropy
    counts, _ = np.histogram(mid_s[mid_s > 0.05], bins=64)
    probs = counts / (counts.sum() + 1e-6)
    entropy = float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    return {"sharpness": sharpness, "total_variation": tv, "entropy": entropy}


def load_or_create_real_mrclip_embeddings(
    cache_path: str | Path,
    clip_model: Any,
    device: str,
    logger: logging.Logger,
    target_count: int = 100,
) -> np.ndarray:
    """
    Loads or extracts genuine 3D MR-CLIP visual embeddings (512-dim) for real MIDI MRI scans.
    Caches results to disk for fast subsequent evaluations.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        logger.info(f"Loading real reference MR-CLIP visual embeddings from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "embeddings" in data:
                embs = np.array(data["embeddings"])
                if embs.ndim == 2 and embs.shape[1] == 512:
                    logger.info(f"Successfully loaded {len(embs)} real visual embeddings from cache.")
                    return embs
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 512:
                logger.info(f"Successfully loaded {len(data)} real visual embeddings from cache.")
                return data
        except Exception as e:
            logger.warning(f"Cache read error: {e}. Re-extracting from raw scans...")

    logger.info(f"Extracting 3D MR-CLIP visual embeddings from verified real MIDI scans (target: {target_count})...")
    candidate_csvs = [
        "/data/axial_train.csv",
        "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids.csv",
        "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv",
    ]
    real_paths: list[Path] = []
    for csv_p in candidate_csvs:
        if os.path.exists(csv_p):
            try:
                df = pd.read_csv(csv_p)
                for col in ["filepath", "file_path", "path"]:
                    if col in df.columns:
                        for fp in df[col].dropna():
                            p = Path(str(fp))
                            if p.exists():
                                real_paths.append(p)
                            elif (Path("/data") / p.name).exists():
                                real_paths.append(Path("/data") / p.name)
                            elif (Path("/data/mr_niftis") / p.name).exists():
                                real_paths.append(Path("/data/mr_niftis") / p.name)
                            if len(real_paths) >= target_count * 2:
                                break
                        break
            except Exception:
                pass
        if len(real_paths) >= target_count * 2:
            break

    if len(real_paths) < target_count:
        globbed = glob.glob("/data/ur_sub-*.nii") + glob.glob("/data/**/*.nii.gz", recursive=True)
        for gp in globbed:
            real_paths.append(Path(gp))
            if len(real_paths) >= target_count * 2:
                break

    real_paths = list(dict.fromkeys(real_paths))
    logger.info(f"Identified {len(real_paths)} candidate real scan files on disk.")

    real_embs = []
    for p in real_paths:
        try:
            nii = nib.load(str(p))
            vol = nii.get_fdata().astype(np.float32)
            if vol.ndim != 3 or vol.shape[0] < 32 or vol.shape[1] < 32 or vol.shape[2] < 16:
                continue
            emb = extract_3d_mrclip_embedding(vol, clip_model, device)
            real_embs.append(emb)
            if len(real_embs) >= target_count:
                break
        except Exception:
            continue

    if not real_embs:
        raise RuntimeError("Could not find or load any real MRI scans to construct MR-CLIP visual FID baseline.")

    feats_real = np.array(real_embs)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump({"embeddings": feats_real}, f)
    logger.info(f"Saved {len(feats_real)} real MR-CLIP visual embeddings to: {cache_file}")
    return feats_real


def run_realism_evaluation(
    args,
    device: str,
    autoencoder: Any,
    model: Any,
    scale_factor: float,
    clip_model: Any,
    tokenizer: Any,
    target_dim: tuple[int, int, int],
    cfg_scale: float,
    embeddings_pkl: str,
    out_dir: Path,
    num_samples: int = 100,
    logger: logging.Logger | None = None,
):
    if logger is None:
        logger = logging.getLogger("eval_realism")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"STARTING 3D MR-CLIP FEATURE SPACE REALISM EVALUATION (N={num_samples})")
    logger.info("=" * 70)

    # Load real test embeddings baseline
    feats_real = load_or_create_real_mrclip_embeddings(
        cache_path=embeddings_pkl,
        clip_model=clip_model,
        device=device,
        logger=logger,
        target_count=max(num_samples, 50),
    )

    conditions = ["T1W", "T2W", "FLAIR"]
    fid_results = {}
    supp_diagnostics = {c: [] for c in conditions}

    base_seed = 54321

    for cond in conditions:
        logger.info(f"\n--- Generating N={num_samples} Scans for Condition: {cond} ---")
        prompt = ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES[cond])
        synth_embs = []

        for i in range(num_samples):
            s = base_seed + i * 19
            vol_np, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt, initial_seed=s,
                logger=logger, max_retries=2
            )
            if vol_np is None:
                continue

            emb = extract_3d_mrclip_embedding(vol_np, clip_model, device)
            synth_embs.append(emb)

            diag = compute_supplementary_diagnostics(vol_np)
            supp_diagnostics[cond].append(diag)

            if (i + 1) % 25 == 0 or (i + 1) == num_samples:
                logger.info(f"  [{cond}] Generated {i + 1}/{num_samples} scans")

        synth_embs = np.array(synth_embs)
        if len(synth_embs) >= 2:
            raw_fid = compute_mrclip_fid(feats_real, synth_embs)
            mean_fid, ci_l, ci_u = bootstrap_fid_ci(feats_real, synth_embs, n_bootstraps=500)
            logger.info(f"  [{cond}] MR-CLIP FID: {raw_fid:.3f} (95% CI: [{ci_l:.3f}, {ci_u:.3f}])")
            fid_results[cond] = {
                "raw_fid": float(raw_fid),
                "bootstrap_mean_fid": float(mean_fid),
                "ci_95_lower": float(ci_l),
                "ci_95_upper": float(ci_u),
                "num_samples": int(len(synth_embs)),
            }
        else:
            logger.warning(f"  [{cond}] Insufficient generated samples ({len(synth_embs)}) to compute FID.")

    # Supplementary diagnostics summary
    supp_summary = {}
    for cond, d_list in supp_diagnostics.items():
        if d_list:
            supp_summary[cond] = {
                "sharpness_mean": float(np.mean([d["sharpness"] for d in d_list])),
                "tv_mean": float(np.mean([d["total_variation"] for d in d_list])),
                "entropy_mean": float(np.mean([d["entropy"] for d in d_list])),
            }

    # Save summary JSON
    full_summary = {
        "mrclip_3d_fid": fid_results,
        "supplementary_diagnostics": supp_summary,
        "note_inception_fid": (
            "2D Inception FID was NOT computed here. "
            "Any fid_and_quality_summary.json with Inception FID values (~3000-5000) "
            "is a legacy artifact from a prior run using ImageNet-pretrained features on MRI slices "
            "and should not be interpreted. The authoritative realism metric is mrclip_3d_fid above."
        ),
    }

    with open(out_dir / "mrclip_fid_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    logger.info(f"Saved FID summary to: {out_dir / 'mrclip_fid_summary.json'}")

    # Publication Figure: MR-CLIP FID with Bootstrap 95% Confidence Intervals
    if fid_results:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        conds = list(fid_results.keys())
        means = [fid_results[c]["bootstrap_mean_fid"] for c in conds]
        yerr_lower = [max(0.0, means[i] - fid_results[c]["ci_95_lower"]) for i, c in enumerate(conds)]
        yerr_upper = [max(0.0, fid_results[c]["ci_95_upper"] - means[i]) for i, c in enumerate(conds)]
        yerr = [yerr_lower, yerr_upper]

        bars = ax.bar(conds, means, yerr=yerr, capsize=5, color=["#1f77b4", "#2ca02c", "#ff7f0e"], edgecolor="black", lw=1.2, width=0.45)
        ax.set_ylabel("3D MR-CLIP Feature Fréchet Distance (FID)", fontsize=11, fontweight="bold")
        ax.set_title("3D MR-CLIP Realism FID with Bootstrap 95% CIs\n(Domain-Native 3D Feature Space, N ≥ 100)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, c in zip(bars, conds):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h * 0.5, f"{h:.2f}\n[{fid_results[c]['ci_95_lower']:.1f}, {fid_results[c]['ci_95_upper']:.1f}]",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        plt.tight_layout()
        plot_path = out_dir / "mrclip_fid_with_bootstrapping.png"
        fig.savefig(str(plot_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved MR-CLIP FID plot to: {plot_path}")
    else:
        logger.warning("No conditions generated sufficient samples to render FID plot.")

    return full_summary


def parse_args():
    p = argparse.ArgumentParser(description="Pillar 6: MR-CLIP Feature Space Realism & FID")
    p.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    p.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    p.add_argument(
        "--clip-checkpoint",
        default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt",
    )
    p.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--embeddings-pkl", default="data/real_mrclip_visual_embeddings.pkl")
    p.add_argument("--output-dir", default="output/eval_quality")
    p.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main():
    args_cli = parse_args()
    logger = setup_logging("eval_realism")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args_cli.device)
    logger.info(f"Target compute device: {device}")

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

    run_realism_evaluation(
        args=args,
        device=str(device),
        autoencoder=autoencoder,
        model=model,
        scale_factor=scale_factor,
        clip_model=clip_model,
        tokenizer=tokenizer,
        target_dim=target_dim,
        cfg_scale=args_cli.cfg_scale,
        embeddings_pkl=args_cli.embeddings_pkl,
        out_dir=out_dir,
        num_samples=args_cli.num_samples,
        logger=logger,
    )


if __name__ == "__main__":
    main()
