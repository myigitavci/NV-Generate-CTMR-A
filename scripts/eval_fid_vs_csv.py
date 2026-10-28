"""
scripts/eval_fid_vs_csv.py
==========================
FID Evaluation: Synthesized Images vs. Real Scans from a Validation CSV

Computes FID per modality (T1W, T2W, FLAIR, PDW) between:
  - Synthetic 3D volumes generated from real CSV text prompts
  - Real NIfTI volumes listed in the CSV

Two FID metrics are computed:
  1. 3D MR-CLIP FID  -- domain-native ViT-B-16 3D visual features (512-dim)
  2. 2D Inception FID -- separately for axial, coronal, and sagittal planes
                         using torchvision Inception-v3 (2048-dim pool3 features)

Usage:
  python -m scripts.eval_fid_vs_csv \\
    --val-csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_val_unseen_OASIS_fixed_desktop.csv \\
    --num-samples 100 \\
    --output-dir output/eval_fid_vs_csv \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

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
import torch
import torch.nn.functional as F
from torchvision import models as tv_models

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.eval_quality_and_fid import compute_mrclip_fid, bootstrap_fid_ci
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    extract_modality,
    generate_complete_brain,
    get_middle_slices,
    resolve_filepath,
)
from scripts.utils import define_instance
from scripts.plot_fid_publication import make_publication_figure


# ---------------------------------------------------------------------------
# Inception-v3 feature extractor (pool3, 2048-dim)
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(torch.nn.Module):
    """Wraps torchvision Inception-v3 and extracts pool3 (2048-dim) features."""

    def __init__(self, device: str):
        super().__init__()
        inc = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1)
        inc.eval()
        self.features = torch.nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> np.ndarray:
        """x: (B,3,H,W) float32 in [0,1] -> (B,2048) numpy."""
        x = x.to(self.device)
        x = x * 2.0 - 1.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        feat = self.features(x)
        return feat.squeeze(-1).squeeze(-1).cpu().numpy()


def slice_to_inception_input(slice_2d: np.ndarray) -> torch.Tensor:
    """Convert (H,W) float32 [0,1] slice to (1,3,H,W) tensor."""
    t = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
    return t.unsqueeze(0).repeat(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def compute_inception_fid(feats_real: np.ndarray, feats_synth: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_s = feats_real.mean(0), feats_synth.mean(0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_s = np.cov(feats_synth, rowvar=False)
    diff = mu_r - mu_s
    covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_s), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_r + offset).dot(sigma_s + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(max(0.0, np.dot(diff, diff) + np.trace(sigma_r) + np.trace(sigma_s) - 2.0 * np.trace(covmean)))


def bootstrap_inception_fid_ci(
    feats_real: np.ndarray,
    feats_synth: np.ndarray,
    n_bootstraps: int = 500,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    n_r, n_s = len(feats_real), len(feats_synth)
    boot_fids = []
    for _ in range(n_bootstraps):
        idx_r = np.random.choice(n_r, size=n_r, replace=True)
        idx_s = np.random.choice(n_s, size=n_s, replace=True)
        try:
            boot_fids.append(compute_inception_fid(feats_real[idx_r], feats_synth[idx_s]))
        except Exception:
            continue
    if not boot_fids:
        raw = compute_inception_fid(feats_real, feats_synth)
        return raw, raw, raw
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.mean(boot_fids)),
        float(np.percentile(boot_fids, 100.0 * alpha)),
        float(np.percentile(boot_fids, 100.0 * (1.0 - alpha))),
    )


# ---------------------------------------------------------------------------
# MR-CLIP visual feature extraction
# ---------------------------------------------------------------------------

def extract_mrclip_visual(vol_np: np.ndarray, clip_model: Any, device: str) -> np.ndarray:
    vol_th = torch.from_numpy(vol_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    v_min = float(np.percentile(vol_np, 1.0))
    v_max = float(np.percentile(vol_np, 99.0))
    if v_max > v_min:
        vol_th = (vol_th - v_min) / (v_max - v_min)
        vol_th = torch.clamp(vol_th, 0.0, 1.0)
    if vol_th.shape[2:] != (128, 128, 128):
        vol_th = F.interpolate(vol_th, size=(128, 128, 128), mode="trilinear", align_corners=False)
    vol_th = vol_th.to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(vol_th) if hasattr(clip_model, "encode_image") else clip_model.visual(vol_th)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_by_modality(
    csv_path: str,
    modalities: list[str],
    max_per_modality: int,
    data_root: str,
    logger: logging.Logger,
) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path)
    groups: dict[str, list[dict]] = {m: [] for m in modalities}
    for _, row in df.iterrows():
        mod = extract_modality(row)
        if mod not in modalities:
            continue
        if len(groups[mod]) >= max_per_modality:
            continue
        raw_fp = str(row.get("filepath", ""))
        resolved = resolve_filepath(raw_fp, data_root)
        if resolved is None or not os.path.exists(resolved):
            logger.debug(f"Skipping missing file: {raw_fp}")
            continue
        groups[mod].append({"filepath": resolved, "text": str(row.get("text", "")).strip()})
    for mod, items in groups.items():
        logger.info(f"  [{mod}] Found {len(items)} real volumes on disk.")
    return groups


# ---------------------------------------------------------------------------
# Feature extraction: real volumes
# ---------------------------------------------------------------------------

def extract_real_features(
    items: list[dict],
    clip_model: Any,
    inception: InceptionFeatureExtractor,
    device: str,
    logger: logging.Logger,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    if cache_path and cache_path.exists():
        logger.info(f"  Loading real feature cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"  Cache read failed ({e}), re-extracting.")

    mrclip_feats, ax_feats, cor_feats, sag_feats = [], [], [], []
    for entry in items:
        fp = entry["filepath"]
        try:
            vol = nib.load(fp).get_fdata().astype(np.float32)
            if vol.ndim != 3 or min(vol.shape) < 16:
                continue
            mrclip_feats.append(extract_mrclip_visual(vol, clip_model, device))
            ax, cor, sag = get_middle_slices(vol)
            ax_feats.append(inception(slice_to_inception_input(ax)).squeeze(0))
            cor_feats.append(inception(slice_to_inception_input(cor)).squeeze(0))
            sag_feats.append(inception(slice_to_inception_input(sag)).squeeze(0))
        except Exception as e:
            logger.warning(f"  Error loading {fp}: {e}")

    result = {
        "mrclip":   np.array(mrclip_feats),
        "axial":    np.array(ax_feats),
        "coronal":  np.array(cor_feats),
        "sagittal": np.array(sag_feats),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"  Saved real feature cache: {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Feature extraction: synthetic volumes
# ---------------------------------------------------------------------------

def generate_synth_features(
    items: list[dict],
    num_synth: int,
    args,
    device: str,
    autoencoder: Any,
    model: Any,
    scale_factor: float,
    clip_model: Any,
    tokenizer: Any,
    target_dim: tuple[int, int, int],
    cfg_scale: float,
    inception: InceptionFeatureExtractor,
    modality: str,
    logger: logging.Logger,
    base_seed: int = 54321,
) -> dict[str, Any]:
    mrclip_feats, ax_feats, cor_feats, sag_feats = [], [], [], []
    fallback = CANONICAL_MIDI_TEMPLATES.get(modality, CANONICAL_MIDI_TEMPLATES["T1W"])

    # If fewer real items than num_synth, wrap around prompts so we generate exactly num_synth with unique seeds
    synth_items = [items[i % len(items)] for i in range(num_synth)]

    for i, entry in enumerate(synth_items):
        prompt = ensure_axial_prompt(entry["text"]) if entry["text"] else ensure_axial_prompt(fallback)
        vol_np, _, _ = generate_complete_brain(
            args=args, device=device, autoencoder=autoencoder, model=model,
            scale_factor=scale_factor, clip_model=clip_model, tokenizer=tokenizer,
            target_dim=target_dim, cfg_scale=cfg_scale, prompt=prompt,
            initial_seed=base_seed + i * 17, logger=logger, max_retries=3,
            fallback_prompt=ensure_axial_prompt(fallback),
        )
        if vol_np is None:
            logger.warning(f"  [{modality}] Sample {i+1}/{num_synth} failed, skipping.")
            continue
        mrclip_feats.append(extract_mrclip_visual(vol_np, clip_model, device))
        ax, cor, sag = get_middle_slices(vol_np)
        ax_feats.append(inception(slice_to_inception_input(ax)).squeeze(0))
        cor_feats.append(inception(slice_to_inception_input(cor)).squeeze(0))
        sag_feats.append(inception(slice_to_inception_input(sag)).squeeze(0))
        if (i + 1) % 10 == 0 or (i + 1) == num_synth:
            logger.info(f"  [{modality}] Generated {i+1}/{num_synth} synthetic volumes.")

    return {
        "mrclip":   np.array(mrclip_feats),
        "axial":    np.array(ax_feats),
        "coronal":  np.array(cor_feats),
        "sagittal": np.array(sag_feats),
    }


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_all_fids(
    real_feats: dict[str, np.ndarray],
    synth_feats: dict[str, np.ndarray],
    modality: str,
    logger: logging.Logger,
    n_bootstraps: int = 100,
    skip_bootstrap: bool = False,
) -> dict[str, Any]:
    fid_labels = {
        "mrclip":   "3D MR-CLIP FID",
        "axial":    "2D Inception FID (axial)",
        "coronal":  "2D Inception FID (coronal)",
        "sagittal": "2D Inception FID (sagittal)",
    }
    results: dict[str, Any] = {}
    for key in ["mrclip", "axial", "coronal", "sagittal"]:
        rf = real_feats.get(key)
        sf = synth_feats.get(key)
        if rf is None or sf is None or len(rf) < 2 or len(sf) < 2:
            logger.warning(f"  [{modality}/{key}] Insufficient samples. Skipping.")
            continue
        if key == "mrclip":
            raw_fid = compute_mrclip_fid(rf, sf)
        else:
            raw_fid = compute_inception_fid(rf, sf)

        if skip_bootstrap:
            mean_fid, ci_l, ci_u = raw_fid, float("nan"), float("nan")
            logger.info(f"  [{modality}] {fid_labels[key]}: {raw_fid:.3f}  (no bootstrap, N_real={len(rf)}, N_synth={len(sf)})")
        else:
            if key == "mrclip":
                mean_fid, ci_l, ci_u = bootstrap_fid_ci(rf, sf, n_bootstraps=n_bootstraps)
            else:
                mean_fid, ci_l, ci_u = bootstrap_inception_fid_ci(rf, sf, n_bootstraps=n_bootstraps)
            logger.info(f"  [{modality}] {fid_labels[key]}: {raw_fid:.3f}  95% CI [{ci_l:.3f}, {ci_u:.3f}]  (N_real={len(rf)}, N_synth={len(sf)})")

        results[key] = {
            "raw_fid":        float(raw_fid),
            "bootstrap_mean": float(mean_fid),
            "ci_95_lower":    float(ci_l),
            "ci_95_upper":    float(ci_u),
            "n_real":         int(len(rf)),
            "n_synth":        int(len(sf)),
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(summary: dict[str, Any], out_dir: Path, logger: logging.Logger) -> None:
    modalities = list(summary.keys())
    fid_keys = ["mrclip", "axial", "coronal", "sagittal"]
    fid_labels = {
        "mrclip":   "3D MR-CLIP FID",
        "axial":    "2D Inception FID — Axial",
        "coronal":  "2D Inception FID — Coronal",
        "sagittal": "2D Inception FID — Sagittal",
    }
    palette = {"T1W": "#4e79a7", "T2W": "#59a14f", "FLAIR": "#f28e2b", "PDW": "#e15759"}

    for key in fid_keys:
        mods = [m for m in modalities if key in summary[m]]
        if not mods:
            continue
        # Use raw_fid if bootstrap was skipped (bootstrap_mean == raw_fid in that case)
        means  = [summary[m][key]["raw_fid"]      for m in mods]
        ci_l   = [summary[m][key]["ci_95_lower"]  for m in mods]
        ci_u   = [summary[m][key]["ci_95_upper"]  for m in mods]
        yerr_l = [max(0.0, means[i] - ci_l[i]) for i in range(len(mods))]
        yerr_u = [max(0.0, ci_u[i] - means[i]) for i in range(len(mods))]
        fig, ax = plt.subplots(figsize=(max(6, len(mods) * 2), 5), dpi=180)
        bars = ax.bar(mods, means, yerr=[yerr_l, yerr_u], capsize=5,
                      color=[palette.get(m, "#aec7e8") for m in mods],
                      edgecolor="black", lw=1.1, width=0.55)
        for bar, m in zip(bars, mods):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h * 0.5,
                    f"{h:.2f}\n[{summary[m][key]['ci_95_lower']:.1f}, {summary[m][key]['ci_95_upper']:.1f}]",
                    ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.set_ylabel("Fréchet Distance", fontsize=11, fontweight="bold")
        ax.set_title(f"{fid_labels[key]}\nSynth vs Real Val CSV — per Modality (Bootstrap 95% CI)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        plt.tight_layout()
        path = out_dir / f"fid_{key}.png"
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    active_keys = [k for k in fid_keys if any(k in summary.get(m, {}) for m in modalities)]
    if len(active_keys) > 1:
        fig, axes = plt.subplots(1, len(active_keys), figsize=(len(active_keys) * 4, 5), dpi=180)
        for ax_obj, key in zip(axes, active_keys):
            mods = [m for m in modalities if key in summary[m]]
            means  = [summary[m][key]["bootstrap_mean"] for m in mods]
            ci_l   = [summary[m][key]["ci_95_lower"]    for m in mods]
            ci_u   = [summary[m][key]["ci_95_upper"]    for m in mods]
            yerr_l = [max(0.0, means[i] - ci_l[i]) for i in range(len(mods))]
            yerr_u = [max(0.0, ci_u[i] - means[i]) for i in range(len(mods))]
            ax_obj.bar(mods, means, yerr=[yerr_l, yerr_u], capsize=4,
                       color=[palette.get(m, "#aec7e8") for m in mods],
                       edgecolor="black", lw=0.9, width=0.55)
            ax_obj.set_title(fid_labels[key], fontsize=9, fontweight="bold")
            ax_obj.grid(axis="y", linestyle="--", alpha=0.35)
        fig.suptitle("Synthesized vs. Real — FID Summary (Bootstrap 95% CI)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = out_dir / "fid_combined.png"
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FID Evaluation: Synthesized vs. Real Val CSV")
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-root", default="/data/mr_niftis")
    p.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "FLAIR", "PDW"])
    p.add_argument("--num-samples", type=int, default=50,
                   help="Number of samples per modality (default 50; cycles prompts with new seeds if fewer available)")
    p.add_argument("--n-bootstraps", type=int, default=100,
                   help="Bootstrap iterations for FID CI (default 100; use --skip-bootstrap for raw FID only)")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap CI entirely — compute only raw FID (fast)")
    p.add_argument("-e", "--env-config",   default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-t", "--model-def",    default="configs/config_network_rflow.json")
    p.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    p.add_argument("--clip-checkpoint",
                   default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt")
    p.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--output-dir", default="output/eval_fid_vs_csv")
    p.add_argument("--cache-dir", default="data/fid_real_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--device",
                   default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main() -> None:
    args_cli = parse_args()
    logger = setup_logging("eval_fid_vs_csv")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(torch.device(args_cli.device))

    logger.info(f"Device: {device}  |  Val CSV: {args_cli.val_csv}")
    logger.info(f"Modalities: {args_cli.modalities}  |  Max samples/modality: {args_cli.num_samples}")

    logger.info("Loading MR-CLIP...")
    clip_model, tokenizer = load_mrclip(
        args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, device)

    logger.info("Loading diffusion model...")
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt.get("state_dict", ckpt))
    autoencoder.eval()

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    diff_model = MRCLIPConditionedUNet(base_unet, clip_embed_dim=512).to(device)
    diff_ckpt = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    sd = diff_ckpt.get("model_state_dict", diff_ckpt.get("state_dict", diff_ckpt))
    diff_model.load_state_dict(sd, strict=False)
    diff_model.eval()

    scale_factor = getattr(args, "scale_factor", 1.0)
    target_dim = tuple(args.diffusion_unet_inference.get(
        "dim", args.diffusion_unet_inference.get("output_size", [256, 256, 128])))

    logger.info("Loading Inception-v3...")
    inception = InceptionFeatureExtractor(device=device)
    inception.eval()

    logger.info("Loading val CSV...")
    modality_items = load_csv_by_modality(
        args_cli.val_csv, args_cli.modalities, args_cli.num_samples, args_cli.data_root, logger)

    summary: dict[str, Any] = {}
    for modality in args_cli.modalities:
        items = modality_items[modality]
        if len(items) < 2:
            logger.warning(f"[{modality}] Only {len(items)} files found, skipping.")
            continue
        logger.info("=" * 70)
        logger.info(f"Modality: {modality}  (Real scans found: {len(items)}, Target synth: {args_cli.num_samples})")
        logger.info("=" * 70)

        cache_path = None if args_cli.no_cache else (Path(args_cli.cache_dir) / f"{modality}_n{len(items)}.pkl")
        logger.info(f"[{modality}] Extracting real features...")
        real_feats = extract_real_features(items, clip_model, inception, device, logger, cache_path)

        logger.info(f"[{modality}] Generating {args_cli.num_samples} synthetic volumes...")
        synth_feats = generate_synth_features(
            items=items, num_synth=args_cli.num_samples, args=args, device=device, autoencoder=autoencoder,
            model=diff_model, scale_factor=scale_factor, clip_model=clip_model,
            tokenizer=tokenizer, target_dim=target_dim, cfg_scale=args_cli.cfg_scale,
            inception=inception, modality=modality, logger=logger,
        )

        logger.info(f"[{modality}] Computing FIDs...")
        summary[modality] = compute_all_fids(
            real_feats, synth_feats, modality, logger,
            n_bootstraps=args_cli.n_bootstraps,
            skip_bootstrap=args_cli.skip_bootstrap,
        )

    with open(out_dir / "fid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {out_dir / 'fid_summary.json'}")

    logger.info("\n" + "=" * 85)
    logger.info(f"{'Modality':<10} {'MR-CLIP FID':>14} {'Inception Axial':>16} {'Inception Coronal':>18} {'Inception Sagittal':>19}")
    logger.info("-" * 80)
    for mod in args_cli.modalities:
        if mod not in summary:
            continue
        r = summary[mod]
        mc  = f"{r['mrclip']['raw_fid']:.2f}"   if "mrclip"   in r else "N/A"
        ax  = f"{r['axial']['raw_fid']:.2f}"    if "axial"    in r else "N/A"
        cor = f"{r['coronal']['raw_fid']:.2f}"  if "coronal"  in r else "N/A"
        sag = f"{r['sagittal']['raw_fid']:.2f}" if "sagittal" in r else "N/A"
        logger.info(f"{mod:<10} {mc:>14} {ax:>16} {cor:>18} {sag:>19}")

    save_plots(summary, out_dir, logger)
    make_publication_figure(summary, out_dir)
    logger.info(f"\nAll outputs & publication figures updated -> {out_dir}")


if __name__ == "__main__":
    main()
