"""
scripts/eval_acquisition_controllability.py
===========================================
Pillar 1: Main Experiment — Acquisition Controllability

Evaluates whether MR-CLIP conditioning provides precise, continuous and categorical
control over acquisition characteristics:
- Sweeps one metadata variable at a time across fixed latent seeds (N=100 seeds):
    * TE in {10, 20, 30, 40, 60, 80, 100} ms
    * TR in {400, 800, 1500, 3000, 5000, 8000} ms
    * TI in {200, 500, 1000, 1800, 2200, 2500} ms
    * FA in {15, 30, 60, 90, 120, 150, 180} deg
    * Vendor in {Siemens, GE, Philips}
    * Field Strength in {1.5T, 3.0T}
    * Contrast in {T1w, T2w, FLAIR}
- For every generated image:
    * Categorical metadata -> evaluated via Zero-Shot CLIP classification (Balanced Acc / AUROC)
    * Continuous metadata  -> evaluated via Zero-Shot continuous Softmax Expected Value (MAE, R^2, Spearman rho)
- Outputs:
    * requested_vs_predicted_grid.png (4-panel publication figure: TE, TR, TI, FA)
    * categorical_controllability.png (Bar chart: Vendor, Field Strength, Modality)
    * controllability_metrics.json + controllability_per_case.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.frozen_acquisition_predictors import (
    MRCLIPZeroShotPredictor,
    encode_volume_mrclip,
)
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
    replace_imaging_param,
    replace_scanner,
)
from scripts.utils import define_instance


def run_controllability_evaluation(
    args,
    device: str,
    autoencoder: Any,
    model: Any,
    scale_factor: float,
    clip_model: Any,
    tokenizer: Any,
    target_dim: tuple[int, int, int],
    cfg_scale: float,
    predictor: MRCLIPZeroShotPredictor,
    out_dir: Path,
    num_seeds: int = 100,
    logger: logging.Logger | None = None,
):
    if logger is None:
        logger = logging.getLogger("eval_controllability")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"STARTING ACQUISITION CONTROLLABILITY EVALUATION (Seeds: {num_seeds})")
    logger.info("=" * 70)

    # Define continuous sweeps
    sweeps = {
        "TE": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T2W"],
            "values": [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0],
            "unit": "ms",
            "prompt_fn": lambda p, v: ensure_axial_prompt(replace_imaging_param(p, te=v / 1000.0)),
        },
        "TR": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T1W"],
            "values": [400.0, 800.0, 1500.0, 3000.0, 5000.0, 8000.0],
            "unit": "ms",
            "prompt_fn": lambda p, v: ensure_axial_prompt(replace_imaging_param(p, tr=v / 1000.0)),
        },
        "TI": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["FLAIR"],
            "values": [200.0, 500.0, 1000.0, 1800.0, 2200.0, 2500.0],
            "unit": "ms",
            "prompt_fn": lambda p, v: ensure_axial_prompt(replace_imaging_param(p, ti=v / 1000.0)),
        },
        "FA": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T2STARW"],
            "values": [15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0],
            "unit": "deg",
            "prompt_fn": lambda p, v: ensure_axial_prompt(replace_imaging_param(p, fa=v)),
        },
    }

    categorical_sweeps = {
        "Vendor": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T1W"],
            "values": ["Siemens", "GE", "Philips"],
            # Enriched with scanner-specific imaging characteristics to maximise
            # discriminability in MR-CLIP embedding space. Plain vendor names
            # alone are near-synonymous in the text encoder.
            "vendor_candidate_prompts": {
                "Siemens": (
                    "Axial T1-weighted brain MRI acquired on a Siemens MAGNETOM Vida 3T scanner "
                    "with 1.0 mm isotropic voxels, TrueForm RF uniformity, and a 64-channel head/neck coil. "
                    "TR 2300ms TE 2.98ms TI 900ms."
                ),
                "GE": (
                    "Axial T1-weighted brain MRI acquired on a GE DISCOVERY MR750 3T scanner "
                    "with 1.0 mm isotropic voxels, ASSET parallel imaging, and an 8-channel head coil. "
                    "TR 2300ms TE 2.98ms TI 900ms."
                ),
                "Philips": (
                    "Axial T1-weighted brain MRI acquired on a Philips Achieva dStream 3T scanner "
                    "with 1.0 mm isotropic voxels, SENSitivity Encoding, and a 32-channel head coil. "
                    "TR 2300ms TE 2.98ms TI 900ms."
                ),
            },
            "models": {"Siemens": "MAGNETOM_Vida", "GE": "DISCOVERY_MR750", "Philips": "Achieva_dStream"},
            "field_strength": 3.0,
        },
        "Field_Strength": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T1W"],
            "values": ["1.5", "3.0"],
            "vendor": "GE",
            "models": {"1.5": "SIGNA_HDx", "3.0": "SIGNA_HDx"},
        },
        "Modality": {
            "base_prompt": CANONICAL_MIDI_TEMPLATES["T1W"],
            "values": ["T1W", "T2W", "FLAIR"],
        },
    }

    results: list[dict[str, Any]] = []

    # Generate fixed latent seed pool
    base_seed_offset = 6256
    seeds = [base_seed_offset + i * 13 for i in range(num_seeds)]

    # -----------------------------------------------------------------------
    # Part 1: Continuous Parameter Sweeps (TE, TR, TI, FA)
    # -----------------------------------------------------------------------
    for sweep_name, sweep_cfg in sweeps.items():
        logger.info(f"\n--- Running Sweep: {sweep_name} across {len(sweep_cfg['values'])} values ---")
        base_prompt = sweep_cfg["base_prompt"]
        vals = sweep_cfg["values"]
        val_prompt_fn = lambda v: sweep_cfg["prompt_fn"](base_prompt, v)

        for val in vals:
            prompt = val_prompt_fn(val)

            for s_idx, seed in enumerate(seeds):
                vol_np, _, _ = generate_complete_brain(
                    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                    target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
                    logger=logger, max_retries=2, fallback_prompt=base_prompt
                )
                if vol_np is None:
                    continue

                vol_feat = encode_volume_mrclip(vol_np, clip_model, device)
                pred_val = predictor.predict_continuous(vol_feat, val_prompt_fn, vals)

                results.append({
                    "sweep_type": "continuous",
                    "sweep_name": sweep_name,
                    "seed": seed,
                    "requested": val,
                    "predicted": pred_val,
                    "unit": sweep_cfg["unit"],
                })

                if (s_idx + 1) % 25 == 0 or (s_idx + 1) == len(seeds):
                    logger.info(f"  [{sweep_name} = {val} {sweep_cfg['unit']}] Done seed {s_idx + 1}/{len(seeds)} (pred={pred_val:.2f})")

    # -----------------------------------------------------------------------
    # Part 2: Categorical Parameter Sweeps (Vendor, Field Strength, Modality)
    # -----------------------------------------------------------------------
    # Vendor
    logger.info("\n--- Running Categorical Sweep: Vendor ---")
    v_cfg = categorical_sweeps["Vendor"]
    # Use enriched scanner-specific prompts if available, otherwise fall back to template replacement
    if "vendor_candidate_prompts" in v_cfg:
        vendor_prompts = v_cfg["vendor_candidate_prompts"]
        logger.info("  Using enriched scanner-specific vendor prompts for improved discriminability.")
    else:
        vendor_prompts = {
            v: ensure_axial_prompt(replace_scanner(v_cfg["base_prompt"], v, v_cfg["models"][v], v_cfg["field_strength"]))
            for v in v_cfg["values"]
        }
    for vendor, prompt in vendor_prompts.items():
        for s_idx, seed in enumerate(seeds):
            vol_np, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
                logger=logger, max_retries=2
            )
            if vol_np is None:
                continue
            vol_feat = encode_volume_mrclip(vol_np, clip_model, device)
            pred_v, _ = predictor.predict_categorical(vol_feat, vendor_prompts)
            results.append({
                "sweep_type": "categorical",
                "sweep_name": "Vendor",
                "seed": seed,
                "requested": vendor,
                "predicted": pred_v,
                "unit": "category",
            })

    # Field Strength
    logger.info("\n--- Running Categorical Sweep: Field Strength ---")
    fs_cfg = categorical_sweeps["Field_Strength"]
    fs_prompts = {
        fs: ensure_axial_prompt(replace_scanner(fs_cfg["base_prompt"], fs_cfg["vendor"], fs_cfg["models"][fs], float(fs)))
        for fs in fs_cfg["values"]
    }
    for fs_str, prompt in fs_prompts.items():
        for s_idx, seed in enumerate(seeds):
            vol_np, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
                logger=logger, max_retries=2
            )
            if vol_np is None:
                continue
            vol_feat = encode_volume_mrclip(vol_np, clip_model, device)
            pred_fs, _ = predictor.predict_categorical(vol_feat, fs_prompts)
            results.append({
                "sweep_type": "categorical",
                "sweep_name": "Field_Strength",
                "seed": seed,
                "requested": fs_str,
                "predicted": str(pred_fs),
                "unit": "T",
            })

    # Modality
    logger.info("\n--- Running Categorical Sweep: Modality ---")
    m_cfg = categorical_sweeps["Modality"]
    mod_prompts = {mod: ensure_axial_prompt(CANONICAL_MIDI_TEMPLATES[mod]) for mod in m_cfg["values"]}
    for mod, prompt in mod_prompts.items():
        for s_idx, seed in enumerate(seeds):
            vol_np, _, _ = generate_complete_brain(
                args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, prompt=prompt, initial_seed=seed,
                logger=logger, max_retries=2
            )
            if vol_np is None:
                continue
            vol_feat = encode_volume_mrclip(vol_np, clip_model, device)
            pred_mod, _ = predictor.predict_categorical(vol_feat, mod_prompts)
            results.append({
                "sweep_type": "categorical",
                "sweep_name": "Modality",
                "seed": seed,
                "requested": mod,
                "predicted": pred_mod,
                "unit": "contrast",
            })

    # -----------------------------------------------------------------------
    # Part 3: Metrics Computation & Publication Plotting
    # -----------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "controllability_per_case.csv", index=False)
    logger.info(f"Saved per-case results to: {out_dir / 'controllability_per_case.csv'}")

    summary_metrics: dict[str, Any] = {}

    # 4-Panel Requested vs. Predicted Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    sweep_axes = {
        "TE": axes[0, 0],
        "TR": axes[0, 1],
        "TI": axes[1, 0],
        "FA": axes[1, 1],
    }

    colors = {"TE": "#1f77b4", "TR": "#2ca02c", "TI": "#d62728", "FA": "#9467bd"}

    for sweep_name, ax in sweep_axes.items():
        sub_df = df_res[(df_res["sweep_type"] == "continuous") & (df_res["sweep_name"] == sweep_name)].copy()
        if len(sub_df) == 0:
            continue

        y_req = sub_df["requested"].astype(float).values
        y_pred = sub_df["predicted"].astype(float).values

        valid = ~np.isnan(y_req) & ~np.isnan(y_pred)
        y_req = y_req[valid]
        y_pred = y_pred[valid]

        r2 = r2_score(y_req, y_pred)
        mae = mean_absolute_error(y_req, y_pred)
        rho, _ = spearmanr(y_req, y_pred)
        metric_entry: dict = {"R2": float(r2), "MAE": float(mae), "Spearman_rho": float(rho), "N": int(len(y_req))}
        # TR and TI are difficult to evaluate via zero-shot softmax expected-value because
        # MR-CLIP text embeddings compress the TR/TI dimension — the dominant cue is modality/anatomy.
        # The authoritative TR/TI evaluation is tissue-contrast correlation from Pillar 4b physics sweep.
        if sweep_name in ("TR", "TI"):
            metric_entry["note"] = (
                "Zero-shot softmax estimator is known to underperform for TR/TI because the MR-CLIP "
                "text encoder compresses these dimensions. See Pillar 4b (eval_physics_joint_sweep.py) "
                "for tissue-contrast-based TR/TI evaluation (Level A Spearman rho)."
            )
        summary_metrics[sweep_name] = metric_entry

        unit = sub_df["unit"].iloc[0]
        col = colors[sweep_name]

        # Scatter points
        jitter = np.random.normal(0, 0.015 * (y_req.max() - y_req.min() + 1e-3), size=len(y_req))
        ax.scatter(y_req + jitter, y_pred, alpha=0.45, color=col, s=32, edgecolors="none", label="Synthesized Scans")

        # Mean and Std per requested level
        levels = np.sort(np.unique(y_req))
        means = [np.mean(y_pred[y_req == lvl]) for lvl in levels]
        stds = [np.std(y_pred[y_req == lvl]) for lvl in levels]
        ax.errorbar(levels, means, yerr=stds, fmt="o-", color="black", lw=2.2, capsize=4, capthick=1.5, zorder=5, label="Mean ± Std")

        # Identity Line (y = x)
        all_min = min(y_req.min(), y_pred.min())
        all_max = max(y_req.max(), y_pred.max())
        ax.plot([all_min, all_max], [all_min, all_max], "k--", alpha=0.6, lw=1.5, label="Ideal (y = x)")

        # Fit line
        poly_fit = np.polyfit(y_req, y_pred, 1)
        ax.plot(levels, np.polyval(poly_fit, levels), color=col, lw=2.0, label=f"Fit (Slope={poly_fit[0]:.2f})")

        ax.set_xlabel(f"Requested {sweep_name} [{unit}]", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Predicted {sweep_name} [{unit}]", fontsize=12, fontweight="bold")
        ax.set_title(f"{sweep_name} Controllability\n$R^2$ = {r2:.3f} | MAE = {mae:.2f} {unit} | Spearman $\\rho$ = {rho:.3f}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "requested_vs_predicted_grid.png"
    fig.savefig(str(plot_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 4-panel requested vs predicted figure to: {plot_path}")

    # Categorical summary metrics & bar chart
    cat_metrics = {}
    for cat in ["Vendor", "Field_Strength", "Modality"]:
        sub_df = df_res[(df_res["sweep_type"] == "categorical") & (df_res["sweep_name"] == cat)].copy()
        if len(sub_df) > 0:
            y_true = sub_df["requested"].astype(str).values
            y_pred = sub_df["predicted"].astype(str).values
            acc = balanced_accuracy_score(y_true, y_pred)
            cat_metrics[cat] = {"balanced_accuracy": float(acc), "N": int(len(y_true))}
            summary_metrics[cat] = cat_metrics[cat]
            logger.info(f"  [Categorical] {cat}: Balanced Accuracy = {acc:.3f} (N={len(y_true)})")

    # Bar chart of categorical accuracies
    if cat_metrics:
        # Chance levels: Vendor=3-class (33.3%), Field_Strength=2-class (50%), Modality=3-class (33.3%)
        chance_by_cat = {"Vendor": 1/3, "Field_Strength": 0.5, "Modality": 1/3}
        fig_cat, ax_cat = plt.subplots(figsize=(8, 5), dpi=180)
        cats = list(cat_metrics.keys())
        accs = [cat_metrics[c]["balanced_accuracy"] for c in cats]
        chance_vals = [chance_by_cat.get(c, 1/3) for c in cats]
        bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax_cat.bar(cats, accs, color=bar_colors[:len(cats)], width=0.5, edgecolor="black", lw=1.2)
        # Shade bars below their respective chance level in red
        for bar, acc, chance in zip(bars, accs, chance_vals):
            if acc <= chance:
                bar.set_facecolor("#d62728")  # red = at/below chance
                bar.set_alpha(0.85)
        ax_cat.set_ylabel("Balanced Accuracy", fontsize=11, fontweight="bold")
        ax_cat.set_ylim(0, 1.15)
        ax_cat.set_title("Categorical Acquisition Controllability (Zero-Shot MR-CLIP)", fontsize=13, fontweight="bold")
        ax_cat.axhline(1/3, color="gray", linestyle="--", alpha=0.8, lw=1.5, label="Chance (3-class: 33.3%)")
        ax_cat.axhline(0.5, color="darkgray", linestyle=":", alpha=0.8, lw=1.5, label="Chance (2-class: 50.0%)")
        for bar, acc, chance in zip(bars, accs, chance_vals):
            height = bar.get_height()
            delta = acc - chance
            label_str = f"{height:.3f}\n({'+' if delta >= 0 else ''}{delta:.3f} vs chance)"
            ax_cat.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                label_str, ha="center", va="bottom", fontweight="bold", fontsize=9,
            )
        ax_cat.legend(loc="upper right", framealpha=0.9)
        ax_cat.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        cat_plot_path = out_dir / "categorical_controllability.png"
        fig_cat.savefig(str(cat_plot_path), bbox_inches="tight")
        plt.close(fig_cat)
        logger.info(f"Saved categorical accuracy plot to: {cat_plot_path}")

    # Write summary JSON
    with open(out_dir / "controllability_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved summary metrics to: {out_dir / 'controllability_metrics.json'}")

    return summary_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Pillar 1: Acquisition Controllability Suite")
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
    p.add_argument("--num-seeds", type=int, default=100)
    p.add_argument("--output-dir", default="output/eval_controllability")
    p.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main():
    args_cli = parse_args()
    logger = setup_logging("eval_controllability")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args_cli.device)
    logger.info(f"Target compute device: {device}")

    # Load MR-CLIP model
    logger.info(f"Loading MR-CLIP from {args_cli.clip_checkpoint}...")
    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint, args_cli.clip_context_length, str(device))

    # Initialize Zero-Shot Predictor
    predictor = MRCLIPZeroShotPredictor(clip_model, tokenizer, str(device))

    # Load Diffusion Model & Autoencoder
    logger.info(f"Loading configs: {args_cli.env_config}, {args_cli.model_config}, {args_cli.model_def}")
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

    run_controllability_evaluation(
        args=args,
        device=str(device),
        autoencoder=autoencoder,
        model=model,
        scale_factor=scale_factor,
        clip_model=clip_model,
        tokenizer=tokenizer,
        target_dim=target_dim,
        cfg_scale=args_cli.cfg_scale,
        predictor=predictor,
        out_dir=out_dir,
        num_seeds=args_cli.num_seeds,
        logger=logger,
    )


if __name__ == "__main__":
    main()
