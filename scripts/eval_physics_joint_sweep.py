"""
scripts/eval_physics_joint_sweep.py
====================================
Pillar 4b: Joint TR x TE x TI Physics Consistency Experiment

Three levels of validation:
  A. 1D sanity checks  – TE, TR, TI sweeps with physics reference overlay
  B. 2D joint sweeps   – TR×TE and TR×TI heatmaps (physics vs. generated)
  C. Full 3D sweep     – N random valid (TR,TE,TI) samples, reporting:
       * Per-tissue Spearman rho vs. Bloch prediction
       * Contrast-ordering accuracy across WM/GM/CSF pairs

Reference tissue parameters at 3T (Stanisz 2005, Lu 2005):
    WM:  T1=850ms,  T2=80ms,  rho=0.70
    GM:  T1=1330ms, T2=90ms,  rho=0.85
    CSF: T1=4300ms, T2=1500ms,rho=1.00
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    generate_complete_brain,
    replace_imaging_param,
)
from scripts.utils import define_instance

# ---------------------------------------------------------------------------
# Tissue relaxation parameters at 3T
# ---------------------------------------------------------------------------
TISSUE_PARAMS = {
    "WM":  {"T1": 850.0,   "T2": 80.0,   "rho": 0.70},
    "GM":  {"T1": 1330.0,  "T2": 90.0,   "rho": 0.85},
    "CSF": {"T1": 4300.0,  "T2": 1500.0, "rho": 1.00},
}

# ---------------------------------------------------------------------------
# Physics signal equations (all params in ms)
# ---------------------------------------------------------------------------

def signal_se(TR: float, TE: float, T1: float, T2: float, rho: float) -> float:
    return rho * (1.0 - np.exp(-TR / T1)) * np.exp(-TE / T2)


def signal_ir(TR: float, TE: float, TI: float, T1: float, T2: float, rho: float) -> float:
    return float(np.abs(rho * (1.0 - 2.0 * np.exp(-TI / T1) + np.exp(-TR / T1)) * np.exp(-TE / T2)))


def physics_signals(TR_ms: float, TE_ms: float, TI_ms: float | None) -> dict[str, float]:
    use_ir = TI_ms is not None and TI_ms > 0
    raw = {}
    for t, p in TISSUE_PARAMS.items():
        if use_ir:
            raw[t] = signal_ir(TR_ms, TE_ms, TI_ms, p["T1"], p["T2"], p["rho"])
        else:
            raw[t] = signal_se(TR_ms, TE_ms, p["T1"], p["T2"], p["rho"])
    total = sum(raw.values()) + 1e-9
    return {t: v / total for t, v in raw.items()}


def csf_null_ti_approx() -> float:
    return TISSUE_PARAMS["CSF"]["T1"] * np.log(2)


def ordering_correct(phys: dict[str, float], gen: dict[str, float]) -> dict[str, bool]:
    return {
        "WM>GM":  (phys["WM"] > phys["GM"]) == (gen["WM"] > gen["GM"]),
        "GM>CSF": (phys["GM"] > phys["CSF"]) == (gen["GM"] > gen["CSF"]),
        "WM>CSF": (phys["WM"] > phys["CSF"]) == (gen["WM"] > gen["CSF"]),
    }

# ---------------------------------------------------------------------------
# Tissue intensity extraction
# ---------------------------------------------------------------------------

def run_synthseg(vol_np: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray | None:
    synthseg_bin = "/usr/local/freesurfer/8.1.0/bin/mri_synthseg"
    if not Path(synthseg_bin).exists():
        synthseg_bin = "mri_synthseg"
    env = os.environ.copy()
    env["FREESURFER_HOME"] = "/usr/local/freesurfer/8.1.0"
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.nii.gz"
        out = Path(tmp) / "out.nii.gz"
        nib.save(nib.Nifti1Image(vol_np.astype(np.float32), np.eye(4)), str(inp))
        try:
            r = subprocess.run(
                [synthseg_bin, "--i", str(inp), "--o", str(out), "--fast", "--keepgeom", "--threads", "4"],
                env=env, capture_output=True, text=True, timeout=180
            )
            if r.returncode == 0 and out.exists():
                seg = np.asanyarray(nib.load(str(out)).dataobj, dtype=np.int32)
                if seg.shape != vol_np.shape:
                    from scipy.ndimage import zoom
                    seg = zoom(seg, [v / s for v, s in zip(vol_np.shape, seg.shape)], order=0).astype(np.int32)
                return seg
        except Exception as e:
            if logger:
                logger.debug(f"SynthSeg failed: {e}")
    return None


def extract_tissue_intensities(vol_np: np.ndarray, seg_np: np.ndarray | None = None) -> dict[str, Any] | None:
    if seg_np is not None:
        wm_m  = np.isin(seg_np, [2, 41])
        gm_m  = np.isin(seg_np, [3, 42])
        csf_m = np.isin(seg_np, [4, 43, 14, 15])
        if wm_m.sum() > 50 and gm_m.sum() > 50 and csf_m.sum() > 50:
            wm  = float(np.mean(vol_np[wm_m]))
            gm  = float(np.mean(vol_np[gm_m]))
            csf = float(np.mean(vol_np[csf_m]))
            total = wm + gm + csf + 1e-9
            wm_n, gm_n, csf_n = wm/total, gm/total, csf/total
            # Sanity check: no single tissue should dominate (degenerate volume)
            if max(wm_n, gm_n, csf_n) > 0.90:
                return None  # Discard: collapsed/pathological measurement
            return {"WM": wm_n, "GM": gm_n, "CSF": csf_n,
                    "brain_mean": float(np.mean(vol_np[wm_m | gm_m | csf_m])),
                    "seg_method": "SynthSeg"}
    # Percentile fallback
    v = vol_np.flatten()
    bg = float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0
    bv = v[v > bg]
    wm, gm, csf = float(np.percentile(bv, 85)), float(np.percentile(bv, 50)), float(np.percentile(bv, 15))
    total = wm + gm + csf + 1e-9
    wm_n, gm_n, csf_n = wm/total, gm/total, csf/total
    # Sanity check: degenerate percentile-fallback volumes produce near-binary distributions
    if max(wm_n, gm_n, csf_n) > 0.90:
        return None  # Discard: percentile fallback collapsed on pathological volume
    return {"WM": wm_n, "GM": gm_n, "CSF": csf_n,
            "brain_mean": float(np.mean(bv)), "seg_method": "Percentile_Fallback"}


def build_prompt(TR_ms: float, TE_ms: float, TI_ms: float | None) -> str:
    use_ir = TI_ms is not None and TI_ms > 0
    if use_ir:
        base = CANONICAL_MIDI_TEMPLATES["FLAIR"]
    elif TE_ms < 30:
        base = CANONICAL_MIDI_TEMPLATES["T1W"]
    else:
        base = CANONICAL_MIDI_TEMPLATES["T2W"]
    p = replace_imaging_param(base, te=TE_ms / 1000.0, tr=TR_ms / 1000.0,
                               ti=(TI_ms / 1000.0 if use_ir else None))
    return ensure_axial_prompt(p)


def generate_and_measure(
    args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
    target_dim, cfg_scale, prompt: str, seed: int, logger
) -> dict[str, Any] | None:
    vol_np, _, _ = generate_complete_brain(
        args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
        target_dim, cfg_scale, prompt=prompt, initial_seed=seed, logger=logger, max_retries=2)
    if vol_np is None:
        return None
    seg_np = run_synthseg(vol_np, logger=logger)
    result = extract_tissue_intensities(vol_np, seg_np)
    if result is None:
        logger.debug(f"Discarding degenerate volume (seed={seed}): single tissue >90% of signal.")
    return result

# ---------------------------------------------------------------------------
# LEVEL A – 1D sanity checks
# ---------------------------------------------------------------------------

def run_level_a(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL A: 1D Sanity Checks with Physics Reference Overlay")
    logger.info("=" * 70)
    results = {}

    # A1: TE sweep
    logger.info("\n[A1] TE sweep (TR=4500ms, TI=None)")
    te_vals = [5, 10, 20, 30, 50, 80, 120, 150]
    phys_te, gen_te = {t: [] for t in ["WM", "GM", "CSF"]}, {t: [] for t in ["WM", "GM", "CSF"]}
    valid_te = []
    for te in te_vals:
        ph = physics_signals(4500.0, float(te), None)
        prompt = build_prompt(4500.0, float(te), None)
        gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                   target_dim, cfg_scale, prompt, seed, logger)
        if gen is None: continue
        valid_te.append(te)
        for t in ["WM", "GM", "CSF"]:
            phys_te[t].append(ph[t]); gen_te[t].append(gen[t])
        logger.info(f"  TE={te}ms: WM phys={ph['WM']:.3f}/gen={gen['WM']:.3f} GM phys={ph['GM']:.3f}/gen={gen['GM']:.3f} CSF phys={ph['CSF']:.3f}/gen={gen['CSF']:.3f}")

    te_spearman = {}
    if len(valid_te) >= 3:
        for t in ["WM", "GM", "CSF"]:
            r, _ = spearmanr(phys_te[t], gen_te[t])
            te_spearman[t] = float(r)
        _plot_1d_comparison(valid_te, "TE (ms)", "T2w regime (TR=4500ms)",
                            phys_te, gen_te, out_dir / "level_a_te_sweep.png", logger)
    results["A1_TE_spearman"] = te_spearman

    # A2: TR sweep
    logger.info("\n[A2] TR sweep (TE=12ms, TI=None)")
    tr_vals = [300, 500, 800, 1200, 2000, 3000, 5000]
    phys_tr, gen_tr = {t: [] for t in ["WM", "GM", "CSF"]}, {t: [] for t in ["WM", "GM", "CSF"]}
    valid_tr = []
    for tr in tr_vals:
        ph = physics_signals(float(tr), 12.0, None)
        prompt = build_prompt(float(tr), 12.0, None)
        gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                   target_dim, cfg_scale, prompt, seed, logger)
        if gen is None: continue
        valid_tr.append(tr)
        for t in ["WM", "GM", "CSF"]:
            phys_tr[t].append(ph[t]); gen_tr[t].append(gen[t])
        logger.info(f"  TR={tr}ms: WM phys={ph['WM']:.3f}/gen={gen['WM']:.3f} GM phys={ph['GM']:.3f}/gen={gen['GM']:.3f}")

    tr_spearman = {}
    if len(valid_tr) >= 3:
        for t in ["WM", "GM", "CSF"]:
            r, _ = spearmanr(phys_tr[t], gen_tr[t])
            tr_spearman[t] = float(r)
        _plot_1d_comparison(valid_tr, "TR (ms)", "T1w regime (TE=12ms)",
                            phys_tr, gen_tr, out_dir / "level_a_tr_sweep.png", logger)
    results["A2_TR_spearman"] = tr_spearman

    # A3: TI sweep (FLAIR)
    ti_null_pred = csf_null_ti_approx()
    logger.info(f"\n[A3] TI sweep (TR=8000ms, TE=90ms, FLAIR) — physics predicts CSF null at {ti_null_pred:.0f}ms")
    ti_vals = [500, 800, 1100, 1400, 1650, 2000, 2500, 3000, 3500]
    phys_ti, gen_ti = {t: [] for t in ["WM", "GM", "CSF"]}, {t: [] for t in ["WM", "GM", "CSF"]}
    valid_ti = []
    for ti in ti_vals:
        ph = physics_signals(8000.0, 90.0, float(ti))
        prompt = build_prompt(8000.0, 90.0, float(ti))
        gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                   target_dim, cfg_scale, prompt, seed, logger)
        if gen is None: continue
        valid_ti.append(ti)
        for t in ["WM", "GM", "CSF"]:
            phys_ti[t].append(ph[t]); gen_ti[t].append(gen[t])
        logger.info(f"  TI={ti}ms: CSF phys={ph['CSF']:.3f}/gen={gen['CSF']:.3f}")

    ti_spearman = {}
    empirical_null = None
    if len(valid_ti) >= 3:
        for t in ["WM", "GM", "CSF"]:
            r, _ = spearmanr(phys_ti[t], gen_ti[t])
            ti_spearman[t] = float(r)
        if gen_ti["CSF"]:
            empirical_null = valid_ti[int(np.argmin(gen_ti["CSF"]))]
        _plot_1d_ti(valid_ti, ti_null_pred, empirical_null, phys_ti, gen_ti,
                    out_dir / "level_a_ti_sweep.png", logger)

    results["A3_TI_spearman"] = ti_spearman
    results["A3_CSF_physics_null_ms"] = int(ti_null_pred)
    results["A3_CSF_empirical_null_ms"] = empirical_null
    return results


def _plot_1d_comparison(x_vals, xlabel, subtitle, phys_d, gen_d, path: Path, logger):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    colors = {"WM": "#ff7f0e", "GM": "#2ca02c", "CSF": "#1f77b4"}
    for t, c in colors.items():
        axes[0].plot(x_vals, phys_d[t], "^-", color=c, lw=2, label=t)
        axes[1].plot(x_vals, gen_d[t], "s--", color=c, lw=2, label=t)
    axes[0].set_title(f"Physics Reference\n({subtitle})", fontweight="bold")
    axes[0].set_xlabel(xlabel, fontweight="bold"); axes[0].set_ylabel("Norm. Signal", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)
    axes[1].set_title("Generated Model Response", fontweight="bold")
    axes[1].set_xlabel(xlabel, fontweight="bold"); axes[1].set_ylabel("Norm. Signal", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)
    ap, ag = [], []
    cv = []
    for t, c in colors.items():
        ap.extend(phys_d[t]); ag.extend(gen_d[t])
        cv.extend([c] * len(phys_d[t]))
    axes[2].scatter(ap, ag, c=cv, s=50, alpha=0.8, zorder=3)
    lo, hi = min(ap + ag) * 0.95, max(ap + ag) * 1.05
    axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    rho_all, _ = spearmanr(ap, ag)
    axes[2].set_title(f"Physics vs. Generated\nSpearman ρ={rho_all:.3f}", fontweight="bold")
    axes[2].set_xlabel("Physics (norm.)", fontweight="bold"); axes[2].set_ylabel("Generated (norm.)", fontweight="bold")
    axes[2].grid(True, alpha=0.4)
    plt.tight_layout(); fig.savefig(str(path), bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_1d_ti(ti_vals, ti_null_pred, empirical_null, phys_d, gen_d, path: Path, logger):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    colors = {"WM": "#ff7f0e", "GM": "#2ca02c", "CSF": "#1f77b4"}
    for t, c in colors.items():
        axes[0].plot(ti_vals, phys_d[t], "^-", color=c, lw=2, label=t)
        axes[1].plot(ti_vals, gen_d[t], "s--", color=c, lw=2, label=t)
    axes[0].axvline(ti_null_pred, color="red", ls="--", lw=1.8, label=f"CSF null ≈{ti_null_pred:.0f}ms")
    axes[0].set_title(f"Physics (TR=8000ms,TE=90ms)\nCSF null at {ti_null_pred:.0f}ms", fontweight="bold")
    axes[0].set_xlabel("TI (ms)", fontweight="bold"); axes[0].set_ylabel("Norm. Signal", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
    if empirical_null:
        axes[1].axvline(empirical_null, color="purple", ls="--", lw=1.8, label=f"Empirical null {empirical_null}ms")
    axes[1].set_title("Generated (FLAIR, TI sweep)", fontweight="bold")
    axes[1].set_xlabel("TI (ms)", fontweight="bold"); axes[1].set_ylabel("Norm. Signal", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)
    ap, ag, cv = [], [], []
    for t, c in colors.items():
        ap.extend(phys_d[t]); ag.extend(gen_d[t]); cv.extend([c] * len(phys_d[t]))
    axes[2].scatter(ap, ag, c=cv, s=50, alpha=0.8, zorder=3)
    lo, hi = min(ap + ag) * 0.95, max(ap + ag) * 1.05
    axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    rho_all, _ = spearmanr(ap, ag)
    axes[2].set_title(f"Physics vs. Generated\nSpearman ρ={rho_all:.3f}", fontweight="bold")
    axes[2].set_xlabel("Physics (norm.)", fontweight="bold"); axes[2].set_ylabel("Generated (norm.)", fontweight="bold")
    axes[2].grid(True, alpha=0.4)
    plt.tight_layout(); fig.savefig(str(path), bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved: {path}")

# ---------------------------------------------------------------------------
# LEVEL B – 2D joint heatmaps
# ---------------------------------------------------------------------------

def run_level_b(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, out_dir: Path, logger, seed: int = 6256) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL B: 2D Joint Heatmaps (TR×TE and TR×TI)")
    logger.info("=" * 70)
    results = {}

    # B1: TR×TE (WM-GM contrast)
    TR_vals = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0]
    TE_vals = [5.0, 20.0, 50.0, 100.0, 150.0]
    logger.info(f"\n[B1] TR×TE grid ({len(TR_vals)}×{len(TE_vals)}, TI=None, measuring WM-GM contrast)")
    phys_b1 = np.zeros((len(TR_vals), len(TE_vals)))
    gen_b1  = np.zeros((len(TR_vals), len(TE_vals)))
    mask_b1 = np.zeros((len(TR_vals), len(TE_vals)), dtype=bool)

    for i, tr in enumerate(TR_vals):
        for j, te in enumerate(TE_vals):
            if te >= tr:
                continue
            ph = physics_signals(tr, te, None)
            phys_b1[i, j] = (ph["WM"] - ph["GM"]) / (ph["WM"] + ph["GM"] + 1e-9)
            gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                       target_dim, cfg_scale,
                                       build_prompt(tr, te, None), seed + i * 100 + j, logger)
            if gen is not None:
                gen_b1[i, j] = (gen["WM"] - gen["GM"]) / (gen["WM"] + gen["GM"] + 1e-9)
                mask_b1[i, j] = True
                logger.info(f"  TR={tr:.0f} TE={te:.0f}: C_WM-GM phys={phys_b1[i,j]:.3f} gen={gen_b1[i,j]:.3f}")

    _plot_2d_heatmap(phys_b1, gen_b1, mask_b1, TR_vals, TE_vals, "TR (ms)", "TE (ms)",
                     r"$C_{WM-GM}$=(WM-GM)/(WM+GM)", "B1: TR×TE — WM/GM Contrast",
                     out_dir / "level_b1_tr_te_heatmap.png", logger)
    rho_b1 = _map_corr(phys_b1, gen_b1, mask_b1)
    acc_b1 = _map_sign_acc(phys_b1, gen_b1, mask_b1)
    results["B1_TR_TE_spearman"] = rho_b1
    results["B1_TR_TE_ordering_acc"] = acc_b1
    logger.info(f"  B1 TR×TE: Spearman rho={rho_b1:.3f}, ordering acc={acc_b1:.3f}")

    # B2: TR×TI (CSF suppression)
    TR_vals2 = [2000.0, 3000.0, 5000.0, 8000.0]
    TI_vals  = [500.0, 800.0, 1100.0, 1400.0, 1650.0, 2000.0, 2500.0, 3000.0]
    TE_b2 = 90.0
    logger.info(f"\n[B2] TR×TI grid ({len(TR_vals2)}×{len(TI_vals)}, TE={TE_b2:.0f}ms, measuring CSF signal)")
    phys_b2 = np.zeros((len(TR_vals2), len(TI_vals)))
    gen_b2  = np.zeros((len(TR_vals2), len(TI_vals)))
    mask_b2 = np.zeros((len(TR_vals2), len(TI_vals)), dtype=bool)

    for i, tr in enumerate(TR_vals2):
        for j, ti in enumerate(TI_vals):
            if ti >= tr:
                continue
            ph = physics_signals(tr, TE_b2, ti)
            phys_b2[i, j] = ph["CSF"]
            gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                       target_dim, cfg_scale,
                                       build_prompt(tr, TE_b2, ti), seed + 1000 + i * 100 + j, logger)
            if gen is not None:
                gen_b2[i, j] = gen["CSF"]
                mask_b2[i, j] = True
                logger.info(f"  TR={tr:.0f} TI={ti:.0f}: CSF phys={phys_b2[i,j]:.3f} gen={gen_b2[i,j]:.3f}")

    _plot_2d_heatmap(phys_b2, gen_b2, mask_b2, TR_vals2, TI_vals, "TR (ms)", "TI (ms)",
                     "CSF norm. signal", f"B2: TR×TI — CSF Suppression (TE={TE_b2:.0f}ms)",
                     out_dir / "level_b2_tr_ti_heatmap.png", logger, cmap="RdBu_r", center_zero=False)
    rho_b2 = _map_corr(phys_b2, gen_b2, mask_b2)
    results["B2_TR_TI_CSF_spearman"] = rho_b2
    logger.info(f"  B2 TR×TI: Spearman rho={rho_b2:.3f}")
    return results


def _plot_2d_heatmap(phys_map, gen_map, mask, row_vals, col_vals, rlabel, clabel,
                     cbarlabel, title, path: Path, logger, cmap="RdBu", center_zero=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    valid_gen = gen_map[mask]
    if center_zero:
        vmax = max(abs(phys_map).max(), abs(valid_gen).max() if len(valid_gen) else 1e-3)
        vmin, vmax = -vmax, vmax
    else:
        all_v = list(phys_map.flatten()) + list(valid_gen)
        vmin, vmax = min(all_v), max(all_v)

    kw = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    im0 = axes[0].imshow(phys_map.T, cmap=cmap, **kw)
    axes[0].set_title(f"Physics: {title}", fontsize=10, fontweight="bold")
    axes[0].set_xticks(range(len(row_vals))); axes[0].set_xticklabels([f"{v:.0f}" for v in row_vals], rotation=45, fontsize=7)
    axes[0].set_yticks(range(len(col_vals))); axes[0].set_yticklabels([f"{v:.0f}" for v in col_vals], fontsize=7)
    axes[0].set_xlabel(rlabel, fontweight="bold"); axes[0].set_ylabel(clabel, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label=cbarlabel, fraction=0.046)

    mg = np.where(mask, gen_map, np.nan)
    im1 = axes[1].imshow(mg.T, cmap=cmap, **kw)
    axes[1].set_title(f"Generated: {title}", fontsize=10, fontweight="bold")
    axes[1].set_xticks(range(len(row_vals))); axes[1].set_xticklabels([f"{v:.0f}" for v in row_vals], rotation=45, fontsize=7)
    axes[1].set_yticks(range(len(col_vals))); axes[1].set_yticklabels([f"{v:.0f}" for v in col_vals], fontsize=7)
    axes[1].set_xlabel(rlabel, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label=cbarlabel, fraction=0.046)

    diff = np.where(mask, gen_map - phys_map, np.nan)
    dmax = float(np.nanmax(np.abs(diff))) if not np.all(np.isnan(diff)) else 1e-3
    im2 = axes[2].imshow(diff.T, cmap="RdBu", vmin=-dmax, vmax=dmax, aspect="auto", origin="lower")
    axes[2].set_title("Δ = Generated − Physics", fontsize=10, fontweight="bold")
    axes[2].set_xticks(range(len(row_vals))); axes[2].set_xticklabels([f"{v:.0f}" for v in row_vals], rotation=45, fontsize=7)
    axes[2].set_yticks(range(len(col_vals))); axes[2].set_yticklabels([f"{v:.0f}" for v in col_vals], fontsize=7)
    axes[2].set_xlabel(rlabel, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], label="Δ Signal", fraction=0.046)

    plt.tight_layout(); fig.savefig(str(path), bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved: {path}")


def _map_corr(phys, gen, mask) -> float:
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(phys[mask].flatten(), gen[mask].flatten())
    return float(rho)


def _map_sign_acc(phys, gen, mask) -> float:
    valid = mask.flatten()
    if valid.sum() < 1:
        return float("nan")
    return float((np.sign(phys.flatten()[valid]) == np.sign(gen.flatten()[valid])).mean())

# ---------------------------------------------------------------------------
# LEVEL C – Full 3D Monte Carlo sweep
# ---------------------------------------------------------------------------

def sample_valid_params(n: int, rng: np.random.Generator) -> list[tuple[float, float, float | None]]:
    """Sample clinically valid (TR, TE, TI) triplets from realistic contrast regimes."""
    regime_counts = {
        "T1_like": int(0.20 * n),
        "PD_like": int(0.15 * n),
        "T2_like": int(0.20 * n),
        "T1_IR":   int(0.15 * n),
        "FLAIR":   int(0.20 * n),
        "Mixed":   n - int(0.20*n) - int(0.15*n) - int(0.20*n) - int(0.15*n) - int(0.20*n),
    }
    samples = []
    for regime, count in regime_counts.items():
        for _ in range(count):
            if regime == "T1_like":
                TR = rng.uniform(300, 800); TE = rng.uniform(5, min(20, TR * 0.8)); TI = None
            elif regime == "PD_like":
                TR = rng.uniform(2000, 5000); TE = rng.uniform(5, 25); TI = None
            elif regime == "T2_like":
                TR = rng.uniform(2000, 6000); TE = rng.uniform(60, min(150, TR * 0.9)); TI = None
            elif regime == "T1_IR":
                TR = rng.uniform(1500, 4000); TE = rng.uniform(5, 30); TI = rng.uniform(300, TR * 0.7)
            elif regime == "FLAIR":
                TR = rng.uniform(5000, 10000); TE = rng.uniform(70, 150)
                TI = float(np.clip(rng.normal(2980, 500), 1200, TR * 0.9)); TE = min(TE, TR * 0.9)
            else:
                TR = rng.uniform(500, 6000); TE = rng.uniform(5, min(150, TR * 0.8))
                TI = float(rng.uniform(200, TR * 0.8)) if rng.random() > 0.5 else None
            TE = min(TE, TR * 0.95)
            if TI is not None:
                TI = min(TI, TR * 0.95)
            samples.append((float(TR), float(TE), float(TI) if TI is not None else None))
    rng.shuffle(samples)
    return samples


def run_level_c(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                target_dim, cfg_scale, out_dir: Path, logger,
                n_samples: int = 60, seed: int = 6256) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"LEVEL C: Full 3D MC Sweep (N={n_samples}) — THE KEY EXPERIMENT")
    logger.info("=" * 70)
    rng = np.random.default_rng(seed)
    param_list = sample_valid_params(n_samples, rng)

    records = []
    ordering = {"WM>GM": [], "GM>CSF": [], "WM>CSF": []}

    for idx, (TR, TE, TI) in enumerate(param_list):
        ph = physics_signals(TR, TE, TI)
        prompt = build_prompt(TR, TE, TI)
        gen = generate_and_measure(args, device, autoencoder, model, scale_factor, clip_model, tokenizer,
                                   target_dim, cfg_scale, prompt, seed + idx * 37, logger)
        if gen is None:
            logger.warning(f"  [{idx+1}/{n_samples}] SKIP (TR={TR:.0f} TE={TE:.0f} TI={TI or 0:.0f})")
            continue

        ord_res = ordering_correct(ph, gen)
        for k, v in ord_res.items():
            ordering[k].append(v)

        row = {
            "idx": idx,
            "TR_ms": TR, "TE_ms": TE, "TI_ms": TI or 0.0, "use_ir": TI is not None and TI > 0,
            "phys_WM": ph["WM"], "phys_GM": ph["GM"], "phys_CSF": ph["CSF"],
            "gen_WM":  gen["WM"],  "gen_GM":  gen["GM"],  "gen_CSF":  gen["CSF"],
            "seg_method": gen.get("seg_method", "?"),
        }
        row.update({f"ord_{k.replace('>','_gt_')}": v for k, v in ord_res.items()})
        records.append(row)
        logger.info(f"  [{idx+1}/{n_samples}] TR={TR:.0f} TE={TE:.0f} TI={TI or 0:.0f} | "
                    f"WM phys={ph['WM']:.3f}/gen={gen['WM']:.3f} | "
                    f"WM>GM={ord_res['WM>GM']} GM>CSF={ord_res['GM>CSF']}")

    if not records:
        logger.error("No Level C samples completed!")
        return {}

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "level_c_3d_sweep.csv", index=False)

    stats: dict[str, Any] = {}
    for t in ["WM", "GM", "CSF"]:
        rho, _ = spearmanr(df[f"phys_{t}"], df[f"gen_{t}"])
        r, _   = pearsonr(df[f"phys_{t}"], df[f"gen_{t}"])
        stats[f"spearman_{t}"] = float(rho)
        stats[f"pearson_{t}"]  = float(r)

    for pair, outcomes in ordering.items():
        acc = float(np.mean(outcomes)) if outcomes else float("nan")
        safe_key = pair.replace(">", "_gt_")
        stats[f"ordering_acc_{safe_key}"] = acc
        logger.info(f"  Ordering [{pair}]: {acc*100:.1f}% ({sum(outcomes)}/{len(outcomes)})")

    overall = float(np.mean([v for vals in ordering.values() for v in vals])) if any(ordering.values()) else float("nan")
    stats["overall_ordering_acc"] = overall
    logger.info(f"  Overall ordering accuracy: {overall*100:.1f}%")
    logger.info(f"  Spearman rho — WM: {stats.get('spearman_WM','?'):.3f}  GM: {stats.get('spearman_GM','?'):.3f}  CSF: {stats.get('spearman_CSF','?'):.3f}")

    # --- Publication figure
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=150)
    tcolors = {"WM": "#ff7f0e", "GM": "#2ca02c", "CSF": "#1f77b4"}
    for ai, t in enumerate(["WM", "GM", "CSF"]):
        ax = axes[ai]
        x = df[f"phys_{t}"].values; y = df[f"gen_{t}"].values
        c_vec = ["#e74c3c" if row["use_ir"] else tcolors[t] for _, row in df.iterrows()]
        ax.scatter(x, y, c=c_vec, s=30, alpha=0.7, zorder=3)
        lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1.5, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{t} Physics (norm.)", fontweight="bold"); ax.set_ylabel(f"{t} Generated (norm.)", fontweight="bold")
        ax.set_title(f"{t}: ρ={stats[f'spearman_{t}']:.3f}, r={stats[f'pearson_{t}']:.3f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.35)
        se_p = mpatches.Patch(color=tcolors[t], label="Spin-Echo"); ir_p = mpatches.Patch(color="#e74c3c", label="Inv. Recovery")
        ax.legend(handles=[se_p, ir_p], fontsize=7)

    ax4 = axes[3]
    pairs = list(ordering.keys()) + ["Overall"]
    accs  = [float(np.mean(ordering[p])) if ordering[p] else 0.0 for p in list(ordering.keys())] + [overall]
    bars = ax4.bar(pairs, accs, color=["#9467bd", "#8c564b", "#e377c2", "#17becf"], edgecolor="black")
    ax4.axhline(0.5, color="gray", ls="--", label="Chance 50%")
    ax4.set_ylim(0, 1.12); ax4.set_ylabel("Ordering Accuracy", fontweight="bold")
    ax4.set_title("Contrast Ordering Accuracy\n(Physics vs. Generated)", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", fontweight="bold", fontsize=9)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_dir / "level_c_summary.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Level C summary figure to: {out_dir / 'level_c_summary.png'}")

    # Parameter space coverage
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), dpi=130)
    df["ord_overall"] = df[["ord_WM_gt_GM", "ord_GM_gt_CSF", "ord_WM_gt_CSF"]].mean(axis=1)
    cmap_ord = plt.cm.RdYlGn; norm_ord = mcolors.Normalize(0, 1)
    sc = axes2[0].scatter(df["TE_ms"], df["TR_ms"], c=df["ord_overall"], cmap=cmap_ord, norm=norm_ord, s=40, alpha=0.8)
    axes2[0].set_xlabel("TE (ms)", fontweight="bold"); axes2[0].set_ylabel("TR (ms)", fontweight="bold")
    axes2[0].set_title("TR vs TE — Ordering Accuracy", fontweight="bold")
    plt.colorbar(sc, ax=axes2[0], label="Ordering Acc.")
    has_ti = df["TI_ms"] > 0
    if has_ti.any():
        sc2 = axes2[1].scatter(df.loc[has_ti, "TI_ms"], df.loc[has_ti, "TR_ms"],
                               c=df.loc[has_ti, "ord_overall"], cmap=cmap_ord, norm=norm_ord, s=40, alpha=0.8)
        axes2[1].set_xlabel("TI (ms)", fontweight="bold"); axes2[1].set_ylabel("TR (ms)", fontweight="bold")
        axes2[1].set_title("TR vs TI — Ordering Accuracy", fontweight="bold")
        plt.colorbar(sc2, ax=axes2[1], label="Ordering Acc.")
    else:
        axes2[1].text(0.5, 0.5, "No IR samples", ha="center", va="center", transform=axes2[1].transAxes)
    axes2[2].hist(df["ord_overall"], bins=np.linspace(0, 1, 11), color="#2c7bb6", edgecolor="black", rwidth=0.85)
    axes2[2].axvline(df["ord_overall"].mean(), color="red", ls="--", lw=2, label=f"Mean={df['ord_overall'].mean():.2f}")
    axes2[2].set_xlabel("Ordering Accuracy per Sample", fontweight="bold"); axes2[2].set_ylabel("Count", fontweight="bold")
    axes2[2].set_title("Distribution of Contrast Ordering Accuracy", fontweight="bold")
    axes2[2].legend()
    plt.tight_layout()
    fig2.savefig(str(out_dir / "level_c_param_coverage.png"), bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved: {out_dir / 'level_c_param_coverage.png'}")

    return stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Pillar 4b: Joint TR×TE×TI Physics Sweep (Levels A/B/C)")
    p.add_argument("-e", "--env-config",   default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    p.add_argument("-t", "--model-def",    default="configs/config_network_rflow.json")
    p.add_argument("--mrclip-ckpt",        default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    p.add_argument("--clip-checkpoint",    default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt")
    p.add_argument("--open-clip-src",      default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    p.add_argument("--clip-context-length", type=int, default=98)
    p.add_argument("--cfg-scale",           type=float, default=4.0)
    p.add_argument("--output-dir",          default="output/eval_physics_joint")
    p.add_argument("--seed",                type=int, default=6256)
    p.add_argument("--level",               choices=["A", "B", "C", "AB", "AC", "BC", "ABC"], default="ABC")
    p.add_argument("--level-c-samples",     type=int, default=60, help="N for Level C MC sweep")
    p.add_argument("--device",              default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    args_cli = p.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("eval_physics_joint")
    device = torch.device(args_cli.device)
    logger.info(f"Device: {device}  |  Levels: {args_cli.level}  |  Output: {out_dir}")

    logger.info("Loading MR-CLIP...")
    clip_model, tokenizer = load_mrclip(args_cli.open_clip_src, args_cli.clip_checkpoint,
                                         args_cli.clip_context_length, str(device))

    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    autoencoder.eval()

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(base_unet, clip_embed_dim=512).to(device)
    sd = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    model.load_state_dict(sd, strict=False)
    model.eval()

    scale_factor = args.scale_factor if hasattr(args, "scale_factor") else 1.0
    target_dim = tuple(args.diffusion_unet_inference.get("dim",
                       args.diffusion_unet_inference.get("output_size", [256, 256, 128])))

    all_results: dict[str, Any] = {}
    lv = args_cli.level

    if "A" in lv:
        all_results.update(run_level_a(args, device, autoencoder, model, scale_factor,
                                        clip_model, tokenizer, target_dim, args_cli.cfg_scale,
                                        out_dir, logger, args_cli.seed))
    if "B" in lv:
        all_results.update(run_level_b(args, device, autoencoder, model, scale_factor,
                                        clip_model, tokenizer, target_dim, args_cli.cfg_scale,
                                        out_dir, logger, args_cli.seed))
    if "C" in lv:
        all_results.update(run_level_c(args, device, autoencoder, model, scale_factor,
                                        clip_model, tokenizer, target_dim, args_cli.cfg_scale,
                                        out_dir, logger, n_samples=args_cli.level_c_samples, seed=args_cli.seed))

    def _serial(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    clean = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean[k] = {kk: _serial(vv) for kk, vv in v.items()}
        else:
            clean[k] = _serial(v)

    out_json = out_dir / "physics_joint_summary.json"
    with open(out_json, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"\nAll results saved to: {out_json}")
    for k, v in clean.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
