#!/usr/bin/env python3
"""
generate_unseen_oasis_scanners_figure.py
========================================
Synthesizes 3D brain MRIs conditioned on real metadata prompts from the
unseen OASIS scanner dataset (data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv).

Layout:
  - 3 T2-Weighted columns (Siemens Avanto 1.5T, Siemens Aera 1.5T, GE Signa HDxt 1.5T)
  - 3 T2-FLAIR columns (GE Discovery MR750 3.0T, GE Signa HDxt 1.5T, GE Discovery MR450 1.5T)
  - 3 Rows (Axial, Coronal, Sagittal) with NO plane text labels.
  - Compact, publication quality.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    ensure_axial_prompt,
    generate_sample,
)
from scripts.generate_scanner_variation_brains import (
    extract_standard_slices,
    save_512_slice,
)
from scripts.utils import define_instance

# Output directories
out_base = Path("output/eval_hardware/unseen_oasis_scanners")
slice_dir = out_base / "slices_512x512"
vol_dir = out_base / "volumes"
panel_dir = out_base / "panels"
for d in [slice_dir, vol_dir, panel_dir]:
    d.mkdir(parents=True, exist_ok=True)


def crop_brain_tight(slice_2d: np.ndarray, pad_ratio: float = 0.04) -> np.ndarray:
    thresh = max(0.06, float(np.percentile(slice_2d[slice_2d > 0], 5.0)) if (slice_2d > 0).any() else 0.06)
    pts = np.where(slice_2d > thresh)
    if len(pts[0]) > 50:
        y_min, y_max = int(np.min(pts[0])), int(np.max(pts[0]))
        x_min, x_max = int(np.min(pts[1])), int(np.max(pts[1]))

        h = y_max - y_min
        w = x_max - x_min
        side = max(h, w)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        y0 = cy - side_padded // 2
        y1 = y0 + side_padded
        x0 = cx - side_padded // 2
        x1 = x0 + side_padded

        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - slice_2d.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - slice_2d.shape[1])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            slice_2d = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left

        return slice_2d[y0:y1, x0:x1]
    return slice_2d


def extract_unseen_slices(vol_np: np.ndarray, modality: str = "T2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = vol_np.astype(np.float32)
    bg = np.percentile(v, 10.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1 = np.percentile(fg, 1.0)
        p99 = np.percentile(fg, 99.0) if modality != "FLAIR" else np.percentile(fg, 99.5)
    else:
        p1, p99 = v.min(), v.max()
    v_norm = np.clip((v - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    thresh = max(0.06, float(np.percentile(v_norm, 15.0)))
    pts = np.where(v_norm > thresh)
    if len(pts[0]) > 500:
        x_min, x_max = int(np.percentile(pts[0], 1.0)), int(np.percentile(pts[0], 99.0))
        y_min, y_max = int(np.percentile(pts[1], 1.0)), int(np.percentile(pts[1], 99.0))
        z_min, z_max = int(np.percentile(pts[2], 1.0)), int(np.percentile(pts[2], 99.0))

        # Axial: max area slice in mid 40%-65% of Z
        z_cands = list(range(int(z_min + (z_max - z_min) * 0.40), int(z_min + (z_max - z_min) * 0.65) + 1))
        z_idx = z_cands[np.argmax([np.sum(v_norm[:, :, z] > thresh) for z in z_cands])]

        # Coronal: slice through mid ventricles (45%-55% of Y)
        y_cands = list(range(int(y_min + (y_max - y_min) * 0.42), int(y_min + (y_max - y_min) * 0.58) + 1))
        y_idx = y_cands[np.argmax([np.sum(v_norm[:, y, :] > thresh) for y in y_cands])]

        # Sagittal: midline (50% of X)
        x_idx = int(x_min + (x_max - x_min) * 0.50)
    else:
        x_idx, y_idx, z_idx = v.shape[0] // 2, v.shape[1] // 2, v.shape[2] // 2

    ax = np.rot90(v_norm[:, :, z_idx])
    cor = np.rot90(v_norm[:, y_idx, :])
    sag = np.rot90(v_norm[x_idx, :, :])

    # Scale Z for isotropic visualization
    z_aspect = 1.36 / 0.94
    im_cor = Image.fromarray((np.clip(cor, 0, 1) * 255).astype(np.uint8))
    cor = np.array(im_cor.resize((cor.shape[1], int(round(cor.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

    im_sag = Image.fromarray((np.clip(sag, 0, 1) * 255).astype(np.uint8))
    sag = np.array(im_sag.resize((sag.shape[1], int(round(sag.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255.0

    return crop_brain_tight(ax), crop_brain_tight(cor), crop_brain_tight(sag)


def save_512_slice(slice_arr: np.ndarray, out_path: Path):
    im = Image.fromarray((np.clip(slice_arr, 0.0, 1.0) * 255).astype(np.uint8))
    im_512 = im.resize((512, 512), resample=Image.Resampling.LANCZOS)
    im_512.save(out_path)


# 6 Scanner Cases with identical 3-scanner ordering across T2 and FLAIR
CASES = [
    # 3 T2 Cases
    {
        "id": "t2_siemens_avanto_1.5T",
        "modality": "T2",
        "scanner_title": "Siemens Avanto (1.5T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), "
            "Acquisition (Description, Sequence, Variant): (t2_tse_tra, SE, SK_SP_OSP), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.095, 3.8, NONE, 150)"
        ),
    },
    {
        "id": "t2_ge_discovery_mr750_3.0T",
        "modality": "T2",
        "scanner_title": "GE Discovery MR750 (3.0T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, DISCOVERY_MR750, 3), "
            "Acquisition (Description, Sequence, Variant): (Ax T2 FRFSE, SE, SK), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.065, 4.38, NONE, 111)"
        ),
    },
    {
        "id": "t2_ge_signa_hdxt_1.5T",
        "modality": "T2",
        "scanner_title": "GE Signa HDxt (1.5T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), "
            "Acquisition (Description, Sequence, Variant): (Ax T2, SE, SK), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.102, 5.2, NONE, 90)"
        ),
    },
    # 3 FLAIR Cases (matching scanner order)
    {
        "id": "flair_siemens_avanto_1.5T",
        "modality": "FLAIR",
        "scanner_title": "Siemens Avanto (1.5T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), "
            "Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fl_pat2, SE_IR, SK_SP_MP_OSP), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.114, 9, 2.5, 150)"
        ),
    },
    {
        "id": "flair_ge_discovery_mr750_3.0T",
        "modality": "FLAIR",
        "scanner_title": "GE Discovery MR750 (3.0T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, DISCOVERY_MR750, 3), "
            "Acquisition (Description, Sequence, Variant): (Ax T2 FLAIR, SE_IR, SK), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.128644, 8, 2, 111)"
        ),
    },
    {
        "id": "flair_ge_signa_hdxt_1.5T",
        "modality": "FLAIR",
        "scanner_title": "GE Signa HDxt (1.5T)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), "
            "Acquisition (Description, Sequence, Variant): (Ax T2 FLAIR, SE_IR, SK), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.12926, 8.002, 2, 90)"
        ),
    },
]


def synthesize_volumes():
    logger = setup_logging("unseen_oasis_gen")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args = load_config(
        "configs/environment_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_network_rflow.json"
    )
    target_dim = tuple(args.diffusion_unet_inference["dim"])

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
    unet_ckpt = torch.load("models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt", map_location=device, weights_only=False)
    state = unet_ckpt.get("model_state_dict", unet_ckpt.get("unet_state_dict", unet_ckpt.get("state_dict", unet_ckpt)))
    cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys()):
        model.load_state_dict(cleaned_state, strict=False)
    else:
        model.unet.load_state_dict(cleaned_state, strict=False)
    model.eval()
    scale_factor = float(unet_ckpt.get("scale_factor", 1.0422))

    clip_ckpt = "/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt"
    clip_model, tokenizer = load_mrclip("/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src", clip_ckpt, 98, str(device))
    clip_model.eval()

    seed = 42

    for case in CASES:
        case_id = case["id"]
        nii_path = vol_dir / f"{case_id}.nii.gz"
        # Check if all 3 slices already exist
        if all((slice_dir / f"{case_id}_{plane}.png").exists() for plane in ["axial", "coronal", "sagittal"]):
            continue

        prompt = ensure_axial_prompt(case["prompt"])
        logger.info(f"Synthesizing {case['scanner_title']} ({case['modality']}) with seed {seed}...")
        with torch.no_grad():
            tok = tokenizer(prompt).to(device)
            emb = clip_model.encode_text(tok).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            
        vol_np, _ = generate_sample(
            args, device, autoencoder, model, scale_factor, emb,
            target_dim, 3.0, seed=seed
        )
        
        if vol_np is not None:
            nib.save(nib.Nifti1Image(vol_np.astype(np.float32), affine=np.eye(4)), nii_path)
            ax_slc, cor_slc, sag_slc = extract_unseen_slices(vol_np, modality=case["modality"])
            save_512_slice(ax_slc, slice_dir / f"{case_id}_axial.png")
            save_512_slice(cor_slc, slice_dir / f"{case_id}_coronal.png")
            save_512_slice(sag_slc, slice_dir / f"{case_id}_sagittal.png")
            logger.info(f"Saved slices for {case_id}")


def render_compact_figure():
    import matplotlib.gridspec as gridspec
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    planes = ["axial", "coronal", "sagittal"]
    n_rows = len(planes)
    
    # 7 columns: 3 T2 cols, 1 spacer gap col (0.14 width), 3 FLAIR cols
    fig = plt.figure(figsize=(17.5, 2.75 * n_rows), dpi=220)
    gs = gridspec.GridSpec(
        n_rows, 7,
        width_ratios=[1.0, 1.0, 1.0, 0.14, 1.0, 1.0, 1.0],
        wspace=0.015, hspace=0.02,
        left=0.01, right=0.99, top=0.88, bottom=0.01
    )

    t2_cases = CASES[:3]
    flair_cases = CASES[3:]

    # Render T2 columns (grid indices 0, 1, 2)
    for c_idx, case in enumerate(t2_cases):
        case_id = case["id"]
        for r_idx, plane in enumerate(planes):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            img_p = slice_dir / f"{case_id}_{plane}.png"
            if img_p.exists():
                im = Image.open(img_p)
                ax.imshow(im, cmap="gray")
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", color="red")
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            
            if r_idx == 0:
                ax.set_title(case["scanner_title"], fontsize=12.5, fontweight="bold", pad=10)

    # Render FLAIR columns (grid indices 4, 5, 6)
    for c_idx, case in enumerate(flair_cases):
        grid_col = c_idx + 4
        case_id = case["id"]
        for r_idx, plane in enumerate(planes):
            ax = fig.add_subplot(gs[r_idx, grid_col])
            img_p = slice_dir / f"{case_id}_{plane}.png"
            if img_p.exists():
                im = Image.open(img_p)
                ax.imshow(im, cmap="gray")
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", color="red")
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            
            if r_idx == 0:
                ax.set_title(case["scanner_title"], fontsize=12.5, fontweight="bold", pad=10)

    # Add Modality Super-Headers centered across the respective sections
    # T2 spans roughly x = 0.01 to 0.48 -> center ~0.245
    # FLAIR spans roughly x = 0.52 to 0.99 -> center ~0.755
    fig.text(0.245, 0.955, "Synthetic T2w", fontsize=18.0, fontweight="bold", color="black", ha="center")
    fig.text(0.755, 0.955, "Synthetic FLAIR", fontsize=18.0, fontweight="bold", color="black", ha="center")

    out_file = panel_dir / "unseen_oasis_scanners_t2_flair_matrix.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    # Also update master path
    master_panel_path = Path("output/eval_hardware/scanner_variation_brains/panels/master_vendor_cross_contrast_axial_matrix.png")
    plt.savefig(master_panel_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved master figure -> {out_file}")


if __name__ == "__main__":
    all_slices_exist = all(
        (slice_dir / f"{case['id']}_{plane}.png").exists()
        for case in CASES
        for plane in ["axial", "coronal", "sagittal"]
    )
    if not all_slices_exist:
        print("Synthesizing unseen OASIS scanner cases...", flush=True)
        synthesize_volumes()
    print("Rendering compact figure...", flush=True)
    render_compact_figure()
    print("Done!", flush=True)
