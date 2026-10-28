#!/usr/bin/env python3
"""
generate_real_vs_synth_figure.py
================================
Publication figure comparing 6 real subjects with corresponding synthetic MRIs
conditioned on the exact same acquisition and hardware metadata prompts.

Layout:
  - 2 Rows (3 subjects per row).
  - Each subject displays [Real, Synthetic] side by side (3 x 2 pairs per row).
  - Row 1: T2-Weighted subjects (Subject 1, Subject 2, Subject 3).
  - Row 2: T2-FLAIR subjects (Subject 4, Subject 5, Subject 6).
  - Clean styling, high resolution (512x512 tiles), no scanner specification headers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from scripts.utils import define_instance

# Output directory
out_dir = Path("output/eval_real_synth")
slice_dir = out_dir / "slices_512x512"
panel_dir = out_dir / "panels"
for d in [slice_dir, panel_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 6 Real Subjects from Unseen OASIS Dataset
SUBJECTS = [
    # Row 1: 3 T2-Weighted Subjects
    {
        "id": "sub_01_t2w",
        "subject_label": "Subject 1 (T2w)",
        "modality": "T2W",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-bb9e7d73d500d9506f1a6466fbb3d8/ses-20131211164020/anat/ur_sub-bb9e7d73d500d9506f1a6466fbb3d8_ses-20131211164020_run-4_T2w_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax PD/T2, SE, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.10184, 5.9, NONE, 90)"
    },
    {
        "id": "sub_02_t2w",
        "subject_label": "Subject 2 (T2w)",
        "modality": "T2W",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-f4f30514bd44be950eb9100127a593/ses-20190311173347/anat/ur_sub-f4f30514bd44be950eb9100127a593_ses-20190311173347_run-3_T2w_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (AXIAL PD/T2, SE, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.11336, 4.52, NONE, 90)"
    },
    {
        "id": "sub_03_t2w",
        "subject_label": "Subject 3 (T2w)",
        "modality": "T2W",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-b3e18fc2f38ef59afdfff82e28cbc2/ses-20140514154406/anat/ur_sub-b3e18fc2f38ef59afdfff82e28cbc2_ses-20140514154406_run-2_T2w_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), Acquisition (Description, Sequence, Variant): (t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.098, 3.25, NONE, 150)"
    },
    # Row 2: 3 FLAIR Subjects
    {
        "id": "sub_04_flair",
        "subject_label": "Subject 4 (FLAIR)",
        "modality": "FLAIR",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-ec20a4c25c87bab6f702c0fb4d6576/ses-20150709132847/anat/ur_sub-ec20a4c25c87bab6f702c0fb4d6576_ses-20150709132847_run-5_FLAIR_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax T2 FLAIR, SE_IR, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.12926, 8.002, 2, 90)"
    },
    {
        "id": "sub_05_flair",
        "subject_label": "Subject 5 (FLAIR)",
        "modality": "FLAIR",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-062664f6e8380a87de426f96d3de72/ses-20140519155043/anat/ur_sub-062664f6e8380a87de426f96d3de72_ses-20140519155043_run-3_FLAIR_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (t2_tirm_tra_dark-fl_pat2, SE_IR, SK_SP_MP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.114, 9, 2.5, 150)"
    },
    {
        "id": "sub_06_flair",
        "subject_label": "Subject 6 (FLAIR)",
        "modality": "FLAIR",
        "real_path": "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-6cabe8f178e39be1e02a62df150396/ses-20151216111049/anat/ur_sub-6cabe8f178e39be1e02a62df150396_ses-20151216111049_run-17_FLAIR_stripped.nii",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, DISCOVERY_MR750, 3), Acquisition (Description, Sequence, Variant): (Ax T2 FLAIR, SE_IR, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.128644, 8, 2, 111)"
    },
]


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


def extract_real_slice(vol_np: np.ndarray, is_flair: bool = False) -> np.ndarray:
    v = vol_np.astype(np.float32)
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1 = np.percentile(fg, 1.0)
        p99 = np.percentile(fg, 99.5 if is_flair else 99.0)
    else:
        p1, p99 = v.min(), v.max()
    v_norm = np.clip((v - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    thresh = max(0.06, float(np.percentile(v_norm, 15.0)))
    pts = np.where(v_norm > thresh)
    z_min, z_max = int(np.percentile(pts[2], 1.0)), int(np.percentile(pts[2], 99.0))

    z_cands = list(range(int(z_min + (z_max - z_min) * 0.52), int(z_min + (z_max - z_min) * 0.68) + 1))
    z_idx = z_cands[np.argmax([np.sum(v_norm[:, :, z] > thresh) for z in z_cands])]

    ax = np.rot90(v_norm[:, :, z_idx])
    return crop_brain_tight(ax)


def extract_syn_slice(vol_np: np.ndarray, is_flair: bool = False) -> np.ndarray:
    v = vol_np.astype(np.float32)
    bg = np.percentile(v, 10.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1 = np.percentile(fg, 1.0)
        p99 = np.percentile(fg, 99.5 if is_flair else 99.0)
    else:
        p1, p99 = v.min(), v.max()
    v_norm = np.clip((v - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    thresh = max(0.06, float(np.percentile(v_norm, 15.0)))
    pts = np.where(v_norm > thresh)
    z_min, z_max = int(np.percentile(pts[2], 1.0)), int(np.percentile(pts[2], 99.0))

    z_cands = list(range(int(z_min + (z_max - z_min) * 0.40), int(z_min + (z_max - z_min) * 0.65) + 1))
    z_idx = z_cands[np.argmax([np.sum(v_norm[:, :, z] > thresh) for z in z_cands])]

    ax = np.rot90(v_norm[:, :, z_idx])
    return crop_brain_tight(ax)


def save_512(arr: np.ndarray, out_path: Path):
    im = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
    im_512 = im.resize((512, 512), resample=Image.Resampling.LANCZOS)
    im_512.save(out_path)


def synthesize_all():
    logger = setup_logging("real_vs_synth_gen")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract all real slices
    for sub in SUBJECTS:
        sid = sub["id"]
        real_p = Path(sub["real_path"])
        real_out = slice_dir / f"{sid}_real.png"
        if not real_out.exists():
            nii = nib.load(str(real_p))
            slc = extract_real_slice(nii.get_fdata(), is_flair=(sub["modality"] == "FLAIR"))
            save_512(slc, real_out)
            logger.info(f"Saved real slice for {sid}")

    # Synthesize missing synthetic cases
    missing_subs = [s for s in SUBJECTS if not (slice_dir / f"{s['id']}_syn.png").exists()]
    if not missing_subs:
        logger.info("All synthetic slices already exist.")
        return

    args = load_config(
        "configs/environment_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_network_rflow.json"
    )
    target_dim = tuple(args.diffusion_unet_inference["dim"])

    logger.info("Loading Autoencoder & UNet...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

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
    for sub in missing_subs:
        sid = sub["id"]
        prompt = ensure_axial_prompt(sub["prompt"])
        logger.info(f"Synthesizing {sub['subject_label']} with seed {seed}...")
        with torch.no_grad():
            tok = tokenizer(prompt).to(device)
            emb = clip_model.encode_text(tok).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)

        vol_np, _ = generate_sample(
            args, device, autoencoder, model, scale_factor, emb,
            target_dim, 3.0, seed=seed
        )

        if vol_np is not None:
            syn_slc = extract_syn_slice(vol_np, is_flair=(sub["modality"] == "FLAIR"))
            save_512(syn_slc, slice_dir / f"{sid}_syn.png")
            logger.info(f"Saved synthetic slice for {sid}")


def render_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 2 Rows (Row 1: Subjects 1-3, Row 2: Subjects 4-6)
    # 8 Grid columns per row:
    # Pair 1: cols 0, 1 | spacer 2 | Pair 2: cols 3, 4 | spacer 5 | Pair 3: cols 6, 7
    fig = plt.figure(figsize=(18.0, 7.2), dpi=220)
    gs = gridspec.GridSpec(
        2, 8,
        width_ratios=[1.0, 1.0, 0.12, 1.0, 1.0, 0.12, 1.0, 1.0],
        wspace=0.015, hspace=0.18,
        left=0.01, right=0.99, top=0.91, bottom=0.02
    )

    pair_col_mapping = [(0, 1), (3, 4), (6, 7)]

    row1_subs = SUBJECTS[:3]
    row2_subs = SUBJECTS[3:]

    for r_idx, row_subs in enumerate([row1_subs, row2_subs]):
        for p_idx, sub in enumerate(row_subs):
            c_real, c_syn = pair_col_mapping[p_idx]
            sid = sub["id"]

            # Real tile
            ax_r = fig.add_subplot(gs[r_idx, c_real])
            im_r_p = slice_dir / f"{sid}_real.png"
            if im_r_p.exists():
                ax_r.imshow(Image.open(im_r_p), cmap="gray")
            ax_r.axis("off")
            ax_r.set_title("Real", fontsize=13.0, fontweight="bold", pad=6)

            # Synthetic tile
            ax_s = fig.add_subplot(gs[r_idx, c_syn])
            im_s_p = slice_dir / f"{sid}_syn.png"
            if im_s_p.exists():
                ax_s.imshow(Image.open(im_s_p), cmap="gray")
            ax_s.axis("off")
            ax_s.set_title("Synthetic", fontsize=13.0, fontweight="bold", pad=6)

    # Super-labels above each subject pair
    # Row 1 Subject labels (T2w)
    fig.text(0.165, 0.965, "Subject 1 (T2w)", fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.500, 0.965, "Subject 2 (T2w)", fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.835, 0.965, "Subject 3 (T2w)", fontsize=14.5, fontweight="bold", color="black", ha="center")

    # Row 2 Subject labels (FLAIR)
    fig.text(0.165, 0.475, "Subject 4 (FLAIR)", fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.500, 0.475, "Subject 5 (FLAIR)", fontsize=14.5, fontweight="bold", color="black", ha="center")
    fig.text(0.835, 0.475, "Subject 6 (FLAIR)", fontsize=14.5, fontweight="bold", color="black", ha="center")

    out_file = panel_dir / "real_vs_synthetic_6subjects_comparison.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved master real vs synth figure -> {out_file}")


if __name__ == "__main__":
    synthesize_all()
    render_figure()
