#!/usr/bin/env python3
"""
Diagnose the half-brain issue.
Shows: original volume, after RAS reorientation, after round_number resize.
Answers: does preprocessing itself crop the top of the brain?
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import monai
import torch
from monai.transforms import Compose


# ---- replicate round_number exactly as in diff_model_create_training_data.py ----
def round_number(number: int, base_number: int = 128) -> int:
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


# ---- minimal intensity normalizer for display ----
def norm(vol):
    v = vol.astype(np.float32)
    p1, p99 = np.percentile(v[v > v.min()], [1, 99]) if (v > v.min()).any() else (v.min(), v.max())
    return np.clip((v - p1) / max(p99 - p1, 1e-6), 0, 1)


# ---- find some real files ----
CSV = "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv"

# Data root prefix: CSV has /nfs/... paths but locally they live under /data/mr_niftis/nfs/...
DATA_ROOT = "/data/mr_niftis"

import pandas as pd
df = pd.read_csv(CSV)

def resolve(fp):
    """Try the raw path, then the local mirror under DATA_ROOT."""
    if os.path.exists(fp):
        return fp
    mirrored = os.path.join(DATA_ROOT, fp.lstrip("/"))
    if os.path.exists(mirrored):
        return mirrored
    return None

# pick 5 files that exist on disk (prefer a mix of sequences)
found = []
for fp in df["filepath"].values:
    resolved = resolve(fp)
    if resolved:
        found.append(resolved)
    if len(found) >= 5:
        break

if not found:
    print("No files found — check CSV paths")
    sys.exit(1)

print(f"Found {len(found)} files")


# ---- plain transforms (no resize) ----
plain_tf = Compose([
    monai.transforms.LoadImaged(keys="image"),
    monai.transforms.EnsureChannelFirstd(keys="image"),
    monai.transforms.Orientationd(keys="image", axcodes="RAS"),
    monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
])


fig, axes = plt.subplots(len(found), 6, figsize=(22, 4 * len(found)))
fig.suptitle("Preprocessing Diagnosis: Original → RAS → Resized\n"
             "Left 3 = coronal view.  Right 3 = sagittal view.", fontsize=13, y=1.01)

for row_i, fp in enumerate(found):
    print(f"\n[{row_i+1}/{len(found)}] {os.path.basename(fp)}")

    # ---- load original via nibabel (no transforms) ----
    nib_img = nib.load(fp)
    orig_data = nib_img.get_fdata().squeeze()
    print(f"  Original shape: {orig_data.shape}, dtype: {orig_data.dtype}")

    # ---- apply plain transforms (→ RAS, no resize) ----
    td = plain_tf({"image": fp})
    ras_vol = td["image"].numpy().squeeze()   # shape (X, Y, Z) in RAS
    print(f"  After RAS: {ras_vol.shape}")

    # ---- compute round_number resize target ----
    sx, sy, sz = ras_vol.shape
    new_dim = (round_number(sx), round_number(sy), round_number(sz))
    print(f"  round_number target: {new_dim}   (from {ras_vol.shape})")

    # ---- apply resize ----
    resize_tf = Compose([
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes="RAS"),
        monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        monai.transforms.Resized(keys="image", spatial_size=new_dim, mode="trilinear"),
    ])
    td2 = resize_tf({"image": fp})
    resized_vol = td2["image"].numpy().squeeze()
    print(f"  After resize: {resized_vol.shape}")

    # ---- measure brain fill fraction in Z ----
    bg = np.percentile(resized_vol, 20)
    mask = resized_vol > max(bg * 1.5, resized_vol.max() * 0.05)
    if mask.any():
        z_min_brain = np.where(mask.any(axis=(0, 1)))[0].min()
        z_max_brain = np.where(mask.any(axis=(0, 1)))[0].max()
        fill_frac = (z_max_brain - z_min_brain) / resized_vol.shape[2]
        top_empty = (resized_vol.shape[2] - z_max_brain) / resized_vol.shape[2]
        print(f"  Brain Z extent: {z_min_brain}-{z_max_brain}/{resized_vol.shape[2]} "
              f"({fill_frac*100:.0f}% fill, {top_empty*100:.0f}% empty top)")
    else:
        z_min_brain, z_max_brain = 0, resized_vol.shape[2] - 1

    # ---- pick slice indices ----
    y_mid_ras = ras_vol.shape[1] // 2
    y_mid_res = resized_vol.shape[1] // 2

    # Coronal = vol[:, y_mid, :]  (X vs Z)
    # Sagittal = vol[x_mid, :, :]  (Y vs Z)
    x_mid_ras = ras_vol.shape[0] // 2
    x_mid_res = resized_vol.shape[0] // 2

    def show(ax, slc, title, highlight_z=None):
        ax.imshow(np.rot90(norm(slc)), cmap="gray", aspect="auto")
        if highlight_z is not None:
            z_frac = highlight_z / slc.shape[0]  # slc cols = Z after rot90
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Original volume — pick middle coronal and sagittal
    o_cor = orig_data[:, orig_data.shape[1] // 2, :]
    o_sag = orig_data[orig_data.shape[0] // 2, :, :]

    # RAS volume
    r_cor = ras_vol[:, y_mid_ras, :]
    r_sag = ras_vol[x_mid_ras, :, :]

    # Resized volume — draw red line at z_max_brain
    res_cor = resized_vol[:, y_mid_res, :]
    res_sag = resized_vol[x_mid_res, :, :]

    show(axes[row_i, 0], o_cor, f"Orig Coronal\n{orig_data.shape}")
    show(axes[row_i, 1], r_cor, f"RAS Coronal\n{ras_vol.shape}")
    show(axes[row_i, 2], res_cor, f"Resized Coronal\n{resized_vol.shape}")
    show(axes[row_i, 3], o_sag, "Orig Sagittal")
    show(axes[row_i, 4], r_sag, "RAS Sagittal")
    show(axes[row_i, 5], res_sag, "Resized Sagittal")

    # Draw red horizontal line at brain top in resized views
    for ax_idx in [2, 5]:
        ax = axes[row_i, ax_idx]
        h = res_cor.shape[1] if ax_idx == 2 else res_sag.shape[1]
        w = res_cor.shape[0] if ax_idx == 2 else res_sag.shape[0]
        # After rot90, Z becomes vertical axis (rows), X/Y is horizontal
        # rot90 maps (row=x, col=z) → (row=z_max→z_min, col=x_min→x_max)
        # z position in image after rot90: row = h - 1 - z_idx
        y_line_brain_top = h - 1 - z_max_brain
        y_line_brain_bot = h - 1 - z_min_brain
        if 0 <= y_line_brain_top < h:
            ax.axhline(y_line_brain_top, color="red", linewidth=1.5, label="brain top")
        if 0 <= y_line_brain_bot < h:
            ax.axhline(y_line_brain_bot, color="cyan", linewidth=1.5, label="brain bot")

    axes[row_i, 0].set_ylabel(f"#{row_i+1}: {os.path.basename(fp)[:25]}", fontsize=7, rotation=0,
                              ha="right", va="center")


plt.tight_layout()
out_path = "output/diagnose_halfbrain.png"
os.makedirs("output", exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved to {out_path}")
