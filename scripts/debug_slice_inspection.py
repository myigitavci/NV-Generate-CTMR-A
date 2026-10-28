#!/usr/bin/env python3
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def normalize(vol):
    v = vol.astype(np.float32)
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    p_low = np.percentile(fg, 1.0) if len(fg) > 100 else np.percentile(v, 1.0)
    p_high = np.percentile(fg, 99.0) if len(fg) > 100 else np.percentile(v, 99.0)
    return np.clip((v - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)

oasis_base = Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS31000/OAS31000_MR_d0072")
files = {
    "real_t1": str(oasis_base / "anat4/NIFTI/ur_sub-OAS31000_ses-d0072_T1w.nii.gz"),
    "syn_t1": "output/validation_experiments/exp2_multimodal_panel/subject_02_T1_Weighted.nii.gz",
    "real_t2": str(oasis_base / "anat2/NIFTI/ur_sub-OAS31000_ses-d0072_acq-TSE_T2w.nii.gz"),
    "syn_t2": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_Weighted.nii.gz",
    "real_flair": str(oasis_base / "anat3/NIFTI/ur_sub-OAS31000_ses-d0072_FLAIR.nii.gz"),
    "syn_flair": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_FLAIR.nii.gz",
}

for name, path in files.items():
    nii = nib.load(path)
    data = normalize(nii.get_fdata())
    print(f"{name}: shape = {data.shape}")
    
    # Let's save a grid of middle slices
    sx, sy, sz = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Try different slice extractions
    ax_slc = np.rot90(data[:, :, int(sz * 0.55)])
    cor_slc = np.rot90(data[:, int(sy * 0.50), :])
    sag_slc = np.rot90(data[int(sx * 0.50), :, :])
    
    # Check aspect ratio for synthetic (sz=128)
    if sz == 128:
        z_aspect = 1.36 / 0.94
        im_cor = Image.fromarray((cor_slc * 255).astype(np.uint8))
        cor_slc = np.array(im_cor.resize((cor_slc.shape[1], int(round(cor_slc.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)) / 255.0
        im_sag = Image.fromarray((sag_slc * 255).astype(np.uint8))
        sag_slc = np.array(im_sag.resize((sag_slc.shape[1], int(round(sag_slc.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)) / 255.0

    axes[0].imshow(ax_slc, cmap="gray")
    axes[0].set_title("Axial")
    axes[1].imshow(cor_slc, cmap="gray")
    axes[1].set_title("Coronal")
    axes[2].imshow(sag_slc, cmap="gray")
    axes[2].set_title("Sagittal")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"output/eval_physics/debug_slices_{name}.png", dpi=150)
    plt.close()
    print(f"Saved debug_slices_{name}.png")
