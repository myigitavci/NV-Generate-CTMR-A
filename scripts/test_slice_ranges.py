#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def normalize(vol):
    v = vol.astype(np.float32)
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    p_low = np.percentile(fg, 1.5) if len(fg) > 100 else np.percentile(v, 1.5)
    p_high = np.percentile(fg, 99.0) if len(fg) > 100 else np.percentile(v, 99.0)
    return np.clip((v - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)

oasis_base = Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS31000/OAS31000_MR_d0072")

# Real T1
nii_r1 = nib.load(oasis_base / "anat4/NIFTI/ur_sub-OAS31000_ses-d0072_T1w.nii.gz")
v_r1 = normalize(nii_r1.get_fdata())

# Real T2
nii_r2 = nib.load(oasis_base / "anat2/NIFTI/ur_sub-OAS31000_ses-d0072_acq-TSE_T2w.nii.gz")
v_r2 = normalize(nii_r2.get_fdata())

# Real FLAIR
nii_rf = nib.load(oasis_base / "anat3/NIFTI/ur_sub-OAS31000_ses-d0072_FLAIR.nii.gz")
v_rf = normalize(nii_rf.get_fdata())

print("Real shapes:", v_r1.shape, v_r2.shape, v_rf.shape)

# Let us check multiple Z slices for Real T1
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
z_fracs = [0.55, 0.60, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78]
for ax, z_f in zip(axes.flat, z_fracs):
    z = int(v_r1.shape[2] * z_f)
    ax.imshow(np.rot90(v_r1[:, :, z]), cmap="gray")
    ax.set_title(f"Z = {z_f:.2f} ({z})")
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/eval_physics/test_real_t1_z_slices.png", dpi=150)
plt.close()
print("Saved test_real_t1_z_slices.png")
