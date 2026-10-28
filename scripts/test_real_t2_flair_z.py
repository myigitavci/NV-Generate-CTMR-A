#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

oasis_base = Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS31000/OAS31000_MR_d0072")
v_t2 = nib.load(oasis_base / "anat2/NIFTI/ur_sub-OAS31000_ses-d0072_acq-TSE_T2w.nii.gz").get_fdata()
v_fl = nib.load(oasis_base / "anat3/NIFTI/ur_sub-OAS31000_ses-d0072_FLAIR.nii.gz").get_fdata()

def norm(v):
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    p1, p99 = np.percentile(fg, 1.5), np.percentile(fg, 99.0)
    return np.clip((v - p1)/(p99 - p1 + 1e-6), 0, 1)

v_t2 = norm(v_t2)
v_fl = norm(v_fl)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
z_fracs = [0.52, 0.58, 0.62, 0.65, 0.68, 0.70, 0.73, 0.76]
for ax, z_f in zip(axes[0], z_fracs):
    z = int(v_t2.shape[2] * z_f)
    ax.imshow(np.rot90(v_t2[:, :, z]), cmap="gray")
    ax.set_title(f"Real T2 Z={z_f:.2f}")
    ax.axis("off")

for ax, z_f in zip(axes[1], z_fracs):
    z = int(v_fl.shape[2] * z_f)
    ax.imshow(np.rot90(v_fl[:, :, z]), cmap="gray")
    ax.set_title(f"Real FLAIR Z={z_f:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("output/eval_physics/test_real_t2_flair_z.png", dpi=150)
plt.close()
print("Saved test_real_t2_flair_z.png")
