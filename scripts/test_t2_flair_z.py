#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v_t2 = nib.load("output/validation_experiments/exp2_multimodal_panel/subject_02_T2_Weighted.nii.gz").get_fdata()
v_fl = nib.load("output/validation_experiments/exp2_multimodal_panel/subject_02_T2_FLAIR.nii.gz").get_fdata()

def norm(v):
    bg = np.percentile(v, 10.0)
    fg = v[v > bg]
    p1, p99 = np.percentile(fg, 1.0), np.percentile(fg, 99.0)
    return np.clip((v - p1)/(p99 - p1 + 1e-6), 0, 1)

v_t2 = norm(v_t2)
v_fl = norm(v_fl)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
z_fracs = [0.45, 0.50, 0.53, 0.56, 0.60, 0.64, 0.68, 0.72]
for ax, z_f in zip(axes[0], z_fracs):
    z = int(v_t2.shape[2] * z_f)
    ax.imshow(np.rot90(v_t2[:, :, z]), cmap="gray")
    ax.set_title(f"T2 Z={z_f:.2f}")
    ax.axis("off")

for ax, z_f in zip(axes[1], z_fracs):
    z = int(v_fl.shape[2] * z_f)
    ax.imshow(np.rot90(v_fl[:, :, z]), cmap="gray")
    ax.set_title(f"FLAIR Z={z_f:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("output/eval_physics/test_t2_flair_z_slices.png", dpi=150)
plt.close()
print("Saved test_t2_flair_z_slices.png")
