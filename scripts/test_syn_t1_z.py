#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v = nib.load("output/validation_experiments/exp2_multimodal_panel/subject_02_T1_Weighted.nii.gz").get_fdata()
bg = np.percentile(v, 10.0)
fg = v[v > bg]
p1, p99 = np.percentile(fg, 1.0), np.percentile(fg, 99.0)
v_norm = np.clip((v - p1)/(p99 - p1 + 1e-6), 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
z_fracs = [0.45, 0.50, 0.53, 0.56, 0.60, 0.64, 0.68, 0.72]
for ax, z_f in zip(axes.flat, z_fracs):
    z = int(v.shape[2] * z_f)
    ax.imshow(np.rot90(v_norm[:, :, z]), cmap="gray")
    ax.set_title(f"Syn T1 Z={z_f:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/eval_physics/test_syn_t1_z_slices.png", dpi=150)
plt.close()
print("Saved test_syn_t1_z_slices.png")
