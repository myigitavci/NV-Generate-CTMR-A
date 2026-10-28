#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p1 = "output/validation_experiments/exp2_multimodal_panel/subject_01_T1-Weighted.nii.gz"
p2 = "output/validation_experiments/exp2_multimodal_panel/subject_01_T1_Weighted.nii.gz"
p3 = "output/validation_experiments/exp2_multimodal_panel/subject_02_T1-Weighted.nii.gz"
p4 = "output/validation_experiments/exp2_multimodal_panel/subject_02_T1_Weighted.nii.gz"

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, p, title in zip(axes.flat, [p1, p2, p3, p4], ["sub1 dash", "sub1 underscore", "sub2 dash", "sub2 underscore"]):
    v = nib.load(p).get_fdata()
    # Normalize
    v_norm = (v - np.percentile(v[v > np.percentile(v, 10)], 1.0)) / (np.percentile(v[v > np.percentile(v, 10)], 99.0) - np.percentile(v[v > np.percentile(v, 10)], 1.0) + 1e-6)
    v_norm = np.clip(v_norm, 0, 1)
    ax.imshow(np.rot90(v_norm[:, :, int(v.shape[2]*0.55)]), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("output/eval_physics/compare_t1_variants.png", dpi=150)
print("Saved compare_t1_variants.png")
