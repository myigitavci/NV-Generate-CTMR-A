#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

cache_dir = Path("output/eval_memorization/diversity_cache")
v = np.load(cache_dir / "vol_T2W_real_seed101.npy")

# Let's save a 2x3 grid of different orientations of coronal and sagittal for seed 101:
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Sagittal is slice along X:
x_mid = 128
y_mid = 128

# Options for sagittal:
# vol[x_mid, :, :] has shape (256, 128). Axis 0 is Y (AP), Axis 1 is Z (IS)
s0 = v[x_mid, :, :].T[::-1, :] # IS vertical, AP horizontal
s1 = np.fliplr(s0)
s2 = np.rot90(v[x_mid, :, :])

axes[0, 0].imshow(s0, cmap="gray")
axes[0, 0].set_title("Sagittal s0 (T[::-1,:])")
axes[0, 1].imshow(s1, cmap="gray")
axes[0, 1].set_title("Sagittal s1 (fliplr(s0))")
axes[0, 2].imshow(s2, cmap="gray")
axes[0, 2].set_title("Sagittal s2 (rot90)")

# Options for coronal:
# vol[:, y_mid, :] has shape (256, 128). Axis 0 is X (LR), Axis 1 is Z (IS)
c0 = v[:, y_mid, :].T[::-1, :] # IS vertical, LR horizontal
c1 = np.fliplr(c0)
c2 = np.rot90(v[:, y_mid, :])

axes[1, 0].imshow(c0, cmap="gray")
axes[1, 0].set_title("Coronal c0 (T[::-1,:])")
axes[1, 1].imshow(c1, cmap="gray")
axes[1, 1].set_title("Coronal c1 (fliplr(c0))")
axes[1, 2].imshow(c2, cmap="gray")
axes[1, 2].set_title("Coronal c2 (rot90)")

plt.tight_layout()
plt.savefig("mpr_check_grid.png", dpi=150)
print("Saved mpr_check_grid.png")
