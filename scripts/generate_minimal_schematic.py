import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Publication typography
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

out_dir = Path("output/eval_physics")
out_dir.mkdir(parents=True, exist_ok=True)

# Load clean synthesized T2 axial brain slice
clean_strip = np.array(Image.open(out_dir / "extracted_te_strip.png").convert("RGB"))
sw_tile = clean_strip.shape[1] / 7.0
sx0 = int(round(5 * sw_tile))
sx1 = int(round(6 * sw_tile))
slice_np = clean_strip[:, sx0:sx1]

# Crop brain tightly
thresh = 15.0
brain_pts = np.where(np.mean(slice_np, axis=-1) > thresh)
y_min, y_max = np.min(brain_pts[0]), np.max(brain_pts[0])
x_min, x_max = np.min(brain_pts[1]), np.max(brain_pts[1])
side = max(y_max - y_min, x_max - x_min)
pad = int(side * 0.02)
cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
slice_cropped = slice_np[cy - side//2 - pad : cy + side//2 + pad, cx - side//2 - pad : cx + side//2 + pad]

# Remove black background (make transparent RGBA)
gray = np.mean(slice_cropped, axis=-1)
mask = gray > thresh
rgba_brain = np.zeros((*slice_cropped.shape[:2], 4), dtype=np.uint8)
rgba_brain[..., :3] = slice_cropped
rgba_brain[..., 3] = np.where(mask, 255, 0).astype(np.uint8)

# ---------------------------------------------------------------------------
# Ultra-Compact Micro Inset (Tight Horizontal Footprint: 2.05 x 0.95 inches)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(2.05, 0.95), dpi=300, facecolor="#ffffff")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")

# 1. Outer subtle border
rect_outer = patches.FancyBboxPatch(
    (1.5, 1.5), 97.0, 33.0,
    boxstyle="round,pad=0.3,rounding_size=1.2",
    fc="#ffffff", ec="#cbd5e1", lw=1.0,
)
ax.add_patch(rect_outer)

# 2. Metadata Box (Compact pill on left)
rect_meta = patches.FancyBboxPatch(
    (3.5, 4.5), 26.0, 27.0,
    boxstyle="round,pad=0.2,rounding_size=0.9",
    fc="#f8fafc", ec="#94a3b8", lw=0.85,
)
ax.add_patch(rect_meta)

ax.text(16.5, 23.0, "Metadata", fontsize=6.2, fontweight="bold", color="#0f172a", ha="center")
ax.text(16.5, 15.0, "Prompt", fontsize=6.2, fontweight="bold", color="#0f172a", ha="center")
ax.text(16.5, 8.0, "(TE, TR)", fontsize=5.2, fontweight="bold", color="#64748b", ha="center")

# 3. Arrow through model
ax.annotate(
    "", xy=(66.0, 18.0), xytext=(30.5, 18.0),
    arrowprops=dict(arrowstyle="->", lw=1.3, color="#2563eb", shrinkA=0, shrinkB=0),
)

# Model badge
badge_core = patches.FancyBboxPatch(
    (31.5, 9.5), 31.0, 17.0,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    fc="#eff6ff", ec="#3b82f6", lw=0.85,
)
ax.add_patch(badge_core)
ax.text(47.0, 20.0, "MR-CLIP", fontsize=6.0, fontweight="bold", color="#1e40af", ha="center")
ax.text(47.0, 13.0, "Diffusion", fontsize=5.6, fontweight="bold", color="#2563eb", ha="center")

# 4. Synthesized MRI (Floating brain on right)
ax_in = fig.add_axes([0.655, 0.22, 0.32, 0.72])
ax_in.imshow(rgba_brain)
ax_in.axis("off")

ax.text(82.0, 4.5, "Synthesized", fontsize=5.2, fontweight="bold", color="#334155", ha="center")

out_file = out_dir / "metadata_conditioned_synthesis_schematic.png"
fig.savefig(str(out_file), bbox_inches="tight", dpi=300, facecolor="#ffffff", edgecolor="none")
plt.close(fig)
print("Saved compact horizontal schematic to:", out_file)
