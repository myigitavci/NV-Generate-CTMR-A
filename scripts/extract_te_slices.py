import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open("output/eval_physics/te_response_consistency.png").convert("RGB"))
h, w, _ = img.shape
print(f"Loaded te_response_consistency shape: {img.shape}")

# Find the mosaic in the middle
# The mosaic has non-zero pixels surrounded by black/white background
# In te_response_consistency, Panel 2 is in the middle
gray = np.mean(img, axis=2)

# Middle region is x from w*0.3 to w*0.75
mid_x0, mid_x1 = int(w * 0.32), int(w * 0.75)
mid_crop = gray[:, mid_x0:mid_x1]

# Find bounding box of the slice mosaic inside the middle region
# Slices have pixels > 10
row_has_brain = np.where(np.sum(mid_crop > 15, axis=1) > 100)[0]
col_has_brain = np.where(np.sum(mid_crop > 15, axis=0) > 100)[0]

print("Row bounds for brain mosaic:", row_has_brain[0], row_has_brain[-1])
print("Col bounds for brain mosaic:", col_has_brain[0], col_has_brain[-1])

y0, y1 = row_has_brain[0], row_has_brain[-1]
x0, x1 = mid_x0 + col_has_brain[0], mid_x0 + col_has_brain[-1]

# Expand slightly to get full black background margin
mosaic = img[y0:y1, x0:x1]
mh, mw, _ = mosaic.shape
print(f"Mosaic shape: {mw}x{mh}, aspect ratio: {mw/mh:.2f}")

# There are 7 slices
slice_w = mw / 7.0
print(f"Slice width: {slice_w}")

te_slices = []
te_vals = [10, 20, 30, 40, 60, 80, 100]
for i, te in enumerate(te_vals):
    sx0 = int(round(i * slice_w))
    sx1 = int(round((i + 1) * slice_w))
    slc = mosaic[:, sx0:sx1]
    te_slices.append((te, slc))
    Image.fromarray(slc).save(f"output/eval_physics/te_slice_{te}ms.png")

print("Successfully extracted all 7 TE slices!")
