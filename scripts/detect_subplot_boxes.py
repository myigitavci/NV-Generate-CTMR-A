#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
arr = np.array(im)
gray = np.mean(arr[:, :, :3], axis=-1)

# Find rows and columns where background is black (< 10)
# A brain scan in matplotlib imshow has a rectangular black box
# Let us profile row minimums or percentiles
row_p10 = np.percentile(gray, 10, axis=1)
col_p10 = np.percentile(gray, 10, axis=0)

# Save a downscaled version of the whole image with horizontal row lines so we can see exact row heights
vis = arr[::4, ::4].copy()
print("Downscaled vis shape:", vis.shape)

# Let us find connected components of black regions (< 15)
from scipy.ndimage import label, find_objects
black_mask = (gray < 20)
# Fill small holes (the brain itself might be > 20 inside the black box)
# Actually the brain is completely surrounded by black box!
# So if we invert (pixels != 255 (white)), that gives all subplots!
not_white = (gray < 250)
lbl, n_lbl = label(not_white)
objs = find_objects(lbl)
print(f"Found {n_lbl} non-white objects")
valid_boxes = []
for idx, slc in enumerate(objs):
    if slc is not None:
        r_slice, c_slice = slc
        h = r_slice.stop - r_slice.start
        w = c_slice.stop - c_slice.start
        if h > 200 and w > 200: # Matplotlib subplots are roughly 400-600 px wide/high
            valid_boxes.append((r_slice.start, r_slice.stop, c_slice.start, c_slice.stop))

# Sort by row then col
valid_boxes = sorted(valid_boxes, key=lambda b: (b[0], b[2]))
print(f"Total valid subplot boxes found: {len(valid_boxes)}")
for i, b in enumerate(valid_boxes[:10]):
    print(f"  Box {i}: row [{b[0]}:{b[1]}], col [{b[2]}:{b[3]}] (h={b[1]-b[0]}, w={b[3]-b[2]})")
