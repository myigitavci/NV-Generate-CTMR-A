#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def crop_brain_tight_square(img_pil, target_size=(512, 512), pad_ratio=0.04):
    arr = np.array(img_pil)
    if len(arr.shape) == 3:
        gray = np.mean(arr[:, :, :3], axis=-1)
    else:
        gray = arr.copy()
        
    thresh = max(15.0, float(np.percentile(gray[gray > 5], 5.0)) if (gray > 5).any() else 15.0)
    pts = np.where(gray > thresh)
    
    if len(pts[0]) > 100:
        y_min, y_max = int(np.min(pts[0])), int(np.max(pts[0]))
        x_min, x_max = int(np.min(pts[1])), int(np.max(pts[1]))
        
        h = y_max - y_min
        w = x_max - x_min
        side = max(h, w)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad
        
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        
        y0 = cy - side_padded // 2
        y1 = y0 + side_padded
        x0 = cx - side_padded // 2
        x1 = x0 + side_padded
        
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - arr.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - arr.shape[1])
        
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            if len(arr.shape) == 3:
                arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
            else:
                arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left
            
        cropped = arr[y0:y1, x0:x1]
        res_pil = Image.fromarray(cropped).resize(target_size, resample=Image.Resampling.LANCZOS)
        return res_pil
    return img_pil.resize(target_size, resample=Image.Resampling.LANCZOS)

# Let us test locating the subplots in case_01
im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
w, h = im.size
print(f"Loaded image: {w}x{h}")

# Total rows = 19
row_h = h / 19.0
col_w = w / 3.0

# Real row is row 0
# Synthetic CFG 4.0 sample 1 is row 1 + 2*3 = row 7
# Synthetic CFG 6.0 sample 1 is row 1 + 3*3 = row 10
for r_idx, label in [(0, "real"), (7, "syn_cfg4"), (10, "syn_cfg6")]:
    for c_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
        box = (
            int(c_idx * col_w),
            int(r_idx * row_h),
            int((c_idx + 1) * col_w),
            int((r_idx + 1) * row_h),
        )
        sub_img = im.crop(box)
        # Crop out title/labels (top 15% and margins)
        sw, sh = sub_img.size
        brain_sub = sub_img.crop((int(sw * 0.05), int(sh * 0.12), int(sw * 0.95), int(sh * 0.98)))
        tight = crop_brain_tight_square(brain_sub, target_size=(512, 512))
        tight.save(f"output/systematic_evaluation_30casesnew/test_extract_{label}_{plane}.png")
        print(f"Saved test_extract_{label}_{plane}.png")
