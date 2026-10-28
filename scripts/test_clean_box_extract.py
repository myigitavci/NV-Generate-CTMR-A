#!/usr/bin/env python3
import numpy as np
from PIL import Image

def extract_clean_brain_from_box(cell_img, target_size=(512, 512), pad_ratio=0.04):
    arr = np.array(cell_img)
    if len(arr.shape) == 3:
        # Convert RGB to grayscale
        gray = np.mean(arr[:, :, :3], axis=-1)
    else:
        gray = arr.copy()
        
    # The brain parenchyma is non-zero (greater than near black, e.g. > 15)
    # But inside the black subplot box, background is near 0.
    # Text on white background is dark on white (> 200 background).
    
    # Let's find pixels that are inside the dark subplot box (where background is < 40)
    # and brain tissue is > 10.
    # Specifically, find the largest contiguous black box or find brain tissue directly:
    # Brain tissue is gray > 15 and surrounding is black (< 30), whereas text is dark surrounded by white (> 200).
    
    # Filter for brain pixels: gray > 12 where the local neighborhood is not white
    is_brain = (gray > 12) & (gray < 240)
    # Also exclude any rows/cols near the top where white background is present
    # Let us find rows where mean intensity is < 150 (the black subplot box)
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    
    black_rows = np.where(row_means < 150)[0]
    black_cols = np.where(col_means < 150)[0]
    
    if len(black_rows) > 20 and len(black_cols) > 20:
        r0, r1 = black_rows[0], black_rows[-1]
        c0, c1 = black_cols[0], black_cols[-1]
        
        # Sub-crop the black box
        black_box = gray[r0:r1, c0:c1]
        orig_box = arr[r0:r1, c0:c1]
        
        # Now find the brain tissue within this black box
        pts = np.where(black_box > 12)
        if len(pts[0]) > 50:
            y_min, y_max = np.min(pts[0]), np.max(pts[0])
            x_min, x_max = np.min(pts[1]), np.max(pts[1])
            
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
            pad_bottom = max(0, y1 - orig_box.shape[0])
            pad_left = max(0, -x0)
            pad_right = max(0, x1 - orig_box.shape[1])
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                if len(orig_box.shape) == 3:
                    orig_box = np.pad(orig_box, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
                else:
                    orig_box = np.pad(orig_box, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
                y0 += pad_top
                y1 += pad_top
                x0 += pad_left
                x1 += pad_left
                
            cropped = orig_box[y0:y1, x0:x1]
            return Image.fromarray(cropped).resize(target_size, resample=Image.Resampling.LANCZOS)
            
    return Image.fromarray(arr).resize(target_size, resample=Image.Resampling.LANCZOS)

im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
w, h = im.size
row_h = h / 19.0
col_w = w / 3.0

# Extract real axial, coronal, sagittal and syn axial, coronal, sagittal
for r_idx, label in [(0, "real"), (10, "syn_cfg6")]:
    for c_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
        box = (
            int(c_idx * col_w),
            int(r_idx * row_h),
            int((c_idx + 1) * col_w),
            int((r_idx + 1) * row_h),
        )
        cell = im.crop(box)
        clean = extract_clean_brain_from_box(cell, target_size=(512, 512))
        clean.save(f"output/systematic_evaluation_30casesnew/clean_{label}_{plane}.png")
        print(f"Saved clean_{label}_{plane}.png")
