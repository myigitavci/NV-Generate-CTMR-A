#!/usr/bin/env python3
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

def crop_brain_tight_square(arr, target_size=(512, 512), pad_ratio=0.03):
    if len(arr.shape) == 3:
        gray = np.mean(arr[:, :, :3], axis=-1)
    else:
        gray = arr.copy()
        
    thresh = max(10.0, float(np.percentile(gray[gray > 5], 5.0)) if (gray > 5).any() else 10.0)
    pts = np.where(gray > thresh)
    
    if len(pts[0]) > 50:
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
        return Image.fromarray(cropped).resize(target_size, resample=Image.Resampling.LANCZOS)
    return Image.fromarray(arr).resize(target_size, resample=Image.Resampling.LANCZOS)

im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
arr = np.array(im)
gray = np.mean(arr[:, :, :3], axis=-1)

not_white = (gray < 250)
lbl, n_lbl = label(not_white)
objs = find_objects(lbl)
valid_boxes = []
for idx, slc in enumerate(objs):
    if slc is not None:
        r_slice, c_slice = slc
        h = r_slice.stop - r_slice.start
        w = c_slice.stop - c_slice.start
        if h > 200 and w > 200:
            valid_boxes.append((r_slice.start, r_slice.stop, c_slice.start, c_slice.stop))

valid_boxes = sorted(valid_boxes, key=lambda b: (b[0], b[2]))

# Test Real (0, 1, 2) and Syn CFG 6.0 (30, 31, 32)
names = ["test_real_ax", "test_real_cor", "test_real_sag", "test_syn_ax", "test_syn_cor", "test_syn_sag"]
indices = [0, 1, 2, 30, 31, 32]

for name, idx in zip(names, indices):
    r0, r1, c0, c1 = valid_boxes[idx]
    sub = arr[r0:r1, c0:c1]
    cropped = crop_brain_tight_square(sub, target_size=(512, 512))
    cropped.save(f"output/systematic_evaluation_30casesnew/{name}.png")
    print(f"Saved {name}.png with size {cropped.size}")
