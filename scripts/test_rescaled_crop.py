#!/usr/bin/env python3
import numpy as np
from PIL import Image

def crop_brain_tight_square_aspect_corrected(arr, is_cor_or_sag=False, target_size=(512, 512), pad_ratio=0.03):
    if len(arr.shape) == 3:
        gray = np.mean(arr[:, :, :3], axis=-1)
    else:
        gray = arr.copy()
        
    thresh = max(10.0, float(np.percentile(gray[gray > 5], 5.0)) if (gray > 5).any() else 10.0)
    pts = np.where(gray > thresh)
    
    if len(pts[0]) > 50:
        y_min, y_max = int(np.min(pts[0])), int(np.max(pts[0]))
        x_min, x_max = int(np.min(pts[1])), int(np.max(pts[1]))
        
        # Subcrop just the brain
        brain_raw = arr[y_min:y_max+1, x_min:x_max+1]
        
        # If coronal or sagittal, apply physical aspect ratio stretch along height (Z axis)
        if is_cor_or_sag:
            z_aspect = 1.36 / 0.94
            h_new = int(round(brain_raw.shape[0] * z_aspect))
            w_new = brain_raw.shape[1]
            im_stretched = Image.fromarray(brain_raw).resize((w_new, h_new), resample=Image.Resampling.BILINEAR)
            brain_raw = np.array(im_stretched)
            
        # Now place into square with padding
        h, w = brain_raw.shape[:2]
        side = max(h, w)
        pad = int(side * pad_ratio)
        side_padded = side + 2 * pad
        
        if len(arr.shape) == 3:
            sq = np.zeros((side_padded, side_padded, arr.shape[2]), dtype=arr.dtype)
        else:
            sq = np.zeros((side_padded, side_padded), dtype=arr.dtype)
            
        y_off = (side_padded - h) // 2
        x_off = (side_padded - w) // 2
        
        sq[y_off:y_off+h, x_off:x_off+w] = brain_raw
        return Image.fromarray(sq).resize(target_size, resample=Image.Resampling.LANCZOS)
        
    return Image.fromarray(arr).resize(target_size, resample=Image.Resampling.LANCZOS)

# Test on synthetic coronal
im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
arr = np.array(im)

from scipy.ndimage import label, find_objects
not_white = (np.mean(arr[:, :, :3], axis=-1) < 250)
lbl, n_lbl = label(not_white)
objs = find_objects(lbl)
valid_boxes = []
for slc in objs:
    if slc is not None:
        r_slice, c_slice = slc
        h = r_slice.stop - r_slice.start
        w = c_slice.stop - c_slice.start
        if h > 200 and w > 200:
            valid_boxes.append((r_slice.start, r_slice.stop, c_slice.start, c_slice.stop))
valid_boxes = sorted(valid_boxes, key=lambda b: (b[0], b[2]))

# Box 31 is Syn CFG 6 Coronal
r0, r1, c0, c1 = valid_boxes[31]
sub = arr[r0:r1, c0:c1]
res = crop_brain_tight_square_aspect_corrected(sub, is_cor_or_sag=True, target_size=(512, 512))
res.save("output/systematic_evaluation_30casesnew/test_syn_cor_rescaled.png")
print("Saved test_syn_cor_rescaled.png")
