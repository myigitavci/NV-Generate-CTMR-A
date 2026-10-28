#!/usr/bin/env python3
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

def standardize_brain_slice(
    raw_sub_img: np.ndarray,
    is_cor_or_sag: bool,
    target_size: tuple[int, int] = (512, 512),
    margin_ratio: float = 0.04
) -> Image.Image:
    """
    Standardizes any brain slice (Axial, Coronal, Sagittal, Real or Syn):
      1. Isolates brain tissue from black background.
      2. If coronal or sagittal, expands height by 1.85x so the vertical brain axis
         is anatomically proportional to its transverse axis.
      3. Centers the brain within a target_size x target_size square with consistent margin.
    """
    if len(raw_sub_img.shape) == 3:
        gray = np.mean(raw_sub_img[:, :, :3], axis=-1)
    else:
        gray = raw_sub_img.copy()

    # Brain tissue threshold
    thresh = max(8.0, float(np.percentile(gray[gray > 3], 4.0)) if (gray > 3).any() else 8.0)
    pts = np.where(gray > thresh)

    if len(pts[0]) > 40:
        y0, y1 = int(np.min(pts[0])), int(np.max(pts[0]))
        x0, x1 = int(np.min(pts[1])), int(np.max(pts[1]))

        # Extract brain bounding box
        brain = raw_sub_img[y0:y1 + 1, x0:x1 + 1]

        # Correct vertical squashing for Coronal & Sagittal
        if is_cor_or_sag:
            h_orig, w_orig = brain.shape[:2]
            # In (128, 256) volumes, height is ~50% of width. 
            # A 1.85x - 1.90x scaling restores true anatomical roundness.
            h_corr = int(round(h_orig * 1.88))
            im_corr = Image.fromarray(brain).resize((w_orig, h_corr), resample=Image.Resampling.BILINEAR)
            brain = np.array(im_corr)

        h, w = brain.shape[:2]
        side = max(h, w)
        pad = int(round(side * margin_ratio))
        side_padded = side + 2 * pad

        # Black background square
        if len(raw_sub_img.shape) == 3:
            sq = np.zeros((side_padded, side_padded, raw_sub_img.shape[2]), dtype=raw_sub_img.dtype)
        else:
            sq = np.zeros((side_padded, side_padded), dtype=raw_sub_img.dtype)

        y_off = (side_padded - h) // 2
        x_off = (side_padded - w) // 2
        sq[y_off:y_off + h, x_off:x_off + w] = brain

        return Image.fromarray(sq).resize(target_size, resample=Image.Resampling.LANCZOS)

    return Image.fromarray(raw_sub_img).resize(target_size, resample=Image.Resampling.LANCZOS)

# Test on case 01
im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
arr = np.array(im)

not_white = (np.mean(arr[:, :, :3], axis=-1) < 250)
lbl, n_lbl = label(not_white)
objs = find_objects(lbl)
valid_boxes = []
for slc in objs:
    if slc is not None:
        r_slice, c_slice = slc
        h = r_slice.stop - r_slice.start
        w = c_slice.stop - c_slice.start
        if h > 180 and w > 180:
            valid_boxes.append((r_slice.start, r_slice.stop, c_slice.start, c_slice.stop))
valid_boxes = sorted(valid_boxes, key=lambda b: (b[0], b[2]))

# Real (0, 1, 2) and Syn (30, 31, 32)
r_ax_box, r_cor_box, r_sag_box = valid_boxes[0], valid_boxes[1], valid_boxes[2]
s_ax_box, s_cor_box, s_sag_box = valid_boxes[30], valid_boxes[31], valid_boxes[32]

r_ax = standardize_brain_slice(arr[r_ax_box[0]:r_ax_box[1], r_ax_box[2]:r_ax_box[3]], is_cor_or_sag=False)
r_cor = standardize_brain_slice(arr[r_cor_box[0]:r_cor_box[1], r_cor_box[2]:r_cor_box[3]], is_cor_or_sag=True)
r_sag = standardize_brain_slice(arr[r_sag_box[0]:r_sag_box[1], r_sag_box[2]:r_sag_box[3]], is_cor_or_sag=True)

s_ax = standardize_brain_slice(arr[s_ax_box[0]:s_ax_box[1], s_ax_box[2]:s_ax_box[3]], is_cor_or_sag=False)
s_cor = standardize_brain_slice(arr[s_cor_box[0]:s_cor_box[1], s_cor_box[2]:s_cor_box[3]], is_cor_or_sag=True)
s_sag = standardize_brain_slice(arr[s_sag_box[0]:s_sag_box[1], s_sag_box[2]:s_sag_box[3]], is_cor_or_sag=True)

r_ax.save("output/systematic_evaluation_30casesnew/test_std_r_ax.png")
r_cor.save("output/systematic_evaluation_30casesnew/test_std_r_cor.png")
r_sag.save("output/systematic_evaluation_30casesnew/test_std_r_sag.png")
s_ax.save("output/systematic_evaluation_30casesnew/test_std_s_ax.png")
s_cor.save("output/systematic_evaluation_30casesnew/test_std_s_cor.png")
s_sag.save("output/systematic_evaluation_30casesnew/test_std_s_sag.png")

print("Saved all standardized test slices!")
