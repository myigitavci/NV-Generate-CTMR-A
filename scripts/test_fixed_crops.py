#!/usr/bin/env python3
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

def clean_crop_and_standardize(
    raw_sub_img: np.ndarray,
    is_cor_or_sag: bool,
    target_size: tuple[int, int] = (512, 512),
    pad_ratio: float = 0.05
) -> Image.Image:
    """
    1. Extracts brain tissue from black background.
    2. If coronal or sagittal, expands height by 1.36/0.94 (1.4468x) to correct physical voxel anisotropy.
    3. Fits the brain tightly and squarely inside a target_size x target_size canvas.
    """
    if len(raw_sub_img.shape) == 3:
        gray = np.mean(raw_sub_img[:, :, :3], axis=-1)
    else:
        gray = raw_sub_img.copy()

    # Find brain tissue pixels (above background black)
    thresh = max(8.0, float(np.percentile(gray[gray > 3], 4.0)) if (gray > 3).any() else 8.0)
    pts = np.where(gray > thresh)

    if len(pts[0]) > 40:
        y0, y1 = int(np.min(pts[0])), int(np.max(pts[0]))
        x0, x1 = int(np.min(pts[1])), int(np.max(pts[1]))

        # Subcrop exact brain bounding box (no cutoffs)
        brain_tissue = raw_sub_img[y0:y1 + 1, x0:x1 + 1]

        # Correct anisotropic voxel aspect ratio for coronal and sagittal planes
        if is_cor_or_sag:
            z_aspect = 1.36 / 0.94  # ~1.4468
            h_orig, w_orig = brain_tissue.shape[:2]
            h_corr = int(round(h_orig * z_aspect))
            im_corr = Image.fromarray(brain_tissue).resize((w_orig, h_corr), resample=Image.Resampling.BILINEAR)
            brain_tissue = np.array(im_corr)

        h, w = brain_tissue.shape[:2]
        side = max(h, w)
        pad = int(round(side * pad_ratio))
        side_padded = side + 2 * pad

        # Create square canvas with black background
        if len(raw_sub_img.shape) == 3:
            square_canvas = np.zeros((side_padded, side_padded, raw_sub_img.shape[2]), dtype=raw_sub_img.dtype)
        else:
            square_canvas = np.zeros((side_padded, side_padded), dtype=raw_sub_img.dtype)

        y_offset = (side_padded - h) // 2
        x_offset = (side_padded - w) // 2
        square_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = brain_tissue

        return Image.fromarray(square_canvas).resize(target_size, resample=Image.Resampling.LANCZOS)

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

# Extract Real (0, 1, 2) and Syn (30, 31, 32)
r_ax_box, r_cor_box, r_sag_box = valid_boxes[0], valid_boxes[1], valid_boxes[2]
s_ax_box, s_cor_box, s_sag_box = valid_boxes[30], valid_boxes[31], valid_boxes[32]

r_ax = clean_crop_and_standardize(arr[r_ax_box[0]:r_ax_box[1], r_ax_box[2]:r_ax_box[3]], is_cor_or_sag=False)
r_cor = clean_crop_and_standardize(arr[r_cor_box[0]:r_cor_box[1], r_cor_box[2]:r_cor_box[3]], is_cor_or_sag=True)
r_sag = clean_crop_and_standardize(arr[r_sag_box[0]:r_sag_box[1], r_sag_box[2]:r_sag_box[3]], is_cor_or_sag=True)

s_ax = clean_crop_and_standardize(arr[s_ax_box[0]:s_ax_box[1], s_ax_box[2]:s_ax_box[3]], is_cor_or_sag=False)
s_cor = clean_crop_and_standardize(arr[s_cor_box[0]:s_cor_box[1], s_cor_box[2]:s_cor_box[3]], is_cor_or_sag=True)
s_sag = clean_crop_and_standardize(arr[s_sag_box[0]:s_sag_box[1], s_sag_box[2]:s_sag_box[3]], is_cor_or_sag=True)

r_ax.save("output/systematic_evaluation_30casesnew/test_fixed_r_ax.png")
r_cor.save("output/systematic_evaluation_30casesnew/test_fixed_r_cor.png")
r_sag.save("output/systematic_evaluation_30casesnew/test_fixed_r_sag.png")
s_ax.save("output/systematic_evaluation_30casesnew/test_fixed_s_ax.png")
s_cor.save("output/systematic_evaluation_30casesnew/test_fixed_s_cor.png")
s_sag.save("output/systematic_evaluation_30casesnew/test_fixed_s_sag.png")

print("Saved all 6 fixed test images!")
