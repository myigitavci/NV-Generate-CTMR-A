#!/usr/bin/env python3
"""
save_10_example_slices.py
==========================
Extracts and saves tightly cropped, aspect-ratio corrected, standardized 512x512
Axial, Coronal, and Sagittal slice images for 10 representative test cases from
output/systematic_evaluation_30casesnew/.

Guarantees:
  - Exact identical 512x512 square image dimensions across all planes and cases.
  - Consistent scaling where the brain fills ~92% of the canvas.
  - Voxel aspect ratio correction on Coronal & Sagittal planes for natural anatomical roundness.
  - Saves both Real Ground Truth and Synthesised MRI slices.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from PIL import Image
import numpy as np
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
      2. If coronal or sagittal, expands height by 1.88x so the vertical brain axis
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


def extract_slices_from_comparison(comp_path: Path):
    """
    Extracts all 57 subplot boxes from a systematic evaluation comparison montage:
    Row 0: Real Ground Truth (Axial, Coronal, Sagittal)
    Row 10: Best Synthetic (CFG 6.0 Sample 1) (Axial, Coronal, Sagittal)
    """
    im = Image.open(comp_path)
    arr = np.array(im)
    gray = np.mean(arr[:, :, :3], axis=-1)

    not_white = (gray < 250)
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
    if len(valid_boxes) < 33:
        raise ValueError(f"Could not find sufficient subplots in {comp_path.name} (found {len(valid_boxes)})")

    # Real Ground Truth: Boxes 0, 1, 2
    r_ax_box = valid_boxes[0]
    r_cor_box = valid_boxes[1]
    r_sag_box = valid_boxes[2]

    # Synthesised: CFG 6.0 Sample 1 is Box 30, 31, 32
    syn_idx = 30 if len(valid_boxes) >= 33 else 21
    s_ax_box = valid_boxes[syn_idx]
    s_cor_box = valid_boxes[syn_idx + 1]
    s_sag_box = valid_boxes[syn_idx + 2]

    r_ax = standardize_brain_slice(arr[r_ax_box[0]:r_ax_box[1], r_ax_box[2]:r_ax_box[3]], is_cor_or_sag=False)
    r_cor = standardize_brain_slice(arr[r_cor_box[0]:r_cor_box[1], r_cor_box[2]:r_cor_box[3]], is_cor_or_sag=True)
    r_sag = standardize_brain_slice(arr[r_sag_box[0]:r_sag_box[1], r_sag_box[2]:r_sag_box[3]], is_cor_or_sag=True)

    s_ax = standardize_brain_slice(arr[s_ax_box[0]:s_ax_box[1], s_ax_box[2]:s_ax_box[3]], is_cor_or_sag=False)
    s_cor = standardize_brain_slice(arr[s_cor_box[0]:s_cor_box[1], s_cor_box[2]:s_cor_box[3]], is_cor_or_sag=True)
    s_sag = standardize_brain_slice(arr[s_sag_box[0]:s_sag_box[1], s_sag_box[2]:s_sag_box[3]], is_cor_or_sag=True)

    return {
        "real_axial": r_ax,
        "real_coronal": r_cor,
        "real_sagittal": r_sag,
        "syn_axial": s_ax,
        "syn_coronal": s_cor,
        "syn_sagittal": s_sag,
    }


def main():
    base_dir = Path("/home/mav24/projects/NV-Generate-CTMR-A/output/systematic_evaluation_30casesnew")
    out_slices_dir = base_dir / "example_slices"
    out_slices_dir.mkdir(parents=True, exist_ok=True)

    comp_files = sorted(list(base_dir.glob("case_*_comparison.png")))
    print(f"Found {len(comp_files)} comparison cases in {base_dir}")

    selected_files = comp_files[:10]
    metadata_manifest = []

    for i, fpath in enumerate(selected_files, start=1):
        print(f"Extracting Standardized Example {i:02d}: {fpath.name}...", flush=True)
        match = re.search(r"case_(\d+)_row(\d+)_([A-Za-z0-9]+)_comparison", fpath.name)
        case_num = int(match.group(1)) if match else i
        row_id = int(match.group(2)) if match else 0
        modality = match.group(3) if match else "MRI"

        case_folder = out_slices_dir / f"case_{i:02d}_{modality}_row{row_id}"
        case_folder.mkdir(parents=True, exist_ok=True)

        slices_dict = extract_slices_from_comparison(fpath)

        saved_paths = {}
        for plane_key, img in slices_dict.items():
            # Save inside dedicated subfolder
            p_sub = case_folder / f"{plane_key}.png"
            img.save(p_sub)

            # Save directly in example_slices/ with clear name
            p_direct = out_slices_dir / f"example_{i:02d}_{modality}_{plane_key}.png"
            img.save(p_direct)

            saved_paths[plane_key] = str(p_direct)

        metadata_manifest.append({
            "example_index": i,
            "case_id": case_num,
            "dataset_row": row_id,
            "modality": modality,
            "source_comparison": str(fpath),
            "slices": saved_paths,
            "dimensions": "512x512",
        })

    manifest_path = out_slices_dir / "metadata_10_examples_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_manifest, f, indent=2)

    print(f"\nSuccessfully extracted and saved 10 standardized example cases to:")
    print(f"  {out_slices_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
