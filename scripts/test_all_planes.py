#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def normalize(vol):
    v = vol.astype(np.float32)
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    p_low = np.percentile(fg, 1.5) if len(fg) > 100 else np.percentile(v, 1.5)
    p_high = np.percentile(fg, 99.0) if len(fg) > 100 else np.percentile(v, 99.0)
    return np.clip((v - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)

def crop_tight_square(slice_2d, is_real=False, plane="axial", pad_ratio=0.03):
    gray = slice_2d.astype(np.float32)
    thresh = max(0.06, np.percentile(gray[gray > 0], 6.0) if (gray > 0).any() else 0.06)
    brain_pts = np.where(gray > thresh)
    
    if len(brain_pts[0]) > 50:
        y_min, y_max = np.min(brain_pts[0]), np.max(brain_pts[0])
        x_min, x_max = np.min(brain_pts[1]), np.max(brain_pts[1])
        
        if is_real and plane in ["coronal", "sagittal"]:
            # Exclude neck/jaw tissue below brainstem
            y_max = int(y_min + (y_max - y_min) * 0.70)
            
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
        pad_bottom = max(0, y1 - slice_2d.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - slice_2d.shape[1])
        
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            slice_2d = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left
            
        return slice_2d[y0:y1, x0:x1]
    return slice_2d

def extract_triplane(vol, is_synthetic=False):
    sx, sy, sz = vol.shape
    if not is_synthetic:
        # Real OASIS3 volume (e.g. 256, 256, 256)
        z_idx = int(sz * 0.64)  # Ventricular level
        y_idx = int(sy * 0.52)  # Mid-coronal
        x_idx = int(sx * 0.50)  # Mid-sagittal
        
        ax = np.rot90(vol[:, :, z_idx])
        cor = np.rot90(vol[:, y_idx, :])
        sag = np.rot90(vol[x_idx, :, :])
    else:
        # Synthetic (256, 256, 128)
        # Find parenchyma bounds
        pts = np.where(vol > 0.08)
        if len(pts[0]) > 100:
            z_min, z_max = np.min(pts[2]), np.max(pts[2])
            y_min, y_max = np.min(pts[1]), np.max(pts[1])
            x_min, x_max = np.min(pts[0]), np.max(pts[0])
            z_idx = int(z_min + (z_max - z_min) * 0.52)
            y_idx = int(y_min + (y_max - y_min) * 0.50)
            x_idx = int(x_min + (x_max - x_min) * 0.50)
        else:
            z_idx, y_idx, x_idx = int(sz * 0.55), sy // 2, sx // 2
            
        ax = np.rot90(vol[:, :, z_idx])
        cor = np.rot90(vol[:, y_idx, :])
        sag = np.rot90(vol[x_idx, :, :])
        
        # Rescale Z for coronal and sagittal
        z_aspect = 1.36 / 0.94
        im_cor = Image.fromarray((cor * 255).astype(np.uint8))
        cor = np.array(im_cor.resize((cor.shape[1], int(round(cor.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)) / 255.0
        
        im_sag = Image.fromarray((sag * 255).astype(np.uint8))
        sag = np.array(im_sag.resize((sag.shape[1], int(round(sag.shape[0] * z_aspect))), resample=Image.Resampling.BILINEAR)) / 255.0
        
    return (
        crop_tight_square(ax, is_real=not is_synthetic, plane="axial"),
        crop_tight_square(cor, is_real=not is_synthetic, plane="coronal"),
        crop_tight_square(sag, is_real=not is_synthetic, plane="sagittal"),
    )

oasis_base = Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS31000/OAS31000_MR_d0072")

cases = [
    {
        "modality": "T1-Weighted (T1w)",
        "real": oasis_base / "anat4/NIFTI/ur_sub-OAS31000_ses-d0072_T1w.nii.gz",
        "syn": "output/validation_experiments/exp2_multimodal_panel/subject_02_T1_Weighted.nii.gz",
    },
    {
        "modality": "T2-Weighted (T2w)",
        "real": oasis_base / "anat2/NIFTI/ur_sub-OAS31000_ses-d0072_acq-TSE_T2w.nii.gz",
        "syn": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_Weighted.nii.gz",
    },
    {
        "modality": "T2-FLAIR",
        "real": oasis_base / "anat3/NIFTI/ur_sub-OAS31000_ses-d0072_FLAIR.nii.gz",
        "syn": "output/validation_experiments/exp2_multimodal_panel/subject_02_T2_FLAIR.nii.gz",
    },
]

fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))
for r_i, c in enumerate(cases):
    vr = normalize(nib.load(c["real"]).get_fdata())
    vs = normalize(nib.load(c["syn"]).get_fdata())
    
    r_ax, r_cor, r_sag = extract_triplane(vr, is_synthetic=False)
    s_ax, s_cor, s_sag = extract_triplane(vs, is_synthetic=True)
    
    for c_i, (img, name) in enumerate([
        (r_ax, "Real Ax"), (r_cor, "Real Cor"), (r_sag, "Real Sag"),
        (s_ax, "Syn Ax"), (s_cor, "Syn Cor"), (s_sag, "Syn Sag")
    ]):
        axes[r_i, c_i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[r_i, c_i].set_title(name if r_i == 0 else "")
        axes[r_i, c_i].axis("off")

plt.tight_layout()
plt.savefig("output/eval_physics/test_all_planes_comparison.png", dpi=150)
plt.close()
print("Saved test_all_planes_comparison.png")
