import os
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image

def crop_brain_tight(slice_2d: np.ndarray, pad_ratio: float = 0.04) -> np.ndarray:
    thresh = max(0.06, float(np.percentile(slice_2d[slice_2d > 0], 5.0)) if (slice_2d > 0).any() else 0.06)
    pts = np.where(slice_2d > thresh)
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

def extract_real_axial(vol_np, is_flair=False):
    v = vol_np.astype(np.float32)
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1 = np.percentile(fg, 1.0)
        p99 = np.percentile(fg, 99.5 if is_flair else 99.0)
    else:
        p1, p99 = v.min(), v.max()
    v_norm = np.clip((v - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    
    thresh = max(0.06, float(np.percentile(v_norm, 15.0)))
    pts = np.where(v_norm > thresh)
    z_min, z_max = int(np.percentile(pts[2], 1.0)), int(np.percentile(pts[2], 99.0))
    
    # Choose optimal mid-ventricular axial slice in 52% - 66% range
    z_cands = list(range(int(z_min + (z_max - z_min) * 0.52), int(z_min + (z_max - z_min) * 0.68) + 1))
    z_idx = z_cands[np.argmax([np.sum(v_norm[:, :, z] > thresh) for z in z_cands])]
    
    # In shape (192, 256, 256), axial slice is v_norm[:, :, z_idx]
    # Let's rotate so frontal lobe is up
    ax = np.rot90(v_norm[:, :, z_idx])
    return crop_brain_tight(ax)

files = [
    ("row0_t2w", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-bb9e7d73d500d9506f1a6466fbb3d8/ses-20131211164020/anat/ur_sub-bb9e7d73d500d9506f1a6466fbb3d8_ses-20131211164020_run-4_T2w_stripped.nii", False),
    ("row3_t2w", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-8dcf9bbc37f4cd5278d3010d1f8296/ses-20170621102001/anat/ur_sub-8dcf9bbc37f4cd5278d3010d1f8296_ses-20170621102001_run-8_T2w_stripped.nii", False),
    ("row10_t2w", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-b3e18fc2f38ef59afdfff82e28cbc2/ses-20140514154406/anat/ur_sub-b3e18fc2f38ef59afdfff82e28cbc2_ses-20140514154406_run-2_T2w_stripped.nii", False),
    ("row13_flair", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-ec20a4c25c87bab6f702c0fb4d6576/ses-20150709132847/anat/ur_sub-ec20a4c25c87bab6f702c0fb4d6576_ses-20150709132847_run-5_FLAIR_stripped.nii", True),
    ("row28_flair", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-062664f6e8380a87de426f96d3de72/ses-20140519155043/anat/ur_sub-062664f6e8380a87de426f96d3de72_ses-20140519155043_run-3_FLAIR_stripped.nii", True),
    ("row96_flair", "/data/mr_niftis/nfs/home/mavci/MR/midi/rawdata/sub-6cabe8f178e39be1e02a62df150396/ses-20151216111049/anat/ur_sub-6cabe8f178e39be1e02a62df150396_ses-20151216111049_run-17_FLAIR_stripped.nii", True),
]

for name, p, is_flair in files:
    nii = nib.load(p)
    d = nii.get_fdata()
    slc = extract_real_axial(d, is_flair=is_flair)
    im = Image.fromarray((slc * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
    im.save(f"output/eval_real_synth/test_slices/{name}_optimal_ax.png")
    print(f"Saved optimal axial for {name}")
