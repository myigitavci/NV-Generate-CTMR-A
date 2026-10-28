import os
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image

def normalize_and_crop(vol_np, is_flair=False):
    v = vol_np.astype(np.float32)
    # Orient properly if needed (192, 256, 256)
    # In MIDI OASIS stripped volumes, axes might be (sagittal, coronal, axial) or (coronal, axial, sagittal)
    # Let's inspect max area slice along each dimension
    shape = v.shape
    print("Volume shape:", shape)
    
    # In shape (192, 256, 256), the axial slices are along axis 2 (or axis 0)
    # Let us extract along axis 2
    bg = np.percentile(v, 8.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1 = np.percentile(fg, 1.0)
        p99 = np.percentile(fg, 99.5 if is_flair else 99.0)
    else:
        p1, p99 = v.min(), v.max()
    v_norm = np.clip((v - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    
    return v_norm

os.makedirs("output/eval_real_synth/test_slices", exist_ok=True)
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
    v_norm = normalize_and_crop(d, is_flair=is_flair)
    
    # Check 3 orthogonal planes mid slice
    ax0 = np.rot90(v_norm[v_norm.shape[0]//2, :, :])
    ax1 = np.rot90(v_norm[:, v_norm.shape[1]//2, :])
    ax2 = np.rot90(v_norm[:, :, v_norm.shape[2]//2])
    
    Image.fromarray((ax0 * 255).astype(np.uint8)).save(f"output/eval_real_synth/test_slices/{name}_ax0.png")
    Image.fromarray((ax1 * 255).astype(np.uint8)).save(f"output/eval_real_synth/test_slices/{name}_ax1.png")
    Image.fromarray((ax2 * 255).astype(np.uint8)).save(f"output/eval_real_synth/test_slices/{name}_ax2.png")
    print(f"Saved test slices for {name}")
