#!/usr/bin/env python3
import pandas as pd
import nibabel as nib
from pathlib import Path

csv_path = "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv"
df = pd.read_csv(csv_path)

for i in [1325, 252, 72, 989, 496, 521, 293, 337, 1408, 1620]:
    row = df.iloc[i]
    fp = row["filepath"]
    full_p = Path("/data/mr_niftis") / fp.lstrip("/")
    if not full_p.exists():
        full_p = Path("/data/mr_niftis") / (fp.lstrip("/") + ".gz")
    print(f"Row {i} [{row['sequence']} | {row['orientation']}]: exists={full_p.exists()}")
    if full_p.exists():
        nii = nib.load(str(full_p))
        print(f"  Shape: {nii.shape}, Zooms (spacing): {nii.header.get_zooms()[:3]}")
