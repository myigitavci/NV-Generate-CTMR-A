import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np

csv_p = "/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv"
df = pd.read_csv(csv_p)

# Let's pick 3 T2W and 3 FLAIR (or 2 T1w, 2 T2w, 2 FLAIR)
# Let's inspect rows with high quality stripped volumes
candidates_t2 = [0, 3, 10, 12, 15]
candidates_flair = [13, 28, 58, 96, 117]
candidates_t1 = [6, 21, 989]

selected = []
for idx in [0, 3, 10, 13, 28, 96]:
    r = df.iloc[idx]
    fp = str(r["filepath"])
    cands = [
        Path(fp),
        Path("/data/mr_niftis") / fp.lstrip("/"),
        Path("/data/mr_niftis") / fp.replace("_stripped", "").lstrip("/"),
        Path("/data/mr_niftis") / (fp.lstrip("/") + ".gz"),
        Path("/data/mr_niftis") / (fp.replace("_stripped", "").lstrip("/") + ".gz"),
    ]
    p_found = None
    for c in cands:
        if c.exists():
            p_found = c
            break
    
    nii = nib.load(str(p_found))
    d = nii.get_fdata()
    print(f"Row {idx} [{r['sequence']}]: path={p_found}, shape={d.shape}, min={d.min()}, max={d.max()}")
    print(f"   Prompt: {r['text']}")
    selected.append({
        "row": idx,
        "seq": r["sequence"],
        "real_path": str(p_found),
        "prompt": r["text"],
        "shape": d.shape
    })
