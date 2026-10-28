import os, glob
from pathlib import Path
import pandas as pd
import nibabel as nib

csv_files = [
    "/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
    "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv",
    "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_val_unseen_OASIS.csv"
]

found_records = []

for csv_p in csv_files:
    if not os.path.exists(csv_p):
        continue
    df = pd.read_csv(csv_p)
    print(f"Checking {csv_p}, rows: {len(df)}")
    
    for idx, r in df.iterrows():
        fp = str(r["filepath"])
        # Check candidate locations
        cands = [
            Path(fp),
            Path("/data") / fp.lstrip("/"),
            Path("/data/mr_niftis") / fp.lstrip("/"),
            Path("/data/mr_niftis") / fp.replace("_stripped", "").lstrip("/"),
            Path("/data/mr_niftis") / (fp.lstrip("/") + ".gz"),
            Path("/data/mr_niftis") / (fp.replace("_stripped", "").lstrip("/") + ".gz"),
            Path("/home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3") / Path(fp).name,
        ]
        
        # Also search in /home/mav24/Projects/AMA-CLIP/data/OASIS3/
        # e.g. /home/mav24/Projects/AMA-CLIP/data/OASIS3/nifti/OASIS3/OAS30001/OAS30001_MR_d0129/...
        matched_path = None
        for c in cands:
            if c.exists():
                matched_path = c
                break
        
        if matched_path is not None:
            found_records.append({
                "csv": Path(csv_p).name,
                "row_idx": idx,
                "sequence": r.get("sequence", "UNK"),
                "text": r.get("text", ""),
                "real_path": str(matched_path)
            })
            if len(found_records) >= 30:
                break

print(f"Total matched real scans: {len(found_records)}")
with open("scratch_matched_real.txt", "w") as f:
    for rec in found_records:
        f.write(f"[{rec['sequence']}] CSV: {rec['csv']} Row {rec['row_idx']}\n")
        f.write(f"  Real path: {rec['real_path']}\n")
        f.write(f"  Prompt: {rec['text']}\n\n")

print("Wrote scratch_matched_real.txt")
