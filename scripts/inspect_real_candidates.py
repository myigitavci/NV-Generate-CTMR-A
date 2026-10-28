import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image

csv_p = "/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv"
df = pd.read_csv(csv_p)

candidates = []
for idx, r in df.iterrows():
    fp = str(r["filepath"])
    cands = [
        Path(fp),
        Path("/data/mr_niftis") / fp.lstrip("/"),
        Path("/data/mr_niftis") / fp.replace("_stripped", "").lstrip("/"),
        Path("/data/mr_niftis") / (fp.lstrip("/") + ".gz"),
        Path("/data/mr_niftis") / (fp.replace("_stripped", "").lstrip("/") + ".gz"),
    ]
    for c in cands:
        if c.exists():
            candidates.append({
                "idx": idx,
                "seq": r["sequence"],
                "path": str(c),
                "text": r["text"],
            })
            break

print(f"Total candidates found: {len(candidates)}")
by_seq = {}
for c in candidates:
    by_seq.setdefault(c["seq"], []).append(c)

for s, lst in by_seq.items():
    print(f"  {s}: {len(lst)} cases")

with open("scratch_candidates_summary.txt", "w") as f:
    for s, lst in by_seq.items():
        f.write(f"=== {s} ({len(lst)}) ===\n")
        for c in lst[:10]:
            f.write(f"  ID {c['idx']}: {c['path']}\n")
            f.write(f"     {c['text']}\n\n")

print("Wrote scratch_candidates_summary.txt")
