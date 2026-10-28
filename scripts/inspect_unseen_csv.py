import os, glob
import pandas as pd
import re

csv_candidates = [
    "data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
    "/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
    "/home/mav24/projects/MR-Styler/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
    "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
]

found = None
for c in csv_candidates:
    if os.path.exists(c):
        found = c
        break

if not found:
    matches = glob.glob("/home/mav24/**/*unseen*.csv", recursive=True) + glob.glob("/data/**/*unseen*.csv", recursive=True)
    if matches:
        found = matches[0]

with open("scratch_unseen.txt", "w") as out:
    out.write(f"Found CSV: {found}\n\n")
    if found:
        df = pd.read_csv(found)
        out.write(f"Total rows: {len(df)}\n")
        out.write(f"Columns: {df.columns.tolist()}\n\n")
        
        # Check modalities and scanners
        if "text" in df.columns:
            for mod in ["T2w", "FLAIR", "T1w", "T2", "t2", "flair"]:
                sub = df[df["text"].str.contains(mod, case=False, na=False)]
                out.write(f"Modality search '{mod}': {len(sub)} rows\n")
                if len(sub) > 0:
                    out.write(f"Sample text for {mod}:\n")
                    for s in sub["text"].head(5):
                        out.write(f"  - {s}\n")
                    out.write("\n")
