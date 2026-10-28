import pandas as pd
import re
from collections import Counter

df = pd.read_csv("/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv")

with open("scratch_unseen_scanners.txt", "w") as out:
    out.write("=== UNIQUE SCANNERS IN UNSEEN OASIS DATASET ===\n\n")
    
    # Extract scanner details from text
    records = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        m_sc = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", text)
        m_acq = re.search(r"Acquisition \(Description, Sequence, Variant\):\s*\(([^)]+)\)", text)
        m_params = re.search(r"Imaging Parameters \(Echo Time, Repetition Time, Inversion Time, Flip Angle\):\s*\(([^)]+)\)", text)
        
        sc_str = m_sc.group(1).strip() if m_sc else "Unknown"
        acq_str = m_acq.group(1).strip() if m_acq else "Unknown"
        params_str = m_params.group(1).strip() if m_params else "Unknown"
        
        mod = "FLAIR" if "FLAIR" in text or "flair" in text else ("T2" if "T2" in text or "t2" in text else "Other")
        
        records.append({
            "idx": idx,
            "modality": mod,
            "scanner": sc_str,
            "acquisition": acq_str,
            "params": params_str,
            "full_text": text
        })
        
    df_rec = pd.DataFrame(records)
    
    out.write("--- T2 Scanners ---\n")
    t2_df = df_rec[df_rec["modality"] == "T2"]
    for sc, grp in t2_df.groupby("scanner"):
        out.write(f"Scanner: {sc} (count={len(grp)})\n")
        for sample_text in grp["full_text"].head(2):
            out.write(f"  Sample prompt: {sample_text}\n")
        out.write("\n")
        
    out.write("\n--- FLAIR Scanners ---\n")
    flair_df = df_rec[df_rec["modality"] == "FLAIR"]
    for sc, grp in flair_df.groupby("scanner"):
        out.write(f"Scanner: {sc} (count={len(grp)})\n")
        for sample_text in grp["full_text"].head(2):
            out.write(f"  Sample prompt: {sample_text}\n")
        out.write("\n")
