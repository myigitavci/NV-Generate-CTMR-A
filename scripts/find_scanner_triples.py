import pandas as pd

df = pd.read_csv("/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv")

scanners = ["Avanto", "DISCOVERY_MR750", "Signa_HDxt", "DISCOVERY_MR450"]
with open("scratch_scanner_triples.txt", "w") as f:
    for sc in scanners:
        for seq in ["T2W", "FLAIR"]:
            sub = df[df["text"].str.contains(sc, case=False, na=False) & (df["sequence"] == seq)]
            f.write(f"=== {sc} | {seq} (count={len(sub)}) ===\n")
            if len(sub) > 0:
                for i in range(min(3, len(sub))):
                    f.write(f"  [{i}] {sub.iloc[i]['text']}\n")
            f.write("\n")

print("Wrote scratch_scanner_triples.txt")
