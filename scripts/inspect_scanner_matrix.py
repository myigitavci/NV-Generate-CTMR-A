import pandas as pd
import glob, re

csv_p = "/data/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_train_unseen_OASIS.csv"
df = pd.read_csv(csv_p)

files = sorted(glob.glob("output/systematic_evaluation_30casesnew2/case_*_comparison.png"))

scanner_cases = []
for f in files:
    m = re.search(r"case_(\d+)_row(\d+)_([A-Z0-9]+)_comparison\.png", f)
    if m:
        c_num = int(m.group(1))
        r_num = int(m.group(2))
        mod = m.group(3)
        row = df.iloc[r_num]
        txt = row["text"]
        sm = re.search(r"Scanner \(Manufacturer, Model, Field Strength\): \(([^)]+)\)", txt)
        scanner_str = sm.group(1) if sm else "Unknown"
        scanner_cases.append({
            "case_num": c_num,
            "row_num": r_num,
            "modality": mod,
            "scanner": scanner_str,
            "file": f,
            "prompt": txt
        })

df_sc = pd.DataFrame(scanner_cases)
with open("scratch_scanner_matrix.txt", "w") as out:
    out.write("=== Cross-tabulation (Scanner x Modality) ===\n")
    out.write(str(pd.crosstab(df_sc["scanner"], df_sc["modality"])) + "\n\n")
    for s, grp in df_sc.groupby("scanner"):
        out.write(f"\n--- Scanner: {s} ---\n")
        for _, r in grp.iterrows():
            out.write(f"  [{r['modality']}] Case #{r['case_num']:02d} (Row {r['row_num']}) -> {r['file']}\n")

print("Wrote scratch_scanner_matrix.txt")
