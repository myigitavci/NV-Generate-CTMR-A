#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("metadata_mrclip_with_oasis.csv")
t1_df = df[df["sequence"] == "T1W"]
print(f"Total T1W entries: {len(t1_df)}")

# Look for axial T1W with complete coverage (e.g. Siemens or GE 3D or standard axial)
for idx, row in t1_df.head(20).iterrows():
    text = str(row["text"])
    if "plane axial" in text.lower():
        print(f"\n--- Row {idx} ---")
        print(text)
