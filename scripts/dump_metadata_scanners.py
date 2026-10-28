import pandas as pd
import re
from collections import Counter

df = pd.read_csv("data/batch00_metadata_processed_ready_age_sex.csv")
with open("scratch_scanners.txt", "w") as f:
    f.write(f"Columns: {df.columns.tolist()}\n\n")
    if "text" in df.columns:
        sc = Counter()
        for t in df["text"].dropna():
            m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", str(t))
            if m:
                sc[m.group(1).strip()] += 1
        f.write("Scanners extracted from text column:\n")
        for k, v in sc.most_common():
            f.write(f"  {v:6d} : {k}\n")
