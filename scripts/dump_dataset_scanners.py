import json
import re
import os
import pandas as pd
from collections import Counter

print("=== CHECKING DATASET SCANNERS ===", flush=True)

# 1. Check dataset_mrclip.json
if os.path.exists("dataset_mrclip.json"):
    print("Reading dataset_mrclip.json...", flush=True)
    with open("dataset_mrclip.json") as f:
        d = json.load(f)
    scanners = Counter()
    for split in ["training", "validation", "test"]:
        for item in d.get(split, []):
            text = item.get("text", "")
            m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", text)
            if m:
                scanners[m.group(1).strip()] += 1
    print("\nScanners in dataset_mrclip.json:", flush=True)
    for sc, count in scanners.most_common(30):
        print(f"  {count:6d} : {sc}", flush=True)

# 2. Check /data CSVs
csv_path = "/data/mr_all_3D_final_et_20_rt_20_sample_20_with_precomputed_text_embeddings_with_anatomical_coordinates.csv"
if os.path.exists(csv_path):
    print(f"\nReading {csv_path}...", flush=True)
    df = pd.read_csv(csv_path, nrows=50000)
    scanners_csv = Counter()
    for t in df["text"].dropna():
        m = re.search(r"Scanner \(Manufacturer, Model, Field Strength\):\s*\(([^)]+)\)", str(t))
        if m:
            scanners_csv[m.group(1).strip()] += 1
    print("\nScanners in CSV:", flush=True)
    for sc, count in scanners_csv.most_common(30):
        print(f"  {count:6d} : {sc}", flush=True)
