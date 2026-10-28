#!/usr/bin/env python3
import pandas as pd
import pickle
import os

csv_path = "output/systematic_evaluation_30casesnew/metrics_per_sample.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Metrics per sample shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 10 cases:")
    print(df[["case_idx", "dataset_idx", "modality", "cfg_scale", "seed", "sim_text", "sim_real"]].head(10))

pkl_path = "output/systematic_evaluation_30casesnew/systematic_embeddings.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print("PKL keys:", data.keys() if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for k in data.keys():
            v = data[k]
            if isinstance(v, (list, tuple)):
                print(f"  {k}: list of len {len(v)}")
                if len(v) > 0:
                    print("    sample element keys:", v[0].keys() if isinstance(v[0], dict) else type(v[0]))
            elif hasattr(v, "shape"):
                print(f"  {k}: shape {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
