#!/usr/bin/env python3
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

pkl_path = "output/validation_experiments/exp4_scanner_clustering/scanner_embeddings.pkl"
with open(pkl_path, "rb") as f:
    all_embeddings = pickle.load(f)

synth_items = [
    item for item in all_embeddings
    if item.get("type", "Synthetic") == "Synthetic" and item.get("modality") != "T2STARW"
]

X = np.array([item["feat"] for item in synth_items])
scanners = [item["scanner"] for item in synth_items]
modalities = [item.get("modality", "MRI") for item in synth_items]

perplexity = min(30, max(3, len(X) // 3))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

print(f"Total synthetic items: {len(synth_items)}")

# Find items where modality is T1W but t-SNE Y > 2.0 (in the PDW zone)
print("\n=== T1W items near PDW cluster (t-SNE Y > 2.0) ===")
for i, (item, (x, y)) in enumerate(zip(synth_items, X_tsne)):
    if item.get("modality") == "T1W" and y > 2.0:
        print(f"Index {i}: Scanner={item['scanner']}, Modality={item['modality']}, X={x:.2f}, Y={y:.2f}")

# Find items where modality is PDW but near T1W cluster (X > -10 and Y < 2.0, or Y < 0)
print("\n=== PDW items near T1W / other cluster (t-SNE Y < 2.0) ===")
for i, (item, (x, y)) in enumerate(zip(synth_items, X_tsne)):
    if item.get("modality") == "PDW" and y < 2.0:
        print(f"Index {i}: Scanner={item['scanner']}, Modality={item['modality']}, X={x:.2f}, Y={y:.2f}")

# Find FLAIR items far from main FLAIR cluster (Y > -10)
print("\n=== FLAIR outliers (t-SNE Y > -10.0) ===")
for i, (item, (x, y)) in enumerate(zip(synth_items, X_tsne)):
    if item.get("modality") == "FLAIR" and y > -10.0:
        print(f"Index {i}: Scanner={item['scanner']}, Modality={item['modality']}, X={x:.2f}, Y={y:.2f}")

# Find T2W items far from main T2W cluster (Y < 5.0)
print("\n=== T2W outliers (t-SNE Y < 5.0) ===")
for i, (item, (x, y)) in enumerate(zip(synth_items, X_tsne)):
    if item.get("modality") == "T2W" and y < 5.0:
        print(f"Index {i}: Scanner={item['scanner']}, Modality={item['modality']}, X={x:.2f}, Y={y:.2f}")
