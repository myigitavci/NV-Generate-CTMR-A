#!/usr/bin/env python3
import glob, os, nibabel as nib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def norm(v):
    bg = np.percentile(v, 10.0)
    fg = v[v > bg]
    if len(fg) > 50:
        p1, p99 = np.percentile(fg, 1.0), np.percentile(fg, 99.0)
        return np.clip((v - p1)/(p99 - p1 + 1e-6), 0, 1)
    return np.clip(v, 0, 1)

files = sorted(glob.glob("output/validation_experiments/**/*.nii.gz", recursive=True)) + \
        sorted(glob.glob("output/eval_memorization/**/*.nii.gz", recursive=True)) + \
        sorted(glob.glob("output/systematic_evaluation*/**/*.nii.gz", recursive=True))

print("Found files count:", len(files))
for f in files[:20]:
    nii = nib.load(f)
    v = nii.get_fdata()
    print(f, v.shape, "mean:", np.mean(v), "max:", np.max(v))
