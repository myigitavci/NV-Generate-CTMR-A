import numpy as np
from PIL import Image
from pathlib import Path

img_path = Path("output/eval_physics/test_symphony_8tr_sweep.png")
if img_path.exists():
    img = np.array(Image.open(img_path).convert("L"))
    h, w = img.shape
    tw = w / 8.0
    tr_vals = [600, 800, 1400, 2200, 3200, 4500, 6000, 8000]
    print(f"Image shape: {w}x{h}, tile width: {tw}")
    
    for i, tr in enumerate(tr_vals):
        x0 = int(round(i * tw))
        x1 = int(round((i + 1) * tw))
        tile = img[:, x0:x1]
        
        # Brain mask: non-black pixels
        brain = tile[tile > 15]
        # CSF: central bright ventricles in this T2-anchored sequence
        # Center of tile
        cy, cx = tile.shape[0] // 2, tile.shape[1] // 2
        center_roi = tile[cy-25:cy+25, cx-25:cx+25]
        ventricle = center_roi[center_roi > np.percentile(center_roi, 75)] if len(center_roi) > 0 else []
        
        csf_mean = np.mean(ventricle) if len(ventricle) > 0 else 0
        brain_mean = np.mean(brain) if len(brain) > 0 else 0
        p85 = np.percentile(brain, 85) if len(brain) > 0 else 0
        p50 = np.percentile(brain, 50) if len(brain) > 0 else 0
        p25 = np.percentile(brain, 25) if len(brain) > 0 else 0
        
        print(f"TR={tr}ms: brain_mean={brain_mean:.1f}, csf={csf_mean:.1f}, p85={p85:.1f}, p50={p50:.1f}, p25={p25:.1f}")
