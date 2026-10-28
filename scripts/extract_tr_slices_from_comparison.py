#!/usr/bin/env python3
from PIL import Image
import numpy as np

img = Image.open("output/eval_physics/screen_step0_tr_comparison.png")
w, h = img.size
print(f"Comparison image size: {w}x{h}")

# The image has 3 rows (GE_Signa_step0, Symphony_empty_step0, Canonical_Vida_step0)
# and 5 columns (TR = 200, 500, 1000, 2500, 6000 ms)
# Row 2 (0-indexed) is Canonical_Vida_step0
row_h = h // 3
col_w = w // 5

tr_vals = [200, 500, 1000, 2500, 6000]

for c_idx, tr in enumerate(tr_vals):
    # Canonical Vida row (row 2)
    box = (c_idx * col_w, 2 * row_h, (c_idx + 1) * col_w, 3 * row_h)
    cell = img.crop(box)
    
    # Inside the cell, crop the inner black canvas (remove title and borders)
    cell_w, cell_h = cell.size
    inner = cell.crop((int(0.05 * cell_w), int(0.12 * cell_h), int(0.95 * cell_w), int(0.96 * cell_h)))
    out_path = f"output/eval_physics/vida_step0_tr_{tr}ms.png"
    inner.save(out_path)
    print(f"Saved {out_path} ({inner.size})")

print("Done extracting!")
