import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.lines as mlines

# 1. Load the original 30-case pristine t-SNE plot
orig_path = "/home/mav24/.gemini/antigravity-ide/brain/0d79d3b3-82cd-4eb3-a80c-faa54f5f554e/.tempmediaStorage/media_1789467623133.png"
img = np.array(Image.open(orig_path).convert("RGB"))
h, w, _ = img.shape
print(f"Loaded image size: {w}x{h}")

# Convert to grayscale
gray = np.mean(img, axis=2)

row_dark = np.sum(gray < 100, axis=1)
col_dark = np.sum(gray < 100, axis=0)

y_top_cands = np.where((row_dark > w * 0.3) & (np.arange(h) < h * 0.3))[0]
y_bot_cands = np.where((row_dark > w * 0.3) & (np.arange(h) > h * 0.7))[0]
x_left_cands = np.where((col_dark > h * 0.3) & (np.arange(w) < w * 0.3))[0]
x_right_cands = np.where((col_dark > h * 0.3) & (np.arange(w) > w * 0.7))[0]

print("y_top_cands:", y_top_cands)
print("y_bot_cands:", y_bot_cands)
print("x_left_cands:", x_left_cands)
print("x_right_cands:", x_right_cands)

y_top = y_top_cands[0] if len(y_top_cands) > 0 else int(h * 0.1)
y_bot = y_bot_cands[-1] if len(y_bot_cands) > 0 else int(h * 0.9)
x_left = x_left_cands[0] if len(x_left_cands) > 0 else int(w * 0.1)
x_right = x_right_cands[-1] if len(x_right_cands) > 0 else int(w * 0.9)

print(f"Inner plot box: x=[{x_left}, {x_right}], y=[{y_top}, {y_bot}]")

# Crop the inside of the plot
inner_canvas = img[y_top+2:y_bot-1, x_left+2:x_right-1].copy()
ih, iw, _ = inner_canvas.shape
print(f"Inner canvas size: {iw}x{ih}")

# Save debug crop
Image.fromarray(inner_canvas).save("output/systematic_evaluation_30cases/debug_crop.png")
