import numpy as np
from PIL import Image

# te_response_consistency.png has:
# 1. Curve plot (left)
# 2. Slices mosaic (middle)
# 3. SynthSeg ROI overlay (right)
img = Image.open("output/eval_physics/te_response_consistency.png")
print("TE response consistency size:", img.size)
