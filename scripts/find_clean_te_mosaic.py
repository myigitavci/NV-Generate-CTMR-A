import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

te_img = np.array(Image.open("output/eval_physics/te_response_consistency.png").convert("RGB"))
h, w, _ = te_img.shape
print("Image shape:", te_img.shape)

# Let's inspect the middle region where the black mosaic is:
# In the middle region, background of brain mosaic is pitch black: [0, 0, 0] or very dark.
# Outside the mosaic, the background of the figure is pure white: [255, 255, 255].
gray = np.mean(te_img, axis=2)

# Black background of the mosaic has gray < 10
# Find rows and cols where there is pitch black pixels (the mosaic box)
is_black = (gray < 5)
black_rows = np.where(np.sum(is_black, axis=1) > w * 0.1)[0]
black_cols = np.where(np.sum(is_black, axis=0) > h * 0.1)[0]

print("Black box rows:", black_rows[0], black_rows[-1])
print("Black box cols:", black_cols[0], black_cols[-1])

y0, y1 = black_rows[0], black_rows[-1]
x0, x1 = black_cols[0], black_cols[-1]

clean_mosaic = te_img[y0:y1, x0:x1]
print("Clean mosaic shape:", clean_mosaic.shape)

# Let's verify clean mosaic
Image.fromarray(clean_mosaic).save("output/eval_physics/clean_te_mosaic.png")

# Now slice clean_mosaic into 7 equal parts
mw = clean_mosaic.shape[1]
sw = mw / 7.0

for i, te in enumerate([10, 20, 30, 40, 60, 80, 100]):
    sx0 = int(round(i * sw))
    sx1 = int(round((i + 1) * sw))
    slc = clean_mosaic[:, sx0:sx1]
    Image.fromarray(slc).save(f"output/eval_physics/clean_te_{te}ms.png")

print("Saved all 7 pristine clean TE slices!")
