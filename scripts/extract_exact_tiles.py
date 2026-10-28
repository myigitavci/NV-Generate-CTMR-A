import numpy as np
from PIL import Image

img = np.array(Image.open("output/eval_physics/te_response_consistency.png").convert("RGB"))
h, w, _ = img.shape
print("Image dimensions:", w, h)

# We want only the black box in the middle horizontal band:
# Central band: y from 0.35*h to 0.65*h, x from 0.3*w to 0.75*w
y_start, y_end = int(0.30 * h), int(0.70 * h)
x_start, x_end = int(0.30 * w), int(0.75 * w)

sub = img[y_start:y_end, x_start:x_end]
gray_sub = np.mean(sub, axis=2)

# Black background is near 0 (< 5)
is_black = (gray_sub < 5)

# The black rectangle forms contiguous black rows and cols
# Let's find the rows that have a long contiguous run of black pixels
black_row_counts = np.sum(is_black, axis=1)
valid_rows = np.where(black_row_counts > sub.shape[1] * 0.5)[0]

black_col_counts = np.sum(is_black, axis=0)
valid_cols = np.where(black_col_counts > sub.shape[0] * 0.4)[0]

print("Valid rows in sub:", valid_rows[0], valid_rows[-1])
print("Valid cols in sub:", valid_cols[0], valid_cols[-1])

strip_y0 = y_start + valid_rows[0]
strip_y1 = y_start + valid_rows[-1]
strip_x0 = x_start + valid_cols[0]
strip_x1 = x_start + valid_cols[-1]

print(f"Exact black strip: Y=[{strip_y0}, {strip_y1}], X=[{strip_x0}, {strip_x1}]")

clean_strip = img[strip_y0:strip_y1, strip_x0:strip_x1]
Image.fromarray(clean_strip).save("output/eval_physics/extracted_te_strip.png")

# Cut into 7 slices
sw = clean_strip.shape[1] / 7.0
for i, te in enumerate([10, 20, 30, 40, 60, 80, 100]):
    sx0 = int(round(i * sw))
    sx1 = int(round((i + 1) * sw))
    slc = clean_strip[:, sx0:sx1]
    # Crop black margin to make it square
    sh, sw_tile, _ = slc.shape
    # If height > width or width > height, crop to square
    dim = min(sh, sw_tile)
    cy, cx = sh // 2, sw_tile // 2
    sq = slc[cy - dim//2 : cy + dim//2, cx - dim//2 : cx + dim//2]
    Image.fromarray(sq).save(f"output/eval_physics/te_tile_{te}ms.png")

print("Successfully saved clean individual square tiles!")
