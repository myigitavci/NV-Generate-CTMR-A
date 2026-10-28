import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.lines as mlines

# Load original image
orig_path = "/home/mav24/.gemini/antigravity-ide/brain/0d79d3b3-82cd-4eb3-a80c-faa54f5f554e/.tempmediaStorage/media_1789467623133.png"
img = np.array(Image.open(orig_path).convert("RGB"))
h, w, _ = img.shape

# Inner plot coordinates found: x=[136, 1687], y=[99, 1332]
# Note: let's verify exact spine positions
gray = np.mean(img, axis=2)

row_dark = np.sum(gray < 100, axis=1)
col_dark = np.sum(gray < 100, axis=0)

y_top_cands = np.where((row_dark > w * 0.3) & (np.arange(h) < h * 0.3))[0]
y_bot_cands = np.where((row_dark > w * 0.3) & (np.arange(h) > h * 0.7))[0]
x_left_cands = np.where((col_dark > h * 0.3) & (np.arange(w) < w * 0.3))[0]
x_right_cands = np.where((col_dark > h * 0.3) & (np.arange(w) > w * 0.7))[0]

y_top = y_top_cands[0]
y_bot = y_bot_cands[-1]
x_left = x_left_cands[0]
x_right = x_right_cands[-1]

print(f"Inner box: x=[{x_left}, {x_right}], y=[{y_top}, {y_bot}]")

# Crop pure inner canvas (excluding border lines)
# Spines are about 1-2 px thick
inner_canvas = img[y_top+2:y_bot-1, x_left+2:x_right-1].copy()
ih, iw, _ = inner_canvas.shape

# Let's locate the old legend box in the top-right
# In the original 30-case plot, where was the legend?
# Let's find any grey frame / legend box in top right
# We can search in y: [0, int(ih*0.4)], x: [int(iw*0.65), iw]
# Let's fill the old legend area with white background (255, 255, 255)
# In matplotlib, default background is (255, 255, 255)
# Let's find the exact bounding box of the legend frame in top-right
sub = inner_canvas[0:int(ih*0.38), int(iw*0.70):iw]
# Legend frame is a rectangular border with grey color ~ 204 or dark text inside
# Let's clean the rectangle where the legend sits:
# We know the legend was in the upper-right corner.
# Let's check if there are any data points in the upper right corner:
# In the 30-case t-SNE plot, the clusters are:
# T1W (bottom left / middle left), T2W (bottom right / middle right), FLAIR (top left / center), PDW (center/bottom).
# The top-right corner [y < 0.35 * ih, x > 0.72 * iw] contains ONLY the legend box and empty white space.

# Let's create a clean copy where the upper right legend is erased to white
clean_canvas = inner_canvas.copy()
# Clear upper-right legend box (y from 0 to 450, x from 1050 to iw)
# Let's verify exactly what is in this box
clean_canvas[0:int(ih*0.35), int(iw*0.72):iw] = 255

# Now let's build the publication figure with matplotlib!
# Dimensions: width 9.5 inches, height 7.0 inches, dpi 300
fig = plt.figure(figsize=(9.2, 7.0), dpi=300)
# Create axes with room on the right for the clean legends
# [left, bottom, width, height] in normalized figure coordinates
ax = fig.add_axes([0.09, 0.09, 0.68, 0.85])

# Display the inner scatter canvas exactly
ax.imshow(clean_canvas, aspect='auto')

# Publication Axes Styling:
# - No tick numbers / no tick marks
ax.set_xticks([])
ax.set_yticks([])

# - Axes labels: t-SNE 1 and t-SNE 2 (clean, elegant, bold/semibold font)
ax.set_xlabel("t-SNE 1", fontsize=15, fontweight='semibold', labelpad=10)
ax.set_ylabel("t-SNE 2", fontsize=15, fontweight='semibold', labelpad=10)

# Spine formatting (clean subtle border)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.2)

# Colors matching the original 30-case plot:
# In the original plot:
# T1W: brown/reddish brown
# T2W: sky blue / light blue
# FLAIR: navy / dark blue
# PDW: green
# T2*W: grey (if applicable)
modality_colors = {
    'T1W': '#8c564b',    # brown
    'T2W': '#0284c7',    # vibrant cyan/sky-blue
    'FLAIR': '#1e3a8a',  # deep navy
    'PDW': '#16a34a',    # emerald green
    'T2*W': '#6b7280'    # neutral grey
}

# 1. Modality Legend Elements
mod_handles = [
    mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                  markersize=9, label=mod)
    for mod, color in modality_colors.items()
]

# 2. Data Type Legend Elements
type_handles = [
    mlines.Line2D([], [], color='#374151', marker='o', linestyle='None',
                  markersize=8, label='Synthetic', fillstyle='full'),
    mlines.Line2D([], [], color='#374151', marker='s', linestyle='None',
                  markersize=8, label='Real', fillstyle='full'),
    mlines.Line2D([], [], color='#374151', marker='*', linestyle='None',
                  markersize=12, label='Metadata', fillstyle='full')
]

# Add Modality Legend (Upper Right outside plot)
leg1 = ax.legend(handles=mod_handles, title="MRI Modality",
                 bbox_to_anchor=(1.03, 0.98), loc='upper left',
                 frameon=True, framealpha=0.95, edgecolor='#cbd5e1',
                 fontsize=11, title_fontsize=12)
leg1.get_title().set_fontweight('bold')
ax.add_artist(leg1)

# Add Data Type Legend (Below Modality Legend outside plot)
leg2 = ax.legend(handles=type_handles, title="Data Type",
                 bbox_to_anchor=(1.03, 0.45), loc='upper left',
                 frameon=True, framealpha=0.95, edgecolor='#cbd5e1',
                 fontsize=11, title_fontsize=12)
leg2.get_title().set_fontweight('bold')

out_path = "/home/mav24/projects/NV-Generate-CTMR-A/output/systematic_evaluation_30cases/tsne_embedding_space.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print("Saved publication-ready t-SNE plot to:", out_path)
