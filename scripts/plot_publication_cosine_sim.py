import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set font family
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]

csv_path = "output/systematic_evaluation_30cases/metrics_per_sample.csv"
if not os.path.exists(csv_path):
    csv_path = "output/systematic_evaluation_30cases/metrics_summary_by_cfg.csv"

df = pd.read_csv(csv_path)

if "cfg_scale" in df.columns and "sim_text" in df.columns:
    grouped = df.groupby("cfg_scale")
    cfg_vals = np.array(sorted(df["cfg_scale"].unique()))
    mean_text = np.array([grouped.get_group(c)["sim_text"].mean() for c in cfg_vals])
    std_text = np.array([grouped.get_group(c)["sim_text"].std() for c in cfg_vals])
    mean_real = np.array([grouped.get_group(c)["sim_real"].mean() for c in cfg_vals])
    std_real = np.array([grouped.get_group(c)["sim_real"].std() for c in cfg_vals])
    baseline_real_text = df["sim_real_text"].mean() if "sim_real_text" in df.columns else 0.402
else:
    cfg_vals = df["cfg_scale"].values
    mean_text = df["mean_sim_text"].values
    std_text = df["std_sim_text"].values
    mean_real = df["mean_sim_real"].values
    std_real = df["std_sim_real"].values
    baseline_real_text = 0.402

# Compact, punchy publication figure (squeezed X-axis, square-proportioned subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300, facecolor="#ffffff")

# ---------------------------------------------------------------------------
# Panel 1: Metadata ↔ Image Alignment
# ---------------------------------------------------------------------------
# Baseline
ax1.axhline(
    baseline_real_text, color="#dc2626", linestyle="--", lw=1.9,
    label=f"Real Baseline ({baseline_real_text:.3f})", zorder=2
)

# Error bar curve
ax1.errorbar(
    cfg_vals, mean_text, yerr=std_text,
    marker="o", markersize=7, color="#2563eb", ecolor="#2563eb",
    capsize=4.5, elinewidth=1.6, capthick=1.6, lw=2.2,
    label="Synthetic ↔ Metadata", zorder=3
)

ax1.set_title("Metadata ↔ Image Alignment", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
ax1.set_xlabel("CFG Guidance Scale ($w$)", fontsize=11.5, fontweight="bold", labelpad=6, color="#334155")
ax1.set_ylabel("Cosine Similarity", fontsize=11.5, fontweight="bold", labelpad=6, color="#334155")
ax1.set_xlim(-0.4, 10.4)
ax1.set_xticks([0, 2, 5, 7, 10])
ax1.set_ylim(0.10, 0.58)
ax1.tick_params(axis="both", which="major", labelsize=10.5, colors="#334155", width=1.1, length=4)
for spine in ax1.spines.values():
    spine.set_color("#334155")
    spine.set_linewidth(1.1)
ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#cbd5e1", zorder=0)
ax1.set_axisbelow(True)
ax1.legend(
    loc="lower right", fontsize=9.8, framealpha=0.96, facecolor="#ffffff",
    edgecolor="#cbd5e1", borderpad=0.5, labelspacing=0.4
)

# ---------------------------------------------------------------------------
# Panel 2: Synthetic ↔ Real Scan Cosine Similarity
# ---------------------------------------------------------------------------
# Error bar curve
ax2.errorbar(
    cfg_vals, mean_real, yerr=std_real,
    marker="s", markersize=7, color="#16a34a", ecolor="#16a34a",
    capsize=4.5, elinewidth=1.6, capthick=1.6, lw=2.2,
    label="Synthetic ↔ Real Scan", zorder=3
)

ax2.set_title("Synthetic ↔ Real Scan Cosine Sim", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
ax2.set_xlabel("CFG Guidance Scale ($w$)", fontsize=11.5, fontweight="bold", labelpad=6, color="#334155")
ax2.set_ylabel("Cosine Similarity", fontsize=11.5, fontweight="bold", labelpad=6, color="#334155")
ax2.set_xlim(-0.4, 10.4)
ax2.set_xticks([0, 2, 5, 7, 10])
ax2.set_ylim(0.05, 0.58)
ax2.tick_params(axis="both", which="major", labelsize=10.5, colors="#334155", width=1.1, length=4)
for spine in ax2.spines.values():
    spine.set_color("#334155")
    spine.set_linewidth(1.1)
ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#cbd5e1", zorder=0)
ax2.set_axisbelow(True)
ax2.legend(
    loc="lower right", fontsize=9.8, framealpha=0.96, facecolor="#ffffff",
    edgecolor="#cbd5e1", borderpad=0.5, labelspacing=0.4
)

plt.tight_layout(w_pad=2.2)

out_file = "output/systematic_evaluation_30cases/cosine_similarity_vs_cfg.png"
fig.savefig(out_file, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved compact publication-ready cosine similarity plot to: {out_file}")
