import sys
sys.path.insert(0, ".")
import numpy as np
from scripts.run_mrclip_validation_experiments import normalize_for_display

for s in [101, 202, 303, 404, 505]:
    vol = np.load(f"output/eval_memorization/diversity_cache/vol_T1W_real_seed{s}.npy")
    norm_vol = normalize_for_display(vol)
    bg_thresh = 0.08
    z_areas = [np.sum(norm_vol[:, :, z] > bg_thresh) for z in range(norm_vol.shape[2])]
    best_z = np.argmax(z_areas)
    print(f"Seed {s}: max area at z={best_z} (area={z_areas[best_z]})")
