import pickle
import numpy as np

with open("output/eval_physics/physics_sweep_cache.pkl", "rb") as f:
    cache = pickle.load(f)

for sweep_name in ["te_data", "tr_data", "ti_data"]:
    d = cache[sweep_name]
    print(f"=== {sweep_name} ===")
    print("Param values:", d["param_vals"])
    print("Spearman rho:", d.get("spearman_rho", "N/A"))
    for r in d["records"]:
        print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
