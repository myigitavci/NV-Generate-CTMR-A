import pickle
from pathlib import Path
import numpy as np

cache_file = Path("output/eval_physics/physics_sweep_cache.pkl")
if cache_file.exists():
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    print("Keys in cache:", list(cache.keys()))
    for k in ["te_data", "tr_data", "ti_data"]:
        if k in cache:
            d = cache[k]
            print(f"\n--- {k} ---")
            print("param_vals:", d.get("param_vals"))
            records = d.get("records", [])
            for r in records:
                print(f"  param={r.get('param_ms')}: csf={r.get('csf'):.1f}, gm={r.get('gm'):.1f}, wm={r.get('wm'):.1f}")
else:
    print("Cache file not found!")
