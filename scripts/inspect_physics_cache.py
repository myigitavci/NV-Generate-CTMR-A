import pickle
import numpy as np

with open("output/eval_physics/physics_sweep_cache.pkl", "rb") as f:
    cache = pickle.load(f)

print("Cache keys:", cache.keys())
for k, v in cache.items():
    if isinstance(v, dict):
        print(f"Key '{k}' subkeys:", v.keys())
        for sk, sv in v.items():
            if hasattr(sv, 'shape'):
                print(f"  {sk}: shape {sv.shape}, dtype {sv.dtype}")
            elif isinstance(sv, list):
                print(f"  {sk}: list of len {len(sv)}, item types {type(sv[0]) if len(sv)>0 else 'empty'}")
            else:
                print(f"  {sk}: {type(sv)} = {sv}")
