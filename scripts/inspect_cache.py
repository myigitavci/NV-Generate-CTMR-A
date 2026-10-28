import pickle
import json

try:
    with open('output/eval_physics/physics_sweep_cache_real.pkl', 'rb') as f:
        d = pickle.load(f)
    print("Keys in physics_sweep_cache_real.pkl:", list(d.keys()))
    if 'tr_records' in d:
        print("TR records:")
        for r in d['tr_records']:
            print(r)
    if 'te_records' in d:
        print("TE records:")
        for r in d['te_records']:
            print(r)
except Exception as e:
    print("Error:", e)
