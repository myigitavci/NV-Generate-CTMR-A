import pickle

with open("output/eval_physics/physics_sweep_cache.pkl", "rb") as f:
    cache = pickle.load(f)

with open("output/eval_physics/cache_dump.txt", "w") as out:
    for sweep_name in ["te_data", "tr_data", "ti_data"]:
        if sweep_name in cache:
            d = cache[sweep_name]
            out.write(f"=== {sweep_name} ===\n")
            out.write(f"Param values: {d.get('param_vals')}\n")
            out.write(f"Spearman rho: {d.get('spearman_rho', 'N/A')}\n")
            for r in d.get("records", []):
                out.write(f"  {r}\n")
            if "slices" in d:
                out.write(f"  Slices count: {len(d['slices'])}\n")
                for s in d["slices"]:
                    out.write(f"    val={s[0]}, shape={s[1].shape}, min={s[1].min()}, max={s[1].max()}\n")
