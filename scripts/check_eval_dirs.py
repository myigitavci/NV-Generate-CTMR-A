import os, glob
from pathlib import Path

for d in ["output/systematic_evaluation_30cases", "output/systematic_evaluation_30casesnew", "output/eval_memorization", "output/validation_experiments"]:
    if os.path.exists(d):
        files = [f for f in glob.glob(f"{d}/**/*", recursive=True) if os.path.isfile(f)]
        print(f"=== {d} (count={len(files)}) ===")
        for f in sorted(files)[:25]:
            print("  ", f)

with open("scratch_eval_dirs.txt", "w") as out:
    for d in ["output/systematic_evaluation_30cases", "output/systematic_evaluation_30casesnew", "output/eval_memorization", "output/validation_experiments"]:
        if os.path.exists(d):
            files = [f for f in glob.glob(f"{d}/**/*", recursive=True) if os.path.isfile(f)]
            out.write(f"=== {d} (count={len(files)}) ===\n")
            for f in sorted(files)[:35]:
                out.write(f"   {f}\n")

print("Wrote scratch_eval_dirs.txt")
