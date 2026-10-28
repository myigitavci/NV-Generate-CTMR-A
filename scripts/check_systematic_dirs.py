#!/usr/bin/env python3
import os
import glob
from pathlib import Path

for pattern in [
    "output/systematic_evaluation_30casesnew*",
    "output/systematic_evaluation*",
    "/home/mav24/projects/NV-Generate-CTMR-A/output/*",
]:
    matches = glob.glob(pattern)
    print(f"=== Pattern: {pattern} ===")
    for m in matches:
        print(" ", m, "(dir)" if os.path.isdir(m) else "(file)")
        if os.path.isdir(m):
            sub = sorted(os.listdir(m))
            print("    Contains", len(sub), "items. Samples:", sub[:10])
