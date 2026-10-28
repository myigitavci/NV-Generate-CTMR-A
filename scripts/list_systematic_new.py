#!/usr/bin/env python3
import os
import glob
from pathlib import Path

p = Path("output/systematic_evaluation_30casesnew")
files = sorted(list(p.glob("*")))
print(f"Total files in {p}: {len(files)}")
for f in files:
    print(f.name)
