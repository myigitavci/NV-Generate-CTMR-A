import os
from pathlib import Path

p = Path("models/frozen_param_predictors.joblib")
if p.exists():
    p.unlink()
    print(f"Removed stale {p}")
else:
    print(f"{p} already gone")
