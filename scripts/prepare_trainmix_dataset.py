from __future__ import annotations

import runpy
import sys
from pathlib import Path


_DISPLAY_TARGET = "scripts/data/prepare_trainmix_dataset.py"
_TARGET = Path(__file__).resolve().parent / "data" / "prepare_trainmix_dataset.py"

print(f"[shim] {Path(__file__).name} moved to {_DISPLAY_TARGET}", file=sys.stderr)
runpy.run_path(str(_TARGET), run_name="__main__")
