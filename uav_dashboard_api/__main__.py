from __future__ import annotations

import importlib.util
from pathlib import Path


_REAL_MAIN = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "dashboard"
    / "backend"
    / "uav_dashboard_api"
    / "__main__.py"
)
_SPEC = importlib.util.spec_from_file_location("_uav_dashboard_api_real_main", _REAL_MAIN)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load dashboard API entrypoint: {_REAL_MAIN}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


if __name__ == "__main__":
    main()
