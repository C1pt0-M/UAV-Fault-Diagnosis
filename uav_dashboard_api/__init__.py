from __future__ import annotations

from pathlib import Path


_REAL_PACKAGE = Path(__file__).resolve().parents[1] / "apps" / "dashboard" / "backend" / "uav_dashboard_api"
__path__ = [str(_REAL_PACKAGE)]
_REAL_INIT = _REAL_PACKAGE / "__init__.py"
_loaded_real_package = False


def _load_real_package() -> None:
    global _loaded_real_package
    if _loaded_real_package:
        return
    _loaded_real_package = True
    globals()["__file__"] = str(_REAL_INIT)
    with open(_REAL_INIT, "r", encoding="utf-8") as fp:
        exec(compile(fp.read(), str(_REAL_INIT), "exec"), globals())


def __getattr__(name: str):
    _load_real_package()
    try:
        return globals()[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
