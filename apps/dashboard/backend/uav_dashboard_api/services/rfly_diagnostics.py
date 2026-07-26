from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

DEFAULT_RFLY_PX4MAVCTRL_PATH = Path(
    r"D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py"
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw_value: Any, *, base_dir: Path) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _safe_name(path: Path | None) -> str | None:
    return path.name if path is not None else None


def _check(ok: bool, *, code: str, label: str, message: str, **extra) -> dict:
    payload = {
        "ok": bool(ok),
        "code": code,
        "label": label,
        "message": message,
    }
    payload.update(extra)
    return payload


def build_rfly_diagnostics(config: dict, override_px4mavctrl_path: str | None = None) -> dict:
    allowed_backends = list(config.get("TELEMETRY_ALLOWED_BACKENDS") or ())
    repo_root = Path(str(config.get("REPO_ROOT") or Path(__file__).resolve().parents[4]))
    bridge_mode = str(config.get("RFLY_BRIDGE_MODE") or "attach").strip().lower()
    if bridge_mode not in {"attach", "buaa"}:
        bridge_mode = "attach"
    sim_map = str(config.get("RFLY_SIM_MAP") or "LowGPU").strip() or "LowGPU"

    px4mavctrl_path = _resolve_path(
        override_px4mavctrl_path or config.get("RFLY_PX4MAVCTRL_PATH"),
        base_dir=repo_root,
    )
    if px4mavctrl_path is None:
        px4mavctrl_path = DEFAULT_RFLY_PX4MAVCTRL_PATH
    bridge_script = _resolve_path(config.get("RFLY_BRIDGE_SCRIPT"), base_dir=repo_root)
    if bridge_script is None:
        script_name = "run_buaa_realtime_bridge.py" if bridge_mode == "buaa" else "run_rfly_live_bridge.py"
        bridge_script = repo_root / "packages" / "rfly_bridge" / "rfly_bridge" / script_name
    buaa_autotest_dir = _resolve_path(
        config.get("RFLY_BUAA_AUTOTEST_DIR"),
        base_dir=repo_root,
    )
    python_bin = str(
        config.get("RFLY_PYTHON_BIN")
        or config.get("TELEMETRY_PYTHON_BIN")
        or config.get("ML_PYTHON_BIN")
        or sys.executable
    ).strip()
    python_path = Path(python_bin).expanduser() if python_bin else None

    port_ok = True
    port_message = "RflySim forward port 可用"
    try:
        forward_port = int(config.get("RFLY_FORWARD_PORT", 15550))
        if forward_port <= 0 or forward_port > 65535:
            port_ok = False
            port_message = "RFLY_FORWARD_PORT 必须在 1-65535 范围内"
    except (TypeError, ValueError):
        forward_port = None
        port_ok = False
        port_message = "RFLY_FORWARD_PORT 必须是数字"

    checks = {
        "backend": _check(
            "rfly" in allowed_backends,
            code="backend_allowed",
            label="RflySim 后端开放",
            message="RflySim 后端已开放" if "rfly" in allowed_backends else "当前环境未开放 RflySim 后端",
            allowed_backends=allowed_backends,
        ),
        "bridge_mode": _check(
            bridge_mode in {"attach", "buaa"},
            code="bridge_mode_valid",
            label="Bridge mode",
            message=f"RflySim bridge mode: {bridge_mode}",
            mode=bridge_mode,
        ),
        "sim_map": _check(
            bool(sim_map),
            code="sim_map_configured",
            label="RflySim 3D map",
            message=f"RflySim 3D map: {sim_map}",
            map=sim_map,
        ),
        "px4mavctrl": _check(
            bridge_mode == "buaa" or (px4mavctrl_path is not None and px4mavctrl_path.exists()),
            code="px4mavctrl_exists",
            label="PX4MavCtrlV4.py",
            message="Buaa mode 使用 AutoTestAPI 内置 PX4MavCtrlV4.py"
            if bridge_mode == "buaa"
            else "PX4MavCtrlV4.py 可用"
            if px4mavctrl_path is not None and px4mavctrl_path.exists()
            else "缺少 PX4MavCtrlV4.py，请在后端 .env 配置或在本次启动时填写",
            configured=px4mavctrl_path is not None,
            name=_safe_name(px4mavctrl_path),
        ),
        "buaa_autotest": _check(
            bridge_mode != "buaa" or (buaa_autotest_dir is not None and buaa_autotest_dir.exists()),
            code="buaa_autotest_exists",
            label="BuaaAutoTest AutoTestAPI",
            message="BuaaAutoTest AutoTestAPI 可用"
            if bridge_mode != "buaa" or (buaa_autotest_dir is not None and buaa_autotest_dir.exists())
            else "Buaa mode 需要配置 RFLY_BUAA_AUTOTEST_DIR",
            name=_safe_name(buaa_autotest_dir),
        ),
        "bridge_script": _check(
            bridge_script.exists(),
            code="bridge_script_exists",
            label="RflySim bridge script",
            message="RflySim bridge script 可用"
            if bridge_script.exists()
            else "RflySim bridge script 不存在",
            name=_safe_name(bridge_script),
        ),
        "python_bin": _check(
            bool(python_bin) and (python_path.exists() if python_path is not None and python_path.is_absolute() else True),
            code="python_bin_exists",
            label="Rfly Bridge Python",
            message="Rfly Bridge Python 已配置" if python_bin else "Rfly Bridge Python 未配置",
            name=_safe_name(python_path) or python_bin,
        ),
        "forward_port": _check(
            port_ok,
            code="forward_port_valid",
            label="Forward UDP port",
            message=port_message,
            host=str(config.get("RFLY_FORWARD_HOST", "127.0.0.1")),
            port=forward_port,
        ),
        "pymav": _check(
            True,
            code="pymav_configured",
            label="PymavLink 增强数据",
            message="PymavLink 增强数据已启用"
            if _as_bool(config.get("RFLY_ENABLE_PYMAV"), True)
            else "PymavLink 增强数据未启用，将使用 RflySim 基础状态",
            enabled=_as_bool(config.get("RFLY_ENABLE_PYMAV"), True),
            address=str(config.get("RFLY_PYMAV_ADDRESS", "udpin:0.0.0.0:14031")),
        ),
    }
    blocking_keys = {
        "backend",
        "bridge_mode",
        "px4mavctrl",
        "buaa_autotest",
        "bridge_script",
        "python_bin",
        "forward_port",
    }
    issues = [
        {
            "code": "backend_not_allowed" if key == "backend" else f"{key}_missing",
            "label": check["label"],
            "message": check["message"],
        }
        for key, check in checks.items()
        if key in blocking_keys and not check["ok"]
    ]
    ready = not issues
    return {
        "ready": ready,
        "summary": {
            "status": "ready" if ready else "not_ready",
            "message": "RflySim 链路配置可用" if ready else issues[0]["message"],
        },
        "checks": checks,
        "issues": issues,
    }
