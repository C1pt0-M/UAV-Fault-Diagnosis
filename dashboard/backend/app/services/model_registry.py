"""Model registry helpers for dashboard integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_OFFLINE_ARCHITECTURES = {
    "lstm",
    "transformer",
    "textcnn",
    "cnn_lstm",
    "cnn_lstm_v2",
    "cnn_lstm_v2_normalproto",
    "cnn_lstm_v2_dg",
    "bilstm_normalproto",
}
SUPPORTED_ONLINE_ARCHITECTURES = {"bilstm", "cnn1d", "transformer", "mstca"}


def _as_name_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _as_name_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data if isinstance(data, dict) else {}


def _infer_architecture(model_dir: Path, config: dict[str, Any]) -> str:
    for key in ("model", "architecture", "model_name"):
        value = str(config.get(key, "")).strip()
        if value:
            return value
    return model_dir.name


def _infer_task(
    model_dir: Path,
    config: dict[str, Any],
    norm_stats: dict[str, Any] | None,
) -> str:
    num_classes = int(config.get("num_classes") or 0)
    label_names = [str(item).strip().lower() for item in config.get("label_names", []) if str(item).strip()]

    if num_classes == 2 or label_names == ["normal", "fault"]:
        return "online_binary"
    if num_classes == 5:
        return "offline_5class"
    if num_classes > 2:
        return "offline_multiclass"
    if norm_stats and norm_stats.get("columns"):
        return "online_binary"
    if model_dir.name.startswith("online_"):
        return "online_binary"
    return "unknown"


def _default_label_names(task: str, num_classes: int) -> list[str]:
    if task == "online_binary" and num_classes == 2:
        return ["normal", "fault"]
    if task == "offline_5class" and num_classes == 5:
        return ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss"]
    if task == "offline_multiclass" and num_classes == 3:
        return ["normal", "motor", "sensor"]
    if task == "offline_multiclass" and num_classes == 4:
        return ["normal", "motor", "propeller", "sensor"]
    return [f"class_{idx}" for idx in range(num_classes)] if num_classes > 0 else []


def _metrics_bundle(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "weighted_f1": metrics.get("weighted_f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_fault": metrics.get("f1_fault"),
        "best_val_f1": config.get("best_val_f1"),
    }


def _model_score(meta: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = meta.get("metrics", {})
    def _num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.0

    return (
        _num(metrics.get("macro_f1")),
        _num(metrics.get("weighted_f1")),
        _num(metrics.get("accuracy")),
        _num(metrics.get("best_val_f1")),
    )


def _build_metadata(model_dir: Path) -> dict[str, Any] | None:
    config_path = model_dir / "config.json"
    best_path = model_dir / "best.pt"
    if not model_dir.is_dir() or not config_path.exists() or not best_path.exists():
        return None

    try:
        config = _read_json(config_path)
    except Exception:
        return None

    metrics_path = model_dir / "metrics_test.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = _read_json(metrics_path)
        except Exception:
            metrics = {}

    metrics_external_path = model_dir / "metrics_external_test.json"
    metrics_external = {}
    if metrics_external_path.exists():
        try:
            metrics_external = _read_json(metrics_external_path)
        except Exception:
            metrics_external = {}

    norm_stats_path = model_dir / "norm_stats.json"
    norm_stats = None
    if norm_stats_path.exists():
        try:
            norm_stats = _read_json(norm_stats_path)
        except Exception:
            norm_stats = None

    architecture = _infer_architecture(model_dir, config)
    task = _infer_task(model_dir, config, norm_stats)
    num_classes = int(config.get("num_classes") or 0)
    label_names = config.get("label_names")
    if not isinstance(label_names, list) or not label_names:
        label_names = _default_label_names(task, num_classes)

    capabilities = {
        "offline_detection": (
            task in {"offline_5class", "offline_multiclass"}
            and architecture in SUPPORTED_OFFLINE_ARCHITECTURES
        ),
        "realtime_detection": (
            task == "online_binary"
            and architecture in SUPPORTED_ONLINE_ARCHITECTURES
            and norm_stats_path.exists()
        ),
        "fault_prob_curve": (
            task == "online_binary"
            and architecture in SUPPORTED_ONLINE_ARCHITECTURES
            and norm_stats_path.exists()
        ),
    }

    return {
        "name": model_dir.name,
        "label": model_dir.name,
        "path": str(model_dir),
        "task": task,
        "architecture": architecture,
        "num_classes": num_classes,
        "label_names": label_names,
        "metrics": _metrics_bundle(config, metrics),
        "metrics_external": _metrics_bundle(config, metrics_external),
        "capabilities": capabilities,
        "has_norm_stats": norm_stats_path.exists(),
        "norm_stats_path": str(norm_stats_path) if norm_stats_path.exists() else None,
        "window_size": norm_stats.get("window_size") if isinstance(norm_stats, dict) else None,
        "feature_count": len(norm_stats.get("columns") or []) if isinstance(norm_stats, dict) else 0,
        "updated_at": int(model_dir.stat().st_mtime),
    }


def list_registered_models(
    config: dict[str, Any],
    *,
    task: str | None = None,
    capability: str | None = None,
    include_unavailable: bool = False,
) -> list[dict[str, Any]]:
    model_root = Path(config["MODEL_ROOT"])
    whitelist = _as_name_set(config.get("MODEL_WHITELIST"))
    if not model_root.exists():
        return []

    items: list[dict[str, Any]] = []
    for model_dir in sorted(model_root.iterdir(), key=lambda item: item.name):
        if whitelist and model_dir.name not in whitelist:
            continue
        meta = _build_metadata(model_dir)
        if meta is None:
            continue
        if task and meta["task"] != task:
            continue
        if capability and not meta["capabilities"].get(capability, False) and not include_unavailable:
            continue
        items.append(meta)

    items.sort(key=lambda item: (_model_score(item), item.get("updated_at", 0), item["name"]), reverse=True)

    preferred_names: list[str] = []
    if task == "online_binary" or capability in {"realtime_detection", "fault_prob_curve"}:
        preferred_names.extend(_as_name_list(config.get("DEFAULT_ONLINE_MODEL")))
        preferred_names.extend(_as_name_list(config.get("ONLINE_MODEL_PRIORITY")))
    elif capability == "offline_detection" or task in {"offline_multiclass", "offline_5class"}:
        preferred_names.extend(_as_name_list(config.get("DEFAULT_OFFLINE_MODEL")))

    if preferred_names:
        preferred_set = {name for name in preferred_names if name}
        items.sort(
            key=lambda item: (item["name"] not in preferred_set, ),
        )
    return items


def _match_requested_model(
    config: dict[str, Any],
    requested_name: str,
) -> Path | None:
    raw = requested_name.strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate

    model_root = Path(config["MODEL_ROOT"])
    direct = model_root / raw
    if direct.exists():
        return direct
    return None


def resolve_registered_model(
    config: dict[str, Any],
    *,
    requested_name: str | None = None,
    task: str | None = None,
    capability: str | None = None,
) -> dict[str, Any] | None:
    if requested_name:
        requested_path = _match_requested_model(config, requested_name)
        if requested_path is not None:
            meta = _build_metadata(requested_path)
            if meta is None:
                return None
            if task and meta["task"] != task:
                return None
            if capability and not meta["capabilities"].get(capability, False):
                return None
            return meta

        models = list_registered_models(
            config,
            task=task,
            capability=capability,
            include_unavailable=False,
        )
        for meta in models:
            if meta["name"] == requested_name:
                return meta
        return None

    preferred_names: list[str] = []
    if task == "online_binary" or capability in {"realtime_detection", "fault_prob_curve"}:
        preferred_names.extend(_as_name_list(config.get("DEFAULT_ONLINE_MODEL")))
        preferred_names.extend(_as_name_list(config.get("ONLINE_MODEL_PRIORITY")))
    elif capability == "offline_detection":
        preferred_names.extend(_as_name_list(config.get("DEFAULT_OFFLINE_MODEL")))

    models = list_registered_models(
        config,
        task=task,
        capability=capability,
        include_unavailable=False,
    )
    if not models:
        return None

    if preferred_names:
        for name in preferred_names:
            for meta in models:
                if meta["name"] == name or meta["path"] == name:
                    return meta

    return models[0]
