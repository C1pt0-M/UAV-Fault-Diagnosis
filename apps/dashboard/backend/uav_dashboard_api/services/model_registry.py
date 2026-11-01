"""Model registry helpers for dashboard integration."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from uav_fault.model_contract import (
    RFLY_SEQ_COLUMNS_V1,
    offline_input_contract_errors,
    offline_label_names,
    offline_norm_contract_errors,
    online_input_contract_errors,
)


SUPPORTED_OFFLINE_ARCHITECTURES = {
    "cnn_lstm_v2_normalproto",
}
SUPPORTED_ONLINE_ARCHITECTURES = {"cnn1d", "mstca", "kg_mstca"}
OFFLINE_MULTICLASS_TASK = "offline_multiclass"
LEGACY_OFFLINE_MULTICLASS_TASK = "offline_5class"
OFFLINE_MULTICLASS_TASK_ALIASES = {
    OFFLINE_MULTICLASS_TASK,
    LEGACY_OFFLINE_MULTICLASS_TASK,
}


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


def _manifest_models(model_root: Path) -> list[tuple[Path, str | None]] | None:
    manifest_path = model_root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Model manifest version must be 1")
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise ValueError("Model manifest models must be a list")

    root = model_root.resolve()
    models: list[tuple[Path, str | None]] = []
    names: set[str] = set()
    directories: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Model manifest entry must be an object")
        raw_path = entry.get("path", entry.get("name"))
        if not isinstance(raw_path, str) or not raw_path or raw_path.strip() != raw_path:
            raise ValueError("Model manifest path must be a nonblank relative path")
        windows_path, posix_path = PureWindowsPath(raw_path), PurePosixPath(raw_path)
        if windows_path.drive or windows_path.root or posix_path.is_absolute() or ".." in windows_path.parts:
            raise ValueError(f"Model manifest path must stay inside MODEL_ROOT: {raw_path}")
        model_dir = (root / raw_path).resolve()
        if model_dir == root or not model_dir.is_relative_to(root):
            raise ValueError(f"Model manifest path must stay inside MODEL_ROOT: {raw_path}")
        name = entry.get("name", model_dir.name)
        if name != model_dir.name:
            raise ValueError(f"Model manifest name must match directory name: {raw_path}")
        if name in names or model_dir in directories:
            raise ValueError(f"Duplicate model manifest entry: {name}")
        runtime = entry.get("runtime")
        if "runtime" in entry and runtime not in ("online", "offline"):
            raise ValueError(f"Invalid model manifest runtime: {name}")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Manifest model directory does not exist: {model_dir}")
        for filename in ("config.json", "best.pt"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Manifest model file does not exist: {model_dir / filename}")
        names.add(name)
        directories.add(model_dir)
        models.append((model_dir, runtime))
    return models


def _task_matches(model_task: str, requested_task: str | None) -> bool:
    if not requested_task:
        return True
    if model_task == requested_task:
        return True
    if (
        requested_task in OFFLINE_MULTICLASS_TASK_ALIASES
        and model_task in OFFLINE_MULTICLASS_TASK_ALIASES
    ):
        return True
    return False


def _as_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, parsed)


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
    if _infer_architecture(model_dir, config) in SUPPORTED_OFFLINE_ARCHITECTURES:
        return OFFLINE_MULTICLASS_TASK
    num_classes = _as_nonnegative_int(config.get("num_classes"))
    raw_label_names = config.get("label_names")
    label_names = (
        [str(item).strip().lower() for item in raw_label_names if str(item).strip()]
        if isinstance(raw_label_names, list)
        else []
    )

    if num_classes == 2 or label_names == ["normal", "fault"]:
        return "online_binary"
    if num_classes > 2:
        return OFFLINE_MULTICLASS_TASK
    if norm_stats and norm_stats.get("columns"):
        return "online_binary"
    if model_dir.name.startswith("online_"):
        return "online_binary"
    return "unknown"


def _default_label_names(task: str, num_classes: int) -> list[str]:
    if task == "online_binary" and num_classes == 2:
        return ["normal", "fault"]
    if task in OFFLINE_MULTICLASS_TASK_ALIASES and num_classes >= 3:
        return offline_label_names({}, num_classes)
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
    num_classes = _as_nonnegative_int(config.get("num_classes", 5 if task in OFFLINE_MULTICLASS_TASK_ALIASES else 0))
    label_names = config.get("label_names")
    if not isinstance(label_names, list) or not label_names:
        label_names = _default_label_names(task, num_classes)

    input_contract_errors: list[str] = []
    if task == "online_binary":
        input_contract_errors = online_input_contract_errors(config, norm_stats)
    elif architecture in SUPPORTED_OFFLINE_ARCHITECTURES:
        input_contract_errors = offline_input_contract_errors(config)
        if not input_contract_errors:
            label_names = offline_label_names(config, num_classes)
        if norm_stats_path.exists():
            input_contract_errors.extend(offline_norm_contract_errors(norm_stats, RFLY_SEQ_COLUMNS_V1, config=config))
    has_valid_input_contract = not input_contract_errors
    capabilities = {
        "offline_detection": (
            task in OFFLINE_MULTICLASS_TASK_ALIASES
            and architecture in SUPPORTED_OFFLINE_ARCHITECTURES
            and has_valid_input_contract
        ),
        "realtime_detection": (
            task == "online_binary"
            and architecture in SUPPORTED_ONLINE_ARCHITECTURES
            and has_valid_input_contract
        ),
        "fault_prob_curve": (
            task == "online_binary"
            and architecture in SUPPORTED_ONLINE_ARCHITECTURES
            and has_valid_input_contract
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
        "input_contract_errors": input_contract_errors,
        "has_norm_stats": norm_stats_path.exists(),
        "norm_stats_path": str(norm_stats_path) if norm_stats_path.exists() else None,
        "window_size": norm_stats.get("window_size") if isinstance(norm_stats, dict) else None,
        "feature_count": (
            len(norm_stats["columns"])
            if isinstance(norm_stats, dict) and isinstance(norm_stats.get("columns"), list)
            else 0
        ),
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

    manifest_models = _manifest_models(model_root)
    models = (
        manifest_models if manifest_models is not None
        else [(directory, None) for directory in sorted(model_root.iterdir(), key=lambda item: item.name)]
    )

    items: list[dict[str, Any]] = []
    for model_dir, runtime in models:
        if whitelist and model_dir.name not in whitelist:
            continue
        meta = _build_metadata(model_dir)
        if meta is None:
            if manifest_models is not None:
                raise ValueError(f"Invalid manifest model metadata: {model_dir}")
            continue
        if runtime is not None:
            expected_tasks = {"online_binary"} if runtime == "online" else OFFLINE_MULTICLASS_TASK_ALIASES
            if meta["task"] not in expected_tasks:
                raise ValueError(f"Model manifest runtime disagrees with model task: {model_dir.name}")
        if not _task_matches(meta["task"], task):
            continue
        if capability and not meta["capabilities"].get(capability, False) and not include_unavailable:
            continue
        items.append(meta)

    items.sort(key=lambda item: (_model_score(item), item.get("updated_at", 0), item["name"]), reverse=True)

    preferred_names: list[str] = []
    if task == "online_binary" or capability in {"realtime_detection", "fault_prob_curve"}:
        preferred_names.extend(_as_name_list(config.get("ONLINE_MODEL_PRIORITY")))
    elif capability == "offline_detection" or task in OFFLINE_MULTICLASS_TASK_ALIASES:
        preferred_names.extend(_as_name_list(config.get("DEFAULT_OFFLINE_MODEL")))

    if preferred_names:
        preferred_rank = {name: idx for idx, name in enumerate(preferred_names) if name}
        items.sort(
            key=lambda item: (preferred_rank.get(item["name"], len(preferred_rank)), ),
        )
    return items


def _match_requested_model(
    config: dict[str, Any],
    requested_name: str,
    *,
    task: str | None = None,
    capability: str | None = None,
) -> dict[str, Any] | None:
    """Resolve only a model already accepted by the registry policy."""
    raw = requested_name.strip()
    if not raw:
        return None

    for meta in list_registered_models(
        config,
        task=task,
        capability=capability,
        include_unavailable=False,
    ):
        if raw == meta["name"]:
            return meta
    return None


def resolve_registered_model(
    config: dict[str, Any],
    *,
    requested_name: str | None = None,
    task: str | None = None,
    capability: str | None = None,
) -> dict[str, Any] | None:
    if requested_name:
        return _match_requested_model(
            config,
            requested_name,
            task=task,
            capability=capability,
        )

    preferred_names: list[str] = []
    if task == "online_binary" or capability in {"realtime_detection", "fault_prob_curve"}:
        preferred_names.extend(_as_name_list(config.get("DEFAULT_ONLINE_MODEL")))
        preferred_names.extend(_as_name_list(config.get("ONLINE_MODEL_PRIORITY")))
    elif capability == "offline_detection" or task in OFFLINE_MULTICLASS_TASK_ALIASES:
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
