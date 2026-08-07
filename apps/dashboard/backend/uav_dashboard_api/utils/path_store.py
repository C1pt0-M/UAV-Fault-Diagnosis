from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from flask import has_app_context, current_app


RUNTIME_PREFIX = 'runtime://'
REPO_PREFIX = 'repo://'
PATH_LIKE_KEYS = {
    'stored_path',
    'prepared_csv_path',
    'csv_path',
    'input_csv',
    'output_csv',
    'source_path',
    'source_csv_path',
    'model_dir',
    'norm_source',
}


def _config_or_current(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        return config
    if has_app_context():
        return current_app.config
    raise RuntimeError('path storage helpers require config or Flask app context')


def _resolved_base(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ''):
        return None
    return Path(path_value).expanduser().resolve()


def _try_relative(target: Path, base: Path | None) -> str | None:
    if base is None:
        return None
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return None


def serialize_local_path(
    raw_path: str | Path | None,
    config: dict[str, Any] | None = None,
) -> str:
    if raw_path in (None, ''):
        return ''
    text = str(raw_path).strip()
    if not text:
        return ''
    if text.startswith(RUNTIME_PREFIX) or text.startswith(REPO_PREFIX):
        return text

    cfg = _config_or_current(config)
    path = Path(text).expanduser()
    resolved = path.resolve() if path.is_absolute() else path.resolve()
    runtime_dir = _resolved_base(cfg.get('RUNTIME_DIR'))
    repo_root = _resolved_base(cfg.get('REPO_ROOT'))

    runtime_rel = _try_relative(resolved, runtime_dir)
    if runtime_rel is not None:
        return f'{RUNTIME_PREFIX}{runtime_rel}'

    repo_rel = _try_relative(resolved, repo_root)
    if repo_rel is not None:
        return f'{REPO_PREFIX}{repo_rel}'

    return str(resolved)


def resolve_local_path(
    raw_path: str | Path | None,
    config: dict[str, Any] | None = None,
) -> str:
    if raw_path in (None, ''):
        return ''
    text = str(raw_path).strip()
    if not text:
        return ''

    cfg = _config_or_current(config)
    runtime_dir = _resolved_base(cfg.get('RUNTIME_DIR'))
    repo_root = _resolved_base(cfg.get('REPO_ROOT'))

    if text.startswith(RUNTIME_PREFIX):
        suffix = text[len(RUNTIME_PREFIX):].strip('/')
        return str(runtime_dir.joinpath(*PurePosixPath(suffix).parts))

    if text.startswith(REPO_PREFIX):
        suffix = text[len(REPO_PREFIX):].strip('/')
        return str(repo_root.joinpath(*PurePosixPath(suffix).parts))

    direct_path = Path(text).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    normalized = text.replace('\\', '/')
    legacy_markers = [
        ('/dashboard/backend/runtime/', runtime_dir),
        ('/runtime/', runtime_dir),
        ('/data/telemetry_sessions/', repo_root / 'data' / 'telemetry_sessions' if repo_root else None),
    ]
    for marker, base in legacy_markers:
        if base is None:
            continue
        idx = normalized.find(marker)
        if idx < 0:
            continue
        suffix = normalized[idx + len(marker):].strip('/')
        if not suffix:
            continue
        return str(base.joinpath(*PurePosixPath(suffix).parts))

    return text


def sanitize_path_payload(
    data: Any,
    config: dict[str, Any] | None = None,
    *,
    key_name: str | None = None,
):
    cfg = _config_or_current(config)
    if isinstance(data, dict):
        return {
            key: sanitize_path_payload(value, cfg, key_name=key)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [sanitize_path_payload(item, cfg, key_name=key_name) for item in data]
    if isinstance(data, str):
        if key_name and (
            key_name in PATH_LIKE_KEYS
            or key_name.endswith('_path')
            or key_name.endswith('_dir')
        ):
            return serialize_local_path(data, cfg)
    return data
