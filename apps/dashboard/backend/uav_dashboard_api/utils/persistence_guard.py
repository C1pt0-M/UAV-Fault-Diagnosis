from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _resolve_sqlite_db_path(database_uri: str | None) -> Path | None:
    uri = str(database_uri or '').strip()
    if not uri.startswith('sqlite:///'):
        return None
    raw_path = uri[len('sqlite:///'):].strip()
    if not raw_path or raw_path == ':memory:':
        return None
    return Path(raw_path)


def backup_sqlite_database(config: dict[str, Any], *, now: datetime | None = None, max_backups: int = 10) -> Path | None:
    db_path = _resolve_sqlite_db_path(config.get('SQLALCHEMY_DATABASE_URI'))
    if db_path is None or not db_path.exists() or not db_path.is_file():
        return None

    runtime_dir = Path(config.get('RUNTIME_DIR') or db_path.parent)
    backup_dir = runtime_dir / 'db_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'{db_path.stem}_{stamp}{db_path.suffix}'
    suffix = 1
    while backup_path.exists():
        backup_path = backup_dir / f'{db_path.stem}_{stamp}_{suffix}{db_path.suffix}'
        suffix += 1

    try:
        source = sqlite3.connect(db_path, timeout=float(config.get('SQLITE_BUSY_TIMEOUT', 30)))
        target = sqlite3.connect(backup_path)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        shutil.copystat(db_path, backup_path)
    except sqlite3.Error:
        backup_path.unlink(missing_ok=True)
        raise

    backups = sorted(backup_dir.glob(f'{db_path.stem}_*{db_path.suffix}'))
    overflow = len(backups) - max(1, int(max_backups))
    for stale in backups[:max(0, overflow)]:
        stale.unlink(missing_ok=True)

    return backup_path


def collect_runtime_persistence_snapshot(
    runtime_dir: Path,
    *,
    run_record_count: int,
    diagnosis_report_count: int,
) -> dict[str, Any]:
    runs_dir = runtime_dir / 'runs'
    run_result_files = len(list(runs_dir.glob('run_*/result.json')))

    warnings: list[str] = []
    if run_result_files > run_record_count:
        warnings.append(
            f'检测到 {runs_dir} 中有 {run_result_files} 个运行结果文件，但 runs 表只有 {run_record_count} 条记录；这通常说明 SQLite 数据库曾被替换、重建或回滚到较新的空状态。'
        )

    return {
        'run_result_files': run_result_files,
        'diagnosis_report_count': diagnosis_report_count,
        'warnings': warnings,
    }


def audit_runtime_persistence(config: dict[str, Any]) -> dict[str, Any]:
    from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
    from uav_dashboard_api.models.run_record import RunRecord

    runtime_dir = Path(config.get('RUNTIME_DIR'))
    return collect_runtime_persistence_snapshot(
        runtime_dir,
        run_record_count=RunRecord.query.count(),
        diagnosis_report_count=DiagnosisReport.query.count(),
    )
