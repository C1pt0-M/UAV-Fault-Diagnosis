import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from uav_dashboard_api import create_app
from uav_dashboard_api.utils.persistence_guard import backup_sqlite_database
from uav_dashboard_api.config import TestingConfig


def _seed_sqlite_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO sentinel(value) VALUES (?)", ("ok",))
    conn.commit()
    conn.close()


def test_create_app_startup_creates_db_backup_and_logs_runtime_gaps(tmp_path: Path, caplog):
    runtime_dir = tmp_path / "runtime"
    db_path = runtime_dir / "dashboard.db"
    _seed_sqlite_file(db_path)

    runs_dir = runtime_dir / "runs"
    reports_dir = runtime_dir / "reports"
    (runs_dir / "run_84").mkdir(parents=True, exist_ok=True)
    (runs_dir / "run_84" / "result.json").write_text("{}", encoding="utf-8")
    (runs_dir / "run_85").mkdir(parents=True, exist_ok=True)
    (runs_dir / "run_85" / "result.json").write_text("{}", encoding="utf-8")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "diagnosis_run_84_20260319163447.pdf").write_bytes(b"%PDF-1.4\n")

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runs_dir
        REPORTS_DIR = reports_dir
        AI_RUNTIME_DIR = runtime_dir / "ai"
        AI_KNOWLEDGE_DIR = runtime_dir / "ai" / "knowledge"
        AI_ENABLE_KNOWLEDGE_BASE = False

    with caplog.at_level(logging.INFO):
        app = create_app(LocalTestingConfig)

    backup_dir = runtime_dir / "db_backups"
    backups = sorted(backup_dir.glob("dashboard_*.db"))
    assert len(backups) == 1
    assert backups[0].exists()
    assert backups[0].read_bytes()[:16] == db_path.read_bytes()[:16]
    normalized_log = caplog.text.replace("\\", "/")
    assert "runtime/runs" in normalized_log
    assert "runtime/reports" in normalized_log

    # 闃叉 app 鐢熷懡鍛ㄦ湡閲岀殑鎵╁睍寮曠敤褰卞搷鍚庣画娴嬭瘯
    assert app.config["RUNTIME_DIR"] == runtime_dir


def test_backup_sqlite_database_includes_committed_wal_changes(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    db_path = runtime_dir / "dashboard.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.execute("INSERT INTO sentinel(value) VALUES (?)", ("from-wal",))
    conn.commit()
    assert db_path.with_name(f"{db_path.name}-wal").exists()
    conn.close()

    backup_path = backup_sqlite_database(
        {
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
            "RUNTIME_DIR": runtime_dir,
            "SQLITE_BUSY_TIMEOUT": 30,
        },
        now=datetime(2026, 4, 27, 10, 0, 0),
    )

    backup_conn = sqlite3.connect(backup_path)
    try:
        rows = backup_conn.execute("SELECT value FROM sentinel").fetchall()
    finally:
        backup_conn.close()

    assert rows == [("from-wal",)]
