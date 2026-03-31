"""
SQLite persistence layer for the (single-machine) UAV fault detection system.

Design goals:
- zero-ops: a single .sqlite file, no external service required
- store "delivery-friendly" artifacts: input assets, model registry, runs, and fault events
- avoid stuffing full raw telemetry frames into SQLite (keep them as files; store paths + indexes here)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    # SQLite can also do this, but generating in Python keeps inserts consistent
    # when we do multi-statement transactions.
    import datetime as _dt

    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat(timespec="milliseconds")


def _to_json(obj: Any | None) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class EventRow:
    run_id: int
    start_t: float
    end_t: float
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None
    n_points: int = 0
    max_fault_prob: float | None = None
    mean_fault_prob: float | None = None
    meta: dict[str, Any] | None = None


class SQLiteStore:
    """Tiny SQLite wrapper with an opinionated schema."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        # Safer defaults for a single-machine service.
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self._ensure_schema()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
              id INTEGER PRIMARY KEY,
              kind TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT,
              size_bytes INTEGER,
              sample_hz REAL,
              t_start REAL,
              t_end REAL,
              meta_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(path)
            );

            CREATE TABLE IF NOT EXISTS models (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              task TEXT NOT NULL,
              model_dir TEXT NOT NULL,
              config_json TEXT,
              input_columns_json TEXT,
              metrics_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(model_dir)
            );

            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY,
              name TEXT,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              asset_id INTEGER,
              model_id INTEGER NOT NULL,
              params_json TEXT,
              summary_json TEXT,
              error TEXT,
              started_at TEXT,
              finished_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(asset_id) REFERENCES assets(id),
              FOREIGN KEY(model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              start_t REAL NOT NULL,
              end_t REAL NOT NULL,
              start_frame_idx INTEGER,
              end_frame_idx INTEGER,
              n_points INTEGER,
              max_fault_prob REAL,
              mean_fault_prob REAL,
              meta_json TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_run_start ON events(run_id, start_t);

            -- Optional table (debug). Not used by default, but keeps the schema extensible.
            CREATE TABLE IF NOT EXISTS prediction_points (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              t REAL,
              frame_idx INTEGER,
              pred_label TEXT NOT NULL,
              fault_prob REAL,
              confidence REAL,
              raw_logits_json TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_points_run_t ON prediction_points(run_id, t);
            """
        )
        self.conn.commit()

    # ── Assets ──────────────────────────────────────────────────────────

    def upsert_asset(
        self,
        *,
        kind: str,
        path: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
        sample_hz: float | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        now = _utc_now_iso()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO assets (kind, path, sha256, size_bytes, sample_hz, t_start, t_end,
                                meta_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              kind=excluded.kind,
              sha256=COALESCE(excluded.sha256, assets.sha256),
              size_bytes=COALESCE(excluded.size_bytes, assets.size_bytes),
              sample_hz=COALESCE(excluded.sample_hz, assets.sample_hz),
              t_start=COALESCE(excluded.t_start, assets.t_start),
              t_end=COALESCE(excluded.t_end, assets.t_end),
              meta_json=COALESCE(excluded.meta_json, assets.meta_json),
              updated_at=excluded.updated_at
            """,
            (
                str(kind),
                str(path),
                sha256,
                size_bytes,
                sample_hz,
                t_start,
                t_end,
                _to_json(meta),
                now,
                now,
            ),
        )
        self.conn.commit()

        row = cur.execute("SELECT id FROM assets WHERE path = ?", (str(path),)).fetchone()
        assert row is not None
        return int(row["id"])

    # ── Models ─────────────────────────────────────────────────────────

    def upsert_model(
        self,
        *,
        name: str,
        task: str,
        model_dir: str,
        config: dict[str, Any] | None = None,
        input_columns: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> int:
        now = _utc_now_iso()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO models (name, task, model_dir, config_json, input_columns_json, metrics_json,
                                created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_dir) DO UPDATE SET
              name=excluded.name,
              task=excluded.task,
              config_json=COALESCE(excluded.config_json, models.config_json),
              input_columns_json=COALESCE(excluded.input_columns_json, models.input_columns_json),
              metrics_json=COALESCE(excluded.metrics_json, models.metrics_json),
              updated_at=excluded.updated_at
            """,
            (
                str(name),
                str(task),
                str(model_dir),
                _to_json(config),
                _to_json(input_columns),
                _to_json(metrics),
                now,
                now,
            ),
        )
        self.conn.commit()

        row = cur.execute("SELECT id FROM models WHERE model_dir = ?", (str(model_dir),)).fetchone()
        assert row is not None
        return int(row["id"])

    # ── Runs ───────────────────────────────────────────────────────────

    def create_run(
        self,
        *,
        kind: str,
        status: str,
        model_id: int,
        asset_id: int | None = None,
        name: str | None = None,
        params: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> int:
        now = _utc_now_iso()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (name, kind, status, asset_id, model_id, params_json,
                              started_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                str(kind),
                str(status),
                asset_id,
                int(model_id),
                _to_json(params),
                started_at,
                now,
                now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        finished_at: str | None = None,
    ):
        now = _utc_now_iso()
        parts = []
        vals: list[Any] = []
        if status is not None:
            parts.append("status = ?")
            vals.append(str(status))
        if summary is not None:
            parts.append("summary_json = ?")
            vals.append(_to_json(summary))
        if error is not None:
            parts.append("error = ?")
            vals.append(str(error))
        if finished_at is not None:
            parts.append("finished_at = ?")
            vals.append(str(finished_at))
        parts.append("updated_at = ?")
        vals.append(now)
        vals.append(int(run_id))

        q = "UPDATE runs SET " + ", ".join(parts) + " WHERE id = ?"
        self.conn.execute(q, tuple(vals))
        self.conn.commit()

    # ── Events ─────────────────────────────────────────────────────────

    def insert_event(self, event: EventRow) -> int:
        now = _utc_now_iso()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO events (run_id, start_t, end_t, start_frame_idx, end_frame_idx,
                                n_points, max_fault_prob, mean_fault_prob, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(event.run_id),
                float(event.start_t),
                float(event.end_t),
                event.start_frame_idx,
                event.end_frame_idx,
                int(event.n_points),
                event.max_fault_prob,
                event.mean_fault_prob,
                _to_json(event.meta),
                now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_events(self, events: list[EventRow]) -> int:
        if not events:
            return 0
        now = _utc_now_iso()
        rows = [
            (
                int(e.run_id),
                float(e.start_t),
                float(e.end_t),
                e.start_frame_idx,
                e.end_frame_idx,
                int(e.n_points),
                e.max_fault_prob,
                e.mean_fault_prob,
                _to_json(e.meta),
                now,
            )
            for e in events
        ]
        self.conn.executemany(
            """
            INSERT INTO events (run_id, start_t, end_t, start_frame_idx, end_frame_idx,
                                n_points, max_fault_prob, mean_fault_prob, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(events)

