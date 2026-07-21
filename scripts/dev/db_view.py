#!/usr/bin/env python3
"""
Small helper to inspect the SQLite DB (runs + events) for demos/debugging.

Examples:
  python scripts/dev/db_view.py --db data/processed/uav_fault.sqlite --list-runs
  python scripts/dev/db_view.py --db data/processed/uav_fault.sqlite --run-id 3 --events
"""

from __future__ import annotations

import argparse
import sqlite3


def _print_table(rows, headers):
    if not rows:
        print("(empty)")
        return
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len("" if v is None else str(v)))
    fmt = "  ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*[("" if v is None else str(v)) for v in r]))


def main() -> int:
    parser = argparse.ArgumentParser(description="View runs/events in the UAV fault SQLite DB.")
    parser.add_argument("--db", required=True, help="SQLite DB path.")
    parser.add_argument("--list-runs", action="store_true", help="List runs (default).")
    parser.add_argument("--run-id", type=int, default=0, help="Target run id.")
    parser.add_argument("--events", action="store_true", help="Show events for --run-id.")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    if args.events and args.run_id > 0:
        rows = cur.execute(
            """
            SELECT id, start_t, end_t, n_points, max_fault_prob, mean_fault_prob
            FROM events
            WHERE run_id = ?
            ORDER BY start_t ASC
            """,
            (int(args.run_id),),
        ).fetchall()
        _print_table(
            rows,
            headers=["event_id", "start_t", "end_t", "n_points", "max_p", "mean_p"],
        )
        return 0

    # Default: list runs
    rows = cur.execute(
        """
        SELECT
          r.id,
          COALESCE(r.name, ''),
          r.kind,
          r.status,
          r.created_at,
          (SELECT COUNT(*) FROM events e WHERE e.run_id = r.id) AS n_events
        FROM runs r
        ORDER BY r.id DESC
        LIMIT 50
        """
    ).fetchall()
    _print_table(rows, headers=["run_id", "name", "kind", "status", "created_at", "n_events"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
