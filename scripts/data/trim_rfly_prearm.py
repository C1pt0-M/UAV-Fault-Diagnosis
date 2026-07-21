from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def trim_prearm(
    path: Path,
    *,
    max_prearm_seconds: float = 30.0,
    keep_prearm_seconds: float = 20.0,
) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or "time_s" not in columns or "armed" not in columns:
        raise ValueError("CSV must contain rows, time_s, and armed")

    times = [float(row["time_s"]) for row in rows]
    armed_indices = [
        idx for idx, row in enumerate(rows) if float(row.get("armed", 0) or 0) > 0.5
    ]
    if not armed_indices:
        raise ValueError("CSV does not contain an armed flight phase")

    first_armed = armed_indices[0]
    prearm_seconds = max(0.0, times[first_armed] - times[0])
    result = {
        "status": "unchanged",
        "file": str(path.resolve()),
        "rows": len(rows),
        "prearm_seconds": round(prearm_seconds, 3),
    }
    if prearm_seconds <= max_prearm_seconds:
        return result

    start_time = times[first_armed] - keep_prearm_seconds
    start_idx = next(idx for idx, value in enumerate(times) if value >= start_time)
    trimmed = rows[start_idx:]
    first_time = float(trimmed[0]["time_s"])
    for idx, row in enumerate(trimmed):
        row["seq"] = str(idx)
        row["time_s"] = str(round(float(row["time_s"]) - first_time, 6))

    backup_dir = path.parent / "raw_long_prearm"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.stem}.long_prearm_raw.csv"
    shutil.copy2(path, backup_path)

    temp_path = path.with_suffix(path.suffix + ".trimmed.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trimmed)
    os.replace(temp_path, path)

    return {
        "status": "trimmed",
        "file": str(path.resolve()),
        "backup": str(backup_path.resolve()),
        "raw_rows": len(rows),
        "rows": len(trimmed),
        "removed_rows": start_idx,
        "raw_prearm_seconds": round(prearm_seconds, 3),
        "kept_prearm_seconds": round(times[first_armed] - times[start_idx], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim excessive pre-Armed telemetry")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max-prearm-seconds", type=float, default=30.0)
    parser.add_argument("--keep-prearm-seconds", type=float, default=20.0)
    args = parser.parse_args()

    result = trim_prearm(
        Path(args.csv),
        max_prearm_seconds=args.max_prearm_seconds,
        keep_prearm_seconds=args.keep_prearm_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
