from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "data" / "trim_rfly_prearm.py"
SPEC = importlib.util.spec_from_file_location("trim_rfly_prearm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trimmer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trimmer)


def _write_session(path: Path, *, first_armed: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq", "time_s", "armed", "value"])
        writer.writeheader()
        for idx in range(5000):
            writer.writerow(
                {
                    "seq": idx,
                    "time_s": idx * 0.02,
                    "armed": int(first_armed <= idx <= 4500),
                    "value": idx,
                }
            )


def test_trim_prearm_keeps_short_session_unchanged(tmp_path):
    path = tmp_path / "short.csv"
    _write_session(path, first_armed=1000)

    result = trimmer.trim_prearm(path)

    assert result["status"] == "unchanged"
    assert result["rows"] == 5000


def test_trim_prearm_backs_up_and_keeps_twenty_seconds(tmp_path):
    path = tmp_path / "long.csv"
    _write_session(path, first_armed=3000)

    result = trimmer.trim_prearm(path)

    assert result["status"] == "trimmed"
    assert result["kept_prearm_seconds"] <= 20.0
    assert Path(result["backup"]).exists()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["seq"] == "0"
    assert float(rows[0]["time_s"]) == 0.0
