from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from validate_rfly_normal import validate


PROFILES = (
    ("normal_takeoff_hover_land_5m_*.csv", "hover_5m", "takeoff_hover_land", 5.0, 0.0),
    ("normal_vertical_3_10m_*.csv", "vertical_3_10m", "vertical_3_10m", 10.0, 0.0),
    ("normal_box_10m_*.csv", "box_10m", "box_10m", 10.0, 5.0),
    ("normal_mixed_5_15m_*.csv", "mixed_5_15m", "mixed_5_15m", 15.0, 5.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_summary(path: Path) -> tuple[int, float]:
    frame = pd.read_csv(path)
    for column in ("time_s", "trueTime"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            duration = float(values.iloc[-1] - values.iloc[0]) if len(values) >= 2 else 0.0
            return len(frame), duration
    return len(frame), 0.0


def package(
    source_dir: Path,
    output_dir: Path,
    *,
    legacy_file: Path | None = None,
    expected_rfly_files: int = 40,
    replace: bool = False,
) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    staging = output_dir.with_name(f"{output_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    entries: list[dict] = []
    for pattern, package_profile, maneuver, altitude, speed in PROFILES:
        for source in sorted(source_dir.glob(pattern)):
            report = validate(
                source,
                require_horizontal_motion=speed > 0.0,
                target_altitude_m=altitude,
            )
            if report["status"] != "pass":
                failed = [name for name, passed in report["checks"].items() if not passed]
                raise ValueError(f"{source.name} failed quality checks: {failed}")
            session_id = source.stem.rsplit("_", 1)[-1]
            package_name = f"nf_rflysim_{package_profile}_{session_id}.csv"
            destination = staging / package_name
            shutil.copy2(source, destination)
            entries.append(
                {
                    "package_name": package_name,
                    "source_name": source.name,
                    "source_type": "rflysim_live_normal",
                    "maneuver": maneuver,
                    "target_altitude_m": altitude,
                    "target_speed_m_s": speed,
                    "rows": report["rows"],
                    "duration_s": report["duration_s"],
                    "sample_hz": report["sample_hz"],
                    "sha256": _sha256(destination),
                    "quality_status": report["status"],
                }
            )

    rfly_count = len(entries)
    if rfly_count != expected_rfly_files:
        raise ValueError(
            f"expected {expected_rfly_files} RflySim files, found {rfly_count}"
        )

    if legacy_file is not None:
        legacy_file = legacy_file.resolve()
        rows, duration = _legacy_summary(legacy_file)
        destination = staging / legacy_file.name
        shutil.copy2(legacy_file, destination)
        entries.append(
            {
                "package_name": destination.name,
                "source_name": legacy_file.name,
                "source_type": "buaa_legacy_normal",
                "maneuver": "buaa_case_189",
                "target_altitude_m": "",
                "target_speed_m_s": "",
                "rows": rows,
                "duration_s": round(duration, 3),
                "sample_hz": 50.0,
                "sha256": _sha256(destination),
                "quality_status": "legacy_verified_normal",
            }
        )

    manifest_path = staging / "extra_normals_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)

    summary = {
        "schema_version": "rfly_extra_normals_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "rflysim_files": rfly_count,
        "legacy_files": len(entries) - rfly_count,
        "total_files": len(entries),
        "total_rows": sum(int(entry["rows"]) for entry in entries),
        "total_duration_s": round(sum(float(entry["duration_s"]) for entry in entries), 3),
        "training_glob": "nf_*.csv",
    }
    (staging / "extra_normals_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    backup = None
    if output_dir.exists():
        if not replace:
            raise FileExistsError(f"output exists: {output_dir}; pass --replace")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = output_dir.with_name(f"{output_dir.name}_backup_{stamp}")
        shutil.move(str(output_dir), str(backup))
    shutil.move(str(staging), str(output_dir))
    summary["output_dir"] = str(output_dir)
    summary["backup_dir"] = str(backup) if backup else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Package verified RflySim extra normals")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-file", default="")
    parser.add_argument("--expected-rfly-files", type=int, default=40)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = package(
        Path(args.source_dir),
        Path(args.output_dir),
        legacy_file=Path(args.legacy_file) if args.legacy_file else None,
        expected_rfly_files=args.expected_rfly_files,
        replace=args.replace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
