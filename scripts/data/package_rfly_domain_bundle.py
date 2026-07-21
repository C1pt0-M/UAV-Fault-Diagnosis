from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_buaa_ulog_case import RFlyMAD_COLUMNS  # noqa: E402
from validate_buaa_motor_fault import validate as validate_motor  # noqa: E402
from validate_buaa_sensor_fault import validate as validate_sensor  # noqa: E402


MOTOR_IDS = range(4501, 4513)
SENSOR_IDS = range(4521, 4533)
FAULT_COLUMNS = [*RFlyMAD_COLUMNS, "is_fault_active", "flight_phase"]
TRAIN_SYNC_FILES = [
    "packages/uav_fault/uav_fault/__init__.py",
    "packages/uav_fault/uav_fault/data/build_online_cache.py",
    "packages/uav_fault/uav_fault/data/build_sequence_cache.py",
    "packages/uav_fault/uav_fault/models/train_online_detector.py",
    "scripts/train_pipeline.py",
    "scripts/eval/eval_rfly_domain_holdout.py",
    "scripts/eval/run_online_backbone_suite.py",
    "configs/data/rfly_fault_adaptation_v1.csv",
    "requirements-autodl-train.txt",
]
NORMAL_REQUIRED_NUMERIC = [
    "seq",
    "time_s",
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_alt_meter",
    "baro_temp",
    "baro_pressure_pa",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "mag_x",
    "mag_y",
    "mag_z",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(case["CaseID"]): case for case in payload["cases"]}


def _case_dir(data_root: Path, metadata: dict[str, Any]) -> Path:
    return data_root / f"TestCase_{metadata['CaseID']}_{metadata['UniqueIndex']}"


def _motor_filename(case: dict[str, Any]) -> str:
    coefficient = str(case["efficiency_coefficient"]).replace(".", "p")
    return f"motor_case_{case['CaseID']}_m{case['motor_index']}_eff{coefficient}.csv"


def _sensor_filename(case: dict[str, Any]) -> str:
    bias = str(case["bias"]).replace(".", "p")
    return (
        f"sensor_case_{case['CaseID']}_{case['sensor']}_{case['axis']}_bias{bias}.csv"
    )


def _validate_fault_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    numeric = frame.drop(columns=["UAVState_data_cmd", "flight_phase"]).apply(
        pd.to_numeric, errors="coerce"
    )
    timestamps = pd.to_numeric(frame["Timestamp"], errors="coerce")
    active = pd.to_numeric(frame["is_fault_active"], errors="coerce").fillna(0) > 0
    duration_s = (float(timestamps.iloc[-1]) - float(timestamps.iloc[0])) / 1_000_000
    sample_hz = (len(frame) - 1) / duration_s if duration_s > 0 else 0.0
    transitions = int((active.astype(int).diff().fillna(0).abs() > 0).sum())
    checks = {
        "columns_49_exact": list(frame.columns) == FAULT_COLUMNS,
        "sample_rate_48_52_hz": 48.0 <= sample_hz <= 52.0,
        "normal_phase_present": bool((~active).any()),
        "fault_phase_present": bool(active.any()),
        "fault_duration_12_20s": 12.0 <= float(active.sum()) / 50.0 <= 20.0,
        "label_transitions_max_2": transitions <= 2,
        "numeric_nan_inf_zero": bool(np.isfinite(numeric.to_numpy()).all()),
        "timestamp_unique": not timestamps.duplicated().any(),
        "timestamp_monotonic": bool(timestamps.is_monotonic_increasing),
    }
    return {
        "rows": len(frame),
        "duration_s": round(duration_s, 3),
        "sample_hz": round(sample_hz, 3),
        "fault_duration_s": round(float(active.sum()) / 50.0, 3),
        "quality_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
    }


def _normal_metadata(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    required = frame[NORMAL_REQUIRED_NUMERIC].apply(pd.to_numeric, errors="coerce")
    duration_s = float(time_s.iloc[-1] - time_s.iloc[0])
    sample_hz = (len(frame) - 1) / duration_s if duration_s > 0 else 0.0
    checks = {
        "columns_75": len(frame.columns) == 75,
        "minimum_rows": len(frame) >= 1000,
        "sample_rate_48_52_hz": 48.0 <= sample_hz <= 52.0,
        "model_input_numeric_finite": bool(np.isfinite(required.to_numpy()).all()),
        "time_monotonic": bool(time_s.is_monotonic_increasing),
        "sequence_unique": not required["seq"].duplicated().any(),
    }
    return {
        "rows": len(frame),
        "duration_s": round(duration_s, 3),
        "sample_hz": round(sample_hz, 3),
        "quality_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
    }


def _manifest_row(
    path: Path,
    relative_path: Path,
    source_type: str,
    class_3: str,
    role: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    strength = metadata.get("efficiency_coefficient", metadata.get("bias", ""))
    return {
        "file": relative_path.as_posix(),
        "source_type": source_type,
        "schema_name": "rfly_live_canonical_v1" if class_3 == "normal" else "rflymad_49_v1",
        "schema_columns": 75 if class_3 == "normal" else 49,
        "class_3": class_3,
        "class_binary": "normal" if class_3 == "normal" else "fault",
        "role": role,
        "case_id": metadata.get("CaseID", ""),
        "unique_index": metadata.get("UniqueIndex", ""),
        "fault_id": metadata.get("fault_id", ""),
        "fault_name": metadata.get("fault_name", ""),
        "component": metadata.get("sensor", f"motor_{metadata.get('motor_index', '')}" if class_3 == "motor" else ""),
        "axis": metadata.get("axis", ""),
        "strength": strength,
        "unit": metadata.get("unit", "coefficient" if class_3 == "motor" else ""),
        "rows": metrics["rows"],
        "duration_s": metrics["duration_s"],
        "sample_hz": metrics["sample_hz"],
        "fault_duration_s": metrics.get("fault_duration_s", ""),
        "sha256": _sha256(path),
        "quality_status": metrics["quality_status"],
    }


def _write_manifest(rows: list[dict[str, Any]], manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifest_dir / "rfly_domain_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, dict[str, float]] = {}
    for class_name in ("normal", "motor", "sensor"):
        selected = [row for row in rows if row["class_3"] == class_name]
        by_class[class_name] = {
            "files": len(selected),
            "rows": sum(int(row["rows"]) for row in selected),
            "duration_s": round(sum(float(row["duration_s"]) for row in selected), 3),
            "fault_duration_s": round(
                sum(float(row["fault_duration_s"] or 0) for row in selected), 3
            ),
        }
    summary = {
        "schema_version": 1,
        "package_name": "rfly_domain_bundle_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_policy": {
            "offline": ["normal", "motor", "sensor"],
            "online": ["normal", "fault"],
            "online_fault_definition": "motor + sensor",
        },
        "roles": {
            "normal": "training_normal",
            "motor": "held_out_rfly_validation",
            "sensor": "held_out_rfly_validation",
        },
        "schemas": {
            "normal": {"name": "rfly_live_canonical_v1", "columns": 75},
            "motor_sensor": {"name": "rflymad_49_v1", "columns": 49},
        },
        "totals": {
            "files": len(rows),
            "rows": sum(int(row["rows"]) for row in rows),
            "quality_pass": sum(row["quality_status"] == "pass" for row in rows),
        },
        "by_class": by_class,
    }
    (manifest_dir / "rfly_domain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_readme(bundle_root: Path) -> None:
    text = """# RflySim domain data bundle v1

This package contains 40 normal live-link sessions and 24 controlled BuaaAutoTestPlatform fault cases.

## Label policy

- Offline: `normal / motor / sensor`
- Online: `normal / fault`, where `fault = motor + sensor`
- `normal/` role: `training_normal`
- `motor/` and `sensor/` role: `held_out_rfly_validation`

Do not add the held-out fault cases to training before measuring the current model on them. If domain adaptation is required, create a new explicit split and retain untouched cases for final testing.

## Schemas

- Normal CSV: `rfly_live_canonical_v1`, 75 columns, generated by the live RflySim receiver.
- Motor/sensor CSV: `rflymad_49_v1`, 49 columns, regenerated at 50 Hz from raw Buaa ULog topics and TrueData labels.

See `manifest/rfly_domain_manifest.csv` for per-file labels, quality status, statistics, and SHA256 values.
"""
    (bundle_root / "README.md").write_text(text, encoding="utf-8")


def _zip_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def _sync_train_min(train_root: Path) -> None:
    for relative in TRAIN_SYNC_FILES:
        source = REPO_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"Required training file is missing: {source}")
        destination = train_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for cache in train_root.rglob("__pycache__"):
        shutil.rmtree(cache)


def package(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    staging = output_root / "rfly_domain_bundle_v1"
    if staging.exists():
        shutil.rmtree(staging)
    for name in ("normal", "motor", "sensor", "manifest"):
        (staging / name).mkdir(parents=True, exist_ok=True)

    motor_cases = _load_cases(Path(args.motor_manifest))
    sensor_cases = _load_cases(Path(args.sensor_manifest))
    rows: list[dict[str, Any]] = []

    normal_files = sorted(Path(args.normal_root).glob("nf_*.csv"))
    if len(normal_files) != 40:
        raise RuntimeError(f"Expected 40 normal CSVs, found {len(normal_files)}")
    for source in normal_files:
        destination = staging / "normal" / source.name
        shutil.copy2(source, destination)
        metrics = _normal_metadata(destination)
        rows.append(
            _manifest_row(
                destination,
                destination.relative_to(staging),
                "rflysim_live",
                "normal",
                "training_normal",
                metrics,
            )
        )

    raw_root = Path(args.buaa_data_root)
    for case_id in MOTOR_IDS:
        metadata = motor_cases[case_id]
        destination = staging / "motor" / _motor_filename(metadata)
        result = validate_motor(_case_dir(raw_root, metadata), metadata, destination)
        if result["status"] != "pass":
            raise RuntimeError(f"Motor case {case_id} failed source validation")
        metrics = _validate_fault_csv(destination)
        rows.append(
            _manifest_row(
                destination,
                destination.relative_to(staging),
                "buaa_rflysim",
                "motor",
                "held_out_rfly_validation",
                metrics,
                metadata,
            )
        )

    for case_id in SENSOR_IDS:
        metadata = sensor_cases[case_id]
        destination = staging / "sensor" / _sensor_filename(metadata)
        result = validate_sensor(_case_dir(raw_root, metadata), metadata, destination)
        if result["status"] != "pass":
            raise RuntimeError(f"Sensor case {case_id} failed source validation")
        metrics = _validate_fault_csv(destination)
        rows.append(
            _manifest_row(
                destination,
                destination.relative_to(staging),
                "buaa_rflysim",
                "sensor",
                "held_out_rfly_validation",
                metrics,
                metadata,
            )
        )

    failures = [row["file"] for row in rows if row["quality_status"] != "pass"]
    if failures:
        raise RuntimeError(f"Unified quality validation failed: {failures}")
    _write_manifest(rows, staging / "manifest")
    _write_readme(staging)

    data_zip = output_root / "rfly_domain_bundle_v1.zip"
    _zip_tree(staging, data_zip)

    train_root = Path(args.train_root)
    _sync_train_min(train_root)
    train_zip = output_root / "uav_fault_train_min_rfly_v3.zip"
    _zip_tree(train_root, train_zip)

    result = {
        "status": "pass",
        "bundle_directory": str(staging),
        "data_zip": str(data_zip),
        "data_zip_bytes": data_zip.stat().st_size,
        "data_zip_sha256": _sha256(data_zip),
        "train_zip": str(train_zip),
        "train_zip_bytes": train_zip.stat().st_size,
        "train_zip_sha256": _sha256(train_zip),
        "files": len(rows),
        "normal_files": sum(row["class_3"] == "normal" for row in rows),
        "motor_files": sum(row["class_3"] == "motor" for row in rows),
        "sensor_files": sum(row["class_3"] == "sensor" for row in rows),
        "rows": sum(int(row["rows"]) for row in rows),
        "quality_pass": sum(row["quality_status"] == "pass" for row in rows),
    }
    (output_root / "rfly_domain_bundle_v1.package.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate, validate, manifest, and package the RflySim domain dataset."
    )
    parser.add_argument("--normal-root", default=r"D:\UAV-Data\extra_normals")
    parser.add_argument("--buaa-data-root", default=r"D:\BuaaAutoTestPlatForm\Data")
    parser.add_argument(
        "--motor-manifest",
        default=r"D:\UAV-Data\rfly_motor_fault_validation\motor_fault_case_manifest.json",
    )
    parser.add_argument(
        "--sensor-manifest",
        default=r"D:\UAV-Data\rfly_sensor_fault_validation\sensor_fault_case_manifest.json",
    )
    parser.add_argument("--train-root", default=r"D:\uav_fault_train_min")
    parser.add_argument("--output-root", default=r"D:\UAV-Data")
    args = parser.parse_args()
    print(json.dumps(package(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
