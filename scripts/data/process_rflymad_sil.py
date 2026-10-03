from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STATUS_CODES = {
    "hover": 0,
    "waypoint": 1,
    "velocity": 2,
    "circling": 3,
    "acce": 4,
    "dece": 5,
}
FAULT_CODES = {
    "motor": 0,
    "propeller": 1,
    "low_voltage": 2,
    "wind_affect": 3,
    "load_lose": 4,
    "accelerometer": 5,
    "gyroscope": 6,
    "magnetometer": 7,
    "barometer": 8,
    "GPS": 9,
    "no_fault": 10,
}
CASE_PREFIX = "TestCase_"
EXPECTED_COLUMN_COUNT = 47
EXPECTED_COLUMN_COUNTS = {"SIL": 47, "HIL": 47, "REAL": 72}
EXPECTED_STEP_US = 20_000.0
PLATFORM_CODES = {"SIL": 1, "HIL": 2, "REAL": 3}
PLATFORM_ROOT_NAMES = {"SIL": "SIL", "HIL": "HIL", "REAL": "Real"}


@dataclass(frozen=True)
class PlannedCase:
    plan_index: int
    flight_status: str
    flight_status_code: int
    fault_type: str
    processed_fault_code: int
    bucket_case_num: int
    raw_case_directory: str
    raw_case_id: str
    source_path: str
    output_filename: str
    output_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name.startswith(CASE_PREFIX):
        try:
            raw_id = path.name.rsplit("_", 1)[1]
            return int(raw_id), 0, path.name
        except (IndexError, ValueError):
            return sys.maxsize, sys.maxsize, path.name
    match = re.fullmatch(r"(\d+)_(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return sys.maxsize, sys.maxsize, path.name


def build_plan(
    input_root: Path, output_root: Path, platform: str = "SIL"
) -> list[PlannedCase]:
    platform = platform.upper()
    if platform not in PLATFORM_CODES:
        raise ValueError(f"unsupported platform: {platform}")
    platform_root = input_root / PLATFORM_ROOT_NAMES[platform]
    if not platform_root.is_dir():
        raise FileNotFoundError(f"{platform} root does not exist: {platform_root}")

    plan: list[PlannedCase] = []
    seen_sources: set[str] = set()
    seen_outputs: set[str] = set()
    for status, status_code in STATUS_CODES.items():
        for fault, fault_code in FAULT_CODES.items():
            bucket = platform_root / status / fault
            if not bucket.is_dir():
                raise FileNotFoundError(f"required bucket does not exist: {bucket}")
            case_dirs = sorted(
                (
                    path
                    for path in bucket.iterdir()
                    if path.is_dir()
                    and (platform == "REAL" or path.name.startswith(CASE_PREFIX))
                ),
                key=_case_sort_key,
            )
            for bucket_case_num, source in enumerate(case_dirs):
                if platform == "REAL":
                    match = re.fullmatch(r"(\d+)_(\d+)", source.name)
                    if not match:
                        raise ValueError(f"invalid Real case directory name: {source}")
                    raw_case_id = match.group(1)
                    testinfo_files = list(source.glob("TestInfo*.xlsx"))
                    log_dirs = list(source.glob("log_*"))
                    ros_dirs = list(source.glob("rfly_real_*"))
                    if len(testinfo_files) != 1:
                        raise FileNotFoundError(
                            f"expected one TestInfo XLSX, found {len(testinfo_files)}: {source}"
                        )
                    if len(log_dirs) != 1 or len(ros_dirs) != 1:
                        raise FileNotFoundError(
                            f"expected one log_* and rfly_real_* directory: {source}"
                        )
                    if len(list(log_dirs[0].glob("*.ulg"))) != 1:
                        raise FileNotFoundError(f"expected one ULog: {source}")
                    if len(list(ros_dirs[0].glob("*.bag"))) != 1:
                        raise FileNotFoundError(f"expected one ROS bag: {source}")
                    if len(list(log_dirs[0].glob("*.csv"))) <= 10:
                        raise FileNotFoundError(f"too few PX4 CSV files: {source}")
                    if len(list(ros_dirs[0].glob("*.csv"))) <= 10:
                        raise FileNotFoundError(f"too few ROS CSV files: {source}")
                else:
                    parts = source.name.rsplit("_", 1)
                    if len(parts) != 2 or len(parts[1]) != 10 or not parts[1].isdigit():
                        raise ValueError(f"invalid case directory name: {source}")
                    raw_case_id = parts[1]
                    for required in ("TestInfo.csv", "Log", "TrueData"):
                        if not (source / required).exists():
                            raise FileNotFoundError(f"missing {required}: {source}")

                processed_case_id = (
                    PLATFORM_CODES[platform] * 1_000_000_000
                    + status_code * 100_000_000
                    + fault_code * 1_000_000
                    + bucket_case_num
                )
                output_filename = f"Case_{processed_case_id}.csv"
                source_key = str(source.resolve()).casefold()
                output_path = output_root / output_filename
                output_key = str(output_path.resolve()).casefold()
                if source_key in seen_sources:
                    raise RuntimeError(f"duplicate source case: {source}")
                if output_key in seen_outputs:
                    raise RuntimeError(f"duplicate output target: {output_path}")
                seen_sources.add(source_key)
                seen_outputs.add(output_key)
                plan.append(
                    PlannedCase(
                        plan_index=len(plan),
                        flight_status=status,
                        flight_status_code=status_code,
                        fault_type=fault,
                        processed_fault_code=fault_code,
                        bucket_case_num=bucket_case_num,
                        raw_case_directory=source.name,
                        raw_case_id=raw_case_id,
                        source_path=str(source.resolve()),
                        output_filename=output_filename,
                        output_path=str(output_path.resolve()),
                    )
                )
    return plan


def write_plan(path: Path, plan: list[PlannedCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in plan)


def _load_reader(tool_root: Path, module_name: str, reader_name: str):
    script_path = tool_root / "Rflytool_main.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"processing script does not exist: {script_path}")

    dependency_names = ("fileprocess", "timetrans", "data_extractor", "get_parse")
    saved_modules = {name: sys.modules.get(name) for name in dependency_names}
    for name in dependency_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(tool_root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load processing script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, reader_name)
    finally:
        sys.path.pop(0)
        for name in dependency_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


def _testinfo_path(source: Path, platform: str) -> Path:
    if platform == "REAL":
        matches = list(source.glob("TestInfo*.xlsx"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one TestInfo XLSX, found {len(matches)}: {source}"
            )
        return matches[0]
    return source / "TestInfo.csv"


def _source_ulog_files(source: Path, platform: str) -> list[Path]:
    if platform == "REAL":
        log_dirs = list(source.glob("log_*"))
        if len(log_dirs) != 1:
            raise FileNotFoundError(f"expected one Real log directory: {source}")
        return sorted(log_dirs[0].glob("*.ulg"))
    return sorted((source / "Log").glob("*.ulg"))


def _copy_processing_inputs(source: Path, staged_case: Path, platform: str) -> None:
    staged_case.mkdir(parents=True)
    testinfo = _testinfo_path(source, platform)
    shutil.copy2(testinfo, staged_case / testinfo.name)
    if platform == "REAL":
        log_dirs = list(source.glob("log_*"))
        ros_dirs = list(source.glob("rfly_real_*"))
        if len(log_dirs) != 1 or len(ros_dirs) != 1:
            raise FileNotFoundError(
                f"expected one log_* and rfly_real_* directory: {source}"
            )
        shutil.copytree(log_dirs[0], staged_case / log_dirs[0].name)
        staged_ros = staged_case / ros_dirs[0].name
        staged_ros.mkdir()
        for csv_path in sorted(ros_dirs[0].glob("*.csv")):
            shutil.copy2(csv_path, staged_ros / csv_path.name)
        return
    shutil.copytree(source / "Log", staged_case / "Log")
    shutil.copytree(source / "TrueData", staged_case / "TrueData")


def _validate_real_staging(staged_case: Path) -> None:
    log_dirs = list(staged_case.glob("log_*"))
    ros_dirs = list(staged_case.glob("rfly_real_*"))
    if len(log_dirs) != 1 or len(ros_dirs) != 1:
        raise FileNotFoundError(f"invalid staged Real case: {staged_case}")
    if len(list(log_dirs[0].glob("*.ulg"))) != 1:
        raise FileNotFoundError(f"expected one staged Real ULog: {staged_case}")
    if len(list(log_dirs[0].glob("*.csv"))) <= 10:
        raise RuntimeError(f"too few staged PX4 CSV files: {staged_case}")
    if len(list(ros_dirs[0].glob("*.csv"))) <= 10:
        raise RuntimeError(f"too few staged ROS CSV files: {staged_case}")


def _run_ulog2csv(staged_case: Path) -> list[Path]:
    log_dir = staged_case / "Log"
    ulg_files = sorted(log_dir.glob("*.ulg"))
    if not ulg_files:
        raise FileNotFoundError(f"no ULog file found: {log_dir}")
    for ulg_path in ulg_files:
        completed = subprocess.run(
            ["ulog2csv", ulg_path.name],
            cwd=log_dir,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise RuntimeError(f"ulog2csv failed for {ulg_path.name}: {detail}")
    generated = sorted(log_dir.glob("*.csv"))
    if len(generated) <= 10:
        raise RuntimeError(f"ulog2csv generated too few CSV files ({len(generated)}): {log_dir}")
    return ulg_files


def validate_output(
    path: Path, expected_column_count: int = EXPECTED_COLUMN_COUNT
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"processed output is missing or empty: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] != expected_column_count:
        raise RuntimeError(
            f"expected {expected_column_count} columns, got {frame.shape[1]}: {path}"
        )
    if len(frame) < 2:
        raise RuntimeError(f"processed output has fewer than two rows: {path}")
    if frame.columns.duplicated().any():
        raise RuntimeError(f"processed output has duplicate columns: {path}")
    if "Timestamp" not in frame.columns:
        raise RuntimeError(f"processed output has no Timestamp column: {path}")
    timestamps = pd.to_numeric(frame["Timestamp"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"Timestamp is invalid or non-monotonic: {path}")
    median_step_us = float(timestamps.diff().dropna().median())
    if abs(median_step_us - EXPECTED_STEP_US) > 1.0:
        raise RuntimeError(
            f"expected 50 Hz ({EXPECTED_STEP_US:.0f} us), got {median_step_us:.3f} us: {path}"
        )
    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "median_step_us": median_step_us,
        "duration_sec": float((timestamps.iloc[-1] - timestamps.iloc[0]) / 1_000_000),
        "missing_ratio": float(frame.isna().mean().mean()),
        "output_size_bytes": int(path.stat().st_size),
        "output_sha256": _sha256(path),
    }


def _write_result(writer: csv.DictWriter, handle, result: dict[str, object]) -> None:
    writer.writerow(result)
    handle.flush()


def _call_official_reader(
    official_reader,
    tool_root: Path,
    staged_case: Path,
    output: Path,
    data_type: int,
) -> None:
    previous_cwd = Path.cwd()
    try:
        os.chdir(tool_root)
        official_reader(str(staged_case), data_type, 50, str(output))
    finally:
        os.chdir(previous_cwd)


def process_plan(
    plan: list[PlannedCase],
    output_root: Path,
    tool_root: Path,
    run_dir: Path,
    resume: bool,
    limit: int | None,
    keep_failed_staging: bool,
    platform: str,
    data_type: int,
    alignment_tool_root: Path | None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = run_dir / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    failed_log = run_dir / "failed_cases.log"
    result_path = run_dir / "case_results.csv"
    reader_name = "Real_data_reader" if platform == "REAL" else "SHIL_data_reader"
    expected_column_count = EXPECTED_COLUMN_COUNTS[platform]
    official_reader = _load_reader(
        tool_root, "rflymad_official_tool", reader_name
    )
    alignment_reader = (
        _load_reader(
            alignment_tool_root, "rflymad_alignment_tool", reader_name
        )
        if alignment_tool_root is not None
        else None
    )
    selected = plan[:limit] if limit is not None else plan
    counts = {"processed": 0, "processed_alignment": 0, "resumed": 0, "failed": 0}

    fieldnames = [
        "plan_index",
        "raw_case_directory",
        "source_path",
        "output_filename",
        "status",
        "started_at",
        "finished_at",
        "elapsed_sec",
        "source_testinfo_sha256",
        "source_ulog_sha256",
        "row_count",
        "column_count",
        "median_step_us",
        "duration_sec",
        "missing_ratio",
        "output_size_bytes",
        "output_sha256",
        "error",
    ]
    with result_path.open("x", newline="", encoding="utf-8-sig") as result_handle, failed_log.open(
        "x", encoding="utf-8"
    ) as failed_handle:
        writer = csv.DictWriter(result_handle, fieldnames=fieldnames)
        writer.writeheader()
        result_handle.flush()

        for item in selected:
            started_at = _utc_now()
            started = time.perf_counter()
            source = Path(item.source_path)
            output_path = Path(item.output_path)
            base_result: dict[str, object] = {
                "plan_index": item.plan_index,
                "raw_case_directory": item.raw_case_directory,
                "source_path": item.source_path,
                "output_filename": item.output_filename,
                "started_at": started_at,
                "source_testinfo_sha256": _sha256(
                    _testinfo_path(source, platform)
                ),
            }
            if output_path.exists():
                if not resume:
                    error = f"output already exists and --resume was not set: {output_path}"
                    failed_handle.write(f"{_utc_now()}\t{item.source_path}\t{error}\n")
                    failed_handle.flush()
                    _write_result(
                        writer,
                        result_handle,
                        {
                            **base_result,
                            "status": "failed",
                            "finished_at": _utc_now(),
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                            "error": error,
                        },
                    )
                    counts["failed"] += 1
                    continue
                try:
                    metrics = validate_output(
                        output_path, expected_column_count=expected_column_count
                    )
                    _write_result(
                        writer,
                        result_handle,
                        {
                            **base_result,
                            "status": "resumed",
                            "finished_at": _utc_now(),
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                            **metrics,
                            "error": "",
                        },
                    )
                    counts["resumed"] += 1
                except Exception as exc:
                    error = f"existing output validation failed: {exc}"
                    failed_handle.write(f"{_utc_now()}\t{item.source_path}\t{error}\n")
                    failed_handle.flush()
                    _write_result(
                        writer,
                        result_handle,
                        {
                            **base_result,
                            "status": "failed",
                            "finished_at": _utc_now(),
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                            "error": error,
                        },
                    )
                    counts["failed"] += 1
                continue

            case_staging = Path(tempfile.mkdtemp(prefix=f"case-{item.plan_index:04d}-", dir=staging_root))
            try:
                staged_case = case_staging / item.raw_case_directory
                _copy_processing_inputs(source, staged_case, platform)
                if platform == "REAL":
                    _validate_real_staging(staged_case)
                else:
                    _run_ulog2csv(staged_case)
                source_ulg_files = _source_ulog_files(source, platform)
                base_result["source_ulog_sha256"] = ";".join(
                    _sha256(ulg_path) for ulg_path in source_ulg_files
                )
                with tempfile.NamedTemporaryFile(
                    prefix=f".{item.output_filename}.",
                    suffix=".partial",
                    dir=output_root,
                    delete=False,
                ) as handle:
                    temp_output = Path(handle.name)
                processing_status = "processed"
                official_error = ""
                try:
                    try:
                        _call_official_reader(
                            official_reader,
                            tool_root,
                            staged_case,
                            temp_output,
                            data_type,
                        )
                        metrics = validate_output(
                            temp_output, expected_column_count=expected_column_count
                        )
                    except Exception as exc:
                        official_error = repr(exc)
                        if alignment_reader is None:
                            raise
                        temp_output.unlink(missing_ok=True)
                        _call_official_reader(
                            alignment_reader,
                            alignment_tool_root,
                            staged_case,
                            temp_output,
                            data_type,
                        )
                        metrics = validate_output(
                            temp_output, expected_column_count=expected_column_count
                        )
                        processing_status = "processed_alignment"
                    if output_path.exists():
                        raise FileExistsError(f"output appeared during processing: {output_path}")
                    temp_output.replace(output_path)
                finally:
                    temp_output.unlink(missing_ok=True)
                _write_result(
                    writer,
                    result_handle,
                    {
                        **base_result,
                        "status": processing_status,
                        "finished_at": _utc_now(),
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                        **metrics,
                        "error": official_error,
                    },
                )
                counts[processing_status] += 1
                shutil.rmtree(case_staging)
            except Exception as exc:
                error = repr(exc)
                failed_handle.write(
                    f"{_utc_now()}\t{item.source_path}\t{error}\n{traceback.format_exc()}\n"
                )
                failed_handle.flush()
                _write_result(
                    writer,
                    result_handle,
                    {
                        **base_result,
                        "status": "failed",
                        "finished_at": _utc_now(),
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                        "error": error,
                    },
                )
                counts["failed"] += 1
                if not keep_failed_staging:
                    shutil.rmtree(case_staging, ignore_errors=True)

    if staging_root.exists() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    return {
        "selected_cases": len(selected),
        **counts,
        "result_path": str(result_path.resolve()),
        "failed_log": str(failed_log.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically process organized RflyMAD SIL/HIL/Real cases at 50 Hz."
    )
    parser.add_argument("--platform", choices=sorted(PLATFORM_CODES), default="SIL")
    parser.add_argument("--input-root", type=Path, default=Path(r"D:/RflyMAD"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--tool-root", type=Path, default=Path(r"D:/RflyMAD/Data_processing_tools")
    )
    parser.add_argument(
        "--alignment-tool-root",
        type=Path,
        default=None,
        help="Optional local tool copy with row-count alignment fixes for failed cases.",
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path(r"D:/UAV-Data/processing_logs")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-failed-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")

    platform = args.platform.upper()
    output_root = args.output_root or Path(
        f"D:/UAV-Data/processed_{PLATFORM_ROOT_NAMES[platform]}_50hz"
    )
    run_id = datetime.now().strftime(
        f"{platform.lower()}-50hz-%Y%m%d-%H%M%S"
    )
    run_dir = args.log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(args.input_root, output_root, platform=platform)
    if not plan:
        raise RuntimeError(f"no {platform} cases found")
    plan_path = run_dir / "case_plan.csv"
    write_plan(plan_path, plan)

    summary: dict[str, object] = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "input_root": str(args.input_root.resolve()),
        "platform": platform,
        "output_root": str(output_root.resolve()),
        "tool_root": str(args.tool_root.resolve()),
        "alignment_tool_root": str(args.alignment_tool_root.resolve())
        if args.alignment_tool_root
        else "",
        "frequency_hz": 50,
        "planned_cases": len(plan),
        "plan_path": str(plan_path.resolve()),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        summary.update(
            process_plan(
                plan=plan,
                output_root=output_root,
                tool_root=args.alignment_tool_root or args.tool_root,
                run_dir=run_dir,
                resume=args.resume,
                limit=args.limit,
                keep_failed_staging=args.keep_failed_staging,
                platform=platform,
                data_type=PLATFORM_CODES[platform],
                alignment_tool_root=args.alignment_tool_root,
            )
        )
    summary["finished_at"] = _utc_now()
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if int(summary.get("failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
