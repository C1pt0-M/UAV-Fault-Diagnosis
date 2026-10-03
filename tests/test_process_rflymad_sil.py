from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "process_rflymad_sil.py"
SPEC = spec_from_file_location("process_rflymad_sil", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _create_case(root: Path, status: str, fault: str, case_name: str) -> Path:
    case = root / "SIL" / status / fault / case_name
    (case / "Log").mkdir(parents=True)
    (case / "TrueData").mkdir()
    (case / "TestInfo.csv").write_text("Fault ID,[123450.]\n", encoding="utf-8")
    (case / "Log" / "case.ulg").write_bytes(b"ulog")
    return case


def _create_all_buckets(root: Path, platform: str = "SIL") -> None:
    platform_root = MODULE.PLATFORM_ROOT_NAMES[platform.upper()]
    for status in MODULE.STATUS_CODES:
        for fault in MODULE.FAULT_CODES:
            (root / platform_root / status / fault).mkdir(parents=True)


def _create_real_case(
    root: Path, status: str, fault: str, case_name: str
) -> Path:
    case = root / "Real" / status / fault / case_name
    log_dir = case / "log_1_2023-1-1-00-00-00"
    ros_dir = case / "rfly_real_2023-01-01-00-00-00"
    log_dir.mkdir(parents=True)
    ros_dir.mkdir()
    (case / "TestInfo_2023-01-01_00-00.xlsx").write_bytes(b"xlsx")
    (log_dir / "case.ulg").write_bytes(b"ulog")
    for index in range(11):
        (log_dir / f"topic_{index}.csv").write_text("timestamp,value\n0,0\n")
        (ros_dir / f"ros_topic_{index}.csv").write_text(
            "rosbagTimestamp,value\n0,0\n"
        )
    (ros_dir / "_slash_mavros_slash_timesync_status.csv").write_text(
        "rosbagTimestamp,remote_timestamp_ns\n0,0\n"
    )
    (ros_dir / "case.bag").write_bytes(b"bag")
    return case


def test_build_plan_is_stable_and_uses_bucket_order_for_processed_ids(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _create_all_buckets(input_root)
    later = _create_case(
        input_root, "hover", "motor", "TestCase_10_1000000009"
    )
    earlier = _create_case(
        input_root, "hover", "motor", "TestCase_2_1000000001"
    )

    plan = MODULE.build_plan(input_root, output_root)
    motor_rows = [row for row in plan if row.flight_status == "hover" and row.fault_type == "motor"]

    assert [Path(row.source_path) for row in motor_rows] == [earlier.resolve(), later.resolve()]
    assert [row.output_filename for row in motor_rows] == [
        "Case_1000000000.csv",
        "Case_1000000001.csv",
    ]


def test_build_real_plan_uses_real_case_order_and_72_column_contract(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _create_all_buckets(input_root, platform="REAL")
    later = _create_real_case(input_root, "hover", "motor", "10_1")
    earlier = _create_real_case(input_root, "hover", "motor", "2_1")

    plan = MODULE.build_plan(input_root, output_root, platform="REAL")
    motor_rows = [
        row
        for row in plan
        if row.flight_status == "hover" and row.fault_type == "motor"
    ]

    assert [Path(row.source_path) for row in motor_rows] == [
        earlier.resolve(),
        later.resolve(),
    ]
    assert [row.output_filename for row in motor_rows] == [
        "Case_3000000000.csv",
        "Case_3000000001.csv",
    ]
    assert MODULE.EXPECTED_COLUMN_COUNTS["REAL"] == 72


def test_copy_real_processing_inputs_omits_large_bag(tmp_path: Path):
    source_root = tmp_path / "input"
    source = _create_real_case(source_root, "hover", "motor", "2_1")
    staged = tmp_path / "staged"

    MODULE._copy_processing_inputs(source, staged, "REAL")
    MODULE._validate_real_staging(staged)

    assert len(list(staged.glob("TestInfo*.xlsx"))) == 1
    assert len(list(staged.glob("log_*/*.ulg"))) == 1
    assert len(list(staged.glob("log_*/*.csv"))) == 11
    assert len(list(staged.glob("rfly_real_*/*.csv"))) == 12
    assert not list(staged.glob("rfly_real_*/*.bag"))


def test_build_plan_rejects_missing_required_case_content(tmp_path: Path):
    input_root = tmp_path / "input"
    _create_all_buckets(input_root)
    case = input_root / "SIL" / "hover" / "motor" / "TestCase_1_1000000000"
    case.mkdir()

    with pytest.raises(FileNotFoundError, match="missing TestInfo.csv"):
        MODULE.build_plan(input_root, tmp_path / "output")


def test_validate_output_requires_50hz_and_expected_schema(tmp_path: Path):
    columns = ["Index", "Timestamp"] + [f"feature_{i}" for i in range(45)]
    frame = pd.DataFrame(
        [[0, 1_000_000, *([0.0] * 45)], [1, 1_020_000, *([1.0] * 45)]],
        columns=columns,
    )
    output = tmp_path / "Case_1000000000.csv"
    frame.to_csv(output, index=False)

    metrics = MODULE.validate_output(output)

    assert metrics["row_count"] == 2
    assert metrics["column_count"] == 47
    assert metrics["median_step_us"] == 20_000.0
    assert len(metrics["output_sha256"]) == 64


def test_validate_real_output_accepts_72_columns(tmp_path: Path):
    columns = ["Index", "Timestamp"] + [f"feature_{i}" for i in range(70)]
    frame = pd.DataFrame(
        [[0, 1_000_000, *([0.0] * 70)], [1, 1_020_000, *([1.0] * 70)]],
        columns=columns,
    )
    output = tmp_path / "Case_3000000000.csv"
    frame.to_csv(output, index=False)

    metrics = MODULE.validate_output(output, expected_column_count=72)

    assert metrics["column_count"] == 72
    assert metrics["median_step_us"] == 20_000.0


def test_validate_output_rejects_non_50hz_data(tmp_path: Path):
    columns = ["Index", "Timestamp"] + [f"feature_{i}" for i in range(45)]
    frame = pd.DataFrame(
        [[0, 1_000_000, *([0.0] * 45)], [1, 1_050_000, *([1.0] * 45)]],
        columns=columns,
    )
    output = tmp_path / "Case_1000000000.csv"
    frame.to_csv(output, index=False)

    with pytest.raises(RuntimeError, match="expected 50 Hz"):
        MODULE.validate_output(output)
