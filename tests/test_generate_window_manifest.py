from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "data" / "generate_window_manifest.py"
)
SPEC = spec_from_file_location("generate_window_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_case(path: Path, fault_states: list[int]) -> None:
    pd.DataFrame(
        {
            "Timestamp": [index * 20_000 for index in range(len(fault_states))],
            "UAVState_data_fault_state": fault_states,
        }
    ).to_csv(path, index=False)


def _contract() -> dict:
    return {"window_size_frames": 4, "stride_frames": 2}


def test_open_set_fault_windows_are_unknown_and_keep_fault_type(tmp_path: Path) -> None:
    csv_path = tmp_path / "case.csv"
    _write_case(csv_path, [0, 0, 0, 0, 1, 1, 1, 1])
    row = {
        "final_output_path": str(csv_path),
        "case_uid": "SIL/hover/propeller/case",
        "domain": "SIL",
        "split": "open_set_test",
        "level1_label": "fault",
        "level2_label": "",
        "level3_label": "",
        "open_set_fault_type": "propeller",
        "flight_status": "hover",
        "normal_window_eligible": "True",
        "closed_set_fault_window_eligible": "False",
        "open_set_fault_window_eligible": "True",
    }

    windows = MODULE.generate_windows_for_case(
        row, _contract(), window_size=4, stride=2
    )

    assert [item["window_usage"] for item in windows] == [
        "normal",
        "open_set_fault",
        "open_set_fault",
    ]
    assert [item["level1_label"] for item in windows] == [
        "normal",
        "fault",
        "fault",
    ]
    assert [item["level2_label"] for item in windows] == [
        "normal",
        "unknown",
        "unknown",
    ]
    assert all(item["open_set_fault_type"] == "propeller" for item in windows)


def test_normal_only_case_never_produces_fault_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "case.csv"
    _write_case(csv_path, [0, 0, 0, 0, 1, 1, 1, 1])
    row = {
        "final_output_path": str(csv_path),
        "case_uid": "REAL/hover/motor/normal-only",
        "domain": "REAL",
        "split": "external_test",
        "level1_label": "normal",
        "level2_label": "normal",
        "level3_label": "normal",
        "open_set_fault_type": "",
        "flight_status": "hover",
        "normal_window_eligible": "True",
        "closed_set_fault_window_eligible": "False",
        "open_set_fault_window_eligible": "False",
    }

    windows = MODULE.generate_windows_for_case(
        row, _contract(), window_size=4, stride=2
    )

    assert {item["window_usage"] for item in windows} == {"normal"}
    assert {item["level1_label"] for item in windows} == {"normal"}
    assert {item["level2_label"] for item in windows} == {"normal"}
    assert {item["level3_label"] for item in windows} == {"normal"}
