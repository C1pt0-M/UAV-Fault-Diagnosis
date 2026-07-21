from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from merge_buaa_ulog_case import merge_case


def _parse_fault_ids(value: str) -> list[int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    values = parsed if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
    return [int(float(value)) for value in values]


def _parse_float_array(value: str) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", value)]
    values = parsed if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
    return [float(value) for value in values]


def _read_testinfo(case_dir: Path) -> dict[str, str]:
    path = case_dir / "TestInfo.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, header=None, names=["key", "value"], dtype=str)
    return dict(zip(frame["key"].fillna(""), frame["value"].fillna("")))


def _load_case_metadata(manifest_path: Path, case_id: int) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        if int(case["CaseID"]) == case_id:
            return case
    raise ValueError(f"CaseID {case_id} not found in {manifest_path}")


def validate(
    case_dir: Path,
    metadata: dict[str, object],
    merged_output: Path,
) -> dict[str, object]:
    true_state_path = case_dir / "TrueData" / "TrueState_data.xlsx"
    uav_state_path = case_dir / "TrueData" / "UAVState_data.xlsx"
    log_dir = case_dir / "Log"
    required_paths = [true_state_path, uav_state_path, log_dir, case_dir / "TestInfo.csv"]
    required_outputs = all(path.exists() for path in required_paths)

    true_state = pd.read_excel(true_state_path, sheet_name="truedataset")
    fault_state = pd.to_numeric(true_state["fault_state"], errors="coerce").fillna(0).astype(int)
    active = fault_state > 0
    inactive = ~active

    motor_index = int(metadata["motor_index"])
    target_column = f"motorRPMs[{motor_index}]"
    other_columns = [f"motorRPMs[{index}]" for index in range(1, 5) if index != motor_index]
    target_active = pd.to_numeric(true_state.loc[active, target_column], errors="coerce")
    others_active = true_state.loc[active, other_columns].apply(pd.to_numeric, errors="coerce")
    target_median = float(target_active.median()) if len(target_active) else 0.0
    others_median = float(others_active.stack().median()) if not others_active.empty else 0.0
    target_vs_others = target_median / others_median if others_median > 0 else 1.0

    testinfo = _read_testinfo(case_dir)
    try:
        sample_hz = float(testinfo.get("Data collected Frequency", "250"))
    except ValueError:
        sample_hz = 250.0
    active_duration_s = float(active.sum()) / sample_hz if sample_hz > 0 else 0.0

    fault_ids = _parse_fault_ids(testinfo.get("Fault ID", ""))
    fault_parameters = _parse_float_array(testinfo.get("Fault Parameter", ""))
    expected_fault_id = int(metadata["fault_id"])
    expected_coefficients = [1.0] * 4
    expected_coefficients[motor_index - 1] = float(metadata["efficiency_coefficient"])
    expected_parameters = [1.0, *expected_coefficients]
    parameters_match = len(fault_parameters) >= len(expected_parameters) and np.allclose(
        fault_parameters[: len(expected_parameters)], expected_parameters, atol=1e-6
    )

    merged = merge_case(case_dir, tolerance_us=50000, target_hz=50.0)
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_output, index=False)
    merged_active = pd.to_numeric(merged["is_fault_active"], errors="coerce").fillna(0) > 0
    merged_fault_duration_s = float(merged_active.sum()) / 50.0
    merged_label_transitions = int(
        (merged_active.astype(int).diff().fillna(0).abs() > 0).sum()
    )
    actuator_columns = [f"_actuator_outputs_0_output[{index}]" for index in range(4)]
    target_actuator = actuator_columns[motor_index - 1]
    other_actuators = [name for name in actuator_columns if name != target_actuator]
    active_target_output = float(
        pd.to_numeric(merged.loc[merged_active, target_actuator], errors="coerce").median()
    )
    active_other_output = float(
        merged.loc[merged_active, other_actuators]
        .apply(pd.to_numeric, errors="coerce")
        .stack()
        .median()
    )
    actuator_compensation_ratio = (
        active_target_output / active_other_output if active_other_output > 0 else 1.0
    )
    actuator_values = merged[actuator_columns].apply(pd.to_numeric, errors="coerce")
    actuator_range = float(actuator_values.max().max() - actuator_values.min().min())

    checks = {
        "required_outputs": required_outputs,
        "minimum_true_rows": len(true_state) >= 2500,
        "normal_phase_present": bool(inactive.any()),
        "fault_phase_present": bool(active.any()),
        "fault_duration_10_20s": 10.0 <= active_duration_s <= 20.0,
        "expected_fault_id": expected_fault_id in fault_ids,
        "expected_fault_parameters": bool(parameters_match),
        "actuator_signal_dynamic": actuator_range > 100.0,
        "merged_normal_phase_present": bool((~merged_active).any()),
        "merged_fault_phase_present": bool(merged_active.any()),
        "merged_fault_duration_12_20s": 12.0 <= merged_fault_duration_s <= 20.0,
        "merged_label_transitions_max_2": merged_label_transitions <= 2,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "case_id": int(metadata["CaseID"]),
        "case_dir": str(case_dir),
        "merged_output": str(merged_output),
        "rows_true": len(true_state),
        "rows_merged": len(merged),
        "motor_index": motor_index,
        "efficiency_coefficient": float(metadata["efficiency_coefficient"]),
        "active_duration_s": round(active_duration_s, 3),
        "merged_fault_duration_s": round(merged_fault_duration_s, 3),
        "merged_label_transitions": merged_label_transitions,
        "target_motor_active_median_rpm": round(target_median, 3),
        "other_motors_active_median_rpm": round(others_median, 3),
        "target_vs_others_ratio": round(target_vs_others, 4),
        "true_motor_rpm_available": bool(others_median > 0),
        "target_actuator_active_median": round(active_target_output, 3),
        "other_actuators_active_median": round(active_other_output, 3),
        "actuator_compensation_ratio": round(actuator_compensation_ratio, 4),
        "actuator_range": round(actuator_range, 3),
        "fault_ids": fault_ids,
        "fault_parameters": fault_parameters,
        "checks": checks,
    }
    report_path = merged_output.with_suffix(".quality.json")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and merge one Buaa motor fault case.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = _load_case_metadata(Path(args.manifest), args.case_id)
    result = validate(Path(args.case_dir), metadata, Path(args.out))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
