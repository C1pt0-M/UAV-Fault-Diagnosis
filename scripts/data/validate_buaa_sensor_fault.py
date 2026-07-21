from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from merge_buaa_ulog_case import merge_case


def _parse_array(value: str) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", value)]
    values = parsed if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
    return [float(value) for value in values]


def _read_testinfo(case_dir: Path) -> dict[str, str]:
    frame = pd.read_csv(case_dir / "TestInfo.csv", header=None, names=["key", "value"], dtype=str)
    return dict(zip(frame["key"].fillna(""), frame["value"].fillna("")))


def _load_metadata(manifest_path: Path, case_id: int) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        if int(case["CaseID"]) == case_id:
            return case
    raise ValueError(f"CaseID {case_id} not found in {manifest_path}")


def validate(case_dir: Path, metadata: dict[str, object], output: Path) -> dict[str, object]:
    required = [
        case_dir / "TestInfo.csv",
        case_dir / "TrueData" / "TrueState_data.xlsx",
        case_dir / "TrueData" / "UAVState_data.xlsx",
        case_dir / "Log",
    ]
    required_outputs = all(path.exists() for path in required)
    true_state = pd.read_excel(required[1], sheet_name="truedataset")
    true_active = pd.to_numeric(true_state["fault_state"], errors="coerce").fillna(0) > 0
    testinfo = _read_testinfo(case_dir)
    sample_hz = float(testinfo.get("Data collected Frequency", "250"))
    true_fault_duration_s = float(true_active.sum()) / sample_hz

    fault_ids = [int(value) for value in _parse_array(testinfo.get("Fault ID", ""))]
    fault_parameters = _parse_array(testinfo.get("Fault Parameter", ""))
    expected_parameters = [float(value) for value in metadata["fault_parameters"]]
    parameters_match = len(fault_parameters) >= len(expected_parameters) and np.allclose(
        fault_parameters[: len(expected_parameters)], expected_parameters, atol=1e-6
    )

    merged = merge_case(case_dir, tolerance_us=50000, target_hz=50.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    active = pd.to_numeric(merged["is_fault_active"], errors="coerce").fillna(0) > 0
    timestamps = pd.to_numeric(merged["Timestamp"], errors="coerce")
    active_start = float(timestamps.loc[active].min())
    active_stop = float(timestamps.loc[active].max())
    pre = (timestamps >= active_start - 5_000_000) & (timestamps < active_start)
    active_core = active & (timestamps >= active_start + 1_000_000) & (
        timestamps <= active_stop - 1_000_000
    )
    signal_column = str(metadata["signal_column"])
    signal = pd.to_numeric(merged[signal_column], errors="coerce")
    pre_median = float(signal.loc[pre].median())
    active_median = float(signal.loc[active_core].median())
    signal_shift = active_median - pre_median
    expected_bias = float(metadata["bias"])
    minimum_shift = float(metadata["minimum_signal_shift"])

    duration_s = float(active.sum()) / 50.0
    transitions = int((active.astype(int).diff().fillna(0).abs() > 0).sum())
    numeric = merged.select_dtypes(include=[np.number])
    checks = {
        "required_outputs": required_outputs,
        "minimum_true_rows": len(true_state) >= 2500,
        "true_fault_duration_10_20s": 10 <= true_fault_duration_s <= 20,
        "expected_fault_id": int(metadata["fault_id"]) in fault_ids,
        "expected_fault_parameters": bool(parameters_match),
        "merged_normal_phase_present": bool((~active).any()),
        "merged_fault_phase_present": bool(active.any()),
        "merged_fault_duration_12_20s": 12 <= duration_s <= 20,
        "merged_label_transitions_max_2": transitions <= 2,
        "numeric_values_finite": bool(np.isfinite(numeric.to_numpy()).all()),
        "target_signal_shift_observed": abs(signal_shift) >= minimum_shift,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "case_id": int(metadata["CaseID"]),
        "case_dir": str(case_dir),
        "merged_output": str(output),
        "sensor": metadata["sensor"],
        "axis": metadata["axis"],
        "bias": expected_bias,
        "unit": metadata["unit"],
        "signal_column": signal_column,
        "rows_true": len(true_state),
        "rows_merged": len(merged),
        "true_fault_duration_s": round(true_fault_duration_s, 3),
        "merged_fault_duration_s": round(duration_s, 3),
        "merged_label_transitions": transitions,
        "pre_signal_median": round(pre_median, 6),
        "active_signal_median": round(active_median, 6),
        "observed_signal_shift": round(signal_shift, 6),
        "fault_ids": fault_ids,
        "fault_parameters": fault_parameters,
        "checks": checks,
    }
    output.with_suffix(".quality.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and merge one Buaa sensor fault case.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = validate(
        Path(args.case_dir),
        _load_metadata(Path(args.manifest), args.case_id),
        Path(args.out),
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
