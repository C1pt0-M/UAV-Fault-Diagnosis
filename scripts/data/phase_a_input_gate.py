#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_import_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "packages" / "uav_fault"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root


REPO_ROOT = _add_import_paths()

SEQ_COLUMNS: list[str] = []
adapt_px4_live_frame = None
normalize_time_columns = None


def _load_runtime_dependencies() -> None:
    global SEQ_COLUMNS, adapt_px4_live_frame, normalize_time_columns
    if SEQ_COLUMNS and adapt_px4_live_frame is not None and normalize_time_columns is not None:
        return

    # 用运行时适配器当前生效的 schema（现在是与离线特征合同一致的 26 通道），
    # 否则覆盖率表会列出适配器已经不产出的列（如 actuator_mean）。
    from uav_fault.telemetry.schema_adapter import (
        SEQ_COLUMNS as active_columns,
        adapt_px4_live_frame as adapt_frame,
        normalize_time_columns as normalize_time,
    )

    SEQ_COLUMNS = list(active_columns)
    adapt_px4_live_frame = adapt_frame
    normalize_time_columns = normalize_time


HIGH_RISK_FIELDS = [
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "rates_sp_roll",
    "rates_sp_pitch",
    "rates_sp_yaw",
    "thrust_z",
]

KEY_STABILITY_FIELDS = [
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "baro_alt",
    "baro_pressure",
    "baro_temp",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "rates_sp_roll",
    "rates_sp_pitch",
    "rates_sp_yaw",
    "thrust_z",
]


def _infer_hz_from_time_s(df: pd.DataFrame) -> float | None:
    if "time_s" not in df.columns or len(df) < 3:
        return None
    try:
        t = np.asarray(df["time_s"], dtype=float)
    except Exception:
        return None
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    dt_med = float(np.median(dt))
    if dt_med <= 0:
        return None
    return float(1.0 / dt_med)


def _validated_time_array(df: pd.DataFrame) -> tuple[np.ndarray | None, bool]:
    if "time_s" not in df.columns or len(df) == 0:
        return None, False
    try:
        t = np.asarray(df["time_s"], dtype=float)
    except Exception:
        return None, False
    if t.shape[0] != len(df) or not np.all(np.isfinite(t)):
        return None, False
    dt = np.diff(t)
    if dt.size and not np.all(dt > 0):
        return None, False
    return t, True


def _safe_constant_rate(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 1.0 if finite.size == 1 else 0.0
    return float(np.mean(np.isclose(finite[1:], finite[:-1], rtol=0.0, atol=1e-12)))


def _coverage_rows(
    coverage_counts: dict[str, dict[str, int]], rows: int
) -> list[dict[str, object]]:
    out = []
    for feature in SEQ_COLUMNS:
        counts = coverage_counts[feature]
        mapped = int(counts["mapped"])
        derived = int(counts["derived"])
        default = int(counts["default"])
        denom = max(rows, 1)
        out.append(
            {
                "feature": feature,
                "mapped_count": mapped,
                "derived_count": derived,
                "default_count": default,
                "mapped_ratio": mapped / denom,
                "derived_ratio": derived / denom,
                "default_ratio": default / denom,
            }
        )
    return out


def _risk_rows(
    canonical_df: pd.DataFrame, coverage_counts: dict[str, dict[str, int]], rows: int
) -> list[dict[str, object]]:
    out = []
    denom = max(rows, 1)
    for feature in KEY_STABILITY_FIELDS:
        values = (
            np.asarray(canonical_df[feature], dtype=float)
            if feature in canonical_df.columns
            else np.array([], dtype=float)
        )
        finite = values[np.isfinite(values)]
        default_count = int(coverage_counts.get(feature, {}).get("default", 0))
        missing_rate = default_count / denom
        constant_rate = _safe_constant_rate(values) if values.size else 0.0
        if feature in HIGH_RISK_FIELDS and missing_rate >= 0.95:
            risk_note = "mostly_default"
        elif feature in HIGH_RISK_FIELDS and constant_rate >= 0.95:
            risk_note = "mostly_constant"
        elif feature in HIGH_RISK_FIELDS:
            risk_note = "observe"
        else:
            risk_note = "key_field"
        out.append(
            {
                "feature": feature,
                "missing_rate": missing_rate,
                "constant_rate": constant_rate,
                "min": float(np.min(finite)) if finite.size else np.nan,
                "max": float(np.max(finite)) if finite.size else np.nan,
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "std": float(np.std(finite)) if finite.size else np.nan,
                "risk_note": risk_note,
            }
        )
    return out


def _valid_window_ratio(
    default_counts: np.ndarray,
    times: np.ndarray,
    window_size: int,
    max_default_ratio: float,
    max_gap_s: float,
) -> float:
    n = len(default_counts)
    if n < window_size or window_size <= 0:
        return 0.0
    valid = 0
    total = 0
    for start in range(0, n - window_size + 1):
        end = start + window_size
        total += 1
        window_default_ratio = float(
            np.mean(default_counts[start:end] / len(SEQ_COLUMNS))
        )
        window_dt = np.diff(times[start:end])
        max_gap = float(np.max(window_dt)) if window_dt.size else 0.0
        if window_default_ratio <= max_default_ratio and max_gap <= max_gap_s:
            valid += 1
    return float(valid / total) if total > 0 else 0.0


def run_phase_a_input_gate(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    window_size: int = 150,
    max_default_ratio: float = 0.30,
    max_gap_s: float = 0.20,
) -> dict[str, object]:
    _load_runtime_dependencies()

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = normalize_time_columns(df)
    rows = df.to_dict(orient="records")

    canonical_rows: list[dict[str, object]] = []
    coverage_counts = {
        feature: {"mapped": 0, "derived": 0, "default": 0} for feature in SEQ_COLUMNS
    }
    default_counts_per_row: list[int] = []

    for row in rows:
        canonical, coverage = adapt_px4_live_frame(row)
        canonical_rows.append(canonical)
        row_default = 0
        for tag in ("mapped", "derived", "default"):
            for feature in coverage.get(tag, []):
                if feature in coverage_counts:
                    coverage_counts[feature][tag] += 1
                    if tag == "default":
                        row_default += 1
        default_counts_per_row.append(row_default)

    canonical_df = pd.DataFrame(canonical_rows)
    times, time_axis_valid = _validated_time_array(canonical_df)
    if time_axis_valid and times is not None:
        dt = np.diff(times)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        dt_median = float(np.median(dt)) if dt.size else None
        dt_p95 = float(np.percentile(dt, 95)) if dt.size else None
        estimated_hz = _infer_hz_from_time_s(canonical_df)
        time_range_s = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    else:
        dt_median = None
        dt_p95 = None
        estimated_hz = None
        time_range_s = None

    total_cells = max(len(canonical_rows) * len(SEQ_COLUMNS), 1)
    mapped_total = sum(v["mapped"] for v in coverage_counts.values())
    derived_total = sum(v["derived"] for v in coverage_counts.values())
    default_total = sum(v["default"] for v in coverage_counts.values())
    if time_axis_valid and times is not None:
        valid_window_ratio = _valid_window_ratio(
            np.asarray(default_counts_per_row, dtype=float),
            times,
            window_size,
            max_default_ratio,
            max_gap_s,
        )
    else:
        valid_window_ratio = 0.0

    summary = {
        "rows": int(len(canonical_rows)),
        "time_range_s": time_range_s,
        "dt_median": dt_median,
        "dt_p95": dt_p95,
        "estimated_hz": estimated_hz,
        "time_axis_valid": int(time_axis_valid),
        "valid_window_ratio": valid_window_ratio,
        "mapped_ratio": float(mapped_total / total_cells),
        "derived_ratio": float(derived_total / total_cells),
        "default_ratio": float(default_total / total_cells),
        "window_size": int(window_size),
        "max_default_ratio": float(max_default_ratio),
        "max_gap_s": float(max_gap_s),
    }

    coverage_rows = _coverage_rows(coverage_counts, len(canonical_rows))
    risk_rows = _risk_rows(canonical_df, coverage_counts, len(canonical_rows))

    with open(output_dir / "phase_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(
        output_dir / "phase_a_feature_coverage.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)

    with open(
        output_dir / "phase_a_risk_fields.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(risk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(risk_rows)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A input compatibility gate for PX4 live CSV."
    )
    parser.add_argument("--csv", required=True, help="Input PX4 live CSV path.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write Phase A outputs."
    )
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--max-default-ratio", type=float, default=0.30)
    parser.add_argument("--max-gap-s", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_phase_a_input_gate(
        csv_path=args.csv,
        output_dir=args.output_dir,
        window_size=args.window_size,
        max_default_ratio=args.max_default_ratio,
        max_gap_s=args.max_gap_s,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
