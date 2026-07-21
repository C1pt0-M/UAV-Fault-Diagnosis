from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RFlyMAD_COLUMNS = [
    "Index",
    "Timestamp",
    "trueTime",
    "_actuator_controls_0_0_control[0]",
    "_actuator_controls_0_0_control[1]",
    "_actuator_controls_0_0_control[2]",
    "_actuator_controls_0_0_control[3]",
    "_actuator_outputs_0_output[0]",
    "_actuator_outputs_0_output[1]",
    "_actuator_outputs_0_output[2]",
    "_actuator_outputs_0_output[3]",
    "_rfly_ctrl_lxl_0_id",
    "_rfly_ctrl_lxl_0_mode",
    "_rfly_ctrl_lxl_0_controls[0]",
    "_rfly_ctrl_lxl_0_controls[1]",
    "_rfly_ctrl_lxl_0_controls[2]",
    "_rfly_ctrl_lxl_0_controls[3]",
    "_rfly_ctrl_lxl_0_controls[4]",
    "_rfly_ctrl_lxl_0_controls[5]",
    "_sensor_combined_0_gyro_rad[0]",
    "_sensor_combined_0_gyro_rad[1]",
    "_sensor_combined_0_gyro_rad[2]",
    "_sensor_combined_0_accelerometer_m_s2[0]",
    "_sensor_combined_0_accelerometer_m_s2[1]",
    "_sensor_combined_0_accelerometer_m_s2[2]",
    "_vehicle_air_data_0_baro_alt_meter",
    "_vehicle_air_data_0_baro_temp_celcius",
    "_vehicle_air_data_0_baro_pressure_pa",
    "_vehicle_attitude_0_q[0]",
    "_vehicle_attitude_0_q[1]",
    "_vehicle_attitude_0_q[2]",
    "_vehicle_attitude_0_q[3]",
    "_vehicle_local_position_0_x",
    "_vehicle_local_position_0_y",
    "_vehicle_local_position_0_z",
    "_vehicle_local_position_0_vx",
    "_vehicle_local_position_0_vy",
    "_vehicle_local_position_0_vz",
    "_vehicle_magnetometer_0_magnetometer_ga[0]",
    "_vehicle_magnetometer_0_magnetometer_ga[1]",
    "_vehicle_magnetometer_0_magnetometer_ga[2]",
    "TrueState_data_motorRPMs[1]",
    "TrueState_data_motorRPMs[2]",
    "TrueState_data_motorRPMs[3]",
    "TrueState_data_motorRPMs[4]",
    "UAVState_data_cmd",
    "UAVState_data_fault_state",
]


def _read_testinfo(case_dir: Path) -> dict[str, str]:
    path = case_dir / "TestInfo.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[str(row[0]).strip()] = str(row[1]).strip()
    return out


def _read_uav_state(case_dir: Path) -> pd.DataFrame | None:
    path = case_dir / "TrueData" / "UAVState_data.xlsx"
    if not path.exists():
        return None
    state = pd.read_excel(
        path,
        sheet_name="extimatedataset",
        usecols=lambda name: name in {"uavTime", "cmd", "fault_state"},
    )
    if "uavTime" not in state.columns or "fault_state" not in state.columns:
        return None
    state = state.rename(
        columns={
            "cmd": "UAVState_data_cmd",
            "fault_state": "UAVState_data_fault_state",
        }
    )
    # uavTime is PX4 boot time in milliseconds; ULog timestamp is microseconds.
    state["timestamp"] = pd.to_numeric(state["uavTime"], errors="coerce") * 1000.0
    state = state.dropna(subset=["timestamp"]).sort_values("timestamp")
    state["timestamp"] = state["timestamp"].round().astype("int64")
    state = state.drop_duplicates("timestamp", keep="last")
    return state.drop(columns=["uavTime"])


def _parse_fault_ids(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value] if value.strip() and value.strip() != "[]" else []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def _topic_file(log_dir: Path, topic: str) -> Path | None:
    matches = sorted(log_dir.glob(f"*_{topic}.csv"))
    return matches[0] if matches else None


def _read_topic(log_dir: Path, topic: str) -> pd.DataFrame | None:
    path = _topic_file(log_dir, topic)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"topic {topic} has no timestamp column: {path}")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df


def _rename_topic(df: pd.DataFrame | None, mapping: dict[str, str]) -> pd.DataFrame | None:
    if df is None:
        return None
    keep = ["timestamp", *[src for src in mapping if src in df.columns]]
    renamed = df[keep].rename(columns=mapping)
    return renamed


def _merge_nearest(
    base: pd.DataFrame,
    other: pd.DataFrame | None,
    *,
    tolerance_us: int | None,
) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    kwargs: dict[str, Any] = {
        "on": "timestamp",
        "direction": "nearest",
        "suffixes": ("", "_drop"),
    }
    if tolerance_us and tolerance_us > 0:
        kwargs["tolerance"] = int(tolerance_us)
    merged = pd.merge_asof(
        base.sort_values("timestamp"),
        other.sort_values("timestamp"),
        **kwargs,
    )
    drop_cols = [name for name in merged.columns if name.endswith("_drop")]
    return merged.drop(columns=drop_cols)


def _infer_controls_from_outputs(out: pd.DataFrame) -> pd.DataFrame:
    motors = [
        "_actuator_outputs_0_output[0]",
        "_actuator_outputs_0_output[1]",
        "_actuator_outputs_0_output[2]",
        "_actuator_outputs_0_output[3]",
    ]
    values = out[motors].astype(float)
    mean_pwm = values.mean(axis=1)
    # Roll/pitch/yaw actuator controls are not exported in this ULog subset.
    # Keep them neutral instead of inventing mixer-specific values. Throttle can
    # be approximated from mean PWM and is much less sensitive to motor ordering.
    out["_actuator_controls_0_0_control[0]"] = 0.0
    out["_actuator_controls_0_0_control[1]"] = 0.0
    out["_actuator_controls_0_0_control[2]"] = 0.0
    out["_actuator_controls_0_0_control[3]"] = ((mean_pwm - 900.0) / 1300.0).clip(0.0, 1.0)
    return out


def _downsample_to_hz(df: pd.DataFrame, target_hz: float | None) -> pd.DataFrame:
    if target_hz is None or target_hz <= 0 or len(df) < 3:
        return df.reset_index(drop=True)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(timestamps)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return df.reset_index(drop=True)
    source_hz = 1_000_000.0 / float(np.median(dt))
    if source_hz <= target_hz * 1.2:
        return df.reset_index(drop=True)

    target_dt_us = 1_000_000.0 / float(target_hz)
    start = float(timestamps[0])
    stop = float(timestamps[-1])
    grid = pd.DataFrame({"timestamp": np.arange(start, stop + 0.5 * target_dt_us, target_dt_us)})
    source = df.sort_values("timestamp").copy()
    source["timestamp"] = pd.to_numeric(source["timestamp"], errors="coerce").astype(float)
    sampled = pd.merge_asof(
        grid,
        source,
        on="timestamp",
        direction="nearest",
        tolerance=int(target_dt_us // 2),
    ).dropna(how="all")
    return sampled.reset_index(drop=True)


def merge_case(
    case_dir: Path,
    *,
    tolerance_us: int | None = 50000,
    target_hz: float | None = 50.0,
) -> pd.DataFrame:
    log_dir = case_dir / "Log"
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    sensor = _read_topic(log_dir, "sensor_combined_0")
    if sensor is None:
        raise FileNotFoundError(f"sensor_combined topic not found under: {log_dir}")

    base = _rename_topic(
        sensor,
        {
            "gyro_rad[0]": "_sensor_combined_0_gyro_rad[0]",
            "gyro_rad[1]": "_sensor_combined_0_gyro_rad[1]",
            "gyro_rad[2]": "_sensor_combined_0_gyro_rad[2]",
            "accelerometer_m_s2[0]": "_sensor_combined_0_accelerometer_m_s2[0]",
            "accelerometer_m_s2[1]": "_sensor_combined_0_accelerometer_m_s2[1]",
            "accelerometer_m_s2[2]": "_sensor_combined_0_accelerometer_m_s2[2]",
        },
    )
    assert base is not None

    topics = [
        (
            "actuator_outputs_0",
            {
                "output[0]": "_actuator_outputs_0_output[0]",
                "output[1]": "_actuator_outputs_0_output[1]",
                "output[2]": "_actuator_outputs_0_output[2]",
                "output[3]": "_actuator_outputs_0_output[3]",
            },
        ),
        (
            "estimator_attitude_0",
            {
                "q[0]": "_vehicle_attitude_0_q[0]",
                "q[1]": "_vehicle_attitude_0_q[1]",
                "q[2]": "_vehicle_attitude_0_q[2]",
                "q[3]": "_vehicle_attitude_0_q[3]",
            },
        ),
        (
            "vehicle_magnetometer_0",
            {
                "magnetometer_ga[0]": "_vehicle_magnetometer_0_magnetometer_ga[0]",
                "magnetometer_ga[1]": "_vehicle_magnetometer_0_magnetometer_ga[1]",
                "magnetometer_ga[2]": "_vehicle_magnetometer_0_magnetometer_ga[2]",
            },
        ),
        (
            "vehicle_gps_position_0",
            {
                "vel_n_m_s": "_vehicle_local_position_0_vx",
                "vel_e_m_s": "_vehicle_local_position_0_vy",
                "vel_d_m_s": "_vehicle_local_position_0_vz",
            },
        ),
        (
            "vehicle_air_data_0",
            {
                "baro_alt_meter": "_vehicle_air_data_0_baro_alt_meter",
                "baro_temp_celcius": "_vehicle_air_data_0_baro_temp_celcius",
                "baro_pressure_pa": "_vehicle_air_data_0_baro_pressure_pa",
            },
        ),
        (
            "vehicle_global_position_0",
            {
                "alt": "_vehicle_air_data_0_baro_alt_meter",
                "lat": "_vehicle_local_position_0_x",
                "lon": "_vehicle_local_position_0_y",
            },
        ),
    ]
    out = base
    for topic, mapping in topics:
        out = _merge_nearest(
            out,
            _rename_topic(_read_topic(log_dir, topic), mapping),
            tolerance_us=tolerance_us,
        )

    uav_state = _read_uav_state(case_dir)
    out = _merge_nearest(out, uav_state, tolerance_us=50000)
    out = _downsample_to_hz(out, target_hz)

    active_interval: tuple[int, int] | None = None
    if uav_state is not None:
        state_values = pd.to_numeric(
            uav_state["UAVState_data_fault_state"], errors="coerce"
        ).fillna(0)
        active_timestamps = uav_state.loc[state_values > 0, "timestamp"]
        if not active_timestamps.empty:
            state_dt = uav_state["timestamp"].diff()
            state_dt = state_dt[state_dt > 0]
            final_step = int(state_dt.median()) if not state_dt.empty else 4000
            active_interval = (
                int(active_timestamps.min()),
                int(active_timestamps.max()) + final_step,
            )

    out["Index"] = np.arange(len(out), dtype=np.int64)
    first_timestamp = float(out["timestamp"].iloc[0]) if len(out) else 0.0
    out["Timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0.0)
    out["trueTime"] = (out["Timestamp"] - first_timestamp) * 1e-6

    for column in RFlyMAD_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0

    for idx in range(4):
        output_col = f"_actuator_outputs_0_output[{idx}]"
        rpm_col = f"TrueState_data_motorRPMs[{idx + 1}]"
        out[rpm_col] = pd.to_numeric(out[output_col], errors="coerce").fillna(0.0) * 3.5

    out = _infer_controls_from_outputs(out)
    out["_vehicle_air_data_0_baro_temp_celcius"] = out["_vehicle_air_data_0_baro_temp_celcius"].replace(0.0, 25.0)
    out["_vehicle_air_data_0_baro_pressure_pa"] = out["_vehicle_air_data_0_baro_pressure_pa"].replace(0.0, 101325.0)
    out["_vehicle_local_position_0_z"] = -pd.to_numeric(
        out["_vehicle_air_data_0_baro_alt_meter"], errors="coerce"
    ).fillna(0.0)

    testinfo = _read_testinfo(case_dir)
    fault_ids = _parse_fault_ids(testinfo.get("Fault ID"))
    case_is_fault = int(bool(fault_ids))
    if "UAVState_data_fault_state" not in out.columns:
        out["UAVState_data_fault_state"] = case_is_fault
    elif active_interval is not None:
        timestamps = pd.to_numeric(out["timestamp"], errors="coerce").fillna(-1)
        out["UAVState_data_fault_state"] = (
            (timestamps >= active_interval[0]) & (timestamps < active_interval[1])
        ).astype(int)
    else:
        out["UAVState_data_fault_state"] = pd.to_numeric(
            out["UAVState_data_fault_state"], errors="coerce"
        ).fillna(0).astype(int)
    if "UAVState_data_cmd" not in out.columns:
        out["UAVState_data_cmd"] = testinfo.get("Control Sequence", "")
    else:
        out["UAVState_data_cmd"] = out["UAVState_data_cmd"].ffill().fillna("unknown")
    out["is_fault_active"] = out["UAVState_data_fault_state"].astype(int)
    out["flight_phase"] = np.where(out["is_fault_active"] > 0, "fault", "normal")

    ordered = out[[*RFlyMAD_COLUMNS, "is_fault_active", "flight_phase"]].copy()
    numeric_columns = [col for col in ordered.columns if col not in {"UAVState_data_cmd", "flight_phase"}]
    for col in numeric_columns:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce").fillna(0.0)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge BuaaAutoTestPlatform exported ULog topic CSVs into the RflyMAD-style 47-column schema."
    )
    parser.add_argument("--case-dir", required=True, help="Buaa Data/TestCase_* directory.")
    parser.add_argument("--out", required=True, help="Output merged CSV path.")
    parser.add_argument(
        "--tolerance-us",
        type=int,
        default=50000,
        help="Nearest-topic merge tolerance in microseconds; <=0 disables tolerance.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=50.0,
        help="Downsample merged output to this frequency; <=0 keeps source rate.",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    tolerance_us = int(args.tolerance_us)
    merged = merge_case(
        case_dir,
        tolerance_us=tolerance_us if tolerance_us > 0 else None,
        target_hz=float(args.target_hz) if float(args.target_hz) > 0 else None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] wrote merged Buaa case CSV: {out_path}")
    print(f"[INFO] rows={len(merged)} columns={len(merged.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
