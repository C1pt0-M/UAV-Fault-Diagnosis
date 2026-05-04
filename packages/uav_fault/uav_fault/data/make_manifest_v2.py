import argparse
import json
import re
from pathlib import Path

import pandas as pd


CASE_RE = re.compile(r"^Case_(\d+)\.csv$")

PLATFORM_MAP = {1: "SIL", 2: "HIL", 3: "Real"}
FLIGHT_MAP = {
    0: "hover",
    1: "waypoint",
    2: "velocity",
    3: "circling",
    4: "acce",
    5: "dece",
}
FAULT_MAP = {
    0: "motor",
    1: "propeller",
    2: "low_voltage",
    3: "wind_affect",
    4: "load_lose",
    5: "accelerometer",
    6: "gyroscope",
    7: "magnetometer",
    8: "barometer",
    9: "GPS",
    10: "no_fault",
}

LABEL_SUBTYPE_ID = {v: k for k, v in FAULT_MAP.items()}
LABEL_MAIN_6 = {
    "no_fault": "normal",
    "motor": "motor",
    "propeller": "propeller",
    "accelerometer": "sensor",
    "gyroscope": "sensor",
    "magnetometer": "sensor",
    "barometer": "sensor",
    "GPS": "sensor",
    "wind_affect": "wind",
    "load_lose": "load",
    "low_voltage": None,
}
LABEL_MAIN_6_ID = {
    "normal": 0,
    "motor": 1,
    "propeller": 2,
    "sensor": 3,
    "wind": 4,
    "load": 5,
}
LABEL_4 = {
    "no_fault": "normal",
    "motor": "motor",
    "propeller": "propeller",
    "accelerometer": "sensor",
    "gyroscope": "sensor",
    "magnetometer": "sensor",
    "barometer": "sensor",
    "GPS": "sensor",
    "wind_affect": None,
    "load_lose": None,
    "low_voltage": None,
}
LABEL_4_ID = {"normal": 0, "motor": 1, "propeller": 2, "sensor": 3}
LABEL_BINARY = {"no_fault": "normal"}
LABEL_BINARY_ID = {"normal": 0, "fault": 1}

SEQ_REQUIRED_COLUMNS_BASE = {
    "_sensor_combined_0_gyro_rad[0]",
    "_sensor_combined_0_gyro_rad[1]",
    "_sensor_combined_0_gyro_rad[2]",
    "_sensor_combined_0_accelerometer_m_s2[0]",
    "_sensor_combined_0_accelerometer_m_s2[1]",
    "_sensor_combined_0_accelerometer_m_s2[2]",
    "_vehicle_air_data_0_baro_alt_meter",
    "_vehicle_air_data_0_baro_pressure_pa",
    "_vehicle_attitude_0_q[0]",
    "_vehicle_attitude_0_q[1]",
    "_vehicle_attitude_0_q[2]",
    "_vehicle_attitude_0_q[3]",
    "_vehicle_magnetometer_0_magnetometer_ga[0]",
    "_vehicle_magnetometer_0_magnetometer_ga[1]",
    "_vehicle_magnetometer_0_magnetometer_ga[2]",
}

LOCAL_VEL_PRIMARY = {
    "_vehicle_local_position_0_vx",
    "_vehicle_local_position_0_vy",
    "_vehicle_local_position_0_vz",
}

LOCAL_VEL_HIL_ALT = {
    "_vehicle_local_position_setpoint_0_vx",
    "_vehicle_local_position_setpoint_0_vy",
    "_vehicle_local_position_setpoint_0_vz",
}


def parse_case_filename(filename: str) -> dict | None:
    m = CASE_RE.match(filename)
    if not m:
        return None
    case_id = int(m.group(1))
    data_type = case_id // 1_000_000_000
    flight_id = (case_id % 1_000_000_000) // 100_000_000
    fault_id = (case_id % 100_000_000) // 1_000_000
    case_num = case_id % 1_000_000
    return {
        "case_id": case_id,
        "case_num": case_num,
        "platform": PLATFORM_MAP.get(data_type),
        "flight_mode": FLIGHT_MAP.get(flight_id),
        "fault_type_canonical": FAULT_MAP.get(fault_id),
    }


def _pick_time_col(df: pd.DataFrame, platform: str) -> str | None:
    if platform in {"SIL", "HIL"}:
        if "trueTime" in df.columns:
            return "trueTime"
        if "Timestamp" in df.columns:
            return "Timestamp"
    if platform == "Real":
        if "rosbagTimestamp" in df.columns:
            return "rosbagTimestamp"
        if "Timestamp" in df.columns:
            return "Timestamp"
    return None


def _time_scale(col: str) -> float:
    if col == "Timestamp":
        return 1e-6
    if col == "rosbagTimestamp":
        return 1e-9
    return 1.0


def _estimate_hz(df: pd.DataFrame, time_col: str) -> float | None:
    t = pd.to_numeric(df[time_col], errors="coerce") * _time_scale(time_col)
    dt = t.diff()
    dt = dt[(dt > 0) & dt.notna()]
    if dt.empty:
        return None
    return float(1.0 / dt.median())


def _find_fault_annotation(df: pd.DataFrame, time_col: str):
    candidates = [
        (
            "TrueState_data_fault_state",
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > 0,
        ),
        (
            "UAVState_data_fault_state",
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > 0,
        ),
        (
            "_rfly_ctrl_lxl_0_id",
            lambda s: (
                (pd.to_numeric(s, errors="coerce").fillna(0) >= 123450)
                & (pd.to_numeric(s, errors="coerce").fillna(0) <= 123549)
            ),
        ),
        (
            "rfly_ctrl_lxl_slash_send_data_id",
            lambda s: (
                (pd.to_numeric(s, errors="coerce").fillna(0) >= 123450)
                & (pd.to_numeric(s, errors="coerce").fillna(0) <= 123549)
            ),
        ),
    ]
    for col, func in candidates:
        if col not in df.columns:
            continue
        mask = (
            func(df[col]).to_numpy()
            if hasattr(func(df[col]), "to_numpy")
            else func(df[col])
        )
        active = [i for i, v in enumerate(mask) if bool(v)]
        if active:
            times = pd.to_numeric(df[time_col], errors="coerce") * _time_scale(time_col)
            return (
                col,
                1,
                active[0],
                active[-1],
                float(times.iloc[active[0]]),
                float(times.iloc[active[-1]]),
            )
    return "none", 0, None, None, None, None


def _has_required_seq_columns(cols: set[str]) -> int:
    base_ok = SEQ_REQUIRED_COLUMNS_BASE.issubset(cols)
    local_ok = LOCAL_VEL_PRIMARY.issubset(cols) or LOCAL_VEL_HIL_ALT.issubset(cols)
    return int(base_ok and local_ok)


def _quality_flag(
    time_col: str | None, n_rows: int, has_seq: bool, has_online: bool
) -> tuple[str, str]:
    if n_rows <= 0:
        return "bad", "empty_csv"
    if time_col is None:
        return "bad", "missing_time_col"
    if not has_seq and not has_online:
        return "bad", "missing_required_columns"
    if not has_seq or not has_online:
        return "warning", "partial_required_columns"
    return "ok", ""


def _load_testinfo_metadata(paths) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    keep_cols = [
        c
        for c in [
            "platform",
            "case_dir",
            "bucket_case_num",
            "case_index_in_dir",
            "testinfo_path",
            "flight_mode",
            "fault_type_canonical",
            "control_sequence_raw",
            "fault_id_raw",
            "fault_id_numeric",
            "fault_parameter_raw",
            "fault_params_json",
            "fault_injection_time_sec",
            "test_end_time_sec",
            "raw_collection_hz",
            "data_num",
        ]
        if c in merged.columns
    ]
    merged = merged[keep_cols].copy()

    def _natural_key(value: str):
        parts = []
        for t in re.split(r"(\d+)", str(value)):
            if t.isdigit():
                parts.append(f"{int(t):010d}")
            else:
                parts.append(t)
        return "".join(parts)

    if {"platform", "flight_mode", "fault_type_canonical", "case_dir"}.issubset(
        merged.columns
    ):
        if "bucket_case_num" in merged.columns:
            merged["bucket_case_num"] = pd.to_numeric(
                merged["bucket_case_num"], errors="coerce"
            )
        if "case_index_in_dir" in merged.columns:
            merged["case_index_in_dir"] = pd.to_numeric(
                merged["case_index_in_dir"], errors="coerce"
            )
        if (
            "bucket_case_num" in merged.columns
            and merged["bucket_case_num"].notna().any()
        ):
            merged = merged.sort_values(
                ["platform", "flight_mode", "fault_type_canonical", "bucket_case_num"]
            ).copy()
        else:
            merged["_case_dir_sort_key"] = merged["case_dir"].map(_natural_key)
            if (
                "case_index_in_dir" in merged.columns
                and merged["case_index_in_dir"].notna().any()
            ):
                merged = merged.sort_values(
                    [
                        "platform",
                        "flight_mode",
                        "fault_type_canonical",
                        "case_index_in_dir",
                        "_case_dir_sort_key",
                    ]
                ).copy()
            else:
                merged = merged.sort_values(
                    [
                        "platform",
                        "flight_mode",
                        "fault_type_canonical",
                        "_case_dir_sort_key",
                    ],
                ).copy()
            merged["bucket_case_num"] = merged.groupby(
                ["platform", "flight_mode", "fault_type_canonical"]
            ).cumcount()
        merged = merged.drop_duplicates(
            subset=[
                "platform",
                "flight_mode",
                "fault_type_canonical",
                "bucket_case_num",
            ],
            keep="first",
        )
        if "_case_dir_sort_key" in merged.columns:
            merged = merged.drop(columns=["_case_dir_sort_key"])
    return merged


def _normalize_csv_dirs(csv_dir):
    if isinstance(csv_dir, (str, Path)):
        return [Path(csv_dir)]
    return [Path(item) for item in csv_dir]


def build_manifest_v2(
    csv_dir,
    output_path,
    summary_output_path,
    schema_version,
    build_version,
    testinfo_metadata_paths=None,
):
    csv_dirs = _normalize_csv_dirs(csv_dir)
    output_path = Path(output_path)
    summary_output_path = Path(summary_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    testinfo_df = _load_testinfo_metadata(testinfo_metadata_paths)

    rows = []
    for one_csv_dir in csv_dirs:
        if not one_csv_dir.exists():
            continue
        for csv_path in sorted(one_csv_dir.glob("Case_*.csv")):
            parsed = parse_case_filename(csv_path.name)
            if parsed is None:
                continue
            df = pd.read_csv(csv_path)
            cols = set(df.columns)
            fault_type = parsed["fault_type_canonical"]
            label_main = LABEL_MAIN_6[fault_type]
            label_4 = LABEL_4[fault_type]
            label_binary = LABEL_BINARY.get(fault_type, "fault")
            time_col = _pick_time_col(df, parsed["platform"])
            time_start = time_end = duration_sec = sample_rate_hz_est = None
            if time_col is not None and not df.empty:
                times = pd.to_numeric(df[time_col], errors="coerce") * _time_scale(time_col)
                time_start = float(times.iloc[0])
                time_end = float(times.iloc[-1])
                duration_sec = float(time_end - time_start)
                sample_rate_hz_est = _estimate_hz(df, time_col)

            has_true_fault_state = int("TrueState_data_fault_state" in cols)
            has_uav_fault_state = int("UAVState_data_fault_state" in cols)
            has_rfly_fault_id = int(
                ("_rfly_ctrl_lxl_0_id" in cols)
                or ("rfly_ctrl_lxl_slash_send_data_id" in cols)
            )
            has_cmd = int("UAVState_data_cmd" in cols)
            has_motor_rpm = int(
                all(f"TrueState_data_motorRPMs[{i}]" in cols for i in range(1, 5))
            )
            has_required_seq = _has_required_seq_columns(cols)
            has_required_online = has_required_seq

            (
                fault_annotation_source,
                has_fault_annotation,
                fault_start_idx,
                fault_end_idx,
                fault_start_sec,
                fault_end_sec,
            ) = (
                _find_fault_annotation(df, time_col)
                if time_col is not None
                else ("none", 0, None, None, None, None)
            )
            quality_flag, exclude_reason = _quality_flag(
                time_col, len(df), bool(has_required_seq), bool(has_required_online)
            )

            rows.append(
                {
                    "sample_id": f"rflymad::50hz::{csv_path.stem}",
                    "filename": csv_path.name,
                    "source_csv_path": str(csv_path),
                    "case_id": parsed["case_id"],
                    "case_num": parsed["case_num"],
                    "bucket_case_num": parsed["case_num"],
                    "platform": parsed["platform"],
                    "flight_mode": parsed["flight_mode"],
                    "fault_type_canonical": fault_type,
                    "label_main_6class": label_main,
                    "label_main_6class_id": LABEL_MAIN_6_ID.get(label_main),
                    "is_in_main_6class": int(label_main is not None),
                    "label_4class": label_4,
                    "label_4class_id": LABEL_4_ID.get(label_4),
                    "is_in_4class_mainline": int(label_4 is not None),
                    "label_binary": label_binary,
                    "label_binary_id": LABEL_BINARY_ID[label_binary],
                    "n_rows": int(len(df)),
                    "time_col": time_col,
                    "time_start": time_start,
                    "time_end": time_end,
                    "duration_sec": duration_sec,
                    "sample_rate_hz_est": sample_rate_hz_est,
                    "has_true_fault_state": has_true_fault_state,
                    "has_uav_fault_state": has_uav_fault_state,
                    "has_rfly_fault_id": has_rfly_fault_id,
                    "has_cmd": has_cmd,
                    "has_motor_rpm": has_motor_rpm,
                    "has_required_seq_columns": has_required_seq,
                    "has_required_online_columns": has_required_online,
                    "fault_annotation_source": fault_annotation_source,
                    "has_fault_annotation": has_fault_annotation,
                    "fault_start_idx": fault_start_idx,
                    "fault_end_idx": fault_end_idx,
                    "fault_start_sec": fault_start_sec,
                    "fault_end_sec": fault_end_sec,
                    "quality_flag": quality_flag,
                    "exclude_reason": exclude_reason,
                    "schema_version": schema_version,
                    "build_version": build_version,
                }
            )

    if not rows:
        raise RuntimeError("没有可用的 RflyMAD processed CSV，无法生成 manifest_v2")

    manifest = pd.DataFrame(rows)
    merge_keys = ["platform", "flight_mode", "fault_type_canonical", "bucket_case_num"]
    if not testinfo_df.empty and set(merge_keys).issubset(testinfo_df.columns):
        manifest = manifest.merge(testinfo_df, how="left", on=merge_keys)
    manifest.to_csv(output_path, index=False)

    summary = {
        "total_cases": int(len(manifest)),
        "platform_counts": {
            k: int(v)
            for k, v in manifest["platform"].value_counts().sort_index().items()
        },
        "label_main_6class_counts": {
            k: int(v)
            for k, v in manifest["label_main_6class"]
            .dropna()
            .value_counts()
            .sort_index()
            .items()
        },
        "in_main_6class_count": int(manifest["is_in_main_6class"].sum()),
        "testinfo_merged_count": int(manifest["control_sequence_raw"].notna().sum())
        if "control_sequence_raw" in manifest.columns
        else 0,
    }
    summary_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="生成 RflyMAD v2 manifest")
    parser.add_argument("--csv-dir", action="append", default=[])
    parser.add_argument("--output", default="data/RflyMAD/manifest/manifest_v2.csv")
    parser.add_argument(
        "--summary-output", default="data/RflyMAD/manifest/manifest_v2.summary.json"
    )
    parser.add_argument("--schema-version", default="rflymad_manifest_v2")
    parser.add_argument("--build-version", default="dev")
    parser.add_argument("--testinfo-metadata", action="append", default=[])
    args = parser.parse_args()
    csv_dirs = args.csv_dir or ["data/RflyMAD/processed_official_50hz"]
    build_manifest_v2(
        csv_dirs,
        args.output,
        args.summary_output,
        args.schema_version,
        args.build_version,
        args.testinfo_metadata,
    )


if __name__ == "__main__":
    main()
