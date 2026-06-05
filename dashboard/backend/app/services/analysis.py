import math

import numpy as np
import pandas as pd


TIME_COLUMNS = ("time_s", "trueTime", "Timestamp", "timestamp", "time", "t")
FIELD_SOURCE_ALIASES = {
    "motor1": ("motor1", "actuator_out_1", "_actuator_outputs_0_output[0]"),
    "motor2": ("motor2", "actuator_out_2", "_actuator_outputs_0_output[1]"),
    "motor3": ("motor3", "actuator_out_3", "_actuator_outputs_0_output[2]"),
    "motor4": ("motor4", "actuator_out_4", "_actuator_outputs_0_output[3]"),
    "actuator_ctrl_1": ("actuator_ctrl_1", "_actuator_controls_0_0_control[0]"),
    "actuator_ctrl_2": ("actuator_ctrl_2", "_actuator_controls_0_0_control[1]"),
    "actuator_ctrl_3": ("actuator_ctrl_3", "_actuator_controls_0_0_control[2]"),
    "actuator_ctrl_4": ("actuator_ctrl_4", "_actuator_controls_0_0_control[3]"),
    "gyro_x": ("gyro_x", "_sensor_combined_0_gyro_rad[0]"),
    "gyro_y": ("gyro_y", "_sensor_combined_0_gyro_rad[1]"),
    "gyro_z": ("gyro_z", "_sensor_combined_0_gyro_rad[2]"),
    "accel_x": ("accel_x", "_sensor_combined_0_accelerometer_m_s2[0]"),
    "accel_y": ("accel_y", "_sensor_combined_0_accelerometer_m_s2[1]"),
    "accel_z": ("accel_z", "_sensor_combined_0_accelerometer_m_s2[2]"),
    "baro_alt": ("baro_alt", "baro_alt_meter", "_vehicle_air_data_0_baro_alt_meter"),
    "baro_temp": ("baro_temp", "baro_temp_celcius", "_vehicle_air_data_0_baro_temp_celcius"),
    "baro_pressure": ("baro_pressure", "baro_pressure_pa", "_vehicle_air_data_0_baro_pressure_pa"),
    "att_q_w": ("att_q_w", "q_w", "_vehicle_attitude_0_q[0]"),
    "att_q_x": ("att_q_x", "q_x", "_vehicle_attitude_0_q[1]"),
    "att_q_y": ("att_q_y", "q_y", "_vehicle_attitude_0_q[2]"),
    "att_q_z": ("att_q_z", "q_z", "_vehicle_attitude_0_q[3]"),
    "roll": ("roll",),
    "pitch": ("pitch",),
    "yaw": ("yaw",),
    "vx": ("vx", "vel_x", "_vehicle_local_position_0_vx", "_vehicle_local_position_setpoint_0_vx"),
    "vy": ("vy", "vel_y", "_vehicle_local_position_0_vy", "_vehicle_local_position_setpoint_0_vy"),
    "vz": ("vz", "vel_z", "_vehicle_local_position_0_vz", "_vehicle_local_position_setpoint_0_vz"),
    "pos_x": ("pos_x", "_vehicle_local_position_0_x"),
    "pos_y": ("pos_y", "_vehicle_local_position_0_y"),
    "pos_z": ("pos_z", "_vehicle_local_position_0_z"),
    "mag_x": ("mag_x", "_vehicle_magnetometer_0_magnetometer_ga[0]"),
    "mag_y": ("mag_y", "_vehicle_magnetometer_0_magnetometer_ga[1]"),
    "mag_z": ("mag_z", "_vehicle_magnetometer_0_magnetometer_ga[2]"),
    "rpm1": ("rpm1", "TrueState_data_motorRPMs[1]"),
    "rpm2": ("rpm2", "TrueState_data_motorRPMs[2]"),
    "rpm3": ("rpm3", "TrueState_data_motorRPMs[3]"),
    "rpm4": ("rpm4", "TrueState_data_motorRPMs[4]"),
    "fault_state": ("fault_state", "UAVState_data_fault_state"),
    "true_fault_state": ("true_fault_state", "TrueState_data_fault_state"),
}
DISPLAY_NAME_OVERRIDES = {
    "motor1": "Motor 1",
    "motor2": "Motor 2",
    "motor3": "Motor 3",
    "motor4": "Motor 4",
    "baro_alt": "Baro Altitude",
    "baro_temp": "Baro Temperature",
    "baro_pressure": "Baro Pressure",
    "pos_x": "Position X",
    "pos_y": "Position Y",
    "pos_z": "Position Z",
    "vx": "Velocity X",
    "vy": "Velocity Y",
    "vz": "Velocity Z",
    "rpm1": "Motor RPM 1",
    "rpm2": "Motor RPM 2",
    "rpm3": "Motor RPM 3",
    "rpm4": "Motor RPM 4",
    "fault_state": "Fault State",
    "true_fault_state": "Ground Truth Fault State",
}
REQUEST_ALIAS_MAP = {
    alias: field
    for field, aliases in FIELD_SOURCE_ALIASES.items()
    for alias in (field, *aliases)
}


def _normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "time_s" in df.columns:
        return df
    df = df.copy()
    if "trueTime" in df.columns:
        df["time_s"] = pd.to_numeric(df["trueTime"], errors="coerce")
    elif "Timestamp" in df.columns:
        df["time_s"] = pd.to_numeric(df["Timestamp"], errors="coerce") * 1e-6
    elif "timestamp" in df.columns:
        df["time_s"] = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "time" in df.columns:
        df["time_s"] = pd.to_numeric(df["time"], errors="coerce")
    elif "t" in df.columns:
        df["time_s"] = pd.to_numeric(df["t"], errors="coerce")
    return df


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _derive_attitude_columns(view_df: pd.DataFrame) -> None:
    quat_cols = ("att_q_w", "att_q_x", "att_q_y", "att_q_z")
    if not all(col in view_df.columns for col in quat_cols):
        return

    w = pd.to_numeric(view_df["att_q_w"], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(view_df["att_q_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(view_df["att_q_y"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(view_df["att_q_z"], errors="coerce").to_numpy(dtype=float)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    valid = np.isfinite(norm) & (norm > 1e-8)
    safe_norm = np.where(valid, norm, np.nan)
    w, x, y, z = (arr / safe_norm for arr in (w, x, y, z))

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    for field, values in (
        ("roll", roll),
        ("pitch", pitch),
        ("yaw", yaw),
    ):
        derived = pd.Series(values, index=view_df.index)
        if field in view_df.columns:
            view_df[field] = view_df[field].combine_first(derived)
        else:
            view_df[field] = derived


def _derive_summary_columns(view_df: pd.DataFrame) -> None:
    motor_cols = [col for col in ("motor1", "motor2", "motor3", "motor4") if col in view_df.columns]
    if motor_cols:
        motors = view_df[motor_cols].apply(pd.to_numeric, errors="coerce")
        view_df["actuator_mean"] = motors.mean(axis=1)
        view_df["actuator_std"] = motors.std(axis=1, ddof=0)

    accel_cols = [col for col in ("accel_x", "accel_y", "accel_z") if col in view_df.columns]
    if len(accel_cols) == 3:
        accel = view_df[accel_cols].apply(pd.to_numeric, errors="coerce")
        view_df["acc_norm"] = np.sqrt((accel * accel).sum(axis=1))
        if "actuator_mean" in view_df.columns:
            view_df["actuator_effort"] = view_df["actuator_mean"] / view_df["acc_norm"].clip(lower=0.1)


def _build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    source_df = _normalize_time_columns(df)
    view_df = pd.DataFrame(index=source_df.index)
    consumed: set[str] = set()

    if "time_s" in source_df.columns:
        view_df["time_s"] = pd.to_numeric(source_df["time_s"], errors="coerce")
        consumed.add("time_s")
    consumed.update(column for column in TIME_COLUMNS if column in source_df.columns)

    for field, aliases in FIELD_SOURCE_ALIASES.items():
        source_name = _first_existing_column(source_df, aliases)
        if source_name is None:
            continue
        view_df[field] = pd.to_numeric(source_df[source_name], errors="coerce")
        consumed.add(source_name)

    _derive_attitude_columns(view_df)
    _derive_summary_columns(view_df)

    for column in source_df.columns:
        if column in consumed or column in view_df.columns:
            continue
        series = pd.to_numeric(source_df[column], errors="coerce")
        if series.notna().any():
            view_df[column] = series

    return view_df


def _load_analysis_frame(csv_path: str, nrows: int | None = None) -> pd.DataFrame:
    return _build_analysis_frame(pd.read_csv(csv_path, nrows=nrows))


def _infer_raw_time_key(columns) -> str:
    for name in TIME_COLUMNS:
        if name in columns:
            return name
    return "__index__"


def _infer_time_key(df: pd.DataFrame) -> str:
    for name in ('time_s', 'timestamp', 'time', 't'):
        if name in df.columns:
            return name
    return '__index__'


def _field_group_name(field: str) -> str:
    key = field.lower()
    mapping = [
        ('motor', '动力系统'),
        ('actuator_out', '动力系统'),
        ('rpm', '动力系统'),
        ('esc_', '动力系统'),
        ('actuator_ctrl', '控制输入'),
        ('battery', '电池系统'),
        ('accel', '惯导传感'),
        ('gyro', '惯导传感'),
        ('mag', '磁力计'),
        ('roll', '姿态控制'),
        ('pitch', '姿态控制'),
        ('yaw', '姿态控制'),
        ('att_q', '姿态控制'),
        ('rates_sp', '姿态控制'),
        ('rate_err', '姿态控制'),
        ('thrust', '控制输出'),
        ('torque', '控制输出'),
        ('hover_', '控制输出'),
        ('baro', '气压高度'),
        ('pos_', '位置速度'),
        ('vx', '位置速度'),
        ('vy', '位置速度'),
        ('vz', '位置速度'),
        ('gps_', 'GPS'),
        ('fault', '故障标签'),
        ('flight_phase', '故障标签'),
    ]
    for prefix, group in mapping:
        if key.startswith(prefix) or prefix in key:
            return group
    return '其他'


def build_fields_response(csv_path: str) -> dict:
    df = _load_analysis_frame(csv_path, nrows=2000)
    time_key = _infer_time_key(df)
    numeric_cols = []
    source_df = pd.read_csv(csv_path, nrows=2000)
    non_numeric_cols = []
    for col in df.columns:
        if col == time_key:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)
    for col in source_df.columns:
        if col in TIME_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(source_df[col]):
            non_numeric_cols.append(col)

    groups: dict[str, list[dict]] = {}
    for col in numeric_cols:
        group = _field_group_name(col)
        groups.setdefault(group, []).append({
            'field': col,
            'display_name': DISPLAY_NAME_OVERRIDES.get(col, col),
            'unit': '',
            'display_unit': '',
            'available': True,
        })

    field_groups = [
        {
            'group': group_name,
            'fields': sorted(fields, key=lambda item: item['field']),
        }
        for group_name, fields in groups.items()
    ]
    field_groups.sort(key=lambda item: item['group'])

    time_info = {'key': time_key, 'available': time_key in df.columns}
    
    n_rows_total = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
    
    raw_columns = list(pd.read_csv(csv_path, nrows=0).columns)
    raw_time_key = _infer_raw_time_key(raw_columns)

    if raw_time_key != '__index__':
        try:
            full_time_col = pd.to_numeric(
                pd.read_csv(csv_path, usecols=[raw_time_key])[raw_time_key],
                errors='coerce',
            )
            if raw_time_key == 'Timestamp':
                full_time_col = full_time_col * 1e-6
            time_info['min'] = float(full_time_col.min())
            time_info['max'] = float(full_time_col.max())
        except Exception:
            time_info['min'] = float(df[time_key].min())
            time_info['max'] = float(df[time_key].max())
    elif time_key == '__index__':
        time_info['min'] = 0.0
        time_info['max'] = float(max(0, n_rows_total - 1))

    return {
        'field_groups': field_groups,
        'time': time_info,
        'sample_rows': max(n_rows_total, len(df)),
        'numeric_only': True,
        'non_numeric_fields': sorted(non_numeric_cols),
    }


def build_timeseries_response(
    csv_path: str,
    fields: list[str],
    start_t: float | None = None,
    end_t: float | None = None,
    max_points: int = 2000,
) -> dict:
    if not fields:
        raise ValueError('fields 不能为空')

    df = _load_analysis_frame(csv_path)
    time_key = _infer_time_key(df)
    raw_rows = len(df)

    if time_key == '__index__':
        df['__index__'] = range(len(df))

    if time_key in df.columns:
        df[time_key] = pd.to_numeric(df[time_key], errors='coerce')
        if start_t is not None:
            df = df[df[time_key] >= start_t]
        if end_t is not None:
            df = df[df[time_key] <= end_t]

    df = df.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return {
            'x': [],
            'series': [],
            'time_key': time_key,
            'downsample_step': 1,
            'n_rows_raw': raw_rows,
            'n_rows_out': 0,
        }

    max_points = max(50, int(max_points))
    step = int(math.ceil(n_rows / max_points)) if n_rows > max_points else 1
    sampled = df.iloc[::step].reset_index(drop=True)

    x_values = sampled[time_key].ffill().bfill().fillna(0).tolist()

    series = []
    for field in fields:
        resolved_field = REQUEST_ALIAS_MAP.get(field, field)
        if resolved_field not in sampled.columns and field in sampled.columns:
            resolved_field = field
        if resolved_field not in sampled.columns:
            continue
        values = pd.to_numeric(sampled[resolved_field], errors='coerce')
        series.append({
            'field': field,
            'values': [None if pd.isna(v) else float(v) for v in values.tolist()],
        })

    return {
        'x': [float(v) for v in x_values],
        'series': series,
        'time_key': time_key,
        'downsample_step': step,
        'n_rows_raw': raw_rows,
        'n_rows_out': len(sampled),
    }
