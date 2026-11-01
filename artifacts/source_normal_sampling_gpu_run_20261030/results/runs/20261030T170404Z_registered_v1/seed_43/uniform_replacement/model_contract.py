"""Model input contracts shared by discovery and inference (no ML imports)."""

from __future__ import annotations

import math
import struct
from typing import Any

RFLY_SEQ_COLUMNS_V1 = [
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
    "baro_alt",
    "baro_temp",
    "baro_pressure",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "actuator_mean",
    "actuator_std",
    "acc_norm",
    "actuator_effort",
]

# Aligned with the offline feature contract v1. Same names and order as
# D:\UAV-Data\feature_contract_v1.json, so offline and online models can share a
# feature set: baro_temp / baro_pressure / actuator_mean / actuator_std /
# actuator_effort are dropped and baro_alt becomes window-relative.
RFLY_SEQ_COLUMNS_V2 = [
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
    "baro_alt_relative",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "acc_norm",
]

RFLY_SEQ_SCHEMAS: dict[str, list[str]] = {
    "rfly_seq_columns_v1": RFLY_SEQ_COLUMNS_V1,
    "rfly_seq_columns_v2": RFLY_SEQ_COLUMNS_V2,
}


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _finite_float32(value: Any, *, positive: bool = False) -> bool:
    # JSON booleans and numeric strings are not normalization coefficients.
    if type(value) not in (int, float):
        return False
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
        return math.isfinite(rounded) and (not positive or rounded > 0)
    except (OverflowError, struct.error):
        return False


def _feature_metadata_errors(data: dict[str, Any], source: str, columns: list[str]) -> list[str]:
    errors: list[str] = []
    if "feature_count" in data and (
        not _positive_int(data["feature_count"]) or data["feature_count"] != len(columns)
    ):
        errors.append(f"{source}.feature_count must match columns length")
    for field in ("feature_schema", "protocol_name"):
        if field not in data:
            continue
        value = data[field]
        if not isinstance(value, str) or not value.strip() or value.strip() != value:
            errors.append(f"{source}.{field} must be a nonblank string")
        elif field == "feature_schema" and value in RFLY_SEQ_SCHEMAS:
            if columns != RFLY_SEQ_SCHEMAS[value]:
                errors.append(f"{source}.feature_schema does not match columns order")
    return errors


def offline_label_names(config: dict[str, Any], num_classes: int) -> list[str]:
    """Resolve labels once for discovery and inference, preserving legacy defaults."""
    if not _positive_int(num_classes) or num_classes < 3:
        raise ValueError("num_classes must be an integer of at least 3 for offline diagnosis")
    if "label_names" in config:
        labels = config["label_names"]
        if not isinstance(labels, list) or not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError("label_names must be a list of nonblank strings")
        labels = [label.strip() for label in labels]
        if len(labels) < num_classes:
            raise ValueError("label_names must provide at least num_classes labels")
        # Older releases could carry extra labels; only output classes are used.
        labels = labels[:num_classes]
        if len(set(labels)) != num_classes:
            raise ValueError("label_names must be unique after whitespace trimming")
        return labels
    if num_classes == 3:
        return ["normal", "motor", "sensor"]
    if num_classes == 4:
        return ["normal", "motor", "propeller", "sensor"]
    labels = ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss"]
    labels.extend(f"class_{index}" for index in range(len(labels), num_classes))
    return labels


def offline_input_contract_errors(config: Any) -> list[str]:
    """Validate config for the existing 31-channel whole-log preprocessor."""
    if not isinstance(config, dict):
        return ["config must be a JSON object"]
    errors: list[str] = []
    if type(config.get("in_channels")) is not int or config["in_channels"] != len(RFLY_SEQ_COLUMNS_V1):
        errors.append("in_channels must be 31 for offline whole-log preprocessing")
    count = config.get("num_classes", 5)
    try:
        offline_label_names(config, count)
    except ValueError as exc:
        errors.append(str(exc))
    if "task" in config and config["task"] not in ("offline_multiclass", "offline_5class"):
        errors.append("task must be offline_multiclass or legacy offline_5class")
    errors.extend(_feature_metadata_errors(config, "config", RFLY_SEQ_COLUMNS_V1))
    return errors


def offline_norm_contract_errors(
    stats: Any, columns: list[str], *, config: dict[str, Any] | None = None
) -> list[str]:
    """Check whole-log statistics, allowing legacy files without column names.

    Zero/tiny positive std retains the offline loader's historical 1e-6 floor;
    online sliding-window normalization intentionally has a different contract.
    """
    if not isinstance(stats, dict):
        return ["norm_stats must be a JSON object"]
    errors: list[str] = []
    if "columns" in stats and stats["columns"] != columns:
        errors.append("norm_stats.columns must match offline preprocessing order")
    for field in ("mean", "std"):
        values = stats.get(field)
        if not isinstance(values, list) or len(values) != len(columns):
            errors.append(f"{field} must match offline preprocessing channel count")
        elif not all(_finite_float32(value) for value in values):
            errors.append(f"{field} must contain finite float32 values")
        elif field == "std" and any(value < 0 for value in values):
            errors.append("std must not contain negative values")
    errors.extend(_feature_metadata_errors(stats, "norm_stats", columns))
    if config is not None:
        for field in ("feature_schema", "protocol_name"):
            if field in config and field in stats and config[field] != stats[field]:
                errors.append(f"{field} differs between config and norm_stats")
    return errors


def online_input_contract_errors(config: Any, stats: Any) -> list[str]:
    """Validate the binary input contract without loading a checkpoint.

    Legacy releases may omit task, labels, schema and window metadata. If
    supplied, these declarations must be consistent; columns always define the
    input order. Architecture support and checkpoint shapes remain loader concerns.
    """
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config must be a JSON object"]
    if not isinstance(stats, dict):
        return ["norm_stats must be a JSON object"]

    if not _positive_int(config.get("in_channels")):
        errors.append("in_channels must be a positive integer")
    if type(config.get("num_classes")) is not int or config["num_classes"] != 2:
        errors.append("num_classes must be 2 for online_binary")
    if "task" in config and config["task"] != "online_binary":
        errors.append("task must be online_binary")
    if "label_names" in config and config["label_names"] != ["normal", "fault"]:
        errors.append("label_names must be ['normal', 'fault'] in that order")

    columns = stats.get("columns")
    valid_columns = (
        isinstance(columns, list)
        and bool(columns)
        and all(isinstance(name, str) and name.strip() == name and bool(name) for name in columns)
        and len(set(columns)) == len(columns)
    )
    if not valid_columns:
        errors.append("columns must be a nonempty list of unique, nonblank feature names")
    count = len(columns) if isinstance(columns, list) else 0
    if config.get("in_channels") != count:
        errors.append("in_channels must match columns length")

    for field in ("mean", "std"):
        values = stats.get(field)
        if not isinstance(values, list) or not values or len(values) != count:
            errors.append(f"{field} must match columns length")
        elif not all(_finite_float32(value, positive=field == "std") for value in values):
            errors.append(f"{field} must contain finite float32 values" + (" greater than zero" if field == "std" else ""))

    for source, data in (("config", config), ("norm_stats", stats)):
        errors.extend(_feature_metadata_errors(data, source, columns if isinstance(columns, list) else []))
        if "window_size" in data and not _positive_int(data["window_size"]):
            errors.append(f"{source}.window_size must be a positive integer")

    for field in ("window_size", "feature_schema", "protocol_name"):
        if field in config and field in stats and config[field] != stats[field]:
            errors.append(f"{field} differs between config and norm_stats")
    return errors

