import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEQ_COLUMN_ALIASES = [
    ("actuator_ctrl_1", ["_actuator_controls_0_0_control[0]"]),
    ("actuator_ctrl_2", ["_actuator_controls_0_0_control[1]"]),
    ("actuator_ctrl_3", ["_actuator_controls_0_0_control[2]"]),
    ("actuator_ctrl_4", ["_actuator_controls_0_0_control[3]"]),
    ("actuator_out_1", ["_actuator_outputs_0_output[0]"]),
    ("actuator_out_2", ["_actuator_outputs_0_output[1]"]),
    ("actuator_out_3", ["_actuator_outputs_0_output[2]"]),
    ("actuator_out_4", ["_actuator_outputs_0_output[3]"]),
    ("gyro_x", ["_sensor_combined_0_gyro_rad[0]"]),
    ("gyro_y", ["_sensor_combined_0_gyro_rad[1]"]),
    ("gyro_z", ["_sensor_combined_0_gyro_rad[2]"]),
    ("accel_x", ["_sensor_combined_0_accelerometer_m_s2[0]"]),
    ("accel_y", ["_sensor_combined_0_accelerometer_m_s2[1]"]),
    ("accel_z", ["_sensor_combined_0_accelerometer_m_s2[2]"]),
    ("baro_alt", ["_vehicle_air_data_0_baro_alt_meter"]),
    ("baro_temp", ["_vehicle_air_data_0_baro_temp_celcius"]),
    ("baro_pressure", ["_vehicle_air_data_0_baro_pressure_pa"]),
    ("att_q_w", ["_vehicle_attitude_0_q[0]"]),
    ("att_q_x", ["_vehicle_attitude_0_q[1]"]),
    ("att_q_y", ["_vehicle_attitude_0_q[2]"]),
    ("att_q_z", ["_vehicle_attitude_0_q[3]"]),
    ("vel_x", ["_vehicle_local_position_0_vx", "_vehicle_local_position_setpoint_0_vx"]),
    ("vel_y", ["_vehicle_local_position_0_vy", "_vehicle_local_position_setpoint_0_vy"]),
    ("vel_z", ["_vehicle_local_position_0_vz", "_vehicle_local_position_setpoint_0_vz"]),
    ("mag_x", ["_vehicle_magnetometer_0_magnetometer_ga[0]"]),
    ("mag_y", ["_vehicle_magnetometer_0_magnetometer_ga[1]"]),
    ("mag_z", ["_vehicle_magnetometer_0_magnetometer_ga[2]"]),
]
DERIVED_COLUMNS = ["actuator_mean", "actuator_std", "acc_norm", "actuator_effort"]
SEQ_COLUMNS = [name for name, _ in SEQ_COLUMN_ALIASES] + DERIVED_COLUMNS

LABEL_3CLASS = {"normal": 0, "motor": 1, "sensor": 2}
DOMAIN_LABELS = {"SIL": 0, "HIL": 1, "Real": 2}
SEQ_LEN = 2048


def interp_resample(data: np.ndarray, target_len: int) -> np.ndarray:
    src_len, n_ch = data.shape
    x_src = np.linspace(0, 1, src_len)
    x_dst = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, n_ch), dtype=np.float32)
    for i in range(n_ch):
        out[:, i] = np.interp(x_dst, x_src, data[:, i])
    return out


def _extract_channels(df: pd.DataFrame) -> np.ndarray:
    data = np.zeros((len(df), len(SEQ_COLUMN_ALIASES)), dtype=np.float32)
    for i, (_, aliases) in enumerate(SEQ_COLUMN_ALIASES):
        for col in aliases:
            if col in df.columns:
                data[:, i] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
                break
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    actuator_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    actuators = data[:, actuator_idx]
    actuator_mean = np.mean(actuators, axis=1, keepdims=True)
    actuator_std = np.std(actuators, axis=1, keepdims=True)
    ax, ay, az = data[:, 11], data[:, 12], data[:, 13]
    acc_norm = np.sqrt(ax**2 + ay**2 + az**2).reshape(-1, 1)
    actuator_effort = actuator_mean / np.clip(acc_norm, 0.1, None)
    return np.concatenate([data, actuator_mean, actuator_std, acc_norm, actuator_effort], axis=1)


def load_and_resample(csv_path: str, seq_len: int) -> np.ndarray:
    df = pd.read_csv(csv_path).reset_index(drop=True)
    data = _extract_channels(df)
    return interp_resample(data, seq_len)


def compute_norm_stats(cache_dir: Path, train_splits: list[str]) -> dict:
    all_data = []
    for split_name in train_splits:
        split_dir = cache_dir / split_name
        if not split_dir.exists():
            continue
        for pt_file in sorted(split_dir.glob("*.pt")):
            sample = torch.load(pt_file, weights_only=True)
            all_data.append(sample["x"])
    if not all_data:
        raise RuntimeError("未找到可用于 norm_stats 的训练样本")
    stacked = torch.stack(all_data)
    mean = stacked.mean(dim=(0, 2)).tolist()
    std = stacked.std(dim=(0, 2)).tolist()
    std = [s if s > 1e-8 else 1.0 for s in std]
    return {"mean": mean, "std": std, "columns": SEQ_COLUMNS}


def build_cache(split_dir: str, output_dir: str, train_splits: tuple[str, ...], label_column: str = "label_main_3class", seq_len: int = SEQ_LEN):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    summary = {"splits": {}, "label_column": label_column, "seq_len": seq_len}
    for csv_file in sorted(split_dir.glob("*.csv")):
        split_name = csv_file.stem
        frame = pd.read_csv(csv_file)
        split_out = output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        saved = 0
        skipped = 0
        for _, row in frame.iterrows():
            src = Path(str(row["source_csv_path"]))
            if not src.exists():
                skipped += 1
                continue
            label = str(row.get(label_column, "") or "").strip()
            if label not in LABEL_3CLASS:
                skipped += 1
                continue
            x = load_and_resample(str(src), seq_len=seq_len)
            sample = {
                "x": torch.from_numpy(x.T),
                "y": LABEL_3CLASS[label],
                "filename": str(row.get("filename", src.name)),
                "sample_id": str(row.get("sample_id", "")),
                "platform": str(row.get("platform", "")),
                "domain_y": DOMAIN_LABELS.get(str(row.get("platform", "")), -1),
                "flight_mode": str(row.get("flight_mode", "")),
                "label_name": label,
                "source_csv_path": str(src),
            }
            torch.save(sample, split_out / f"{src.stem}.pt")
            saved += 1
            processed += 1
        summary["splits"][split_name] = {"saved": saved, "skipped": skipped}

    stats = compute_norm_stats(output_dir, list(train_splits))
    (output_dir / "norm_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["processed"] = processed
    summary["train_splits_for_norm"] = list(train_splits)
    (output_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="构建 transfer 3class sequence cache")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--label-column", default="label_main_3class")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    build_cache(
        split_dir=args.split_dir,
        output_dir=args.output,
        train_splits=tuple(args.train_splits),
        label_column=args.label_column,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
