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

LABEL_MAIN6 = {
    "normal": 0,
    "motor": 1,
    "propeller": 2,
    "sensor": 3,
    "wind": 4,
    "load": 5,
}
DOMAIN_LABELS = {"SIL": 0, "HIL": 1, "Real": 2}

SEQ_LEN = 2048
MIN_FAULT_WINDOW = 4096
FAULT_PRE_MULT = 1.0
FAULT_POST_MULT = 0.5


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
    return np.concatenate(
        [data, actuator_mean, actuator_std, acc_norm, actuator_effort],
        axis=1,
    )


def _fault_focused_crop(
    df: pd.DataFrame,
    min_window_len: int = MIN_FAULT_WINDOW,
    pre_mult: float = FAULT_PRE_MULT,
    post_mult: float = FAULT_POST_MULT,
) -> pd.DataFrame:
    fault_col = None
    for candidate in (
        "flight_phase",
        "UAVState_data_fault_state",
        "TrueState_data_fault_state",
        "rfly_ctrl_lxl_slash_send_data_id",
    ):
        if candidate in df.columns:
            fault_col = candidate
            break
    if fault_col is None:
        return df

    if fault_col == "flight_phase":
        fault_mask = df[fault_col] == "fault"
    elif fault_col in {"UAVState_data_fault_state", "TrueState_data_fault_state"}:
        fault_mask = pd.to_numeric(df[fault_col], errors="coerce").fillna(0) > 0
    else:
        ids = pd.to_numeric(df[fault_col], errors="coerce").fillna(0).astype(np.int64)
        fault_mask = ((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))

    if int(fault_mask.sum()) == 0:
        return df

    fault_indices = np.where(fault_mask.values)[0]
    fault_start = fault_indices[0]
    fault_end = fault_indices[-1] + 1
    fault_len = fault_end - fault_start
    pre_len = int(pre_mult * fault_len)
    post_len = int(post_mult * fault_len)
    window_start = max(0, fault_start - pre_len)
    window_end = min(len(df), fault_end + post_len)
    cur_len = window_end - window_start
    if cur_len < min_window_len:
        need = min_window_len - cur_len
        left = min(window_start, need // 2)
        right = need - left
        window_start -= left
        window_end = min(len(df), window_end + right)
        window_start = max(0, window_end - min_window_len)
    return df.iloc[window_start:window_end].reset_index(drop=True)


def load_and_resample(
    csv_path: str,
    seq_len: int,
    fault_focus: bool = True,
    min_fault_window: int = MIN_FAULT_WINDOW,
    fault_pre_mult: float = FAULT_PRE_MULT,
    fault_post_mult: float = FAULT_POST_MULT,
) -> np.ndarray:
    raw_df = pd.read_csv(csv_path)
    df = raw_df.reset_index(drop=True)
    if fault_focus:
        df = _fault_focused_crop(
            df,
            min_window_len=min_fault_window,
            pre_mult=fault_pre_mult,
            post_mult=fault_post_mult,
        )
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
        raise RuntimeError("未找到可用于计算归一化统计的训练样本")
    stacked = torch.stack(all_data)
    mean = stacked.mean(dim=(0, 2)).tolist()
    std = stacked.std(dim=(0, 2)).tolist()
    std = [s if s > 1e-8 else 1.0 for s in std]
    return {"mean": mean, "std": std, "columns": SEQ_COLUMNS}


def _load_split_frames(split_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv_file in sorted(split_dir.glob("*.csv")):
        frames[csv_file.stem] = pd.read_csv(csv_file)
    if not frames:
        raise RuntimeError(f"split 目录为空: {split_dir}")
    return frames


def _label_from_row(row: pd.Series, label_column: str) -> int:
    label = str(row.get(label_column, "") or "").strip()
    if label not in LABEL_MAIN6:
        raise ValueError(f"未知 main_6class 标签: {label}")
    return LABEL_MAIN6[label]


def build_cache_v2(
    split_dir: str,
    output_dir: str,
    seq_len: int = SEQ_LEN,
    label_column: str = "label_main_6class",
    train_splits: tuple[str, ...] = ("train",),
    fault_focus: bool = True,
    min_fault_window: int = MIN_FAULT_WINDOW,
    fault_pre_mult: float = FAULT_PRE_MULT,
    fault_post_mult: float = FAULT_POST_MULT,
):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"已清空旧缓存: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames = _load_split_frames(split_dir)
    processed = 0
    skipped = 0
    build_summary = {"splits": {}, "label_column": label_column, "seq_len": seq_len}

    for split_name, split_df in split_frames.items():
        if "source_csv_path" not in split_df.columns:
            raise ValueError(f"{split_name}.csv 缺少 source_csv_path 字段")
        split_out_dir = output_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        count_ok = 0
        count_skip = 0
        for _, row in split_df.iterrows():
            src = Path(str(row["source_csv_path"]))
            if not src.exists():
                print(f"[WARN] 缺失源文件，跳过: {src}")
                skipped += 1
                count_skip += 1
                continue

            try:
                x = load_and_resample(
                    str(src),
                    seq_len=seq_len,
                    fault_focus=fault_focus,
                    min_fault_window=min_fault_window,
                    fault_pre_mult=fault_pre_mult,
                    fault_post_mult=fault_post_mult,
                )
                sample = {
                    "x": torch.from_numpy(x.T),
                    "y": _label_from_row(row, label_column),
                    "filename": str(row.get("filename", src.name)),
                    "sample_id": str(row.get("sample_id", "")),
                    "platform": str(row.get("platform", "")),
                    "domain_y": DOMAIN_LABELS.get(str(row.get("platform", "")), -1),
                    "flight_mode": str(row.get("flight_mode", "")),
                    "label_name": str(row.get(label_column, "")),
                    "source_csv_path": str(src),
                }
                dst_name = src.stem + ".pt"
                torch.save(sample, split_out_dir / dst_name)
                processed += 1
                count_ok += 1
            except Exception as exc:
                print(f"[WARN] 样本构建失败，跳过: {src} | {exc!r}")
                skipped += 1
                count_skip += 1

        build_summary["splits"][split_name] = {
            "saved": count_ok,
            "skipped": count_skip,
        }

    stats = compute_norm_stats(output_dir, list(train_splits))
    with open(output_dir / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    build_summary["processed"] = processed
    build_summary["skipped"] = skipped
    build_summary["train_splits_for_norm"] = list(train_splits)
    (output_dir / "build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n缓存构建完成: {output_dir}")
    for split_name, stats_row in build_summary["splits"].items():
        print(f"  {split_name}: saved={stats_row['saved']} skipped={stats_row['skipped']}")
    print(f"  total_saved={processed}, total_skipped={skipped}")
    print(f"  channels={len(SEQ_COLUMNS)}, seq_len={seq_len}")


def main():
    parser = argparse.ArgumentParser(description="构建 RflyMAD v2 sequence cache")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--label-column", default="label_main_6class")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="用于计算 norm_stats 的 split 名称列表",
    )
    parser.add_argument("--disable-fault-focus", action="store_true", default=False)
    parser.add_argument("--min-fault-window", type=int, default=MIN_FAULT_WINDOW)
    parser.add_argument("--fault-pre-mult", type=float, default=FAULT_PRE_MULT)
    parser.add_argument("--fault-post-mult", type=float, default=FAULT_POST_MULT)
    args = parser.parse_args()

    build_cache_v2(
        split_dir=args.split_dir,
        output_dir=args.output,
        seq_len=args.seq_len,
        label_column=args.label_column,
        train_splits=tuple(args.train_splits),
        fault_focus=not args.disable_fault_focus,
        min_fault_window=args.min_fault_window,
        fault_pre_mult=args.fault_pre_mult,
        fault_post_mult=args.fault_post_mult,
    )


if __name__ == "__main__":
    main()
