"""
从 Rfly processed CSV 构建在线检测用的短窗口序列缓存（二分类 + 31 通道）。

复用 build_sequence_cache.py 中的 Rfly 特征 schema 与通道提取逻辑，输出合同保持不变：
  data/processed/online_cache_rfly/
    train.pt   — {'x': (N, C, T), 'y': (N,)}
    val.pt
    test.pt
    norm_stats.json

标签为二分类：0=normal, 1=fault。
窗口标签仍基于“窗口内 fault-active 占比超过阈值”。
"""
import argparse
import json
import math
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sequence_cache import DERIVED_COLUMNS, SEQ_COLUMN_ALIASES, SEQ_COLUMNS, _extract_channels

WINDOW_SIZE = 150
STRIDE = 50
FAULT_THRESHOLD = 0.5
TARGET_HZ = 50.0
FAULT_COLUMNS = (
    'flight_phase',
    'UAVState_data_fault_state',
    'TrueState_data_fault_state',
    'rfly_ctrl_lxl_slash_send_data_id',
    '_rfly_ctrl_lxl_0_id',
)


def _infer_fault_active(df: pd.DataFrame) -> np.ndarray:
    if 'flight_phase' in df.columns:
        return (df['flight_phase'] == 'fault').to_numpy(dtype=np.float32)
    for col in ('UAVState_data_fault_state', 'TrueState_data_fault_state'):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return (values > 0).to_numpy(dtype=np.float32)
    for col in ('rfly_ctrl_lxl_slash_send_data_id', '_rfly_ctrl_lxl_0_id'):
        if col in df.columns:
            ids = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)
            mask = ((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))
            return mask.to_numpy(dtype=np.float32)
    return np.zeros(len(df), dtype=np.float32)


MAX_WINDOWS_PER_FILE = 8


def _cap_windows(x_windows: np.ndarray, y_windows: np.ndarray, max_windows_per_file: int):
    if max_windows_per_file <= 0 or len(y_windows) <= max_windows_per_file:
        return x_windows, y_windows
    idx = np.linspace(0, len(y_windows) - 1, num=max_windows_per_file, dtype=int)
    idx = np.unique(idx)
    return x_windows[idx], y_windows[idx]


def _infer_source_hz(df: pd.DataFrame) -> float | None:
    time_col = None
    scale = 1.0
    if "time_s" in df.columns:
        time_col = "time_s"
    elif "trueTime" in df.columns:
        time_col = "trueTime"
    elif "Timestamp" in df.columns:
        time_col = "Timestamp"
        scale = 1e-6
    if time_col is None:
        return None
    t = pd.to_numeric(df[time_col], errors="coerce") * scale
    dt = t.diff()
    dt = dt[(dt > 0) & dt.notna()]
    if dt.empty:
        return None
    hz = 1.0 / float(dt.median())
    return float(hz)


def _downsample_df(df: pd.DataFrame, target_hz: float, source_hz_hint: float | None = None):
    """按固定步长下采样（优先使用整数倍率，避免昂贵的插值）。"""
    if target_hz <= 0:
        return df, None, 1

    source_hz = float(source_hz_hint) if source_hz_hint else _infer_source_hz(df)
    if source_hz is None or not np.isfinite(source_hz) or source_hz <= 0:
        return df, None, 1

    ratio = source_hz / float(target_hz)
    step = int(round(ratio))
    if step <= 1:
        return df, source_hz, 1

    # 允许轻微数值误差，但拒绝非整数倍率（否则下采样会引入时间漂移）。
    if abs(ratio - step) > 0.02:
        raise ValueError(
            f"Unsupported hz ratio for simple downsample: source_hz={source_hz:.3f}, "
            f"target_hz={target_hz:.3f}, ratio={ratio:.3f}. Consider adding interpolation resampling."
        )

    return df.iloc[::step].reset_index(drop=True), source_hz, step


def extract_windows(
    csv_path: Path,
    window_size: int,
    stride: int,
    *,
    target_hz: float = TARGET_HZ,
    source_hz_hint: float | None = None,
    fault_threshold: float = FAULT_THRESHOLD,
    force_label: int | None = None,
    max_windows_per_file: int = MAX_WINDOWS_PER_FILE,
):
    """从单个 CSV 文件提取滑动窗口片段。"""
    df = pd.read_csv(csv_path)

    if 'flight_phase' in df.columns:
        df = df[df['flight_phase'].isin(['normal', 'fault'])].reset_index(drop=True)

    df, _, _ = _downsample_df(df, target_hz=target_hz, source_hz_hint=source_hz_hint)
    if len(df) < window_size:
        return None, None

    data = _extract_channels(df)
    fault_active = _infer_fault_active(df)

    x_list, y_list = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window_data = data[start:end].T
        if force_label is not None:
            label = int(force_label)
        else:
            fault_ratio = float(fault_active[start:end].mean())
            label = 1 if fault_ratio > fault_threshold else 0
        x_list.append(window_data)
        y_list.append(label)

    if not x_list:
        return None, None

    x_windows = np.stack(x_list)
    y_windows = np.array(y_list, dtype=np.int64)
    return _cap_windows(x_windows, y_windows, max_windows_per_file)


def build_cache(
    csv_dir: str,
    splits_path: str,
    output_dir: str,
    window_size: int,
    stride: int,
    *,
    target_hz: float = TARGET_HZ,
    source_hz_hint: float | None = None,
    fault_threshold: float = FAULT_THRESHOLD,
    limit_files: int = 0,
    extra_normal_dir: str | None = None,
    extra_normal_glob: str = "nf_*.csv",
    extra_normal_split: str = "train",
    extra_normal_limit_files: int = 0,
    max_windows_per_file: int = MAX_WINDOWS_PER_FILE,
    manifest_path: str | None = None,
):
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    # 清空旧缓存
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"已清空旧缓存: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)

    resolved_manifest_path = Path(manifest_path) if manifest_path else (csv_dir / 'manifest.csv')
    if not resolved_manifest_path.exists():
        resolved_manifest_path = csv_dir.parent / 'processed' / 'manifest.csv'
    if not resolved_manifest_path.exists():
        resolved_manifest_path = csv_dir.parent / 'manifest' / 'manifest_v2.csv'
    manifest = pd.read_csv(resolved_manifest_path)
    label_col = None
    for candidate in ("label", "label_binary", "label_main_6class"):
        if candidate in manifest.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(
            f"manifest 缺少标签列，期望包含 label / label_binary / label_main_6class 之一: {resolved_manifest_path}"
        )
    fname_to_label = dict(zip(manifest['filename'], manifest[label_col]))
    fname_to_source = {}
    if 'source_csv_path' in manifest.columns:
        fname_to_source = {
            str(row['filename']): str(row['source_csv_path'])
            for _, row in manifest[['filename', 'source_csv_path']].dropna().iterrows()
        }

    stats = {split: {'normal': 0, 'fault': 0, 'files': 0, 'source_normal_files': 0, 'source_fault_files': 0} for split in splits}

    # 用于写入 norm_stats.json 的运行配置
    cache_cfg = {
        "source_hz_hint": float(source_hz_hint) if source_hz_hint else None,
        "target_hz": float(target_hz),
        "window_size": int(window_size),
        "stride": int(stride),
        "fault_threshold": float(fault_threshold),
        "feature_schema": "rfly_seq_columns_v1",
        "max_windows_per_file": int(max_windows_per_file),
        "fault_columns": list(FAULT_COLUMNS),
        "extra_normal_dir": str(extra_normal_dir) if extra_normal_dir else None,
        "extra_normal_glob": str(extra_normal_glob) if extra_normal_dir else None,
        "extra_normal_split": str(extra_normal_split) if extra_normal_dir else None,
    }

    # 估计源采样率（只读一个文件，避免后续重复打印/猜测）
    source_hz_est = None
    downsample_factor_est = 1
    if target_hz > 0:
        example = None
        for split in ("train", "val", "test"):
            for fname in splits.get(split, []):
                p = Path(fname_to_source.get(fname, str(csv_dir / fname)))
                if p.exists():
                    example = p
                    break
            if example is not None:
                break
        if example is not None:
            try:
                df0 = pd.read_csv(example, usecols=["Timestamp"])
                source_hz_est = _infer_source_hz(df0)
                if source_hz_est is not None and np.isfinite(source_hz_est) and source_hz_est > 0:
                    ratio = float(source_hz_est) / float(target_hz)
                    downsample_factor_est = int(round(ratio)) if ratio >= 1.0 else 1
            except Exception:
                source_hz_est = None
                downsample_factor_est = 1

    cache_cfg["source_hz_est"] = float(source_hz_est) if source_hz_est is not None else None
    cache_cfg["downsample_factor"] = int(downsample_factor_est)

    # Extra negative examples (live normals) for domain alignment.
    extra_files: list[Path] = []
    if extra_normal_dir:
        p = Path(extra_normal_dir)
        if p.exists():
            extra_files = sorted(p.glob(extra_normal_glob))
            if extra_normal_limit_files and extra_normal_limit_files > 0:
                extra_files = extra_files[: int(extra_normal_limit_files)]
        else:
            print(f"[WARN] extra_normal_dir not found, ignoring: {p}")
            extra_files = []

    for split_name, filenames in splits.items():
        if limit_files and limit_files > 0:
            filenames = list(filenames)[: int(limit_files)]
        all_x, all_y = [], []

        for idx, fname in enumerate(filenames):
            csv_path = Path(fname_to_source.get(fname, str(csv_dir / fname)))
            if not csv_path.exists():
                print(f"[WARN] 跳过: {fname}")
                continue

            x_windows, y_windows = extract_windows(
                csv_path,
                window_size,
                stride,
                target_hz=target_hz,
                source_hz_hint=source_hz_hint,
                fault_threshold=fault_threshold,
                max_windows_per_file=max_windows_per_file,
            )
            if x_windows is None:
                continue

            all_x.append(x_windows)
            all_y.append(y_windows)
            stats[split_name]['files'] += 1
            label_str = str(fname_to_label.get(fname, '')).strip().lower()
            if label_str == 'normal':
                stats[split_name]['source_normal_files'] += 1
            elif label_str:
                stats[split_name]['source_fault_files'] += 1

            if (idx + 1) % 100 == 0:
                print(f"  {split_name}: {idx + 1}/{len(filenames)} files processed...")

        # Append extra normal files into one split (default: train).
        if extra_files and split_name == extra_normal_split:
            for j, csv_path in enumerate(extra_files):
                if not csv_path.exists() or csv_path.stat().st_size == 0:
                    continue
                try:
                    x_windows, y_windows = extract_windows(
                        csv_path,
                        window_size,
                        stride,
                        target_hz=target_hz,
                        source_hz_hint=source_hz_hint,
                        fault_threshold=fault_threshold,
                        force_label=0,
                        max_windows_per_file=max_windows_per_file,
                    )
                except Exception as e:
                    print(f"[WARN] extra normal failed: {csv_path.name}: {e}")
                    continue
                if x_windows is None:
                    continue
                all_x.append(x_windows)
                all_y.append(y_windows)
                stats[split_name]['files'] += 1
                stats[split_name]['source_normal_files'] += 1
                if (j + 1) % 50 == 0:
                    print(f"  {split_name} (extra normals): {j + 1}/{len(extra_files)} files processed...")

        if not all_x:
            print(f"[WARN] {split_name}: 没有生成任何窗口样本（可能 window_size 过大或文件过少）")
            out_path = output_dir / f'{split_name}.pt'
            torch.save(
                {
                    'x': torch.zeros((0, len(SEQ_COLUMNS), window_size), dtype=torch.float32),
                    'y': torch.zeros((0,), dtype=torch.int64),
                },
                out_path,
            )
            continue

        x_cat = np.concatenate(all_x, axis=0)
        y_cat = np.concatenate(all_y, axis=0)

        n_normal = int((y_cat == 0).sum())
        n_fault = int((y_cat == 1).sum())
        stats[split_name]['normal'] = n_normal
        stats[split_name]['fault'] = n_fault

        out_path = output_dir / f'{split_name}.pt'
        torch.save({
            'x': torch.from_numpy(x_cat),
            'y': torch.from_numpy(y_cat),
        }, out_path)
        print(f"  {split_name}.pt 已保存: {len(y_cat)} 样本, "
              f"{out_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n计算归一化统计量...")
    train_data = torch.load(output_dir / 'train.pt', weights_only=True)
    x_train = train_data['x']
    mean = x_train.float().mean(dim=(0, 2)).tolist()
    std = x_train.float().std(dim=(0, 2)).tolist()
    std = [s if s > 1e-8 else 1.0 for s in std]

    norm_stats = {
        'mean': mean,
        'std': std,
        'columns': SEQ_COLUMNS,
        **cache_cfg,
    }
    with open(output_dir / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"在线检测缓存构建完成: {output_dir}")
    print(f"{'='*60}")
    print(
        f"通道数: {len(SEQ_COLUMNS)}, window_size: {window_size}, stride: {stride}, "
        f"target_hz: {target_hz} (downsample_factor≈{downsample_factor_est})"
    )
    print(f"\n{'split':<8} {'files':>6} {'src_nf':>8} {'src_ff':>8} {'normal':>8} {'fault':>8} {'total':>8}")
    print('-' * 70)
    for split_name, s in stats.items():
        total = s['normal'] + s['fault']
        print(
            f"{split_name:<8} {s['files']:>6} {s['source_normal_files']:>8} {s['source_fault_files']:>8} "
            f"{s['normal']:>8} {s['fault']:>8} {total:>8}"
        )

    total_all = sum(s['normal'] + s['fault'] for s in stats.values())
    total_fault = sum(s['fault'] for s in stats.values())
    print('-' * 70)
    print(f"{'total':<8} {'':>6} {'':>8} {'':>8} {total_all - total_fault:>8} {total_fault:>8} {total_all:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='构建 Rfly 在线检测短窗口缓存')
    parser.add_argument('--csv-dir', default='data/processed')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--output', default='data/processed/online_cache_rfly')
    parser.add_argument('--manifest-path', default='',
                        help='可选：显式指定 manifest 或 manifest_v2 CSV 路径')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--target-hz', type=float, default=TARGET_HZ,
                        help='下采样到目标频率后再做滑窗 (0=不下采样)')
    parser.add_argument('--source-hz-hint', type=float, default=0.0,
                        help='源数据采样率提示 (0=自动推断)')
    parser.add_argument('--fault-threshold', type=float, default=FAULT_THRESHOLD,
                        help='窗口内 fault-active 占比阈值，超过则标记为 fault')
    parser.add_argument('--limit-files', type=int, default=0,
                        help='每个 split 仅处理前 N 个文件（调试用，0=全量）')
    parser.add_argument('--extra-normal-dir', default='',
                        help='可选：额外的 normal CSV 目录，用于降低线上误报')
    parser.add_argument('--extra-normal-glob', default='nf_*.csv',
                        help='extra-normal-dir 下的文件 glob（默认 nf_*.csv）')
    parser.add_argument('--extra-normal-split', default='train', choices=['train', 'val', 'test'],
                        help='将额外 normal 文件合并到哪个 split（默认 train）')
    parser.add_argument('--extra-normal-limit-files', type=int, default=0,
                        help='extra-normal-dir 仅处理前 N 个文件（0=全量）')
    parser.add_argument('--max-windows-per-file', type=int, default=MAX_WINDOWS_PER_FILE,
                        help='每个文件最多保留多少个窗口（0=不限制）')
    args = parser.parse_args()

    build_cache(args.csv_dir, args.splits, args.output,
                args.window_size, args.stride,
                target_hz=args.target_hz,
                source_hz_hint=(args.source_hz_hint if args.source_hz_hint > 0 else None),
                fault_threshold=args.fault_threshold,
                limit_files=args.limit_files,
                extra_normal_dir=(args.extra_normal_dir if args.extra_normal_dir else None),
                extra_normal_glob=args.extra_normal_glob,
                extra_normal_split=args.extra_normal_split,
                extra_normal_limit_files=args.extra_normal_limit_files,
                max_windows_per_file=args.max_windows_per_file,
                manifest_path=(args.manifest_path if args.manifest_path else None))
