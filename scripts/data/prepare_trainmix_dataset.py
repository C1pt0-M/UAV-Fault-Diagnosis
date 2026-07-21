#!/usr/bin/env python3
"""
组装在线检测训练混合集（离线 + 在线标注）并生成 splits。

默认策略：
- 离线样本沿用 `data/processed/splits.json`（train/val/test 不变）。
- 在线样本（如 telemetry_live/csv_labeled）全部加入 train。
- 仅构建 trainmix，不包含 holdout。

示例：
  python scripts/data/prepare_trainmix_dataset.py \
    --offline-csv-dir data/csv_raw \
    --offline-meta-dir data/metadata \
    --online-csv-dir data/telemetry_live/csv_labeled \
    --online-meta-dir data/telemetry_live/metadata \
    --base-splits data/processed/splits.json \
    --output-root data/trainmix_labeled \
    --output-splits data/processed/splits_trainmix_labeled.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


SPLIT_KEYS = ("train", "val", "test")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_split_keys(splits: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k in SPLIT_KEYS:
        v = splits.get(k, [])
        if not isinstance(v, list):
            raise ValueError(f"split '{k}' 必须是 list，当前类型: {type(v).__name__}")
        out[k] = list(v)
    return out


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"未知 mode: {mode}")


def _prepare_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"目录已存在: {path}（如需覆盖请加 --force）")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _validate_files_exist(root: Path, names: List[str], suffix: str) -> None:
    missing = []
    for name in names:
        stem = Path(name).stem
        expect = root / f"{stem}{suffix}"
        if not expect.exists():
            missing.append(str(expect))
    if missing:
        preview = "\n".join(missing[:10])
        extra = f"\n... 另有 {len(missing) - 10} 个缺失" if len(missing) > 10 else ""
        raise FileNotFoundError(f"缺失文件（前10个）:\n{preview}{extra}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 trainmix（离线+在线）并生成 splits。")
    p.add_argument("--offline-csv-dir", default="data/csv_raw")
    p.add_argument("--offline-meta-dir", default="data/metadata")
    p.add_argument("--online-csv-dir", default="data/telemetry_live/csv_labeled")
    p.add_argument("--online-meta-dir", default="data/telemetry_live/metadata")
    p.add_argument("--base-splits", default="data/processed/splits.json")
    p.add_argument("--output-root", default="data/trainmix_labeled")
    p.add_argument("--output-splits", default="data/processed/splits_trainmix_labeled.json")
    p.add_argument("--online-split", default="train", choices=SPLIT_KEYS, help="在线样本并入哪个 split。")
    p.add_argument("--mode", default="symlink", choices=("symlink", "copy"), help="输出文件组织方式。")
    p.add_argument("--glob", default="*.csv", help="在线 CSV 文件匹配模式。")
    p.add_argument("--force", action="store_true", help="覆盖已有输出目录。")
    p.add_argument("--report-json", default="", help="可选：写出汇总报告 JSON。")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    offline_csv_dir = Path(args.offline_csv_dir)
    offline_meta_dir = Path(args.offline_meta_dir)
    online_csv_dir = Path(args.online_csv_dir)
    online_meta_dir = Path(args.online_meta_dir)
    base_splits_path = Path(args.base_splits)
    out_root = Path(args.output_root)
    out_csv_dir = out_root / "csv_raw"
    out_meta_dir = out_root / "metadata"
    out_splits_path = Path(args.output_splits)

    for p in [offline_csv_dir, offline_meta_dir, online_csv_dir, online_meta_dir, base_splits_path]:
        if not p.exists():
            raise SystemExit(f"[ERROR] 路径不存在: {p}")

    base_splits = _ensure_split_keys(_load_json(base_splits_path))
    offline_names = sorted({x for v in base_splits.values() for x in v})
    online_names = sorted(p.name for p in online_csv_dir.glob(args.glob))

    if not online_names:
        raise SystemExit(f"[ERROR] 在线目录未匹配到 CSV: {online_csv_dir}/{args.glob}")

    dup_names = sorted(set(offline_names).intersection(online_names))
    if dup_names:
        preview = ", ".join(dup_names[:8])
        raise SystemExit(f"[ERROR] 在线/离线文件名冲突({len(dup_names)}): {preview}")

    _validate_files_exist(offline_csv_dir, offline_names, ".csv")
    _validate_files_exist(offline_meta_dir, offline_names, ".json")
    _validate_files_exist(online_csv_dir, online_names, ".csv")
    _validate_files_exist(online_meta_dir, online_names, ".json")

    _prepare_dir(out_root, force=args.force)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # 1) 写入离线样本
    for name in offline_names:
        stem = Path(name).stem
        _link_or_copy(offline_csv_dir / f"{stem}.csv", out_csv_dir / f"{stem}.csv", args.mode)
        _link_or_copy(offline_meta_dir / f"{stem}.json", out_meta_dir / f"{stem}.json", args.mode)

    # 2) 追加在线样本（默认并入 train）
    for name in online_names:
        stem = Path(name).stem
        _link_or_copy(online_csv_dir / f"{stem}.csv", out_csv_dir / f"{stem}.csv", args.mode)
        _link_or_copy(online_meta_dir / f"{stem}.json", out_meta_dir / f"{stem}.json", args.mode)

    out_splits = {k: list(v) for k, v in base_splits.items()}
    out_splits[args.online_split].extend(online_names)
    out_splits[args.online_split] = sorted(set(out_splits[args.online_split]))

    out_splits_path.parent.mkdir(parents=True, exist_ok=True)
    out_splits_path.write_text(json.dumps(out_splits, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "offline_files": len(offline_names),
        "online_files": len(online_names),
        "total_linked_csv": len(list(out_csv_dir.glob("*.csv"))),
        "total_linked_meta": len(list(out_meta_dir.glob("*.json"))),
        "mode": args.mode,
        "online_split": args.online_split,
        "split_counts": {k: len(v) for k, v in out_splits.items()},
        "output_root": str(out_root),
        "output_splits": str(out_splits_path),
    }

    print("[OK] trainmix 构建完成")
    print(f"  output_root   : {out_root}")
    print(f"  output_splits : {out_splits_path}")
    print(f"  split counts  : {report['split_counts']}")
    print(f"  totals        : csv={report['total_linked_csv']}, meta={report['total_linked_meta']}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  report_json   : {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
