#!/usr/bin/env python3
"""
为实时 CSV 补充故障时序标签（is_fault_active / flight_phase）。

用途：
- 在线实时 CSV 通常没有 is_fault_active，直接用于 build_online_cache 会把故障样本当 normal。
- 本脚本基于 metadata 中的 fault_injection 信息，自动生成时序标签列。

默认行为：
- 输出到新目录，不改动原始 CSV。
- normal (nf) -> is_fault_active 全 0。
- fault (bf/mf/pi/tf) -> 按 fault_time_rel 切分，故障后标 1。

示例：
  python scripts/data/annotate_fault_activity.py \
    --csv-dir data/telemetry_live/csv_raw \
    --meta-dir data/telemetry_live/metadata \
    --output-dir data/telemetry_live/csv_labeled
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FAULT_PREFIXES = {"bf", "mf", "pi", "tf"}
NORMAL_PREFIXES = {"nf"}


@dataclass
class FileResult:
    file: str
    prefix: str
    status: str
    start_t: float | None
    fault_rows: int
    total_rows: int
    note: str = ""


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _infer_fault_start_time(meta: dict[str, Any], ramp_onset_frac: float) -> float | None:
    fi = meta.get("fault_injection") or {}
    ft = _safe_float(fi.get("fault_time_rel"))

    if ft is None:
        sim_time = _safe_float(meta.get("sim_time"))
        fault_ratio = _safe_float(fi.get("fault_ratio"))
        if sim_time is not None and fault_ratio is not None:
            ft = sim_time * fault_ratio

    if ft is None:
        return None

    inject_mode = str(fi.get("inject_mode", "step")).lower()
    if inject_mode == "ramp":
        ramp_duration = _safe_float(fi.get("ramp_duration")) or 0.0
        ft = ft + ramp_duration * float(ramp_onset_frac)

    return max(0.0, float(ft))


def _build_fault_mask(df: pd.DataFrame, start_t: float) -> np.ndarray:
    if "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce")
        if t.notna().any():
            t_fill = t.ffill().bfill().fillna(0.0)
            return (t_fill.to_numpy(dtype=np.float64) >= float(start_t)).astype(np.int64)

    # fallback: 无 time_s 时按行比例切
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.arange(n, dtype=np.float64)
    # 这里把 start_t 视作“相对秒数”，fallback 退化为中点切分。
    cut = int(n * 0.5)
    return (idx >= cut).astype(np.int64)


def annotate_one(
    csv_path: Path,
    meta_path: Path | None,
    add_flight_phase: bool,
    ramp_onset_frac: float,
    overwrite_existing: bool,
) -> tuple[pd.DataFrame | None, FileResult]:
    prefix = csv_path.name.split("_", 1)[0].lower()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, FileResult(
            file=csv_path.name,
            prefix=prefix,
            status="read_error",
            start_t=None,
            fault_rows=0,
            total_rows=0,
            note=type(e).__name__,
        )

    n = len(df)
    if n == 0:
        return df, FileResult(
            file=csv_path.name,
            prefix=prefix,
            status="empty",
            start_t=None,
            fault_rows=0,
            total_rows=0,
        )

    has_label_col = "is_fault_active" in df.columns
    if has_label_col and not overwrite_existing:
        fault_rows = int((pd.to_numeric(df["is_fault_active"], errors="coerce").fillna(0) > 0.5).sum())
        return df, FileResult(
            file=csv_path.name,
            prefix=prefix,
            status="skipped_existing",
            start_t=None,
            fault_rows=fault_rows,
            total_rows=n,
        )

    is_fault = np.zeros((n,), dtype=np.int64)
    start_t: float | None = None

    if prefix in NORMAL_PREFIXES:
        pass
    elif prefix in FAULT_PREFIXES:
        meta: dict[str, Any] = {}
        if meta_path is None or not meta_path.exists():
            return None, FileResult(
                file=csv_path.name,
                prefix=prefix,
                status="missing_meta",
                start_t=None,
                fault_rows=0,
                total_rows=n,
                note="fault file without metadata",
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            return None, FileResult(
                file=csv_path.name,
                prefix=prefix,
                status="bad_meta",
                start_t=None,
                fault_rows=0,
                total_rows=n,
                note=type(e).__name__,
            )

        has_fault = bool(meta.get("has_fault", True))
        if has_fault:
            start_t = _infer_fault_start_time(meta, ramp_onset_frac=ramp_onset_frac)
            if start_t is None:
                return None, FileResult(
                    file=csv_path.name,
                    prefix=prefix,
                    status="missing_fault_time",
                    start_t=None,
                    fault_rows=0,
                    total_rows=n,
                    note="fault_injection lacks fault_time_rel/fault_ratio",
                )
            is_fault = _build_fault_mask(df, start_t=start_t)
    else:
        return None, FileResult(
            file=csv_path.name,
            prefix=prefix,
            status="unknown_prefix",
            start_t=None,
            fault_rows=0,
            total_rows=n,
        )

    df["is_fault_active"] = is_fault
    if add_flight_phase:
        df["flight_phase"] = np.where(is_fault > 0, "fault", "normal")

    return df, FileResult(
        file=csv_path.name,
        prefix=prefix,
        status="ok",
        start_t=start_t,
        fault_rows=int(is_fault.sum()),
        total_rows=n,
    )


def run(args: argparse.Namespace) -> int:
    csv_dir = Path(args.csv_dir)
    meta_dir = Path(args.meta_dir)
    if not csv_dir.exists():
        raise SystemExit(f"[ERROR] csv dir not found: {csv_dir}")
    if not meta_dir.exists():
        raise SystemExit(f"[ERROR] meta dir not found: {meta_dir}")

    if args.in_place:
        out_dir = csv_dir
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(csv_dir.glob(args.glob))
    if args.limit_files and args.limit_files > 0:
        files = files[: int(args.limit_files)]
    if not files:
        print(f"[WARN] no csv files matched: {csv_dir}/{args.glob}")
        return 0

    results: list[FileResult] = []
    for p in files:
        meta_path = meta_dir / f"{p.stem}.json"
        out_df, r = annotate_one(
            csv_path=p,
            meta_path=meta_path,
            add_flight_phase=args.add_flight_phase,
            ramp_onset_frac=args.ramp_onset_frac,
            overwrite_existing=args.overwrite_existing,
        )
        results.append(r)

        if out_df is None:
            continue
        out_path = p if args.in_place else (out_dir / p.name)
        out_df.to_csv(out_path, index=False)

    # summary
    by_status: dict[str, int] = {}
    by_prefix: dict[str, int] = {}
    total_fault_rows = 0
    total_rows = 0
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_prefix[r.prefix] = by_prefix.get(r.prefix, 0) + 1
        total_fault_rows += r.fault_rows
        total_rows += r.total_rows

    print(f"[INFO] processed files: {len(results)}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] status counts: {by_status}")
    print(f"[INFO] prefix counts: {dict(sorted(by_prefix.items()))}")
    if total_rows > 0:
        print(
            "[INFO] fault-row ratio: "
            f"{total_fault_rows}/{total_rows} = {total_fault_rows / total_rows:.4f}"
        )

    bad = [r for r in results if r.status not in {"ok", "skipped_existing", "empty"}]
    if bad:
        print("\n[WARN] problematic files:")
        for r in bad[:20]:
            print(f"  - {r.file}: {r.status} ({r.note})")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")

    # optional report
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "file": r.file,
                "prefix": r.prefix,
                "status": r.status,
                "start_t": r.start_t,
                "fault_rows": r.fault_rows,
                "total_rows": r.total_rows,
                "note": r.note,
            }
            for r in results
        ]
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] report json: {report_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate telemetry CSV with is_fault_active using metadata.")
    parser.add_argument("--csv-dir", required=True, help="Input CSV directory.")
    parser.add_argument("--meta-dir", required=True, help="Metadata directory (same stem .json).")
    parser.add_argument("--glob", default="*.csv", help="Filename glob under --csv-dir.")
    parser.add_argument("--output-dir", default="", help="Output directory (required unless --in-place).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite CSV in place.")
    parser.add_argument("--add-flight-phase", action="store_true", help="Also write flight_phase column.")
    parser.add_argument(
        "--ramp-onset-frac",
        type=float,
        default=0.5,
        help="For ramp injection, fault starts at fault_time_rel + ramp_duration * frac (default 0.5).",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing is_fault_active.")
    parser.add_argument("--limit-files", type=int, default=0, help="Only process first N files (0=all).")
    parser.add_argument("--report-json", default="", help="Optional path to write per-file report JSON.")
    args = parser.parse_args()

    if not args.in_place and not args.output_dir:
        parser.error("Either --in-place or --output-dir must be provided.")
    if args.ramp_onset_frac < 0.0 or args.ramp_onset_frac > 1.0:
        parser.error("--ramp-onset-frac must be in [0, 1].")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
