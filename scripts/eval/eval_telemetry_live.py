#!/usr/bin/env python3
"""
Evaluate an online binary fault detector (bilstm/cnn1d/transformer) on telemetry_live CSVs.

This script is meant for "online-like" validation: it replays telemetry frames in order,
runs sliding-window inference every N frames, then summarizes both point-level scores and
event-level aggregations (hysteresis).

Typical usage (recommended: run with the conda env that has torch installed):
  python scripts/eval_telemetry_live.py \\
    --csv-dir csv_raw_nf \\
    --model-dir assets/models/online_detector_cnn1d_50hz \\
    --every 10 \\
    --out logs/telemetry_live_eval.csv \\
    --plots
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


def _add_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


_add_import_paths()

from uav_fault.telemetry.replay_metrics import gt_event_metrics as _gt_event_metrics
from uav_fault.telemetry.schema_adapter import (
    adapt_px4_live_frame as _adapt_px4_live_frame,
    canonicalize_rfly_records as _canonicalize_rfly_records,
    normalize_time_columns as _normalize_time_columns,
)


def _infer_expected_from_name(name: str) -> tuple[str, str]:
    """Best-effort label from filename prefix (weak label).

    Returns:
      (prefix, expected) where expected is one of: 'normal', 'fault', 'unknown'.
    """
    prefix = name.split("_", 1)[0].lower()
    if prefix == "nf":
        return prefix, "normal"
    if prefix in {"bf", "mf", "pi", "tf"}:
        return prefix, "fault"
    return prefix, "unknown"


def _infer_hz_from_time_s(df) -> float | None:
    if "time_s" not in df.columns or len(df) < 3:
        return None
    t = np.asarray(df["time_s"], dtype=float)
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    dt_med = float(np.median(dt))
    if dt_med <= 0:
        return None
    return float(1.0 / dt_med)


def _time_array(df) -> np.ndarray:
    if "time_s" not in df.columns or len(df) == 0:
        return np.arange(len(df), dtype=float)
    try:
        t = np.asarray(df["time_s"], dtype=float)
    except Exception:
        return np.arange(len(df), dtype=float)
    if t.shape[0] != len(df) or not np.all(np.isfinite(t)):
        return np.arange(len(df), dtype=float)
    return t


def _infer_fault_active_from_df(df) -> np.ndarray | None:
    if "is_fault_active" in df.columns:
        return (np.asarray(df["is_fault_active"], dtype=float) > 0.5).astype(bool)
    if "flight_phase" in df.columns:
        return (df["flight_phase"].astype(str) == "fault").to_numpy(dtype=bool)
    for col in ("UAVState_data_fault_state", "TrueState_data_fault_state"):
        if col in df.columns:
            values = np.asarray(df[col], dtype=float)
            return values > 0
    for col in ("rfly_ctrl_lxl_slash_send_data_id", "_rfly_ctrl_lxl_0_id"):
        if col in df.columns:
            ids = np.asarray(df[col], dtype=float)
            ids = np.nan_to_num(ids, nan=0.0).astype(np.int64)
            return ((ids >= 123450) & (ids <= 123459)) | (
                (ids >= 123540) & (ids <= 123549)
            )
    return None


def _records_for_backend(df, backend: str) -> list[dict]:
    if backend == "px4":
        rows = df.to_dict(orient="records")
        return [_adapt_px4_live_frame(row)[0] for row in rows]
    return _canonicalize_rfly_records(df)


def _maybe_plot(
    points, events, out_path: Path, onset_th: float, clear_th: float
) -> None:
    """Plot fault_prob over time and shade aggregated events (best-effort)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not points:
        return

    t = [p["time_s"] for p in points]
    fp = [p["fault_prob"] for p in points]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, fp, lw=1.2, label="fault_prob")
    ax.axhline(
        float(onset_th), ls="--", lw=1.0, color="tab:red", alpha=0.8, label="onset_th"
    )
    ax.axhline(
        float(clear_th),
        ls="--",
        lw=1.0,
        color="tab:orange",
        alpha=0.8,
        label="clear_th",
    )

    for e in events:
        ax.axvspan(e.start_t, e.end_t, color="tab:red", alpha=0.15)

    ax.set_xlabel("time_s")
    ax.set_ylabel("fault_prob")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _replay_points_legacy(detector, records, every_n: int) -> list[dict]:
    window = deque(maxlen=int(detector.window_size))
    points = []
    for idx, row in enumerate(records):
        window.append(row)
        if (idx + 1) % every_n != 0:
            continue
        if len(window) < int(detector.window_size):
            continue
        pred = detector.predict(window)
        if pred is None:
            continue
        t_val = row.get("time_s", idx)
        try:
            t = float(t_val)
        except Exception:
            t = float(idx)
        points.append(
            {
                "time_s": t,
                "frame_idx": int(idx),
                "fault_prob": float(pred.get("fault_prob", 0.0)),
                "label": str(pred.get("label", "")),
                "confidence": float(pred.get("confidence", 0.0)),
            }
        )
    return points


def _replay_points_fast(detector, records, every_n: int, batch_size: int) -> list[dict]:
    outputs = detector.predict_records_batch(
        records, every_n=every_n, batch_size=batch_size
    )
    points = []
    for end, pred in outputs:
        idx = int(end - 1)
        row = records[idx]
        t_val = row.get("time_s", idx)
        try:
            t = float(t_val)
        except Exception:
            t = float(idx)
        points.append(
            {
                "time_s": t,
                "frame_idx": idx,
                "fault_prob": float(pred.get("fault_prob", 0.0)),
                "label": str(pred.get("label", "")),
                "confidence": float(pred.get("confidence", 0.0)),
            }
        )
    return points


def main() -> int:
    _add_import_paths()
    # Local imports: keep script import-time light.
    from uav_fault.online.event_aggregator import FaultEventAggregator
    from uav_fault.online.fault_detector import FaultDetector

    parser = argparse.ArgumentParser(
        description="Replay telemetry_live CSVs and summarize online detector outputs."
    )
    parser.add_argument("--csv-dir", default="csv_raw_nf", help="Input CSV dir.")
    parser.add_argument(
        "--glob", default="*.csv", help="Filename glob under --csv-dir."
    )
    parser.add_argument(
        "--file-list",
        default="",
        help="Optional text file of filenames to replay (one per line).",
    )
    parser.add_argument(
        "--backend",
        choices=["replay", "px4"],
        default="replay",
        help="Input schema backend.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model directory (best.pt/config.json/norm_stats.json).",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help="Comma-separated canonical feature names to zero out during evaluation.",
    )
    parser.add_argument("--device", default="cpu", help="cpu/cuda.")
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window size override (0=use model/normalization config).",
    )
    parser.add_argument(
        "--every", type=int, default=10, help="Run inference every N frames."
    )
    parser.add_argument(
        "--limit-files", type=int, default=0, help="Process only first N files (0=all)."
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Replay only first N rows per file (0=all).",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["fast", "legacy"],
        default="fast",
        help="Replay path: fast=batch inference, legacy=per-window predict().",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for --replay-mode fast."
    )
    parser.add_argument(
        "--out",
        default="",
        help="Write summary CSV to this path (default: logs/telemetry_live_eval_<ts>.csv).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write per-file fault_prob plots (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-dir", default="", help="Plot output dir (default: alongside --out)."
    )

    # Event aggregation parameters (same semantics as uav_fault.online.run_detector)
    parser.add_argument(
        "--onset-th",
        type=float,
        default=0.6,
        help="Fault onset threshold (fault_prob).",
    )
    parser.add_argument(
        "--clear-th",
        type=float,
        default=0.4,
        help="Fault clear threshold (fault_prob).",
    )
    parser.add_argument(
        "--k-on",
        type=int,
        default=3,
        help="Consecutive points >= onset_th to start an event.",
    )
    parser.add_argument(
        "--k-off",
        type=int,
        default=3,
        help="Consecutive points < clear_th to end an event.",
    )
    parser.add_argument(
        "--min-event-s",
        type=float,
        default=0.5,
        help="Drop events shorter than this (seconds).",
    )
    parser.add_argument(
        "--merge-gap-s",
        type=float,
        default=0.5,
        help="Merge events with small gaps (seconds).",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        raise SystemExit(f"[ERROR] csv dir not found: {csv_dir}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"[ERROR] model dir not found: {model_dir}")

    exclude_features = [
        item.strip() for item in str(args.exclude_features).split(",") if item.strip()
    ]

    detector = FaultDetector(
        model_dir=str(model_dir),
        device=args.device,
        window_size=(args.window if args.window > 0 else None),
        exclude_features=exclude_features,
    )
    window_size = int(detector.window_size)
    every_n = int(args.every)
    if every_n <= 0:
        raise SystemExit("[ERROR] --every must be positive.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = (
        Path(args.out) if args.out else Path("logs") / f"telemetry_live_eval_{ts}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = (
        Path(args.plot_dir) if args.plot_dir else out_csv.with_suffix("").as_posix()
    )
    plot_dir = Path(plot_dir)

    # Collect files.
    if args.file_list:
        names = [
            line.strip()
            for line in Path(args.file_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        files = [csv_dir / name for name in names]
    else:
        files = sorted(csv_dir.glob(args.glob))
    if args.limit_files and args.limit_files > 0:
        files = files[: int(args.limit_files)]

    if not files:
        print(f"[WARN] no files matched: {csv_dir}/{args.glob}")
        return 0

    import pandas as pd

    rows = []

    # File-level "weak label" eval (by filename prefix).
    eval_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0}

    gt_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unlabeled": 0}
    gt_event_counts = {"hits": 0, "misses": 0, "false_alarms": 0, "labeled_files": 0}
    gt_latencies: list[float] = []

    print(
        f"[INFO] model={detector.config.get('model')} window_size={window_size} every={every_n} device={args.device}"
    )
    print(f"[INFO] replay_mode={args.replay_mode} batch_size={int(args.batch_size)}")
    print(f"[INFO] scanning {len(files)} files under: {csv_dir}")

    for p in files:
        name = p.name
        prefix, expected = _infer_expected_from_name(name)

        if p.stat().st_size == 0:
            rows.append(
                {
                    "file": name,
                    "prefix": prefix,
                    "expected": expected,
                    "status": "empty",
                }
            )
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            rows.append(
                {
                    "file": name,
                    "prefix": prefix,
                    "expected": expected,
                    "status": f"read_error:{type(e).__name__}",
                }
            )
            continue

        if args.limit_rows and args.limit_rows > 0:
            df = df.iloc[: int(args.limit_rows)].copy()

        df = _normalize_time_columns(df)
        hz = _infer_hz_from_time_s(df)
        t0 = (
            float(df["time_s"].iloc[0])
            if "time_s" in df.columns and len(df)
            else float("nan")
        )
        t1 = (
            float(df["time_s"].iloc[-1])
            if "time_s" in df.columns and len(df)
            else float("nan")
        )

        records = _records_for_backend(df, args.backend)
        if args.replay_mode == "fast":
            points = _replay_points_fast(
                detector, records, every_n=every_n, batch_size=int(args.batch_size)
            )
        else:
            points = _replay_points_legacy(detector, records, every_n=every_n)

        fps = np.asarray([pt["fault_prob"] for pt in points], dtype=float)
        n_inf = int(len(points))

        if n_inf == 0:
            row_out = {
                "file": name,
                "prefix": prefix,
                "expected": expected,
                "status": "no_inference",
                "n_rows": int(len(df)),
                "sample_hz": hz,
                "t_start": t0,
                "t_end": t1,
                "window_size": window_size,
                "every": every_n,
                "n_inferences": 0,
            }
            rows.append(row_out)
            continue

        agg = FaultEventAggregator(
            onset_th=float(args.onset_th),
            clear_th=float(args.clear_th),
            k_on=int(args.k_on),
            k_off=int(args.k_off),
            min_event_s=float(args.min_event_s),
            merge_gap_s=float(args.merge_gap_s),
        )
        events = []
        for pt in points:
            events.extend(
                agg.update(
                    t=pt["time_s"],
                    fault_prob=pt["fault_prob"],
                    frame_idx=pt["frame_idx"],
                )
            )
        events.extend(agg.flush())

        total_event_s = float(sum(max(0.0, e.end_t - e.start_t) for e in events))
        max_event_fp = float(max((e.max_fault_prob for e in events), default=0.0))
        pred_fault_by_event = bool(len(events) > 0)
        pred_fault_by_max05 = bool(float(np.max(fps)) >= 0.5)

        gt_metrics = _gt_event_metrics(df, events)
        if gt_metrics["gt_has_labels"]:
            gt_event_counts["labeled_files"] += 1
            gt_fault = bool(gt_metrics["gt_fault"])
            pred_fault = bool(len(events) > 0)
            if gt_fault and pred_fault:
                gt_counts["tp"] += 1
            elif gt_fault and (not pred_fault):
                gt_counts["fn"] += 1
            elif (not gt_fault) and pred_fault:
                gt_counts["fp"] += 1
            else:
                gt_counts["tn"] += 1
            gt_event_counts["hits"] += int(gt_metrics["hit"] or 0)
            gt_event_counts["misses"] += int(gt_metrics["miss"] or 0)
            gt_event_counts["false_alarms"] += int(gt_metrics["false_alarm"] or 0)
            latency = gt_metrics["detection_latency_s"]
            if latency is not None and bool(gt_metrics["hit"]):
                gt_latencies.append(float(latency))
        else:
            gt_counts["unlabeled"] += 1

        if expected == "unknown":
            eval_counts["unknown"] += 1
        else:
            exp_fault = expected == "fault"
            if exp_fault and pred_fault_by_event:
                eval_counts["tp"] += 1
            elif exp_fault and (not pred_fault_by_event):
                eval_counts["fn"] += 1
            elif (not exp_fault) and pred_fault_by_event:
                eval_counts["fp"] += 1
            else:
                eval_counts["tn"] += 1

        row_out = {
            "file": name,
            "prefix": prefix,
            "expected": expected,
            "status": "ok",
            "n_rows": int(len(df)),
            "sample_hz": hz,
            "t_start": t0,
            "t_end": t1,
            "window_size": window_size,
            "every": every_n,
            "n_inferences": n_inf,
            "mean_fault_prob": float(np.mean(fps)),
            "max_fault_prob": float(np.max(fps)),
            "p_fault_prob_ge_05": float(np.mean(fps >= 0.5)),
            "p_fault_prob_ge_06": float(np.mean(fps >= 0.6)),
            "n_events": int(len(events)),
            "total_event_s": total_event_s,
            "max_event_fault_prob": max_event_fp,
            "pred_fault_by_event": int(pred_fault_by_event),
            "pred_fault_by_max05": int(pred_fault_by_max05),
            **gt_metrics,
        }
        if events:
            # Keep only the most informative event for quick inspection.
            best = max(events, key=lambda e: float(e.max_fault_prob))
            row_out.update(
                {
                    "best_event_start_t": float(best.start_t),
                    "best_event_end_t": float(best.end_t),
                    "best_event_dur_s": float(best.end_t - best.start_t),
                    "best_event_max_fp": float(best.max_fault_prob),
                    "best_event_mean_fp": float(best.mean_fault_prob),
                }
            )

        rows.append(row_out)

        if args.plots:
            out_path = plot_dir / f"{Path(name).stem}_fault_prob.png"
            _maybe_plot(
                points,
                events,
                out_path,
                onset_th=float(args.onset_th),
                clear_th=float(args.clear_th),
            )

    # Write CSV + print a compact table.
    import pandas as pd

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote summary: {out_csv}")

    # Console table (best-effort compact view)
    show_cols = [
        "file",
        "expected",
        "status",
        "n_rows",
        "sample_hz",
        "n_inferences",
        "mean_fault_prob",
        "max_fault_prob",
        "p_fault_prob_ge_05",
        "p_fault_prob_ge_06",
        "n_events",
        "total_event_s",
    ]
    present = [c for c in show_cols if c in df_out.columns]
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        try:
            print(df_out[present].to_string(index=False))
        except Exception:
            pass

    # Weak-label file-level stats.
    known = (
        eval_counts["tp"] + eval_counts["tn"] + eval_counts["fp"] + eval_counts["fn"]
    )
    if known > 0:
        acc = (eval_counts["tp"] + eval_counts["tn"]) / known
        prec = eval_counts["tp"] / max(1, (eval_counts["tp"] + eval_counts["fp"]))
        rec = eval_counts["tp"] / max(1, (eval_counts["tp"] + eval_counts["fn"]))
        print(
            "\n[WEAK-LABEL FILE-LEVEL] (expected label inferred from filename prefix; predicted fault = n_events>0)\n"
            f"  files_known={known} files_unknown={eval_counts['unknown']}\n"
            f"  TP={eval_counts['tp']} FP={eval_counts['fp']} TN={eval_counts['tn']} FN={eval_counts['fn']}\n"
            f"  accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f}"
        )

    labeled = gt_event_counts["labeled_files"]
    if labeled > 0:
        gt_known = gt_counts["tp"] + gt_counts["tn"] + gt_counts["fp"] + gt_counts["fn"]
        gt_acc = (gt_counts["tp"] + gt_counts["tn"]) / max(1, gt_known)
        gt_prec = gt_counts["tp"] / max(1, (gt_counts["tp"] + gt_counts["fp"]))
        gt_rec = gt_counts["tp"] / max(1, (gt_counts["tp"] + gt_counts["fn"]))
        latency_mean = float(np.mean(gt_latencies)) if gt_latencies else float("nan")
        latency_median = (
            float(np.median(gt_latencies)) if gt_latencies else float("nan")
        )
        print(
            "\n[GT EVENT-LEVEL] (ground truth from is_fault_active; predicted fault = n_events>0)\n"
            f"  files_labeled={labeled} files_unlabeled={gt_counts['unlabeled']}\n"
            f"  TP={gt_counts['tp']} FP={gt_counts['fp']} TN={gt_counts['tn']} FN={gt_counts['fn']}\n"
            f"  file_accuracy={gt_acc:.3f} file_precision={gt_prec:.3f} file_recall={gt_rec:.3f}\n"
            f"  hits={gt_event_counts['hits']} misses={gt_event_counts['misses']} false_alarms={gt_event_counts['false_alarms']}\n"
            f"  latency_mean={latency_mean:.3f}s latency_median={latency_median:.3f}s"
        )

    if args.plots:
        print(f"[OK] wrote plots under: {plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
