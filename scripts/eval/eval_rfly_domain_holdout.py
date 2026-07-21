from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "packages" / "uav_fault"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from uav_fault.online.fault_detector import FaultDetector  # noqa: E402
from uav_fault.telemetry.schema_adapter import (  # noqa: E402
    assess_canonical_input_contract,
    canonicalize_rfly_records,
    normalize_time_columns,
)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(truth)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _fault_mask(frame: pd.DataFrame) -> np.ndarray:
    if "is_fault_active" in frame.columns:
        return pd.to_numeric(frame["is_fault_active"], errors="coerce").fillna(0).to_numpy() > 0
    if "flight_phase" in frame.columns:
        return frame["flight_phase"].astype(str).eq("fault").to_numpy()
    for column in ("UAVState_data_fault_state", "TrueState_data_fault_state"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy() > 0
    return np.zeros(len(frame), dtype=bool)


def _time_values(frame: pd.DataFrame) -> np.ndarray:
    normalized = normalize_time_columns(frame)
    if "time_s" in normalized.columns:
        values = pd.to_numeric(normalized["time_s"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).all():
            return values
    return np.arange(len(frame), dtype=float) / 50.0


def _first_consecutive_time(
    values: np.ndarray,
    times: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    count: int,
) -> float | None:
    run = 0
    for value, time_value, allowed in zip(values, times, eligible):
        run = run + 1 if allowed and value >= threshold else 0
        if run >= count:
            return float(time_value)
    return None


def evaluate_file(
    path: Path,
    detector: FaultDetector,
    *,
    every: int,
    point_threshold: float,
    onset_threshold: float,
    k_on: int,
) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(path)
    records = canonicalize_rfly_records(frame)
    active = _fault_mask(frame)
    times = _time_values(frame)
    predictions = detector.predict_records_batch(records, every_n=every, batch_size=256)

    rows = []
    for end, prediction in predictions:
        start = end - detector.window_size
        contract = assess_canonical_input_contract(records[start:end])
        if contract.get("detection_blocked"):
            continue
        ratio = float(active[start:end].mean())
        rows.append(
            {
                "end": int(end),
                "time_s": float(times[end - 1]),
                "fault_prob": float(prediction["fault_prob"]),
                "fault_ratio": ratio,
                "gt_window_fault": int(ratio > 0.5),
                "gt_endpoint_fault": int(active[end - 1]),
                "clean_normal_window": int(ratio == 0.0),
            }
        )
    points = pd.DataFrame(rows)
    fault_class = "normal" if not active.any() else path.parent.name
    result = {
        "file": path.name,
        "fault_class": fault_class,
        "rows": len(frame),
        "inferences": len(points),
        "status": "ok" if len(points) else "no_inferences",
    }
    if points.empty:
        return result, points

    probs = points["fault_prob"].to_numpy(dtype=float)
    pred = probs >= point_threshold
    clean = points["clean_normal_window"].to_numpy(dtype=bool)
    endpoint_fault = points["gt_endpoint_fault"].to_numpy(dtype=bool)
    window_fault = points["gt_window_fault"].to_numpy(dtype=bool)
    point_times = points["time_s"].to_numpy(dtype=float)

    window_metrics = _binary_metrics(window_fault, pred)
    result.update(
        {
            "mean_fault_prob": float(probs.mean()),
            "max_fault_prob": float(probs.max()),
            "clean_normal_windows": int(clean.sum()),
            "clean_normal_false_positive_rate": float(pred[clean].mean()) if clean.any() else None,
            "clean_normal_mean_fault_prob": float(probs[clean].mean()) if clean.any() else None,
            "active_windows": int(endpoint_fault.sum()),
            "active_window_recall": float(pred[endpoint_fault].mean()) if endpoint_fault.any() else None,
            "active_mean_fault_prob": float(probs[endpoint_fault].mean()) if endpoint_fault.any() else None,
            "window_label_accuracy": window_metrics["accuracy"],
            "window_label_f1": window_metrics["f1"],
        }
    )

    if active.any():
        fault_start = float(times[np.flatnonzero(active)[0]])
        fault_end = float(times[np.flatnonzero(active)[-1]])
        pre_fault = point_times < fault_start
        during_fault = (point_times >= fault_start) & (point_times <= fault_end)
        false_alarm_time = _first_consecutive_time(
            probs, point_times, pre_fault, onset_threshold, k_on
        )
        detection_time = _first_consecutive_time(
            probs, point_times, during_fault, onset_threshold, k_on
        )
        result.update(
            {
                "fault_start_s": fault_start,
                "fault_end_s": fault_end,
                "pre_fault_false_alarm": int(false_alarm_time is not None),
                "first_pre_fault_alarm_s": false_alarm_time,
                "fault_detected": int(detection_time is not None),
                "first_detection_s": detection_time,
                "detection_latency_s": (
                    float(detection_time - fault_start) if detection_time is not None else None
                ),
            }
        )
    else:
        alarm_time = _first_consecutive_time(
            probs, point_times, np.ones(len(points), dtype=bool), onset_threshold, k_on
        )
        result.update(
            {
                "pre_fault_false_alarm": int(alarm_time is not None),
                "first_pre_fault_alarm_s": alarm_time,
                "fault_detected": 0,
            }
        )
    return result, points


def _aggregate(model: str, rows: pd.DataFrame, points: pd.DataFrame) -> list[dict]:
    output = []
    for class_name in ("normal", "motor", "sensor", "fault_all", "all"):
        if class_name == "fault_all":
            selected_rows = rows[rows["fault_class"].isin(["motor", "sensor"])]
        elif class_name == "all":
            selected_rows = rows
        else:
            selected_rows = rows[rows["fault_class"] == class_name]
        selected_points = points[points["file"].isin(selected_rows["file"])]
        if selected_rows.empty or selected_points.empty:
            continue
        y_true = selected_points["gt_window_fault"].to_numpy(dtype=int)
        y_pred = (selected_points["fault_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
        metrics = _binary_metrics(y_true, y_pred)
        clean = selected_points["clean_normal_window"].to_numpy(dtype=bool)
        active = selected_points["gt_endpoint_fault"].to_numpy(dtype=bool)
        latencies = pd.to_numeric(selected_rows.get("detection_latency_s"), errors="coerce").dropna()
        output.append(
            {
                "model": model,
                "class": class_name,
                "files": len(selected_rows),
                "windows": len(selected_points),
                "accuracy": metrics["accuracy"],
                "precision_fault": metrics["precision"],
                "recall_fault": metrics["recall"],
                "f1_fault": metrics["f1"],
                "clean_normal_fpr": float(y_pred[clean].mean()) if clean.any() else None,
                "active_window_recall": float(y_pred[active].mean()) if active.any() else None,
                "files_detected": int(pd.to_numeric(selected_rows["fault_detected"], errors="coerce").fillna(0).sum()),
                "files_with_pre_fault_alarm": int(pd.to_numeric(selected_rows["pre_fault_false_alarm"], errors="coerce").fillna(0).sum()),
                "median_detection_latency_s": float(latencies.median()) if len(latencies) else None,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay RflySim domain normal and held-out fault cases.")
    parser.add_argument("--bundle-root", default=r"D:\UAV-Data\rfly_domain_bundle_v1")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--include-split", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--point-threshold", type=float, default=0.5)
    parser.add_argument("--onset-threshold", type=float, default=0.6)
    parser.add_argument("--k-on", type=int, default=3)
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root)
    models_root = Path(args.models_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_manifest:
        split_frame = pd.read_csv(args.split_manifest)
        if not {"file", "split"}.issubset(split_frame.columns):
            raise ValueError("split manifest must contain file and split columns")
        if args.include_split:
            split_frame = split_frame[split_frame["split"] == args.include_split]
        files = [bundle_root / str(item) for item in split_frame["file"].tolist()]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"split manifest references missing files: {missing}")
    else:
        files = [
            *sorted((bundle_root / "normal").glob("nf_*.csv")),
            *sorted((bundle_root / "motor").glob("*.csv")),
            *sorted((bundle_root / "sensor").glob("*.csv")),
        ]
    model_dirs = sorted(path for path in models_root.glob("online_detector_*_rfly") if path.is_dir())
    all_aggregate = []
    for model_dir in model_dirs:
        detector = FaultDetector(str(model_dir), device=args.device, window_size=None)
        model = str(detector.config["model"])
        file_rows = []
        point_frames = []
        for path in files:
            row, points = evaluate_file(
                path,
                detector,
                every=args.every,
                point_threshold=args.point_threshold,
                onset_threshold=args.onset_threshold,
                k_on=args.k_on,
            )
            file_rows.append(row)
            if not points.empty:
                points.insert(0, "fault_class", row["fault_class"])
                points.insert(0, "file", path.name)
                point_frames.append(points)
        file_frame = pd.DataFrame(file_rows)
        point_frame = pd.concat(point_frames, ignore_index=True)
        file_frame.to_csv(output_dir / f"{model}_files.csv", index=False)
        point_frame.to_csv(output_dir / f"{model}_windows.csv", index=False)
        all_aggregate.extend(_aggregate(model, file_frame, point_frame))

    aggregate = pd.DataFrame(all_aggregate)
    aggregate.to_csv(output_dir / "held_out_replay_summary.csv", index=False)
    (output_dir / "held_out_replay_summary.json").write_text(
        json.dumps(all_aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(aggregate.to_string(index=False))
    print(f"[OK] wrote replay outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
