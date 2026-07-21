from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UAV_FAULT_ROOT = PROJECT_ROOT / "packages" / "uav_fault"
if str(UAV_FAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(UAV_FAULT_ROOT))

from uav_fault.online.fault_detector import FaultDetector
from uav_fault.telemetry.schema_adapter import (
    assess_canonical_input_contract,
    adapt_px4_live_frame,
    canonicalize_rfly_records,
)


def _records_for_backend(df: pd.DataFrame, backend: str) -> list[dict[str, Any]]:
    rows = df.to_dict(orient="records")
    if backend == "px4":
        return [adapt_px4_live_frame(row)[0] for row in rows]
    return canonicalize_rfly_records(df)


def _infer_hz(records: list[dict[str, Any]]) -> float | None:
    times = np.asarray([row.get("time_s") for row in records], dtype=float)
    if times.size < 3 or not np.all(np.isfinite(times)):
        return None
    dt = np.diff(times)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    return float(1.0 / median_dt) if median_dt > 0 else None


def _fault_state_summary(df: pd.DataFrame) -> str:
    for column in ("is_fault_active", "flight_phase", "UAVState_data_fault_state", "TrueState_data_fault_state"):
        if column in df.columns:
            counts = df[column].value_counts(dropna=False).head(8).to_dict()
            return f"{column}:{counts}"
    return ""


def _top_feature_text(columns: list[str], values: np.ndarray, *, limit: int) -> str:
    parts = []
    for idx in np.argsort(values)[::-1][:limit]:
        parts.append(f"{columns[int(idx)]}={float(values[int(idx)]):.3f}")
    return ";".join(parts)


def _window_rows(
    *,
    detector: FaultDetector,
    records: list[dict[str, Any]],
    features_z_abs: np.ndarray,
    every_n: int,
    columns: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for end, pred in detector.predict_records_batch(records, every_n=every_n):
        window_z = features_z_abs[end - detector.window_size : end].mean(axis=0)
        top_idx = int(window_z.argmax())
        contract = assess_canonical_input_contract(records[end - detector.window_size : end])
        rows.append(
            {
                "frame_end": int(end),
                "time_s": float(records[end - 1].get("time_s", end)),
                "fault_prob": float(pred.get("fault_prob", 0.0)),
                "label": pred.get("label"),
                "mean_abs_z": float(window_z.mean()),
                "top_z_feature": columns[top_idx],
                "top_z": float(window_z[top_idx]),
                "input_contract_detection_blocked": int(bool(contract.get("detection_blocked"))),
                "input_contract_zero_signal_groups": ",".join(contract.get("zero_signal_groups", [])),
            }
        )
    return rows


def diagnose_file(
    *,
    path: Path,
    detector: FaultDetector,
    columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    backend: str,
    every_n: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = pd.read_csv(path)
    records = _records_for_backend(df, backend)
    contract = assess_canonical_input_contract(records)
    features = detector._build_feature_matrix(records)
    features_z_abs = np.abs((features - mean[np.newaxis, :]) / std[np.newaxis, :])
    full_z = features_z_abs.mean(axis=0) if len(features_z_abs) else np.zeros(len(columns))
    windows = _window_rows(
        detector=detector,
        records=records,
        features_z_abs=features_z_abs,
        every_n=every_n,
        columns=columns,
    )

    summary: dict[str, Any] = {
        "file": str(path),
        "n_rows": int(len(df)),
        "sample_hz": _infer_hz(records),
        "time_start_s": float(records[0].get("time_s")) if records else None,
        "time_end_s": float(records[-1].get("time_s")) if records else None,
        "fault_state_summary": _fault_state_summary(df),
        "input_contract_status": contract.get("status"),
        "input_contract_reason": contract.get("reason"),
        "input_contract_detection_blocked": int(bool(contract.get("detection_blocked"))),
        "input_contract_zero_signal_groups": ",".join(contract.get("zero_signal_groups", [])),
        "full_file_top_z": _top_feature_text(columns, full_z, limit=top_k),
        "n_windows": int(len(windows)),
    }
    if windows:
        probs = np.asarray([row["fault_prob"] for row in windows], dtype=float)
        blocked = np.asarray(
            [row["input_contract_detection_blocked"] for row in windows], dtype=float
        )
        max_idx = int(probs.argmax())
        max_row = windows[max_idx]
        summary.update(
            {
                "first_fault_prob": float(probs[0]),
                "mean_fault_prob": float(probs.mean()),
                "max_fault_prob": float(probs[max_idx]),
                "p_fault_prob_ge_05": float(np.mean(probs >= 0.5)),
                "n_blocked_windows": int(blocked.sum()),
                "p_blocked_windows": float(blocked.mean()),
                "max_fault_time_s": max_row["time_s"],
                "max_window_top_z_feature": max_row["top_z_feature"],
                "max_window_top_z": max_row["top_z"],
            }
        )
    return summary, windows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose live telemetry schema drift against an online detector contract."
    )
    parser.add_argument("--csv", required=True, nargs="+", help="Telemetry CSV file(s).")
    parser.add_argument("--model-dir", default="assets/models/online_detector_kg_mstca_rfly")
    parser.add_argument("--backend", choices=["replay", "px4"], default="replay")
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="runtime/live_schema_drift_summary.csv")
    parser.add_argument("--windows-out", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    stats = json.loads((model_dir / "norm_stats.json").read_text(encoding="utf-8"))
    columns = list(stats["columns"])
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    detector = FaultDetector(str(model_dir), device=args.device)

    summaries = []
    all_windows = []
    for item in args.csv:
        path = Path(item)
        summary, windows = diagnose_file(
            path=path,
            detector=detector,
            columns=columns,
            mean=mean,
            std=std,
            backend=args.backend,
            every_n=int(args.every),
            top_k=int(args.top_k),
        )
        summaries.append(summary)
        for row in windows:
            all_windows.append({"file": str(path), **row})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_path, index=False)
    print(f"[OK] wrote summary: {out_path}")
    print(pd.DataFrame(summaries).to_string(index=False))

    if args.windows_out:
        windows_path = Path(args.windows_out)
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_windows).to_csv(windows_path, index=False)
        print(f"[OK] wrote windows: {windows_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
