from __future__ import annotations

import numpy as np


def infer_fault_active_from_df(df) -> np.ndarray | None:
    if "is_fault_active" in df.columns:
        return (np.asarray(df["is_fault_active"], dtype=float) > 0.5).astype(bool)
    if "flight_phase" in df.columns:
        return (df["flight_phase"].astype(str) == "fault").to_numpy(dtype=bool)
    for col in ("UAVState_data_fault_state", "TrueState_data_fault_state"):
        if col in df.columns:
            return np.asarray(df[col], dtype=float) > 0
    for col in ("rfly_ctrl_lxl_slash_send_data_id", "_rfly_ctrl_lxl_0_id"):
        if col in df.columns:
            ids = np.nan_to_num(np.asarray(df[col], dtype=float), nan=0.0).astype(np.int64)
            return ((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))
    return None


def time_array(df) -> np.ndarray:
    if "time_s" not in df.columns or len(df) == 0:
        return np.arange(len(df), dtype=float)
    try:
        t = np.asarray(df["time_s"], dtype=float)
    except Exception:
        return np.arange(len(df), dtype=float)
    if t.shape[0] != len(df) or not np.all(np.isfinite(t)):
        return np.arange(len(df), dtype=float)
    return t


def event_spans_from_mask(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, active in enumerate(mask.astype(bool)):
        if active and start_idx is None:
            start_idx = idx
        elif (not active) and start_idx is not None:
            spans.append((float(times[start_idx]), float(times[idx - 1])))
            start_idx = None
    if start_idx is not None and len(mask) > 0:
        spans.append((float(times[start_idx]), float(times[len(mask) - 1])))
    return spans


def gt_event_metrics(df, events) -> dict[str, float | int | None]:
    fault_mask = infer_fault_active_from_df(df)
    if fault_mask is None or len(df) == 0:
        return {
            "gt_has_labels": 0,
            "gt_fault": None,
            "gt_event_count": None,
            "gt_onset_t": None,
            "gt_total_event_s": None,
            "pred_event_count": int(len(events)),
            "hit": None,
            "miss": None,
            "false_alarm": None,
            "first_detection_time": None,
            "detection_latency_s": None,
        }
    mask = np.asarray(fault_mask, dtype=bool)
    times = time_array(df)
    gt_spans = event_spans_from_mask(mask, times)
    gt_fault = bool(gt_spans)
    gt_onset_t = float(gt_spans[0][0]) if gt_spans else None
    first_detection_time = float(events[0].start_t) if events else None
    hit = False
    if gt_spans and events:
        for pred in events:
            for gt_start, gt_end in gt_spans:
                if float(pred.end_t) >= gt_start and float(pred.start_t) <= gt_end:
                    hit = True
                    break
            if hit:
                break
    false_alarm = bool(events) and not hit
    miss = gt_fault and not hit
    latency = None
    if gt_onset_t is not None and first_detection_time is not None:
        latency = float(first_detection_time - gt_onset_t)
    return {
        "gt_has_labels": 1,
        "gt_fault": int(gt_fault),
        "gt_event_count": int(len(gt_spans)),
        "gt_onset_t": gt_onset_t,
        "gt_total_event_s": float(sum(max(0.0, end - start) for start, end in gt_spans)),
        "pred_event_count": int(len(events)),
        "hit": int(hit),
        "miss": int(bool(miss)),
        "false_alarm": int(bool(false_alarm)),
        "first_detection_time": first_detection_time,
        "detection_latency_s": latency,
    }
