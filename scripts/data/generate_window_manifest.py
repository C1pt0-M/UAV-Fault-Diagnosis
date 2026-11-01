"""Generate window-level manifest from case-level split and feature contract.

Outputs a CSV with one row per sliding window, recording:
- case identity, domain, split, labels
- window start/end row indices into the source CSV
- fault timing metadata for topology and early-detection experiments

Does NOT write individual window files to disk.
Does NOT compute normalization statistics (separate step).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_contract(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fault_column_priority():
    """Return ordered list of columns to check for fault state."""
    return [
        "UAVState_data_fault_state",
        "TrueState_data_fault_state",
        "_rfly_ctrl_lxl_0_id",
        "rfly_ctrl_lxl_slash_send_data_id",
    ]


def _infer_fault_mask(df: pd.DataFrame) -> np.ndarray:
    """Return historical binary state; reject masked repaired data in this legacy API."""
    if any(name in df.columns for name in ("labeltime_supervision_valid", "labeltime_input_clock_review")):
        raise ValueError(
            "Label-time repaired CSV requires its versioned window links and supervision mask; "
            "this legacy boolean-only reader cannot safely consume it. "
            "Use the verified train.pt/val.pt and window links in the repaired version."
        )
    for col in _fault_column_priority():
        if col not in df.columns:
            continue
        if col in ("UAVState_data_fault_state", "TrueState_data_fault_state"):
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return (values > 0).to_numpy(dtype=bool)
        else:
            # rfly_ctrl id-based
            ids = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
            mask = ((ids >= 123450) & (ids <= 123459)) | (
                (ids >= 123540) & (ids <= 123549)
            )
            return mask.to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _find_fault_onset(fault_mask: np.ndarray) -> int | None:
    """Return the first row index where fault becomes active, or None."""
    indices = np.flatnonzero(fault_mask)
    return int(indices[0]) if indices.size > 0 else None


def generate_windows_for_case(
    case_row: dict,
    contract: dict,
    *,
    window_size: int,
    stride: int,
) -> list[dict]:
    """Generate window records for a single case."""
    csv_path = Path(case_row["final_output_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    n_rows = len(df)
    if n_rows < window_size:
        return []

    # Determine fault mask and onset
    fault_mask = _infer_fault_mask(df)
    fault_onset_row = _find_fault_onset(fault_mask)

    # Case metadata
    case_uid = case_row["case_uid"]
    domain = case_row["domain"]
    split = case_row["split"]
    case_level1 = case_row["level1_label"]
    case_level2 = case_row["level2_label"]
    case_level3 = case_row["level3_label"]
    open_set_fault_type = case_row.get("open_set_fault_type", "")
    flight_status = case_row.get("flight_status", "")
    normal_window_eligible = (
        case_row.get("normal_window_eligible", "True") == "True"
    )
    closed_fault_eligible = (
        case_row.get("closed_set_fault_window_eligible", "False") == "True"
    )
    open_fault_eligible = (
        case_row.get("open_set_fault_window_eligible", "False") == "True"
    )
    normal_only = normal_window_eligible and not (
        closed_fault_eligible or open_fault_eligible
    )

    windows = []
    for start in range(0, n_rows - window_size + 1, stride):
        end = start + window_size  # exclusive

        # Window-level fault statistics
        window_fault_mask = fault_mask[start:end]
        fault_frames_in_window = int(window_fault_mask.sum())
        fault_fraction = fault_frames_in_window / window_size

        # Window label: based on endpoint (causal online convention)
        endpoint_fault = bool(fault_mask[end - 1])

        # Determine window labels and downstream role.
        if normal_only or not endpoint_fault:
            window_level1 = "normal"
            window_level2 = "normal"
            window_level3 = "normal"
            window_usage = "normal"
        elif closed_fault_eligible:
            window_level1 = "fault"
            window_level2 = case_level2
            window_level3 = case_level3
            window_usage = "closed_set_fault"
        elif open_fault_eligible:
            window_level1 = "fault"
            window_level2 = "unknown"
            window_level3 = "unknown"
            window_usage = "open_set_fault"
        else:
            raise ValueError(
                f"fault endpoint has no eligible role: {case_uid}"
            )

        # Phase classification
        if fault_onset_row is None:
            phase = "normal_flight"
        elif end <= fault_onset_row:
            phase = "pre_fault"
        elif start >= fault_onset_row:
            phase = "stable_fault"
        else:
            phase = "transition"

        # Time since fault onset (in frames)
        if fault_onset_row is not None and end > fault_onset_row:
            frames_since_onset = end - 1 - fault_onset_row
        else:
            frames_since_onset = -1  # not applicable

        windows.append({
            "case_uid": case_uid,
            "domain": domain,
            "split": split,
            "flight_status": flight_status,
            "window_start_row": start,
            "window_end_row": end,
            "window_size": window_size,
            "case_level1_label": case_level1,
            "case_level2_label": case_level2,
            "case_level3_label": case_level3,
            "open_set_fault_type": open_set_fault_type,
            "level1_label": window_level1,
            "level2_label": window_level2,
            "level3_label": window_level3,
            "window_usage": window_usage,
            "fault_fraction": round(fault_fraction, 4),
            "endpoint_fault": int(endpoint_fault),
            "phase": phase,
            "fault_onset_row": fault_onset_row if fault_onset_row is not None else -1,
            "frames_since_onset": frames_since_onset,
            "source_csv": str(csv_path),
        })

    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate window-level manifest")
    parser.add_argument(
        "--split-manifest",
        default=r"D:\UAV-Data\splits\rflymad_case_split_v2\rflymad_case_level_split_manifest.csv",
    )
    parser.add_argument(
        "--contract",
        default=r"D:\UAV-Data\feature_contract_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\UAV-Data\window_manifest_v1",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit cases for pilot run")
    parser.add_argument("--splits", default="", help="Comma-separated splits to process (empty=all)")
    args = parser.parse_args()

    contract = _load_contract(Path(args.contract))
    window_size = contract["window_size_frames"]
    stride = contract["stride_frames"]

    manifest = pd.read_csv(args.split_manifest, dtype=str).fillna("")
    if args.splits:
        selected_splits = [s.strip() for s in args.splits.split(",")]
        manifest = manifest[manifest["split"].isin(selected_splits)]

    if args.limit > 0:
        manifest = manifest.head(args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "window_manifest.csv"
    partial_path = manifest_path.with_suffix(".csv.partial")
    summary_path = output_dir / "window_manifest_summary.json"
    partial_path.unlink(missing_ok=True)

    processed = 0
    total_windows = 0
    counters = {
        "split": Counter(),
        "level1_label": Counter(),
        "level2_label": Counter(),
        "level3_label": Counter(),
        "phase": Counter(),
        "window_usage": Counter(),
        "open_set_fault_type": Counter(),
    }
    fieldnames: list[str] | None = None

    try:
        with partial_path.open("w", newline="", encoding="utf-8") as handle:
            writer = None
            for _, row in manifest.iterrows():
                case_dict = row.to_dict()
                windows = generate_windows_for_case(
                    case_dict, contract, window_size=window_size, stride=stride
                )
                if not windows:
                    raise ValueError(
                        f"case produced no windows: {case_dict.get('case_uid', '?')}"
                    )
                if writer is None:
                    fieldnames = list(windows[0].keys())
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerows(windows)
                processed += 1
                total_windows += len(windows)
                for window in windows:
                    for key, counter in counters.items():
                        value = str(window[key])
                        if key != "open_set_fault_type" or value:
                            counter[value] += 1

                if processed % 200 == 0 or processed == len(manifest):
                    print(
                        f"processed {processed}/{len(manifest)} cases, "
                        f"{total_windows} windows",
                        flush=True,
                    )
        if fieldnames is None:
            raise ValueError("no windows generated")
        partial_path.replace(manifest_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": "rflymad_window_manifest_v1",
        "cases_processed": processed,
        "cases_skipped": 0,
        "total_windows": total_windows,
        "window_size": window_size,
        "stride": stride,
        "label_policy": "causal_endpoint",
        "contract": str(args.contract),
        "split_manifest": str(args.split_manifest),
        "output_manifest": str(manifest_path),
        "by_split": dict(counters["split"]),
        "by_level1_label": dict(counters["level1_label"]),
        "by_level2_label": dict(counters["level2_label"]),
        "by_level3_label": dict(counters["level3_label"]),
        "by_phase": dict(counters["phase"]),
        "by_window_usage": dict(counters["window_usage"]),
        "by_open_set_fault_type": dict(counters["open_set_fault_type"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
