"""Compute exact train-window normalization statistics for feature contract v1.

The computation preserves the weighting induced by overlapping windows without
materializing every (window, frame, channel) value. Per-case prefix sums produce
exact sums and squared sums over the same window starts as the window manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("SIL", "HIL", "Real", "RflySim")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    features = contract.get("features", [])
    if len(features) != int(contract.get("feature_count", -1)):
        raise ValueError("feature_count does not match features")
    if [item["index"] for item in features] != list(range(len(features))):
        raise ValueError("feature indices must be contiguous and ordered")
    return contract


def _source_columns(contract: dict, domain: str) -> tuple[list[str], list[str]]:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    canonical: list[str] = []
    source: list[str] = []
    for feature in contract["features"]:
        name = feature["canonical_name"]
        column = feature["source_columns"][domain]
        if column not in {"derived", "derived_from_euler_rad"}:
            canonical.append(name)
            source.append(column)
    return canonical, source


def _extract_case_features(path: Path, domain: str, contract: dict) -> np.ndarray:
    canonical, source = _source_columns(contract, domain)
    frame = pd.read_csv(path, usecols=source, low_memory=False)
    output = np.empty((len(frame), int(contract["feature_count"])), dtype=np.float64)
    assigned: set[str] = set()

    for name, column in zip(canonical, source):
        index = next(
            item["index"]
            for item in contract["features"]
            if item["canonical_name"] == name
        )
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        output[:, index] = values
        assigned.add(name)

    derived_names = {
        item["canonical_name"]
        for item in contract["features"]
        if item["source_columns"][domain] == "derived"
    }
    if derived_names != {"acc_norm"}:
        raise ValueError(f"unexpected derived features for {domain}: {derived_names}")
    feature_index = {
        item["canonical_name"]: int(item["index"])
        for item in contract["features"]
    }
    ax = output[:, feature_index["accel_x"]]
    ay = output[:, feature_index["accel_y"]]
    az = output[:, feature_index["accel_z"]]
    output[:, feature_index["acc_norm"]] = np.sqrt(ax * ax + ay * ay + az * az)
    assigned.add("acc_norm")

    expected = {item["canonical_name"] for item in contract["features"]}
    if assigned != expected:
        raise ValueError(f"unassigned features: {sorted(expected - assigned)}")
    if not np.isfinite(output).all():
        bad = np.argwhere(~np.isfinite(output))[0]
        name = contract["features"][int(bad[1])]["canonical_name"]
        raise ValueError(f"non-finite feature {name} at row {int(bad[0])}: {path}")
    return output


def _window_starts(n_rows: int, window_size: int, stride: int) -> np.ndarray:
    if n_rows < window_size:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)


def aggregate_case_windows(
    values: np.ndarray,
    *,
    window_size: int,
    stride: int,
    relative_feature_index: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return windows, scalar observations, sums, and squared sums per feature."""
    starts = _window_starts(len(values), window_size, stride)
    n_windows = int(len(starts))
    n_features = values.shape[1]
    if n_windows == 0:
        return 0, 0, np.zeros(n_features), np.zeros(n_features)

    ends = starts + window_size
    prefix = np.vstack(
        [np.zeros((1, n_features), dtype=np.float64), np.cumsum(values, axis=0)]
    )
    prefix_sq = np.vstack(
        [np.zeros((1, n_features), dtype=np.float64), np.cumsum(values * values, axis=0)]
    )
    window_sums = prefix[ends] - prefix[starts]
    window_sums_sq = prefix_sq[ends] - prefix_sq[starts]

    # baro_alt_relative[t] = raw[t] - raw[window_start]
    baseline = values[starts, relative_feature_index]
    raw_sum = window_sums[:, relative_feature_index].copy()
    raw_sum_sq = window_sums_sq[:, relative_feature_index].copy()
    window_sums[:, relative_feature_index] = raw_sum - window_size * baseline
    window_sums_sq[:, relative_feature_index] = (
        raw_sum_sq
        - 2.0 * baseline * raw_sum
        + window_size * baseline * baseline
    )

    observations = n_windows * window_size
    return (
        n_windows,
        observations,
        window_sums.sum(axis=0),
        window_sums_sq.sum(axis=0),
    )


def _expected_train_windows(summary_path: Path) -> int:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary["by_split"]["train"])


def compute_stats(
    *,
    split_manifest: Path,
    contract_path: Path,
    window_summary_path: Path,
    output_dir: Path,
) -> dict:
    contract = _load_contract(contract_path)
    window_size = int(contract["window_size_frames"])
    stride = int(contract["stride_frames"])
    feature_names = [item["canonical_name"] for item in contract["features"]]
    relative_indices = [
        int(item["index"])
        for item in contract["features"]
        if item.get("preprocessing") == "window_relative_subtract_first"
    ]
    if len(relative_indices) != 1:
        raise ValueError(f"expected one window-relative feature, found {relative_indices}")
    relative_index = relative_indices[0]

    manifest = pd.read_csv(split_manifest, dtype=str).fillna("")
    manifest = manifest[manifest["split"] == "train"].copy()
    if manifest.empty:
        raise ValueError("split manifest contains no train cases")
    if manifest["case_uid"].duplicated().any():
        raise ValueError("duplicate train case_uid")

    n_features = len(feature_names)
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sum_sq = np.zeros(n_features, dtype=np.float64)
    total_windows = 0
    total_observations = 0
    total_source_rows = 0
    cases_by_domain: dict[str, int] = {}
    windows_by_domain: dict[str, int] = {}
    case_rows: list[dict] = []

    for number, (_, row) in enumerate(manifest.iterrows(), 1):
        domain = str(row["domain"])
        path = Path(row["final_output_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        values = _extract_case_features(path, domain, contract)
        n_windows, observations, sums, sums_sq = aggregate_case_windows(
            values,
            window_size=window_size,
            stride=stride,
            relative_feature_index=relative_index,
        )
        if n_windows == 0:
            raise ValueError(f"train case shorter than one window: {path}")
        total_windows += n_windows
        total_observations += observations
        total_source_rows += len(values)
        total_sum += sums
        total_sum_sq += sums_sq
        cases_by_domain[domain] = cases_by_domain.get(domain, 0) + 1
        windows_by_domain[domain] = windows_by_domain.get(domain, 0) + n_windows
        case_rows.append(
            {
                "case_uid": row["case_uid"],
                "domain": domain,
                "source_csv": str(path),
                "source_rows": len(values),
                "windows": n_windows,
                "window_observations": observations,
            }
        )
        if number % 200 == 0 or number == len(manifest):
            print(
                f"processed {number}/{len(manifest)} train cases, "
                f"{total_windows} windows",
                flush=True,
            )

    expected_windows = _expected_train_windows(window_summary_path)
    if total_windows != expected_windows:
        raise ValueError(
            f"train window count mismatch: computed={total_windows}, "
            f"manifest_summary={expected_windows}"
        )
    if total_observations != total_windows * window_size:
        raise AssertionError("window observation count is inconsistent")

    mean = total_sum / total_observations
    variance = total_sum_sq / total_observations - mean * mean
    tiny_negative = (variance < 0) & (variance > -1e-9)
    variance[tiny_negative] = 0.0
    if (variance < 0).any():
        bad = [feature_names[index] for index in np.flatnonzero(variance < 0)]
        raise ValueError(f"negative variances after aggregation: {bad}")
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite normalization statistics")
    constant = [feature_names[index] for index in np.flatnonzero(std <= 1e-8)]
    if constant:
        raise ValueError(f"constant train-window features: {constant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "schema_version": "rflymad_feature_norm_v1",
        "feature_contract_version": contract["version"],
        "feature_contract_sha256": _sha256(contract_path),
        "split_protocol": "rflymad_case_split_v2",
        "split_manifest_sha256": _sha256(split_manifest),
        "window_manifest_summary_sha256": _sha256(window_summary_path),
        "fit_partition": "train",
        "method": "z_score_population",
        "window_weighting": "every overlapping window frame contributes once",
        "window_size_frames": window_size,
        "stride_frames": stride,
        "train_cases": int(len(manifest)),
        "train_windows": total_windows,
        "train_window_observations": total_observations,
        "source_rows_read": total_source_rows,
        "cases_by_domain": cases_by_domain,
        "windows_by_domain": windows_by_domain,
        "columns": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    stats_path = output_dir / "norm_stats_train_v1.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    audit_rows = []
    for index, feature in enumerate(contract["features"]):
        audit_rows.append(
            {
                "index": index,
                "feature": feature["canonical_name"],
                "unit": feature["unit"],
                "preprocessing": feature["preprocessing"],
                "mean": mean[index],
                "std": std[index],
                "variance": variance[index],
                "constant": bool(std[index] <= 1e-8),
                "finite": bool(np.isfinite(mean[index]) and np.isfinite(std[index])),
            }
        )
    pd.DataFrame(audit_rows).to_csv(output_dir / "norm_stats_train_v1_audit.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "norm_stats_train_v1_case_inventory.csv", index=False)

    summary = {
        "status": "pass",
        "stats_path": str(stats_path),
        "train_cases": int(len(manifest)),
        "train_windows": total_windows,
        "expected_train_windows": expected_windows,
        "train_window_observations": total_observations,
        "source_rows_read": total_source_rows,
        "feature_count": n_features,
        "constant_features": constant,
        "non_finite_features": [],
        "cases_by_domain": cases_by_domain,
        "windows_by_domain": windows_by_domain,
    }
    (output_dir / "norm_stats_train_v1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        default=r"D:\UAV-Data\splits\rflymad_case_split_v2\rflymad_case_level_split_manifest.csv",
    )
    parser.add_argument(
        "--contract", default=r"D:\UAV-Data\feature_contract_v1.json"
    )
    parser.add_argument(
        "--window-summary",
        default=r"D:\UAV-Data\window_manifest_v1\window_manifest_summary.json",
    )
    parser.add_argument(
        "--output-dir", default=r"D:\UAV-Data\window_manifest_v1"
    )
    args = parser.parse_args()
    summary = compute_stats(
        split_manifest=Path(args.split_manifest),
        contract_path=Path(args.contract),
        window_summary_path=Path(args.window_summary),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
