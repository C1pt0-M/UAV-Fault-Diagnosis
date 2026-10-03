"""Topology change pilot: verify whether fault onset induces a stable
change in inter-channel dependency structure.

For each fault case:
  A_pre  = correlation matrix of 3s window ending just before fault onset
  A_post = correlation matrix of 3s window starting at fault onset
  Delta_A = A_post - A_pre

For each normal case:
  Use two adjacent non-overlapping 3s windows from mid-flight as pseudo pre/post.
  This establishes the baseline "natural variability" of Delta_A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "packages" / "uav_fault"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from uav_fault.data.build_sequence_cache import SEQ_COLUMN_ALIASES

CONTRACT_PATH = Path(r"D:\UAV-Data\feature_contract_v1.json")
SPLIT_PATH = Path(
    r"D:\UAV-Data\splits\rflymad_case_split_v2\rflymad_case_level_split_manifest.csv"
)
OUTPUT_DIR = Path(r"D:\UAV-Data\experiments\topology_change_pilot")

WINDOW_FRAMES = 150  # 3 seconds at 50 Hz
N_CASES_PER_CLASS = 50
SEED = 42


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def extract_features(path: Path, domain: str, contract: dict) -> np.ndarray:
    """Extract 26-channel feature matrix from a case CSV."""
    features = contract["features"]
    n_features = len(features)
    feature_index = {item["canonical_name"]: item["index"] for item in features}

    # Determine source columns needed
    source_to_canonical = {}
    for item in features:
        src = item["source_columns"][domain]
        if src not in ("derived", "derived_from_euler_rad"):
            source_to_canonical[src] = item["canonical_name"]

    df = pd.read_csv(path, usecols=list(source_to_canonical.keys()), low_memory=False)
    output = np.empty((len(df), n_features), dtype=np.float64)

    for src_col, canon_name in source_to_canonical.items():
        idx = feature_index[canon_name]
        output[:, idx] = pd.to_numeric(df[src_col], errors="coerce").to_numpy(dtype=np.float64)

    # Derived: acc_norm
    ax = output[:, feature_index["accel_x"]]
    ay = output[:, feature_index["accel_y"]]
    az = output[:, feature_index["accel_z"]]
    output[:, feature_index["acc_norm"]] = np.sqrt(ax**2 + ay**2 + az**2)

    # baro_alt stays absolute here; we don't do window-relative for correlation
    # (relative would kill the cross-channel correlation signal with altitude)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output


def fault_onset_row(path: Path) -> int | None:
    """Find the first row where fault state becomes active."""
    priority = [
        "UAVState_data_fault_state",
        "TrueState_data_fault_state",
        "_rfly_ctrl_lxl_0_id",
    ]
    df = pd.read_csv(path, low_memory=False)
    for col in priority:
        if col not in df.columns:
            continue
        if col in ("UAVState_data_fault_state", "TrueState_data_fault_state"):
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mask = values > 0
        else:
            ids = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
            mask = ((ids >= 123450) & (ids <= 123459)) | (
                (ids >= 123540) & (ids <= 123549)
            )
        indices = np.flatnonzero(mask.to_numpy())
        if indices.size > 0:
            return int(indices[0])
    return None


def correlation_matrix(window: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix (26x26)."""
    # Standardize each channel
    std = window.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0  # avoid division by zero for constant channels
    centered = (window - window.mean(axis=0, keepdims=True)) / std
    corr = (centered.T @ centered) / window.shape[0]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def compute_delta_a(
    features: np.ndarray, onset: int, window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute A_pre, A_post, Delta_A around fault onset."""
    pre_end = onset
    pre_start = pre_end - window_size
    post_start = onset
    post_end = post_start + window_size

    if pre_start < 0 or post_end > len(features):
        return None

    a_pre = correlation_matrix(features[pre_start:pre_end])
    a_post = correlation_matrix(features[post_start:post_end])
    delta_a = a_post - a_pre
    return a_pre, a_post, delta_a


def compute_normal_delta_a(
    features: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """For normal cases: two adjacent windows from mid-flight."""
    mid = len(features) // 2
    start1 = mid - window_size
    end1 = mid
    start2 = mid
    end2 = mid + window_size

    if start1 < 0 or end2 > len(features):
        return None

    a1 = correlation_matrix(features[start1:end1])
    a2 = correlation_matrix(features[start2:end2])
    delta_a = a2 - a1
    return a1, a2, delta_a


def select_cases(manifest: pd.DataFrame, n_per_class: int, seed: int) -> dict[str, pd.DataFrame]:
    """Select n cases per class from SIL train for pilot."""
    sil_train = manifest[
        (manifest["split"] == "train") & (manifest["domain"] == "SIL")
    ]
    selected = {}
    for label in ("normal", "motor", "sensor"):
        pool = sil_train[sil_train["level2_label"] == label]
        n = min(n_per_class, len(pool))
        selected[label] = pool.sample(n=n, random_state=seed)
    return selected


def main() -> int:
    contract = load_contract()
    manifest = pd.read_csv(SPLIT_PATH, dtype=str).fillna("")
    cases = select_cases(manifest, N_CASES_PER_CLASS, SEED)
    feature_names = [item["canonical_name"] for item in contract["features"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {"normal": [], "motor": [], "sensor": []}
    delta_a_by_class: dict[str, list[np.ndarray]] = {
        "normal": [], "motor": [], "sensor": []
    }

    for label, case_df in cases.items():
        print(f"\nProcessing {label}: {len(case_df)} cases", flush=True)
        for _, row in case_df.iterrows():
            path = Path(row["final_output_path"])
            features = extract_features(path, row["domain"], contract)

            if label == "normal":
                result = compute_normal_delta_a(features, WINDOW_FRAMES)
            else:
                onset = fault_onset_row(path)
                if onset is None:
                    continue
                result = compute_delta_a(features, onset, WINDOW_FRAMES)

            if result is None:
                continue

            a_pre, a_post, delta_a = result
            frob_norm = float(np.linalg.norm(delta_a, "fro"))
            max_abs_change = float(np.abs(delta_a).max())
            mean_abs_change = float(np.abs(delta_a).mean())

            results[label].append({
                "case_uid": row["case_uid"],
                "level3_label": row["level3_label"],
                "flight_status": row["flight_status"],
                "frobenius_norm": frob_norm,
                "max_abs_change": max_abs_change,
                "mean_abs_change": mean_abs_change,
            })
            delta_a_by_class[label].append(delta_a)

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("TOPOLOGY CHANGE PILOT RESULTS")
    print("=" * 60)

    summary = {}
    for label in ("normal", "motor", "sensor"):
        if not results[label]:
            print(f"  {label}: NO VALID CASES")
            continue
        frob_norms = [item["frobenius_norm"] for item in results[label]]
        mean_abs = [item["mean_abs_change"] for item in results[label]]
        summary[label] = {
            "n_cases": len(results[label]),
            "frobenius_norm": {
                "mean": float(np.mean(frob_norms)),
                "std": float(np.std(frob_norms)),
                "median": float(np.median(frob_norms)),
                "min": float(np.min(frob_norms)),
                "max": float(np.max(frob_norms)),
            },
            "mean_abs_change": {
                "mean": float(np.mean(mean_abs)),
                "std": float(np.std(mean_abs)),
            },
        }
        print(f"\n  {label} (n={len(results[label])})")
        print(f"    Frobenius norm of ΔA: {np.mean(frob_norms):.4f} ± {np.std(frob_norms):.4f}")
        print(f"    Mean |ΔA| per edge:   {np.mean(mean_abs):.6f} ± {np.std(mean_abs):.6f}")

    # Compute mean ΔA per class
    mean_delta_a = {}
    for label in ("normal", "motor", "sensor"):
        if delta_a_by_class[label]:
            mean_delta_a[label] = np.mean(delta_a_by_class[label], axis=0)

    # Effect size: motor vs normal, sensor vs normal
    print("\n" + "-" * 60)
    print("EFFECT SIZE (Cohen's d of Frobenius norms)")
    if "normal" in summary and "motor" in summary:
        d_motor = (
            summary["motor"]["frobenius_norm"]["mean"]
            - summary["normal"]["frobenius_norm"]["mean"]
        ) / max(
            np.sqrt(
                (summary["motor"]["frobenius_norm"]["std"] ** 2
                 + summary["normal"]["frobenius_norm"]["std"] ** 2) / 2
            ),
            1e-8,
        )
        print(f"  motor vs normal: d = {d_motor:.3f}")
        summary["effect_size_motor_vs_normal"] = d_motor

    if "normal" in summary and "sensor" in summary:
        d_sensor = (
            summary["sensor"]["frobenius_norm"]["mean"]
            - summary["normal"]["frobenius_norm"]["mean"]
        ) / max(
            np.sqrt(
                (summary["sensor"]["frobenius_norm"]["std"] ** 2
                 + summary["normal"]["frobenius_norm"]["std"] ** 2) / 2
            ),
            1e-8,
        )
        print(f"  sensor vs normal: d = {d_sensor:.3f}")
        summary["effect_size_sensor_vs_normal"] = d_sensor

    if "motor" in summary and "sensor" in summary:
        d_ms = (
            summary["motor"]["frobenius_norm"]["mean"]
            - summary["sensor"]["frobenius_norm"]["mean"]
        ) / max(
            np.sqrt(
                (summary["motor"]["frobenius_norm"]["std"] ** 2
                 + summary["sensor"]["frobenius_norm"]["std"] ** 2) / 2
            ),
            1e-8,
        )
        print(f"  motor vs sensor: d = {d_ms:.3f}")
        summary["effect_size_motor_vs_sensor"] = d_ms

    # Save
    (OUTPUT_DIR / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for label in ("normal", "motor", "sensor"):
        if results[label]:
            pd.DataFrame(results[label]).to_csv(
                OUTPUT_DIR / f"cases_{label}.csv", index=False
            )
        if label in mean_delta_a:
            np.save(OUTPUT_DIR / f"mean_delta_a_{label}.npy", mean_delta_a[label])

    # Save feature names for visualization
    (OUTPUT_DIR / "feature_names.json").write_text(
        json.dumps(feature_names) + "\n", encoding="utf-8"
    )

    print("\n" + "-" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
