"""Build the channel-topology graph used by the topology-guided fault diagnosis model.

Outputs (all under D:\\UAV-Data\\experiments\\topology_graph):

  adjacency_normal_mean.npy   26x26  mean Pearson correlation over normal train windows
  adjacency_normal_std.npy    26x26  per-edge std of the same (edge-level reference scale)
  adjacency_normal_count.npy  26x26  how many windows contributed to each edge
  adjacency_physical.npy      26x26  physical prior edges (domain knowledge)
  adjacency_fused.npy         26x26  normalised combination of the two
  edge_table.csv              edge list with data-driven value, std and prior flag
  feature_names.json          canonical channel order
  summary.json                provenance: split, domains, cases, windows

Design notes
------------
The pilot (topology_change_pilot.py) showed:
  * motor faults  -> large GLOBAL change in the dependency structure (d = 1.27)
  * sensor faults -> GLOBAL change indistinguishable from normal jitter (d = -0.05),
                     but SPARSE LOCAL edges change a lot (up to 145x)
Therefore the graph is built from NORMAL flight only, and the model must be able to
detect *local edge* deviation rather than rely on a global summary. Keeping the
per-edge std is what makes that possible.

Pearson correlation is invariant to the constant shift introduced by window-relative
altitude preprocessing, so the raw altitude column is fine here.
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

# Reuse the verified feature extraction from the pilot so the contract handling
# stays in exactly one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology_change_pilot import extract_features, load_contract  # noqa: E402

CONTRACT_PATH = Path(r"D:\UAV-Data\feature_contract_v1.json")
SPLIT_PATH = Path(
    r"D:\UAV-Data\splits\rflymad_case_split_v2\rflymad_case_level_split_manifest.csv"
)
WINDOW_MANIFEST_PATH = Path(r"D:\UAV-Data\window_manifest_v1\window_manifest.csv")
OUTPUT_DIR = Path(r"D:\UAV-Data\experiments\topology_graph")

WINDOW_FRAMES = 150
MAX_CASES_PER_DOMAIN = 120
WINDOW_SAMPLE_STEP = 10  # use every N-th window of a case
SEED = 42
MANIFEST_CHUNK_ROWS = 500_000

MANIFEST_COLUMNS = [
    "case_uid",
    "domain",
    "split",
    "level2_label",
    "window_usage",
    "window_start_row",
    "window_end_row",
    "source_csv",
]

# ---------------------------------------------------------------------------
# Physical prior edges.
#
# The first version of this list paired ctrl_i with out_i one-to-one. Normal-flight
# data rejected that: the off-diagonal ctrl/out correlations are STRONGER than the
# diagonal (mean |corr| 0.351 vs 0.320), which is exactly what a quadrotor mixing
# matrix predicts (every motor is a signed combination of the four commands).
#
# The list below is the structure that survived inspection of the data:
#
#   ctrl x gyro       ctrl_1<->gyro_x -0.564 | ctrl_2<->gyro_y -0.507 | ctrl_3<->gyro_z -0.380
#                     clean near-diagonal rate-control law: the strongest structure present
#   ctrl_4 x out_*    +0.555 .. +0.642 for all four motors: collective thrust raises all
#   ctrl_4 x accel_z  -0.433 and x acc_norm +0.446: thrust drives vertical acceleration
#   mag x quaternion  mag_x<->att_q_y -0.553, mag_y<->att_q_x +0.394, mag_z<->att_q_y +0.493
#   out x out         +0.22 .. +0.41: rotors couple through the airframe
#
# Tier 1: physically expected AND observable while hovering (this graph).
# Tier 2: physically real but dormant in hover, because attitude is nearly constant
#         so quaternion correlations vanish (gyro<->att_q_x was 0.021). Kept separate
#         because faults excite exactly these modes.
# ---------------------------------------------------------------------------
PHYSICAL_EDGES_TIER1: dict[str, list[tuple[str, str]]] = {
    # roll/pitch/yaw command -> corresponding body rate
    "rate_command_to_rate": [
        ("actuator_ctrl_1", "gyro_x"),
        ("actuator_ctrl_2", "gyro_y"),
        ("actuator_ctrl_3", "gyro_z"),
    ],
    # collective thrust reaches every motor (mixing matrix)
    "thrust_to_motors": [
        ("actuator_ctrl_4", f"actuator_out_{i}") for i in range(1, 5)
    ],
    # differential commands reach every motor too (mixing matrix, signed)
    "differential_command_to_motors": [
        (f"actuator_ctrl_{i}", f"actuator_out_{j}")
        for i in (1, 2, 3)
        for j in range(1, 5)
    ],
    # thrust -> vertical acceleration
    "thrust_to_vertical_accel": [
        ("actuator_ctrl_4", "accel_z"),
        ("actuator_ctrl_4", "acc_norm"),
    ],
    # rotors are mechanically coupled through the airframe
    "rotor_coupling": [
        (f"actuator_out_{i}", f"actuator_out_{j}")
        for i in range(1, 5)
        for j in range(i + 1, 5)
    ],
    # magnetometer senses the earth field in the body frame -> attitude
    "magnetic_attitude": [
        ("mag_x", "att_q_y"),
        ("mag_y", "att_q_x"),
        ("mag_z", "att_q_y"),
        ("mag_z", "att_q_z"),
    ],
    # acc_norm is derived from the accelerometer
    "derived_acc_norm": [
        (f"accel_{a}", "acc_norm") for a in ("x", "y", "z")
    ],
}

# Physically real, but not observable in a hover window: attitude barely changes,
# so a correlation over 3 s is dominated by noise. They should become informative
# once a fault excites the rate/attitude modes.
PHYSICAL_EDGES_TIER2: dict[str, list[tuple[str, str]]] = {
    # body rates integrate into attitude
    "rate_to_attitude": [
        (f"gyro_{g}", f"att_q_{q}")
        for g, q in (("x", "x"), ("y", "y"), ("z", "z"), ("z", "w"))
    ],
    # gravity projects onto the accelerometer through attitude
    "gravity_projection": [
        (f"accel_{a}", f"att_q_{q}")
        for a in ("x", "y", "z")
        for q in ("w", "x", "y", "z")
    ],
    # barometric altitude and vertical velocity describe the same motion
    "altitude_velocity": [("baro_alt_relative", "vel_z")],
    # horizontal motion shows up in the accelerometer
    "velocity_accel": [
        (f"vel_{v}", f"accel_{a}") for v in ("x", "y", "z") for a in ("x", "y", "z")
    ],
    # gyro and accelerometer sit on the same IMU and share vibration
    "imu_consistency": [
        (f"gyro_{g}", f"accel_{a}") for g in ("x", "y", "z") for a in ("x", "y", "z")
    ],
}


def correlation_matrix(window: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of a (frames, channels) window."""
    std = window.std(axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    centered = (window - window.mean(axis=0, keepdims=True)) / std
    corr = (centered.T @ centered) / window.shape[0]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def load_normal_train_windows() -> pd.DataFrame:
    """Read only the normal/train rows of the window manifest (the file is ~940 MB)."""
    keep: list[pd.DataFrame] = []
    reader = pd.read_csv(
        WINDOW_MANIFEST_PATH,
        usecols=MANIFEST_COLUMNS,
        dtype={
            "case_uid": "string",
            "domain": "category",
            "split": "category",
            "level2_label": "category",
            "window_usage": "category",
            "source_csv": "string",
        },
        chunksize=MANIFEST_CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        mask = (
            (chunk["split"] == "train")
            & (chunk["level2_label"] == "normal")
            & (chunk["window_usage"] == "normal")
        )
        if mask.any():
            keep.append(chunk.loc[mask].copy())
    if not keep:
        raise RuntimeError("no normal/train windows found in the window manifest")
    return pd.concat(keep, ignore_index=True)


def select_normal_cases(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cap the number of cases per domain so extraction stays cheap."""
    frames = []
    rng = np.random.default_rng(seed)
    for domain, group in pool.groupby("domain", observed=True):
        cases = group["case_uid"].unique()
        if len(cases) > MAX_CASES_PER_DOMAIN:
            cases = rng.choice(cases, size=MAX_CASES_PER_DOMAIN, replace=False)
        selected = group[group["case_uid"].isin(set(cases))]
        frames.append(selected)
        print(f"  {domain}: {len(cases)} cases, {len(selected)} windows")
    return pd.concat(frames, ignore_index=True) if frames else pool.iloc[:0]


def build_physical_adjacency(
    names: list[str], groups: dict[str, list[tuple[str, str]]]
) -> tuple[np.ndarray, dict[str, int]]:
    index = {name: i for i, name in enumerate(names)}
    adjacency = np.zeros((len(names), len(names)), dtype=np.float64)
    edge_counts: dict[str, int] = {}
    for group, edges in groups.items():
        added = 0
        for left, right in edges:
            if left not in index or right not in index:
                raise KeyError(f"unknown channel in physical edge: {left} - {right}")
            i, j = index[left], index[right]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
            added += 1
        edge_counts[group] = added
    return adjacency, edge_counts


def prior_vs_data_report(
    edge_table: pd.DataFrame, tier1_pairs: set[frozenset], tier2_pairs: set[frozenset]
) -> dict:
    """Quantify how well the prior lines up with the measured normal-flight structure."""
    pairs_of = lambda frame: [
        frozenset((row["left"], row["right"])) for _, row in frame.iterrows()
    ]
    edge_pairs = pairs_of(edge_table)
    rows = []
    for name, pairs in (("tier1", tier1_pairs), ("tier2", tier2_pairs)):
        mask = [p in pairs for p in edge_pairs]
        subset = edge_table[mask]
        rows.append(
            {
                "tier": name,
                "pairs": int(len(subset)),
                "mean_abs_corr": float(subset["data_abs_mean"].mean()),
                "median_abs_corr": float(subset["data_abs_mean"].median()),
                "frac_above_0.15": float((subset["data_abs_mean"] > 0.15).mean()),
                "frac_above_0.30": float((subset["data_abs_mean"] > 0.30).mean()),
            }
        )
    known = tier1_pairs | tier2_pairs
    outside = edge_table[[p not in known for p in edge_pairs]]
    rows.append(
        {
            "tier": "not_in_prior",
            "pairs": int(len(outside)),
            "mean_abs_corr": float(outside["data_abs_mean"].mean()),
            "median_abs_corr": float(outside["data_abs_mean"].median()),
            "frac_above_0.15": float((outside["data_abs_mean"] > 0.15).mean()),
            "frac_above_0.30": float((outside["data_abs_mean"] > 0.30).mean()),
        }
    )

    ranked = edge_table.sort_values("data_abs_mean", ascending=False).reset_index(drop=True)
    topk = {}
    for k in (20, 40, 80, 100):
        hits = sum(1 for p in pairs_of(ranked.head(k)) if p in tier1_pairs)
        topk[f"top{k}_tier1_hits"] = int(hits)
    return {"tiers": rows, "topk_overlap": topk}


def main() -> int:
    contract = load_contract()
    names = [item["canonical_name"] for item in contract["features"]]
    n = len(names)

    manifest = load_normal_train_windows()
    print(f"Normal train windows available: {len(manifest)}")
    print("Selecting normal train cases by domain:")
    cases = select_normal_cases(manifest, SEED)

    sums = np.zeros((n, n), dtype=np.float64)
    sums_sq = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    total_windows = 0
    skipped = 0

    by_source = cases.groupby("source_csv")
    print(f"\nProcessing {len(by_source)} source CSVs ...")
    for source_csv, group in by_source:
        path = Path(source_csv)
        if not path.exists():
            print(f"  MISSING: {source_csv}")
            skipped += 1
            continue
        domain = str(group["domain"].iloc[0])
        try:
            features = extract_features(path, domain, contract)
        except Exception as exc:  # keep going, report at the end
            print(f"  FAILED {path.name}: {exc}")
            skipped += 1
            continue

        starts = group["window_start_row"].astype(int).to_numpy()
        ends = group["window_end_row"].astype(int).to_numpy()
        picked = 0
        for start, end in zip(starts[::WINDOW_SAMPLE_STEP], ends[::WINDOW_SAMPLE_STEP]):
            if start < 0 or end > len(features):
                continue
            corr = correlation_matrix(features[start:end])
            sums += corr
            sums_sq += corr * corr
            counts += 1.0
            picked += 1
            total_windows += 1
        if picked == 0:
            print(f"  no usable windows for {path.name}")

    if total_windows == 0:
        print("No windows accumulated; aborting.")
        return 1

    denominator = np.maximum(counts, 1.0)
    mean = sums / denominator
    variance = np.maximum(sums_sq / denominator - mean * mean, 0.0)
    std = np.sqrt(variance)
    np.fill_diagonal(mean, 1.0)
    np.fill_diagonal(std, 0.0)

    physical_t1, tier1_groups = build_physical_adjacency(names, PHYSICAL_EDGES_TIER1)
    physical_t2, tier2_groups = build_physical_adjacency(names, PHYSICAL_EDGES_TIER2)
    physical = np.clip(physical_t1 + physical_t2, 0.0, 1.0)

    # Fused adjacency: data-driven magnitude gated by physical prior, normalised to [0, 1].
    abs_mean = np.abs(mean)
    data_norm = abs_mean / max(abs_mean.max(), 1e-12)
    prior_gate = 0.5 + 0.5 * physical  # prior edges get a boost, others are kept
    fused = data_norm * prior_gate
    np.fill_diagonal(fused, 0.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "adjacency_normal_mean.npy", mean)
    np.save(OUTPUT_DIR / "adjacency_normal_std.npy", std)
    np.save(OUTPUT_DIR / "adjacency_normal_count.npy", counts)
    np.save(OUTPUT_DIR / "adjacency_physical_tier1.npy", physical_t1)
    np.save(OUTPUT_DIR / "adjacency_physical_tier2.npy", physical_t2)
    np.save(OUTPUT_DIR / "adjacency_physical.npy", physical)
    np.save(OUTPUT_DIR / "adjacency_fused.npy", fused)
    (OUTPUT_DIR / "feature_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Edge table (upper triangle) for inspection and plotting.
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "left": names[i],
                    "right": names[j],
                    "data_mean": float(mean[i, j]),
                    "data_abs_mean": float(abs(mean[i, j])),
                    "data_std": float(std[i, j]),
                    "n_windows": float(counts[i, j]),
                    "physical_tier1": float(physical_t1[i, j]),
                    "physical_tier2": float(physical_t2[i, j]),
                    "physical_prior": float(physical[i, j]),
                    "fused": float(fused[i, j]),
                }
            )
    edge_table = pd.DataFrame(rows).sort_values("data_abs_mean", ascending=False)
    edge_table.to_csv(OUTPUT_DIR / "edge_table.csv", index=False)

    tier1_pairs = {
        frozenset(p) for edges in PHYSICAL_EDGES_TIER1.values() for p in edges
    }
    tier2_pairs = {
        frozenset(p) for edges in PHYSICAL_EDGES_TIER2.values() for p in edges
    }
    prior_report = prior_vs_data_report(edge_table, tier1_pairs, tier2_pairs)

    top = edge_table.head(12)
    print("\n" + "=" * 68)
    print("TOPOLOGY GRAPH BUILT")
    print("=" * 68)
    print(f"  cases processed : {len(by_source) - skipped}")
    print(f"  cases skipped   : {skipped}")
    print(f"  windows used    : {total_windows}")
    print(f"  channels        : {n}")
    print(f"  edges (pairs)   : {n * (n - 1) // 2}")
    print("\n  Tier 1 prior groups (physically expected AND hover-observable):")
    for group, count in tier1_groups.items():
        print(f"    {group:32s} {count}")
    print(f"    -> tier1 pairs total: {int(physical_t1.sum() // 2)}")
    print("\n  Tier 2 prior groups (physically real, dormant in hover):")
    for group, count in tier2_groups.items():
        print(f"    {group:32s} {count}")
    print(f"    -> tier2 pairs total: {int(physical_t2.sum() // 2)}")
    print("\n  Strongest data-driven couplings (|corr| on normal flight):")
    for _, row in top.iterrows():
        mark = ""
        if row["physical_tier1"]:
            mark = " [tier1]"
        elif row["physical_tier2"]:
            mark = " [tier2]"
        print(
            f"    {row['left']:20s} <-> {row['right']:20s} "
            f"corr={row['data_mean']:+.3f} std={row['data_std']:.3f}{mark}"
        )

    print("\n" + "-" * 68)
    print("PRIOR vs MEASURED STRUCTURE")
    print("-" * 68)
    for row in prior_report["tiers"]:
        print(
            f"  {row['tier']:12s} pairs={row['pairs']:4d}  "
            f"mean|corr|={row['mean_abs_corr']:.4f}  "
            f">0.15: {row['frac_above_0.15'] * 100:5.1f}%  "
            f">0.30: {row['frac_above_0.30'] * 100:5.1f}%"
        )
    print("  top-K overlap (tier1 hits):", prior_report["topk_overlap"])

    summary = {
        "schema_version": "topology_graph_v1",
        "contract": str(CONTRACT_PATH),
        "split_manifest": str(SPLIT_PATH),
        "window_manifest": str(WINDOW_MANIFEST_PATH),
        "split": "train",
        "label_filter": "level2_label == normal",
        "window_size_frames": WINDOW_FRAMES,
        "window_sample_step": WINDOW_SAMPLE_STEP,
        "max_cases_per_domain": MAX_CASES_PER_DOMAIN,
        "seed": SEED,
        "channels": names,
        "cases_processed": len(by_source) - skipped,
        "cases_skipped": skipped,
        "windows_used": total_windows,
        "physical_prior_pairs": int(physical.sum() // 2),
        "physical_prior_tier1_pairs": int(physical_t1.sum() // 2),
        "physical_prior_tier2_pairs": int(physical_t2.sum() // 2),
        "physical_edge_groups": {"tier1": tier1_groups, "tier2": tier2_groups},
        "prior_vs_data": prior_report,
        "mean_abs_correlation": float(np.abs(mean - np.eye(n)).mean()),
        "strongest_edge": {
            "left": str(edge_table.iloc[0]["left"]),
            "right": str(edge_table.iloc[0]["right"]),
            "corr": float(edge_table.iloc[0]["data_mean"]),
        },
        "note": (
            "Graph is a NORMAL-flight reference. Sensor faults leave the global "
            "structure unchanged, so the model must use edge-level deviation."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
