"""Build the online binary detection cache from the 26-channel window manifest.

Replaces the old path (build_online_cache.py -> build_sequence_cache.SEQ_COLUMNS),
which carried its own hard-coded 31-channel list and a separate case-level selection.
Here the window manifest is the single source of truth, so the online and offline
tasks are guaranteed to share one feature set, one split discipline and one set of
train-only normalisation statistics.

Task: binary. normal -> 0, motor|sensor -> 1. Open-set fault windows
(level2_label == unknown) are excluded, because the online detector is a closed-set
normal/fault discriminator.

Labels are taken from the manifest's *window-level* level2_label, which already
handles fault onset: pre-fault windows inside a fault case are labelled normal.

Splits map straight onto the manifest:
    train          -> train.pt
    val            -> val.pt
    test           -> test.pt
    external_test  -> real_holdout_test.pt      (Real; the honest deployment test)

train_online_detector.py detects real_holdout_test.pt automatically and reports it as
the external test.

Output layout (default D:\\UAV-Data\\online_cache_v2):
    train.pt / val.pt / test.pt / real_holdout_test.pt   {'x': (N, 26, 150), 'y': (N,)}
    norm_stats.json                                      train-only stats, schema v2
    build_summary.json                                   provenance and counts
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfly_feature_contract import (  # noqa: E402
    contract_columns,
    load_contract,
    read_case_features,
    window_batch,
)

sys.path.insert(0, str(REPO / "packages" / "uav_fault"))
from uav_fault.telemetry.schema_adapter import canonicalize_rfly_frame  # noqa: E402

DEFAULT_MANIFEST = Path(r"D:\UAV-Data\window_manifest_v1\window_manifest.csv")
DEFAULT_NORM_STATS = Path(r"D:\UAV-Data\window_manifest_v1\norm_stats_train_v1.json")
DEFAULT_OUTPUT = Path(r"D:\UAV-Data\online_cache_v2")

# The 40 verified RflySim live normal sessions (takeoff-hover-land, vertical, box,
# mixed; quality_status=pass). They are the only samples from the live telemetry
# distribution, and without them the live path reported fault_prob ~1 on normal
# flights (see docs/autodl_rfly40_retrain.md, acceptance item 4).
DEFAULT_EXTRA_NORMAL_DIR = Path(r"D:\UAV-Data(old)\extra_normals")
EXTRA_NORMAL_GLOB = "nf_*.csv"

MANIFEST_COLUMNS = [
    "case_uid",
    "domain",
    "split",
    "level2_label",
    "window_usage",
    "phase",
    "window_start_row",
    "window_end_row",
    "source_csv",
]

SPLIT_TO_FILE = {
    "train": "train.pt",
    "val": "val.pt",
    "test": "test.pt",
    "external_test": "real_holdout_test.pt",
}

NORMAL_LABEL = "normal"
FAULT_LABELS = {"motor", "sensor"}

CHUNK_ROWS = 500_000


def load_closed_set_windows(manifest_path: Path) -> pd.DataFrame:
    """Read only the closed-set normal/fault windows of the window manifest."""
    keep: list[pd.DataFrame] = []
    reader = pd.read_csv(
        manifest_path,
        usecols=MANIFEST_COLUMNS,
        dtype={
            "case_uid": "string",
            "domain": "category",
            "split": "category",
            "level2_label": "category",
            "window_usage": "category",
            "phase": "category",
            "source_csv": "string",
        },
        chunksize=CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        mask = chunk["level2_label"].isin([NORMAL_LABEL, *FAULT_LABELS]) & chunk[
            "split"
        ].isin(list(SPLIT_TO_FILE))
        if mask.any():
            keep.append(chunk.loc[mask].copy())
    if not keep:
        raise RuntimeError("no closed-set windows found in the window manifest")
    return pd.concat(keep, ignore_index=True)


def subsample_per_case(frame: pd.DataFrame, max_per_case: int) -> pd.DataFrame:
    """Evenly thin each case's windows so the phase mix survives the cap.

    Windows come out of the manifest in time order per case, so taking an even stride
    keeps pre-fault, transition and stable-fault windows rather than only the first N.
    """
    if max_per_case <= 0:
        return frame
    out = []
    for _, group in frame.groupby("case_uid", observed=True, sort=False):
        ordered = group.sort_values("window_start_row")
        if len(ordered) <= max_per_case:
            out.append(ordered)
            continue
        picks = np.unique(
            np.linspace(0, len(ordered) - 1, max_per_case).round().astype(int)
        )
        out.append(ordered.iloc[picks])
    return pd.concat(out, ignore_index=True) if out else frame.iloc[:0]


def build_split(
    frame: pd.DataFrame,
    contract: dict,
    columns: list[str],
    *,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Materialise one split as (x, y) tensors plus a provenance summary."""
    by_case = frame.groupby("case_uid", observed=True, sort=False)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    domains: dict[str, int] = defaultdict(int)
    phases: dict[str, int] = defaultdict(int)
    skipped = 0

    for case_uid, group in by_case:
        source = Path(str(group["source_csv"].iloc[0]))
        domain = str(group["domain"].iloc[0])
        if not source.exists():
            skipped += 1
            if verbose:
                print(f"  MISSING {source}")
            continue
        try:
            matrix, actual_columns = read_case_features(source, domain, contract)
        except Exception as exc:
            skipped += 1
            if verbose:
                print(f"  FAILED {source.name}: {exc}")
            continue
        if actual_columns != columns:
            raise RuntimeError(f"contract column order changed for {source.name}")

        spans: list[tuple[int, int]] = []
        labels: list[int] = []
        for _, row in group.iterrows():
            start = int(row["window_start_row"])
            end = int(row["window_end_row"])
            if start < 0 or end > len(matrix):
                continue
            spans.append((start, end))
            labels.append(0 if row["level2_label"] == NORMAL_LABEL else 1)
        if not spans:
            continue

        xs.append(window_batch(matrix, columns, spans))
        ys.extend(labels)
        domains[domain] += len(spans)
        for phase, count in group["phase"].value_counts().items():
            phases[str(phase)] += int(count)

    if not xs:
        return (
            torch.zeros((0, len(columns), 150), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            {"windows": 0, "cases": 0, "skipped_cases": skipped},
        )

    x = torch.from_numpy(np.concatenate(xs, axis=0))
    y = torch.tensor(ys, dtype=torch.int64)
    summary = {
        "windows": int(x.shape[0]),
        "cases": int(len(xs)),
        "skipped_cases": skipped,
        "normal": int((y == 0).sum()),
        "fault": int((y == 1).sum()),
        "fault_ratio": float((y == 1).float().mean()) if len(y) else 0.0,
        "by_domain": dict(domains),
        "by_phase": dict(phases),
    }
    return x, y, summary


def read_live_session(path: Path, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a live-link recording into the 26 contract columns, plus its timestamps.

    Live recordings already use the bridge's canonical naming (motor1..4,
    baro_alt_meter, roll/pitch/yaw in radians, actuator_out_1..4 as PWM), so they go
    through the same adapter the live path uses. That is the point: these samples must
    carry exactly the features the detector sees at inference time.
    """
    frame = pd.read_csv(path, low_memory=False)
    matrix = np.empty((len(frame), len(columns)), dtype=np.float64)
    for i, record in enumerate(frame.to_dict(orient="records")):
        canonical, _ = canonicalize_rfly_frame(record)
        matrix[i] = [canonical[name] for name in columns]
    times = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=np.float64)
    return matrix, times


def resample_uniform(
    matrix: np.ndarray, times: np.ndarray, target_hz: float
) -> np.ndarray:
    """Put a jittery live recording on a uniform timeline.

    The recordings are not uniformly sampled: ~70% of frames are 15.9 ms apart and
    ~28% are 30.4 ms apart (63 Hz and 33 Hz alternating), averaging 49.9 Hz. Without
    resampling a 150-sample window spans anywhere from 2.4 s to 4.5 s, while a training
    window is exactly 3 s. Interpolating onto a 50 Hz grid restores that equivalence.
    """
    finite = np.isfinite(times)
    if finite.sum() < 2:
        return matrix
    times = times[finite]
    matrix = matrix[finite]
    order = np.argsort(times)
    times, matrix = times[order], matrix[order]
    keep = np.concatenate(([True], np.diff(times) > 0))
    times, matrix = times[keep], matrix[keep]
    if len(times) < 2:
        return matrix

    count = int(np.floor((times[-1] - times[0]) * target_hz)) + 1
    if count < 2:
        return matrix
    grid = times[0] + np.arange(count) / target_hz
    out = np.empty((count, matrix.shape[1]), dtype=np.float64)
    for channel in range(matrix.shape[1]):
        out[:, channel] = np.interp(grid, times, matrix[:, channel])
    return out


def airborne_mask(matrix: np.ndarray, columns: list[str], margin_m: float = 0.5) -> np.ndarray:
    """Which frames are actually flying.

    The recordings are takeoff-hover-land: ~17% of the recorded time is on the ground
    with the motors idling. Ground windows must not be labelled "normal flight", and
    the last airborne moments carry the landing impact (acc_z spikes to 40-60 m/s2).

    Altitude is the reliable discriminator. Motor PWM is not: the first segment sits at
    PWM ~1300 while still at ground level, so a PWM threshold misses it.
    """
    if "baro_alt_relative" not in columns:
        return np.ones(len(matrix), dtype=bool)
    alt = matrix[:, columns.index("baro_alt_relative")]
    if not np.isfinite(alt).any():
        return np.ones(len(matrix), dtype=bool)
    return alt > (np.nanmin(alt) + margin_m)


def build_extra_normal_windows(
    directory: Path,
    glob: str,
    columns: list[str],
    *,
    window_size: int,
    stride: int,
    max_per_file: int,
    resample_hz: float,
    exclude_ground: bool = True,
    airborne_fraction: float = 0.95,
    files: list[Path] | None = None,
) -> tuple[torch.Tensor, list[str], dict]:
    """Windows for the RflySim live normal sessions; all labelled normal.

    Returns (windows, used files, per-file window counts). Pass `files` to restrict the
    set, which is how the held-out sessions are built separately from the training ones.
    """
    if files is None:
        files = sorted(directory.glob(glob))
    empty = torch.zeros((0, len(columns), window_size), dtype=torch.float32)
    if not files:
        return empty, [], {}

    batches: list[np.ndarray] = []
    used: list[str] = []
    counts: dict[str, int] = {}
    dropped_ground = 0
    for path in files:
        matrix, times = read_live_session(path, columns)
        if resample_hz > 0:
            matrix = resample_uniform(matrix, times, resample_hz)
        flying = (
            airborne_mask(matrix, columns)
            if exclude_ground
            else np.ones(len(matrix), dtype=bool)
        )

        spans = []
        for start in range(0, len(matrix) - window_size + 1, stride):
            end = start + window_size
            if exclude_ground and flying[start:end].mean() < airborne_fraction:
                dropped_ground += 1
                continue
            spans.append((start, end))
        if max_per_file > 0 and len(spans) > max_per_file:
            picks = np.unique(
                np.linspace(0, len(spans) - 1, max_per_file).round().astype(int)
            )
            spans = [spans[i] for i in picks]
        if not spans:
            print(f"  no usable windows in {path.name} ({len(matrix)} rows)")
            continue
        batches.append(window_batch(matrix, columns, spans))
        used.append(path.name)
        counts[path.name] = len(spans)

    info = {
        "files": len(used),
        "windows": int(sum(counts.values())),
        "ground_windows_dropped": dropped_ground,
        "per_file": counts,
    }
    if not batches:
        return empty, [], info
    return torch.from_numpy(np.concatenate(batches, axis=0)), used, info


def maneuver_of(name: str) -> str:
    """nf_rflysim_hover_5m_01.csv -> hover"""
    parts = Path(name).stem.split("_")
    return parts[2] if len(parts) > 2 else "unknown"


def split_holdout(files: list[str], per_maneuver: int) -> tuple[list[str], list[str]]:
    """Hold out `per_maneuver` sessions from each maneuver for honest evaluation.

    Without this, checking that the 40 sessions no longer trigger alarms measures
    training-set performance: they were all in the train split.
    """
    if per_maneuver <= 0:
        return files, []
    groups: dict[str, list[str]] = {}
    for name in files:
        groups.setdefault(maneuver_of(name), []).append(name)
    train, holdout = [], []
    for _, names in sorted(groups.items()):
        names = sorted(names)
        holdout.extend(names[:per_maneuver])
        train.extend(names[per_maneuver:])
    return sorted(train), sorted(holdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--norm-stats", default=str(DEFAULT_NORM_STATS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--max-windows-per-case",
        type=int,
        default=64,
        help="0 keeps every closed-set window (tens of GB); 64 keeps the cache ~GB",
    )
    parser.add_argument(
        "--extra-normal-dir",
        default=str(DEFAULT_EXTRA_NORMAL_DIR),
        help="RflySim live normal sessions added to the train split; empty to skip",
    )
    parser.add_argument("--extra-normal-glob", default=EXTRA_NORMAL_GLOB)
    parser.add_argument(
        "--extra-normal-max-windows-per-file",
        type=int,
        default=128,
        help="matches the historical extra_normal_max_windows_per_file",
    )
    parser.add_argument("--extra-normal-stride", type=int, default=30)
    parser.add_argument(
        "--extra-normal-holdout-per-maneuver",
        type=int,
        default=2,
        help=(
            "sessions per maneuver reserved for evaluation (written to "
            "live_holdout.pt, never trained on). 0 disables the holdout, which makes "
            "the false-alarm check circular."
        ),
    )
    parser.add_argument(
        "--extra-normal-keep-ground",
        action="store_true",
        help=(
            "keep ground windows (motors idling, aircraft on the ground). Off by "
            "default: ~17%% of each recording is ground time and is not normal flight."
        ),
    )
    parser.add_argument(
        "--extra-normal-airborne-fraction",
        type=float,
        default=0.95,
        help="minimum fraction of a window that must be airborne to keep it",
    )
    parser.add_argument(
        "--extra-normal-resample-hz",
        type=float,
        default=50.0,
        help=(
            "resample live recordings onto a uniform grid before windowing; 0 disables. "
            "The recordings alternate between ~63 Hz and ~33 Hz, so without this a "
            "150-sample window is not 3 s."
        ),
    )
    parser.add_argument(
        "--splits",
        default=",".join(SPLIT_TO_FILE),
        help="comma-separated subset to rebuild, e.g. external_test",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    args = parser.parse_args()
    wanted = [name.strip() for name in args.splits.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in SPLIT_TO_FILE]
    if unknown:
        parser.error(f"unknown splits {unknown}; choose from {sorted(SPLIT_TO_FILE)}")

    contract = load_contract()
    columns = contract_columns(contract)

    norm_stats = json.loads(Path(args.norm_stats).read_text(encoding="utf-8"))
    if list(norm_stats["columns"]) != columns:
        raise ValueError(
            "norm stats columns do not match the feature contract; "
            f"stats={len(norm_stats['columns'])} contract={len(columns)}"
        )

    print(f"contract columns : {len(columns)}")
    print(f"window manifest  : {args.manifest}")
    manifest = load_closed_set_windows(Path(args.manifest))
    print(f"closed-set windows: {len(manifest)}")
    print(f"split distribution: {manifest['split'].value_counts().to_dict()}")

    manifest = subsample_per_case(manifest, args.max_windows_per_case)
    print(f"after per-case cap ({args.max_windows_per_case}): {len(manifest)}")

    if args.dry_run:
        print("\n[dry-run] per-split plan:")
        for split in SPLIT_TO_FILE:
            part = manifest[manifest["split"] == split]
            counts = part["level2_label"].value_counts().to_dict()
            print(
                f"  {split:14s} windows={len(part):7d} "
                f"cases={part['case_uid'].nunique():5d} labels={counts}"
            )
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge with an existing summary so a long build can be run split by split.
    summary_path = output_dir / "build_summary.json"
    summary: dict[str, dict] = {}
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(previous.get("splits", {}))
        except (json.JSONDecodeError, OSError):
            pass
    for split, filename in SPLIT_TO_FILE.items():
        if split not in wanted:
            continue
        part = manifest[manifest["split"] == split]
        if part.empty:
            print(f"\n[{split}] no windows, skipped")
            continue
        print(f"\n[{split}] {len(part)} windows over {part['case_uid'].nunique()} cases")
        x, y, info = build_split(part, contract, columns, verbose=True)

        if split == "train" and args.extra_normal_dir:
            extra_dir = Path(args.extra_normal_dir)
            all_sessions = sorted(extra_dir.glob(args.extra_normal_glob))
            train_files, holdout_files = split_holdout(
                [p.name for p in all_sessions],
                args.extra_normal_holdout_per_maneuver,
            )
            by_name = {p.name: p for p in all_sessions}
            common = dict(
                window_size=150,
                stride=args.extra_normal_stride,
                max_per_file=args.extra_normal_max_windows_per_file,
                resample_hz=args.extra_normal_resample_hz,
                exclude_ground=not args.extra_normal_keep_ground,
                airborne_fraction=args.extra_normal_airborne_fraction,
            )
            extra_x, used_files, extra_info = build_extra_normal_windows(
                extra_dir,
                args.extra_normal_glob,
                columns,
                files=[by_name[name] for name in train_files if name in by_name],
                **common,
            )

            if extra_x.shape[0]:
                x = torch.cat([x, extra_x], dim=0)
                y = torch.cat([y, torch.zeros(extra_x.shape[0], dtype=torch.int64)])
                info["manifest_windows"] = info["windows"]
                info["windows"] = int(x.shape[0])
                info["extra_normal_dir"] = str(extra_dir)
                info["extra_normal"] = {
                    "windows": int(extra_x.shape[0]),
                    "files": extra_info["files"],
                    "ground_windows_dropped": extra_info["ground_windows_dropped"],
                    "holdout_files": len(holdout_files),
                }
                info["normal"] = int((y == 0).sum())
                info["fault"] = int((y == 1).sum())
                info["fault_ratio"] = float((y == 1).float().mean())
                print(
                    f"  + {extra_x.shape[0]} live normal windows from "
                    f"{extra_info['files']} sessions "
                    f"({extra_info['ground_windows_dropped']} dropped as ground)"
                )
            else:
                print(f"  no extra normals found in {extra_dir}")

            # Held-out live sessions: written separately so the false-alarm check runs
            # on sessions the model never saw.
            if holdout_files:
                hold_x, hold_used, hold_info = build_extra_normal_windows(
                    extra_dir,
                    args.extra_normal_glob,
                    columns,
                    files=[by_name[name] for name in holdout_files if name in by_name],
                    **common,
                )
                if hold_x.shape[0]:
                    # Per-window maneuver labels, so the acceptance report can be broken
                    # down by flight profile. per_file preserves file iteration order.
                    groups: list[str] = []
                    for name, count in hold_info["per_file"].items():
                        groups.extend([maneuver_of(name)] * count)
                    torch.save(
                        {
                            "x": hold_x,
                            "y": torch.zeros(hold_x.shape[0], dtype=torch.int64),
                            "groups": groups,
                        },
                        output_dir / "live_holdout.pt",
                    )
                    by_maneuver: dict[str, int] = {}
                    for name, count in hold_info["per_file"].items():
                        key = maneuver_of(name)
                        by_maneuver[key] = by_maneuver.get(key, 0) + count
                    summary["live_holdout"] = {
                        "windows": int(hold_x.shape[0]),
                        "files": hold_info["files"],
                        "file_names": sorted(hold_used),
                        "by_maneuver": by_maneuver,
                        "ground_windows_dropped": hold_info["ground_windows_dropped"],
                        "label": "all normal (live RflySim sessions)",
                    }
                    print(
                        f"  holdout: {hold_x.shape[0]} windows from "
                        f"{hold_info['files']} sessions -> live_holdout.pt"
                    )

        torch.save({"x": x, "y": y}, output_dir / filename)
        summary[split] = info
        print(
            f"  -> {filename}  x={tuple(x.shape)} "
            f"normal={info['normal']} fault={info['fault']} "
            f"({info['fault_ratio'] * 100:.1f}% fault)"
        )

    # Normalisation statistics are reused verbatim: they were fitted on train windows
    # only, which is exactly what the online cache needs.
    stats = {
        "columns": columns,
        "mean": list(norm_stats["mean"]),
        "std": list(norm_stats["std"]),
        "window_size": 150,
        "target_hz": 50.0,
        "stride": int(norm_stats.get("stride", 5)),
        "feature_schema": "rfly_seq_columns_v2",
        "protocol_name": "rflymad_online_binary_v2",
        "label_space": "normal_fault",
        "source_fault_classes": sorted(FAULT_LABELS),
        "source_norm_stats": str(args.norm_stats),
        "sampling": {
            "mode": "even_per_case",
            "max_windows_per_case": args.max_windows_per_case,
            "manifest_stride_frames": int(norm_stats.get("stride", 5)),
        },
    }
    (output_dir / "norm_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "online_cache_v2",
                "manifest": str(args.manifest),
                "max_windows_per_case": args.max_windows_per_case,
                "extra_normal_dir": args.extra_normal_dir,
                "extra_normal_glob": args.extra_normal_glob,
                "extra_normal_stride_frames": args.extra_normal_stride,
                "extra_normal_resample_hz": args.extra_normal_resample_hz,
                "extra_normal_exclude_ground": not args.extra_normal_keep_ground,
                "extra_normal_airborne_fraction": args.extra_normal_airborne_fraction,
                "extra_normal_holdout_per_maneuver": args.extra_normal_holdout_per_maneuver,
                "extra_normal_max_windows_per_file": args.extra_normal_max_windows_per_file,
                "extra_normal_split": "train",
                "columns": columns,
                "splits": summary,
                "notes": (
                    "Derived from the 26-channel window manifest. Window-level labels: "
                    "normal->0, motor|sensor->1, open-set (unknown) excluded. "
                    "Pre-fault windows inside fault cases stay labelled normal. "
                    "RflySim live normal sessions are appended to train only."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 66)
    print("ONLINE CACHE BUILT")
    print("=" * 66)
    for split, info in summary.items():
        if split not in SPLIT_TO_FILE:
            continue
        print(
            f"  {split:16s} windows={info['windows']:7d}  cases={info['cases']:5d}  "
            f"normal={info['normal']:7d}  fault={info['fault']:7d}  "
            f"skipped={info['skipped_cases']}"
        )
    print(f"\nSaved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
