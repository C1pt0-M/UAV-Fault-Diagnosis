"""Build a three-class, contract-driven window cache for offline topology experiments.

Reuses the binary builder's manifest selection and deterministic per-case cap, but
preserves normal/motor/sensor labels and per-window case/domain provenance. No live
normal recordings or Real adaptation data enter training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_online_cache_from_windows import load_closed_set_windows, subsample_per_case, SPLIT_TO_FILE
from rfly_feature_contract import load_contract, contract_columns, read_case_features, window_batch

LABEL_MAP = {"normal": 0, "motor": 1, "sensor": 2}
DEFAULT_ROOT = Path(r"D:\UAV-Data")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_partitions(cases: pd.DataFrame) -> None:
    if cases["case_uid"].duplicated().any():
        raise ValueError("duplicate case_uid in case split manifest")
    paired = cases[cases["simulation_pair_key"].fillna("") != ""]
    if (paired.groupby("simulation_pair_key")["split"].nunique() > 1).any():
        raise ValueError("a SIL/HIL simulation pair crosses partitions")
    real = cases["domain"].str.upper() == "REAL"
    if (cases.loc[real, "split"] != "external_test").any():
        raise ValueError("Real cases must remain external_test only")


def materialize(frame: pd.DataFrame, contract: dict) -> dict:
    columns = contract_columns(contract)
    xs, labels, case_ids, domains, pair_keys = [], [], [], [], []
    starts = []
    for case_uid, group in frame.groupby("case_uid", sort=False, observed=True):
        if group["source_csv"].nunique() != 1 or group["domain"].nunique() != 1:
            raise ValueError(f"inconsistent source/domain for {case_uid}")
        matrix, actual = read_case_features(Path(str(group["source_csv"].iloc[0])),
                                            str(group["domain"].iloc[0]), contract)
        if actual != columns:
            raise ValueError(f"channel order mismatch for {case_uid}")
        spans = []
        for row in group.itertuples(index=False):
            start, end = int(row.window_start_row), int(row.window_end_row)
            if start < 0 or end > len(matrix) or end - start != 150:
                raise ValueError(f"invalid window {case_uid}: [{start}, {end})")
            spans.append((start, end))
            labels.append(LABEL_MAP[row.level2_label])
            case_ids.append(str(case_uid))
            domains.append(str(row.domain).upper())
            pair_keys.append(str(row.simulation_pair_key))
            starts.append(start)
        windows = window_batch(matrix, columns, spans)
        if not np.isfinite(windows).all():
            raise ValueError(f"nonfinite windows for {case_uid}")
        xs.append(windows)
    if not xs:
        raise ValueError("empty split")
    return {
        "x": torch.from_numpy(np.concatenate(xs)),
        "y": torch.tensor(labels, dtype=torch.long),
        "case_ids": case_ids, "domains": domains, "pair_keys": pair_keys,
        "window_start_rows": torch.tensor(starts, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "window_manifest_v1/window_manifest.csv")
    parser.add_argument("--case-split", type=Path, default=DEFAULT_ROOT / "splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv")
    parser.add_argument("--contract", type=Path, default=DEFAULT_ROOT / "feature_contract_v1.json")
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_ROOT / "window_manifest_v1/norm_stats_train_v1.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "topology_cache_v1")
    parser.add_argument("--max-windows-per-case", type=int, default=64)
    parser.add_argument("--splits", nargs="+", choices=list(SPLIT_TO_FILE), default=list(SPLIT_TO_FILE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_windows_per_case <= 0:
        parser.error("--max-windows-per-case must be positive")
    contract = load_contract(args.contract)
    columns = contract_columns(contract)
    if len(columns) != 26 or contract["window_size_frames"] != 150 or contract["sampling_rate_hz"] != 50:
        raise ValueError("expected the 26-channel / 150-frame / 50-Hz contract")
    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    if stats["columns"] != columns or stats.get("fit_partition") != "train":
        raise ValueError("normalization must match the contract and be fit on train only")
    if stats.get("feature_contract_sha256") != sha256(args.contract):
        raise ValueError("normalization contract hash mismatch")
    if stats.get("split_manifest_sha256") != sha256(args.case_split):
        raise ValueError("normalization case split hash mismatch")
    for key in ("mean", "std"):
        values = np.asarray(stats[key])
        if values.shape != (26,) or not np.isfinite(values).all():
            raise ValueError(f"invalid normalization {key}")
    if (np.asarray(stats["std"]) <= 0).any():
        raise ValueError("normalization std must be positive")
    cases = pd.read_csv(args.case_split, dtype=str).fillna("")
    validate_partitions(cases)
    print("loading window manifest...", flush=True)
    frame = load_closed_set_windows(args.manifest)
    joined = frame.merge(cases[["case_uid", "split", "domain", "simulation_pair_key"]],
                         on="case_uid", how="left", validate="many_to_one", suffixes=("", "_case"))
    if joined["split_case"].isna().any() or (joined["split"] != joined["split_case"]).any():
        raise ValueError("window splits differ from authoritative case splits")
    if (joined["domain"].str.upper() != joined["domain_case"].str.upper()).any():
        raise ValueError("window domains differ from case split manifest")
    frame = subsample_per_case(joined, args.max_windows_per_case)
    plans = {}
    for split in SPLIT_TO_FILE:
        part = frame[frame["split"] == split]
        plans[split] = {"windows": len(part), "cases": part["case_uid"].nunique(),
                        "labels": part["level2_label"].value_counts().to_dict()}
        print(f"{split}: {plans[split]}", flush=True)
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": "topology_cache_v1", "label_map": LABEL_MAP,
        "manifest": str(args.manifest), "case_split_sha256": sha256(args.case_split),
        "contract_sha256": sha256(args.contract), "norm_stats_sha256": sha256(args.norm_stats),
        "max_windows_per_case": args.max_windows_per_case,
        "sampling": "even_per_case", "real_role": "external_test_only", "live_normals": False,
    }
    summary_path = args.output_dir / "build_summary.json"
    summary = {**provenance, "splits": {}}
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError("existing cache uses different provenance; choose a new output directory")
        summary["splits"].update(previous["splits"])
    for split in args.splits:
        if (args.output_dir / SPLIT_TO_FILE[split]).exists() and not args.overwrite:
            raise FileExistsError(f"{split} already exists; use --overwrite explicitly")
    for split in args.splits:
        part = frame[frame["split"] == split]
        print(f"building {split}...", flush=True)
        payload = materialize(part, contract)
        target = args.output_dir / SPLIT_TO_FILE[split]
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        summary["splits"][split] = plans[split]
        norm = {**stats, "feature_schema": "rfly_seq_columns_v2", "window_size": 150,
                "target_hz": 50.0, "label_names": list(LABEL_MAP), "label_space": "normal_motor_sensor"}
        (args.output_dir / "norm_stats.json").write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
        tmp_summary = summary_path.with_suffix(".json.tmp")
        tmp_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        tmp_summary.replace(summary_path)
        print(f"saved {target}: {tuple(payload['x'].shape)}", flush=True)
        del payload


if __name__ == "__main__":
    main()
