from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLIT_PROTOCOL_VERSION = "2.0"
DEFAULT_SEED = "rflymad-case-split-v2-20260812"
INTERNAL_RATIO_WEIGHTS = {"train": 70, "val": 15, "test": 15}
INTERNAL_RATIOS = {
    split: weight / 100 for split, weight in INTERNAL_RATIO_WEIGHTS.items()
}
INTERNAL_SPLITS = tuple(INTERNAL_RATIOS)
ALL_SPLITS = (*INTERNAL_SPLITS, "external_test", "open_set_test")
SIMULATION_DOMAINS = {"SIL", "HIL"}

SPLIT_FIELDS = (
    "split_protocol_version",
    "split_seed",
    "split",
    "split_role",
    "split_stratum",
    "simulation_pair_key",
    "simulation_pair_status",
    "simulation_pair_group_size",
    "normalization_fit_eligible",
    "model_selection_eligible",
    "threshold_tuning_eligible",
    "final_evaluation_only",
)


def parse_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def stable_score(seed: str, stratum: str, group_key: str) -> str:
    payload = f"{seed}\0{stratum}\0{group_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def simulation_pair_key(row: dict[str, str]) -> str:
    domain = row["domain"]
    if domain not in SIMULATION_DOMAINS:
        return ""
    output_filename = row["output_filename"]
    stem = Path(output_filename).stem
    parts = stem.split("_", 1)
    if len(parts) != 2 or len(parts[1]) != 10 or not parts[1].isdigit():
        raise ValueError(f"invalid simulation output filename: {output_filename}")
    platform_code = {"SIL": "1", "HIL": "2"}[domain]
    if parts[1][0] != platform_code:
        raise ValueError(
            f"unexpected platform code for {domain}: {output_filename}"
        )
    return parts[1][1:]


def closed_set_stratum(row: dict[str, str]) -> str:
    if not parse_bool(row["closed_set_eligible"]):
        raise ValueError(f"closed-set stratum requested for open-set case: {row['case_uid']}")
    return f"{row['level3_label']}|{row['flight_status']}"


def apportion_group_counts(group_count: int) -> dict[str, int]:
    if group_count < 0:
        raise ValueError("group_count must be non-negative")
    counts = {
        split: group_count * weight // 100
        for split, weight in INTERNAL_RATIO_WEIGHTS.items()
    }
    remainders = {
        split: group_count * weight % 100
        for split, weight in INTERNAL_RATIO_WEIGHTS.items()
    }
    remaining = group_count - sum(counts.values())
    priority = {split: index for index, split in enumerate(INTERNAL_SPLITS)}
    order = sorted(
        INTERNAL_SPLITS,
        key=lambda split: (-remainders[split], priority[split]),
    )
    for split in order[:remaining]:
        counts[split] += 1
    if sum(counts.values()) != group_count:
        raise RuntimeError("split apportionment did not preserve group count")
    return counts


def rebalance_stratum_counts(
    counts_by_stratum: dict[str, dict[str, int]],
    group_counts_by_stratum: dict[str, int],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    total_groups = sum(group_counts_by_stratum.values())
    targets = apportion_group_counts(total_groups)
    current = {
        split: sum(counts[split] for counts in counts_by_stratum.values())
        for split in INTERNAL_SPLITS
    }
    split_priority = {split: index for index, split in enumerate(INTERNAL_SPLITS)}

    while current != targets:
        donors = [
            split for split in INTERNAL_SPLITS if current[split] > targets[split]
        ]
        receivers = [
            split for split in INTERNAL_SPLITS if current[split] < targets[split]
        ]
        candidates: list[tuple[int, str, int, int, str, str]] = []
        for stratum, counts in counts_by_stratum.items():
            group_count = group_counts_by_stratum[stratum]
            for donor in donors:
                if counts[donor] <= 1:
                    continue
                for receiver in receivers:
                    donor_weight = INTERNAL_RATIO_WEIGHTS[donor]
                    receiver_weight = INTERNAL_RATIO_WEIGHTS[receiver]
                    before = (
                        100 * counts[donor] - donor_weight * group_count
                    ) ** 2 + (
                        100 * counts[receiver] - receiver_weight * group_count
                    ) ** 2
                    after = (
                        100 * (counts[donor] - 1) - donor_weight * group_count
                    ) ** 2 + (
                        100 * (counts[receiver] + 1)
                        - receiver_weight * group_count
                    ) ** 2
                    candidates.append(
                        (
                            after - before,
                            stratum,
                            split_priority[donor],
                            split_priority[receiver],
                            donor,
                            receiver,
                        )
                    )
        if not candidates:
            raise RuntimeError(
                f"cannot satisfy global split targets {targets} from {current} "
                "while preserving per-stratum coverage"
            )
        _, stratum, _, _, donor, receiver = min(candidates)
        counts_by_stratum[stratum][donor] -= 1
        counts_by_stratum[stratum][receiver] += 1
        current[donor] -= 1
        current[receiver] += 1

    return counts_by_stratum, targets


def group_simulation_rows(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["domain"] not in SIMULATION_DOMAINS:
            continue
        key = simulation_pair_key(row)
        groups[key].append(row)

    for key, group in groups.items():
        if len(group) > 2:
            raise ValueError(f"simulation pair group has more than two rows: {key}")
        domains = [row["domain"] for row in group]
        if len(domains) != len(set(domains)):
            raise ValueError(f"simulation pair group repeats a domain: {key}")
        identities = {
            (
                row["flight_status"],
                row["source_fault_type"],
                row["level1_label"],
                row["level2_label"],
                row["level3_label"],
                row["open_set_fault_type"],
            )
            for row in group
        }
        if len(identities) != 1:
            raise ValueError(
                f"simulation pair group has inconsistent labels: {key} -> {identities}"
            )
    return dict(groups)


def build_internal_assignments(
    groups: dict[str, list[dict[str, str]]], seed: str
) -> dict[str, str]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key, group in groups.items():
        first = group[0]
        if not parse_bool(first["closed_set_eligible"]):
            continue
        by_stratum[closed_set_stratum(first)].append(key)

    counts_by_stratum = {
        stratum: apportion_group_counts(len(group_keys))
        for stratum, group_keys in by_stratum.items()
    }
    group_counts_by_stratum = {
        stratum: len(group_keys) for stratum, group_keys in by_stratum.items()
    }
    counts_by_stratum, _ = rebalance_stratum_counts(
        counts_by_stratum, group_counts_by_stratum
    )

    assignments: dict[str, str] = {}
    for stratum, group_keys in sorted(by_stratum.items()):
        ordered = sorted(
            group_keys,
            key=lambda key: (stable_score(seed, stratum, key), key),
        )
        counts = counts_by_stratum[stratum]
        cursor = 0
        for split in INTERNAL_SPLITS:
            next_cursor = cursor + counts[split]
            for key in ordered[cursor:next_cursor]:
                assignments[key] = split
            cursor = next_cursor
        if cursor != len(ordered):
            raise RuntimeError(f"stratum assignment is incomplete: {stratum}")
    return assignments


def split_role(split: str) -> str:
    roles = {
        "train": "closed_set_training",
        "val": "closed_set_model_selection",
        "test": "closed_set_internal_final_evaluation",
        "external_test": "closed_set_real_external_final_evaluation",
        "open_set_test": "simulation_unknown_fault_final_evaluation",
    }
    return roles[split]


def attach_split_fields(
    rows: list[dict[str, str]], seed: str
) -> list[dict[str, Any]]:
    groups = group_simulation_rows(rows)
    assignments = build_internal_assignments(groups, seed)
    result: list[dict[str, Any]] = []

    for row in rows:
        domain = row["domain"]
        pair_key = simulation_pair_key(row) if domain in SIMULATION_DOMAINS else ""
        if domain == "REAL":
            if not parse_bool(row["closed_set_eligible"]):
                raise ValueError(f"REAL contains an open-set case: {row['case_uid']}")
            split = "external_test"
            stratum = closed_set_stratum(row)
            pair_status = "not_applicable"
            group_size = 1
        elif parse_bool(row["open_set_eligible"]):
            split = "open_set_test"
            stratum = f"{row['open_set_fault_type']}|{row['flight_status']}"
            group_size = len(groups[pair_key])
            pair_status = "paired" if group_size == 2 else "singleton"
        else:
            split = assignments[pair_key]
            stratum = closed_set_stratum(row)
            group_size = len(groups[pair_key])
            pair_status = "paired" if group_size == 2 else "singleton"

        enriched: dict[str, Any] = dict(row)
        enriched.update(
            {
                "split_protocol_version": SPLIT_PROTOCOL_VERSION,
                "split_seed": seed,
                "split": split,
                "split_role": split_role(split),
                "split_stratum": stratum,
                "simulation_pair_key": pair_key,
                "simulation_pair_status": pair_status,
                "simulation_pair_group_size": group_size,
                "normalization_fit_eligible": split == "train",
                "model_selection_eligible": split == "val",
                "threshold_tuning_eligible": split == "val",
                "final_evaluation_only": split in {
                    "test",
                    "external_test",
                    "open_set_test",
                },
            }
        )
        result.append(enriched)

    validate_split_rows(result)
    return result


def validate_split_rows(rows: list[dict[str, Any]]) -> None:
    seen_uids: set[str] = set()
    seen_paths: set[str] = set()
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        uid = str(row["case_uid"])
        path = str(row["final_output_path"]).casefold()
        split = str(row["split"])
        if uid in seen_uids:
            raise ValueError(f"duplicate case UID: {uid}")
        if path in seen_paths:
            raise ValueError(f"duplicate output path: {path}")
        if split not in ALL_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        seen_uids.add(uid)
        seen_paths.add(path)

        domain = row["domain"]
        closed = parse_bool(row["closed_set_eligible"])
        open_set = parse_bool(row["open_set_eligible"])
        if domain == "REAL" and split != "external_test":
            raise ValueError(f"REAL case is not external_test: {uid}")
        if open_set and split != "open_set_test":
            raise ValueError(f"open-set case is not open_set_test: {uid}")
        if domain in SIMULATION_DOMAINS and closed and split not in INTERNAL_SPLITS:
            raise ValueError(f"simulation closed-set case has invalid split: {uid}")
        if parse_bool(row["normalization_fit_eligible"]) != (split == "train"):
            raise ValueError(f"normalization eligibility mismatch: {uid}")
        if parse_bool(row["model_selection_eligible"]) != (split == "val"):
            raise ValueError(f"model-selection eligibility mismatch: {uid}")
        pair_key = str(row["simulation_pair_key"])
        if pair_key:
            pair_splits[pair_key].add(split)

    leaking = {key: values for key, values in pair_splits.items() if len(values) != 1}
    if leaking:
        raise ValueError(f"simulation pair groups cross splits: {leaking}")


def read_contract(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"contract has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counter(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize(rows: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in ALL_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        by_split[split] = {
            "cases": len(split_rows),
            "domain_counts": counter([row["domain"] for row in split_rows]),
            "level1_counts": counter([row["level1_label"] for row in split_rows]),
            "level2_counts": counter(
                [row["level2_label"] for row in split_rows if row["level2_label"]]
            ),
            "level3_counts": counter(
                [row["level3_label"] for row in split_rows if row["level3_label"]]
            ),
            "open_set_counts": counter(
                [
                    row["open_set_fault_type"]
                    for row in split_rows
                    if row["open_set_fault_type"]
                ]
            ),
            "flight_status_counts": counter(
                [row["flight_status"] for row in split_rows]
            ),
            "paired_topology_cases": sum(
                parse_bool(row["paired_topology_eligible"])
                for row in split_rows
            ),
            "normal_only_cases": sum(
                row["window_usage"] == "normal_windows_only"
                for row in split_rows
            ),
        }

    simulation_rows = [row for row in rows if row["domain"] in SIMULATION_DOMAINS]
    pair_groups = group_simulation_rows(simulation_rows)
    internal_rows = [row for row in rows if row["split"] in INTERNAL_SPLITS]
    internal_groups_by_split = {
        split: len(
            {
                row["simulation_pair_key"]
                for row in internal_rows
                if row["split"] == split
            }
        )
        for split in INTERNAL_SPLITS
    }
    return {
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": seed,
        "protocol": {
            "simulation_closed_set": {
                "domains": ["SIL", "HIL"],
                "ratios": INTERNAL_RATIOS,
                "global_group_targets": apportion_group_counts(
                    sum(internal_groups_by_split.values())
                ),
                "stratification": "level3_label x flight_status",
                "assignment_unit": "simulation_pair_group",
                "allocation": (
                    "global quota constrained; each stratum retains train/val/test "
                    "coverage and squared ratio deviation is minimized"
                ),
            },
            "real_closed_set": {
                "split": "external_test",
                "used_for_training": False,
                "used_for_model_selection": False,
                "used_for_threshold_tuning": False,
            },
            "simulation_open_set": {
                "split": "open_set_test",
                "used_for_closed_set_training": False,
            },
            "window_generation": "after this case-level split",
            "normalization_fit": "train split only",
        },
        "total_cases": len(rows),
        "case_uid_unique": len({row["case_uid"] for row in rows}) == len(rows),
        "output_path_unique": len(
            {str(row["final_output_path"]).casefold() for row in rows}
        )
        == len(rows),
        "simulation_pair_groups": len(pair_groups),
        "simulation_pair_group_sizes": counter(
            [len(group) for group in pair_groups.values()]
        ),
        "simulation_singleton_groups": sum(
            len(group) == 1 for group in pair_groups.values()
        ),
        "internal_group_counts": internal_groups_by_split,
        "by_split": by_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh leakage-controlled RflyMAD case-level split."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            r"D:\UAV-Data\quality_reports\rflymad_hierarchical_fault_contract.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\UAV-Data\splits\rflymad_case_split_v2"),
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_fields, contract_rows = read_contract(args.contract)
    rows = attach_split_fields(contract_rows, args.seed)
    output_fields = [*contract_fields, *SPLIT_FIELDS]
    output_root = args.output_root

    combined_path = output_root / "rflymad_case_level_split_manifest.csv"
    write_csv(combined_path, output_fields, rows)

    split_paths: dict[str, str] = {}
    for split in ALL_SPLITS:
        path = output_root / f"rflymad_case_level_split_{split}.csv"
        write_csv(path, output_fields, [row for row in rows if row["split"] == split])
        split_paths[split] = str(path)

    domain_paths: dict[str, str] = {}
    for domain in ("SIL", "HIL", "REAL"):
        path = output_root / f"rflymad_case_level_split_{domain.lower()}.csv"
        write_csv(path, output_fields, [row for row in rows if row["domain"] == domain])
        domain_paths[domain] = str(path)

    summary = summarize(rows, args.seed)
    summary["inputs"] = {"hierarchical_contract": str(args.contract)}
    summary["outputs"] = {
        "combined_manifest": str(combined_path),
        "split_manifests": split_paths,
        "domain_manifests": domain_paths,
    }
    summary_path = output_root / "rflymad_case_level_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
