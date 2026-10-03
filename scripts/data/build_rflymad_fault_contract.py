from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "1.0"
DOMAINS = ("SIL", "HIL", "REAL")
SENSOR_TYPES = (
    "accelerometer",
    "gyroscope",
    "magnetometer",
    "barometer",
    "GPS",
)
OPEN_SET_TYPES = ("propeller", "low_voltage", "wind_affect", "load_lose")
CLOSED_SET_TYPES = ("no_fault", "motor", *SENSOR_TYPES)
LEVEL1_IDS = {"normal": 0, "fault": 1}
LEVEL2_IDS = {"normal": 0, "motor": 1, "sensor": 2}
LEVEL3_IDS = {
    "normal": 0,
    "motor": 1,
    "accelerometer": 2,
    "gyroscope": 3,
    "magnetometer": 4,
    "barometer": 5,
    "GPS": 6,
}
REAL_ID_TO_SUBTYPE = {
    123450: "motor",
    123452: "accelerometer",
    123453: "gyroscope",
    123454: "barometer",
    123455: "magnetometer",
    123456: "GPS",
}

OUTPUT_FIELDS = (
    "contract_version",
    "domain",
    "case_uid",
    "plan_index",
    "flight_status",
    "source_fault_type",
    "raw_case_directory",
    "source_path",
    "output_filename",
    "final_output_path",
    "row_count",
    "column_count",
    "median_step_us",
    "output_sha256",
    "source_disposition",
    "source_metadata_status",
    "level1_label",
    "level1_id",
    "level2_label",
    "level2_id",
    "level3_label",
    "level3_id",
    "closed_set_eligible",
    "open_set_eligible",
    "open_set_fault_type",
    "normal_window_eligible",
    "closed_set_fault_window_eligible",
    "open_set_fault_window_eligible",
    "paired_topology_eligible",
    "event_detection_eligible",
    "window_usage",
    "window_label_source",
    "included_in_final_dataset",
)


def parse_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def require(row: dict[str, str], field: str, domain: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{domain} manifest row is missing {field}: {row}")
    return value


def normalize_sha256(value: str, domain: str, output_filename: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(
            f"invalid SHA-256 for {domain}/{output_filename}: {value!r}"
        )
    return digest


def base_source_fields(row: dict[str, str], domain: str) -> dict[str, str]:
    if domain == "REAL":
        source_fault_type = require(row, "source_fault_type", domain)
        included = parse_bool(row.get("included_in_final_dataset", ""))
        disposition = require(row, "disposition", domain)
        metadata_status = row.get("metadata_status", "").strip()
    else:
        source_fault_type = require(row, "fault_type", domain)
        included = parse_bool(row.get("included", ""))
        disposition = "processed_official"
        metadata_status = "ok"

    final_output_path = require(row, "final_output_path", domain)
    output_filename = require(row, "output_filename", domain)
    raw_case_directory = require(row, "raw_case_directory", domain)
    flight_status = require(row, "flight_status", domain)
    source_path = require(row, "source_path", domain)
    return {
        "source_fault_type": source_fault_type,
        "included": str(included),
        "source_disposition": disposition,
        "source_metadata_status": metadata_status,
        "final_output_path": final_output_path,
        "output_filename": output_filename,
        "raw_case_directory": raw_case_directory,
        "flight_status": flight_status,
        "source_path": source_path,
    }


def labels_for_simulation(source_fault_type: str) -> dict[str, Any]:
    if source_fault_type == "no_fault":
        return {
            "level1_label": "normal",
            "level2_label": "normal",
            "level3_label": "normal",
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": False,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": False,
            "event_detection_eligible": False,
            "window_usage": "normal_windows_only",
        }
    if source_fault_type == "motor":
        return {
            "level1_label": "fault",
            "level2_label": "motor",
            "level3_label": "motor",
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": True,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": True,
            "event_detection_eligible": True,
            "window_usage": "row_level_closed_set_windows",
        }
    if source_fault_type in SENSOR_TYPES:
        return {
            "level1_label": "fault",
            "level2_label": "sensor",
            "level3_label": source_fault_type,
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": True,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": True,
            "event_detection_eligible": True,
            "window_usage": "row_level_closed_set_windows",
        }
    if source_fault_type in OPEN_SET_TYPES:
        return {
            "level1_label": "fault",
            "level2_label": "",
            "level3_label": "",
            "closed_set_eligible": False,
            "open_set_eligible": True,
            "open_set_fault_type": source_fault_type,
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": False,
            "open_set_fault_window_eligible": True,
            "paired_topology_eligible": False,
            "event_detection_eligible": False,
            "window_usage": "row_level_open_set_windows",
        }
    raise ValueError(f"unsupported SIL/HIL fault type: {source_fault_type}")


def labels_for_real(row: dict[str, str]) -> dict[str, Any]:
    main_label = require(row, "effective_main3_label", "REAL")
    subtype = require(row, "effective_fault_subtype", "REAL")
    disposition = require(row, "disposition", "REAL")
    window_usage = require(row, "window_usage", "REAL")

    if main_label == "normal":
        if subtype != "normal":
            raise ValueError(f"REAL normal case has non-normal subtype: {subtype}")
        expected_usage_by_disposition = {
            "retain_normal": {"row_level_main3_windows", "normal_windows_only"},
            "retain_normal_only": {"normal_windows_only"},
        }
        allowed_usage = expected_usage_by_disposition.get(disposition)
        if allowed_usage is None or window_usage not in allowed_usage:
            raise ValueError(
                "REAL normal case has unsupported disposition/window usage: "
                f"{disposition}/{window_usage}"
            )
        return {
            "level1_label": "normal",
            "level2_label": "normal",
            "level3_label": "normal",
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": False,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": False,
            "event_detection_eligible": False,
            "window_usage": "normal_windows_only",
        }
    if main_label == "motor":
        if subtype != "motor" or disposition != "retain_motor_fault":
            raise ValueError(
                f"REAL motor contract mismatch: subtype={subtype}, disposition={disposition}"
            )
        return {
            "level1_label": "fault",
            "level2_label": "motor",
            "level3_label": "motor",
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": True,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": True,
            "event_detection_eligible": True,
            "window_usage": "row_level_closed_set_windows",
        }
    if main_label == "sensor":
        if subtype not in SENSOR_TYPES:
            raise ValueError(f"REAL sensor case has unsupported subtype: {subtype}")
        if disposition not in {
            "retain_sensor_fault",
            "retain_sensor_family_subtype_conflict",
        }:
            raise ValueError(
                f"REAL sensor case has unsupported disposition: {disposition}"
            )
        return {
            "level1_label": "fault",
            "level2_label": "sensor",
            "level3_label": subtype,
            "closed_set_eligible": True,
            "open_set_eligible": False,
            "open_set_fault_type": "",
            "normal_window_eligible": True,
            "closed_set_fault_window_eligible": True,
            "open_set_fault_window_eligible": False,
            "paired_topology_eligible": True,
            "event_detection_eligible": True,
            "window_usage": "row_level_closed_set_windows",
        }
    raise ValueError(f"unsupported REAL main label: {main_label}")


def build_contract_row(
    row: dict[str, str], domain: str, *, check_output_exists: bool = True
) -> dict[str, Any] | None:
    domain = domain.upper()
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    included_field = (
        "included_in_final_dataset" if domain == "REAL" else "included"
    )
    if not parse_bool(row.get(included_field, "")):
        return None
    source = base_source_fields(row, domain)

    labels = labels_for_real(row) if domain == "REAL" else labels_for_simulation(
        source["source_fault_type"]
    )
    final_output_path = Path(source["final_output_path"])
    if check_output_exists and not final_output_path.is_file():
        raise FileNotFoundError(
            f"final output does not exist: {final_output_path}"
        )

    level1 = labels["level1_label"]
    level2 = labels["level2_label"]
    level3 = labels["level3_label"]
    output_filename = source["output_filename"]
    result: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "domain": domain,
        "case_uid": (
            f"{domain}/{source['flight_status']}/{source['source_fault_type']}/"
            f"{source['raw_case_directory']}"
        ),
        "plan_index": require(row, "plan_index", domain),
        "flight_status": source["flight_status"],
        "source_fault_type": source["source_fault_type"],
        "raw_case_directory": source["raw_case_directory"],
        "source_path": source["source_path"],
        "output_filename": output_filename,
        "final_output_path": str(final_output_path),
        "row_count": require(row, "row_count", domain),
        "column_count": require(row, "column_count", domain),
        "median_step_us": row.get("median_step_us", "20000").strip() or "20000",
        "output_sha256": normalize_sha256(
            require(row, "output_sha256", domain), domain, output_filename
        ),
        "source_disposition": source["source_disposition"],
        "source_metadata_status": source["source_metadata_status"],
        "level1_label": level1,
        "level1_id": LEVEL1_IDS[level1],
        "level2_label": level2,
        "level2_id": "" if not level2 else LEVEL2_IDS[level2],
        "level3_label": level3,
        "level3_id": "" if not level3 else LEVEL3_IDS[level3],
        "closed_set_eligible": labels["closed_set_eligible"],
        "open_set_eligible": labels["open_set_eligible"],
        "open_set_fault_type": labels["open_set_fault_type"],
        "normal_window_eligible": labels["normal_window_eligible"],
        "closed_set_fault_window_eligible": labels[
            "closed_set_fault_window_eligible"
        ],
        "open_set_fault_window_eligible": labels[
            "open_set_fault_window_eligible"
        ],
        "paired_topology_eligible": labels["paired_topology_eligible"],
        "event_detection_eligible": labels["event_detection_eligible"],
        "window_usage": labels["window_usage"],
        "window_label_source": (
            "rfly_ctrl_lxl_slash_send_data_id"
            if domain == "REAL"
            else "UAVState_data_fault_state"
        ),
        "included_in_final_dataset": True,
    }
    validate_contract_row(result)
    return result


def validate_contract_row(row: dict[str, Any]) -> None:
    closed = bool(row["closed_set_eligible"])
    open_set = bool(row["open_set_eligible"])
    if closed == open_set:
        raise ValueError(
            f"case must belong to exactly one task role: {row['case_uid']}"
        )
    if closed:
        if not row["level2_label"] or not row["level3_label"]:
            raise ValueError(f"closed-set labels are incomplete: {row['case_uid']}")
        if row["open_set_fault_type"]:
            raise ValueError(f"closed-set case has open-set label: {row['case_uid']}")
    else:
        if row["level2_label"] or row["level3_label"]:
            raise ValueError(f"open-set case has closed-set labels: {row['case_uid']}")
        if not row["open_set_fault_type"]:
            raise ValueError(f"open-set type is missing: {row['case_uid']}")
    if row["paired_topology_eligible"] and not row[
        "closed_set_fault_window_eligible"
    ]:
        raise ValueError(
            f"paired topology case cannot produce fault windows: {row['case_uid']}"
        )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def counter_dict(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize(rows: list[dict[str, Any]], inputs: dict[str, Path]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        closed_rows = [row for row in domain_rows if row["closed_set_eligible"]]
        open_rows = [row for row in domain_rows if row["open_set_eligible"]]
        by_domain[domain] = {
            "final_cases": len(domain_rows),
            "closed_set_cases": len(closed_rows),
            "open_set_cases": len(open_rows),
            "level1_counts": counter_dict(
                [row["level1_label"] for row in domain_rows]
            ),
            "level2_counts": counter_dict(
                [row["level2_label"] for row in closed_rows]
            ),
            "level3_counts": counter_dict(
                [row["level3_label"] for row in closed_rows]
            ),
            "open_set_counts": counter_dict(
                [row["open_set_fault_type"] for row in open_rows]
            ),
            "paired_topology_cases": sum(
                bool(row["paired_topology_eligible"]) for row in domain_rows
            ),
            "normal_only_cases": sum(
                row["window_usage"] == "normal_windows_only"
                for row in domain_rows
            ),
        }
    closed_rows = [row for row in rows if row["closed_set_eligible"]]
    open_rows = [row for row in rows if row["open_set_eligible"]]
    return {
        "contract_version": CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {domain: str(path) for domain, path in inputs.items()},
        "final_case_count": len(rows),
        "closed_set_case_count": len(closed_rows),
        "open_set_case_count": len(open_rows),
        "closed_set_level2_counts": counter_dict(
            [row["level2_label"] for row in closed_rows]
        ),
        "closed_set_level3_counts": counter_dict(
            [row["level3_label"] for row in closed_rows]
        ),
        "open_set_fault_counts": counter_dict(
            [row["open_set_fault_type"] for row in open_rows]
        ),
        "paired_topology_case_count": sum(
            bool(row["paired_topology_eligible"]) for row in rows
        ),
        "by_domain": by_domain,
        "rules": {
            "level1": LEVEL1_IDS,
            "level2_closed_set": LEVEL2_IDS,
            "level3_closed_set": LEVEL3_IDS,
            "open_set_fault_types": list(OPEN_SET_TYPES),
            "sensor_family_types": list(SENSOR_TYPES),
            "real_row_id_mapping": {
                str(key): value for key, value in REAL_ID_TO_SUBTYPE.items()
            },
            "split_unit": "complete flight case before window generation",
            "normalization_fit": "training partition only",
            "real_normal_only_policy": (
                "normal windows only; excluded from fault windows and fault-case metrics"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the hierarchical RflyMAD fault-label contract."
    )
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=Path(r"D:\UAV-Data\quality_reports"),
    )
    parser.add_argument("--sil-manifest", type=Path)
    parser.add_argument("--hil-manifest", type=Path)
    parser.add_argument("--real-manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\UAV-Data\quality_reports"),
    )
    parser.add_argument(
        "--skip-output-existence-check",
        action="store_true",
        help="Only for isolated tests; production runs must check every output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quality_root = args.quality_root
    inputs = {
        "SIL": args.sil_manifest
        or quality_root / "sil_50hz_official_final_case_manifest_20260809.csv",
        "HIL": args.hil_manifest
        or quality_root / "hil_50hz_official_final_case_manifest_20260809.csv",
        "REAL": args.real_manifest
        or quality_root / "real_50hz_final_case_manifest_20260809.csv",
    }
    rows: list[dict[str, Any]] = []
    seen_case_uids: set[str] = set()
    seen_output_paths: set[str] = set()
    for domain in DOMAINS:
        for source_row in read_manifest(inputs[domain]):
            result = build_contract_row(
                source_row,
                domain,
                check_output_exists=not args.skip_output_existence_check,
            )
            if result is None:
                continue
            case_uid = result["case_uid"]
            output_path = str(result["final_output_path"]).casefold()
            if case_uid in seen_case_uids:
                raise RuntimeError(f"duplicate case UID: {case_uid}")
            if output_path in seen_output_paths:
                raise RuntimeError(f"duplicate final output path: {output_path}")
            seen_case_uids.add(case_uid)
            seen_output_paths.add(output_path)
            rows.append(result)

    domain_order = {domain: index for index, domain in enumerate(DOMAINS)}
    rows.sort(
        key=lambda row: (
            domain_order[row["domain"]],
            int(row["plan_index"]),
            row["case_uid"],
        )
    )
    output_root = args.output_root
    combined_path = output_root / "rflymad_hierarchical_fault_contract.csv"
    write_csv(combined_path, rows)
    domain_paths: dict[str, str] = {}
    for domain in DOMAINS:
        domain_path = output_root / f"rflymad_hierarchical_fault_contract_{domain.lower()}.csv"
        write_csv(domain_path, [row for row in rows if row["domain"] == domain])
        domain_paths[domain] = str(domain_path)

    summary = summarize(rows, inputs)
    summary["outputs"] = {
        "combined_manifest": str(combined_path),
        "domain_manifests": domain_paths,
    }
    summary_path = output_root / "rflymad_hierarchical_fault_contract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mapping_path = output_root / "rflymad_hierarchical_fault_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "level1": LEVEL1_IDS,
                "level2_closed_set": LEVEL2_IDS,
                "level3_closed_set": LEVEL3_IDS,
                "source_to_level2": {
                    "no_fault": "normal",
                    "motor": "motor",
                    **{sensor: "sensor" for sensor in SENSOR_TYPES},
                },
                "source_to_level3": {
                    "no_fault": "normal",
                    "motor": "motor",
                    **{sensor: sensor for sensor in SENSOR_TYPES},
                },
                "open_set_fault_types": list(OPEN_SET_TYPES),
                "real_row_id_to_level3": {
                    "0": "normal",
                    "1500": "normal",
                    **{
                        str(key): value
                        for key, value in REAL_ID_TO_SUBTYPE.items()
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
