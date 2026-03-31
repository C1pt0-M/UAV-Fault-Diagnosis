import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


CASE_RE = re.compile(r"^TestCase_(\d+)_(\d+)$")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

PLATFORM_MAP = {1: "SIL", 2: "HIL", 3: "Real"}
FLIGHT_MAP = {
    0: "hover",
    1: "waypoint",
    2: "velocity",
    3: "circling",
    4: "acce",
    5: "dece",
}
FAULT_MAP = {
    0: "motor",
    1: "propeller",
    2: "low_voltage",
    3: "wind_affect",
    4: "load_lose",
    5: "accelerometer",
    6: "gyroscope",
    7: "magnetometer",
    8: "barometer",
    9: "GPS",
    10: "no_fault",
}


def parse_case_id(case_id: int) -> dict:
    data_type = case_id // 1_000_000_000
    flight_id = (case_id % 1_000_000_000) // 100_000_000
    fault_id = (case_id % 100_000_000) // 1_000_000
    case_num = case_id % 1_000_000
    return {
        "platform": PLATFORM_MAP.get(data_type),
        "flight_mode": FLIGHT_MAP.get(flight_id),
        "fault_type_canonical": FAULT_MAP.get(fault_id),
        "case_num": case_num,
    }


def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in NUM_RE.findall(text or "")]


def read_testinfo(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    out = {}
    for row in rows:
        if not row:
            continue
        key = row[0].strip()
        value = ",".join(row[1:]).strip() if len(row) > 1 else ""
        out[key] = value
    return out


def _candidate_testinfo_paths(mapping_row: dict) -> list[Path]:
    candidates = []
    testinfo_path = str(mapping_row.get("testinfo_path", "") or "").strip()
    if testinfo_path:
        candidates.append(Path(testinfo_path))
    raw_case_path = str(mapping_row.get("raw_case_path", "") or "").strip()
    if raw_case_path:
        raw_case_dir = Path(raw_case_path)
        candidates.append(raw_case_dir / "TestInfo.csv")
        candidates.append(raw_case_dir / "TestInfo.xlsx")
    unique = []
    seen = set()
    for path in candidates:
        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))
    return unique


def load_case_mapping(path: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        raise RuntimeError(f"case_mapping 为空: {path}")
    if "processed_case_id" not in df.columns:
        raise RuntimeError(f"case_mapping 缺少 processed_case_id: {path}")

    stats = {
        "input_rows": int(len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_processed_case_id_rows": int(
            df.duplicated(subset=["processed_case_id"]).sum()
        ),
    }

    df = df.drop_duplicates().copy()
    text = df["processed_case_id"].astype("string").str.strip()
    valid = text.str.fullmatch(r"-?\d+")
    parsed_ids = pd.Series([None] * len(df), index=df.index, dtype="object")
    parsed_ids.loc[valid.fillna(False)] = text.loc[valid.fillna(False)].map(int)
    df["processed_case_id"] = parsed_ids
    df = df[df["processed_case_id"].notna()].copy()
    df = df.drop_duplicates(subset=["processed_case_id"], keep="last").copy()
    df = df.sort_values("processed_case_id").reset_index(drop=True)

    stats["output_rows"] = int(len(df))
    return df.to_dict(orient="records"), stats


def _safe_int(value, default=None):
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return int(num)


def build_row(testinfo_path: Path, platform_name: str, mapping_row: dict | None = None) -> dict:
    case_dir = testinfo_path.parent
    m = CASE_RE.match(case_dir.name)
    if not m:
        raise ValueError(f"无法解析 case 目录名: {case_dir.name}")
    case_idx = int(m.group(1))
    raw_case_id = int(m.group(2))
    raw_meta = parse_case_id(raw_case_id)
    raw = read_testinfo(testinfo_path)
    fault_dirname = case_dir.parent.name
    flight_dirname = case_dir.parent.parent.name

    processed_case_id = None
    build_batch_id = ""
    processed_csv_path = ""
    raw_case_path = str(case_dir)
    raw_case_index_in_dir = case_idx
    bucket_case_num = raw_meta["case_num"]
    platform = platform_name
    flight_mode = flight_dirname
    fault_type_canonical = fault_dirname

    if mapping_row is not None:
        processed_case_id = int(mapping_row["processed_case_id"])
        processed_meta = parse_case_id(processed_case_id)
        build_batch_id = str(mapping_row.get("build_batch_id", "") or "")
        processed_csv_path = str(mapping_row.get("processed_csv_path", "") or "")
        raw_case_path = str(mapping_row.get("raw_case_path", "") or raw_case_path)
        raw_case_index_in_dir = _safe_int(
            mapping_row.get("raw_case_index_in_dir"), case_idx
        )
        bucket_case_num = processed_meta["case_num"]
        platform = (
            str(mapping_row.get("platform", "") or "").strip()
            or processed_meta["platform"]
            or platform_name
        )
        flight_mode = (
            str(mapping_row.get("flight_mode", "") or "").strip()
            or processed_meta["flight_mode"]
            or flight_dirname
        )
        fault_type_canonical = (
            str(mapping_row.get("fault_type_canonical", "") or "").strip()
            or processed_meta["fault_type_canonical"]
            or fault_dirname
        )

    fault_id_list = parse_numbers(raw.get("Fault ID", ""))
    fault_params = parse_numbers(raw.get("Fault Parameter", ""))
    inject_time = parse_numbers(raw.get("Fault injection time", ""))
    end_time = parse_numbers(raw.get("Test end time", ""))
    collect_hz = parse_numbers(raw.get("Data collected Frequency", ""))
    data_num = parse_numbers(raw.get("Data num", ""))

    return {
        "processed_case_id": processed_case_id,
        "processed_csv_path": processed_csv_path,
        "testinfo_path": str(testinfo_path),
        "build_batch_id": build_batch_id,
        "case_dir": case_dir.name,
        "case_index_in_dir": raw_case_index_in_dir,
        "bucket_case_num": bucket_case_num,
        "raw_case_index_in_dir": raw_case_index_in_dir,
        "case_id": raw_case_id,
        "platform": platform,
        "flight_mode": flight_mode,
        "fault_type_canonical": fault_type_canonical,
        "case_num": raw_meta["case_num"],
        "flight_mode_from_case_id": raw_meta["flight_mode"],
        "fault_type_from_case_id": raw_meta["fault_type_canonical"],
        "fault_dirname": fault_dirname,
        "flight_dirname": flight_dirname,
        "raw_case_path": raw_case_path,
        "raw_case_id": raw_case_id,
        "case_id_from_testinfo": parse_numbers(raw.get("Case ID", ""))[0]
        if parse_numbers(raw.get("Case ID", ""))
        else None,
        "control_sequence_raw": raw.get("Control Sequence", ""),
        "fault_id_raw": raw.get("Fault ID", ""),
        "fault_id_numeric": int(fault_id_list[0]) if fault_id_list else None,
        "fault_parameter_raw": raw.get("Fault Parameter", ""),
        "fault_params_json": json.dumps(fault_params, ensure_ascii=False),
        "fault_injection_time_sec": inject_time[0] if inject_time else None,
        "test_end_time_sec": end_time[0] if end_time else None,
        "raw_collection_hz": collect_hz[0] if collect_hz else None,
        "data_num": int(data_num[0]) if data_num else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 SHIL TestInfo.csv 元数据")
    parser.add_argument("--root", default="/home/jack/RflyMAD/HIL")
    parser.add_argument("--case-mapping", default="")
    parser.add_argument(
        "--output", default="data/RflyMAD/manifest/hil_testinfo_metadata.csv"
    )
    parser.add_argument(
        "--summary", default="data/RflyMAD/manifest/hil_testinfo_metadata.summary.json"
    )
    parser.add_argument("--platform", default="HIL")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    missing_testinfo = []
    if args.case_mapping:
        mapping_rows, mapping_stats = load_case_mapping(Path(args.case_mapping))
        rows = []
        for mapping_row in mapping_rows:
            selected_path = None
            for candidate in _candidate_testinfo_paths(mapping_row):
                if candidate.exists() and candidate.name.lower().startswith("testinfo"):
                    selected_path = candidate
                    break
            if selected_path is None:
                missing_testinfo.append(
                    {
                        "processed_case_id": int(mapping_row["processed_case_id"]),
                        "raw_case_path": mapping_row.get("raw_case_path", ""),
                    }
                )
                continue
            rows.append(build_row(selected_path, args.platform, mapping_row))
    else:
        mapping_stats = {}
        rows = [
            build_row(p, args.platform)
            for p in sorted(root.glob("*/*/TestCase_*/TestInfo.csv"))
        ]

    if not rows:
        raise RuntimeError(f"未找到 TestInfo.csv: {root}")

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "total_rows": len(rows),
        "platform_counts": {args.platform: len(rows)},
        "case_mapping_used": bool(args.case_mapping),
        "case_mapping_path": args.case_mapping,
        "case_mapping_stats": mapping_stats,
        "missing_testinfo_count": len(missing_testinfo),
        "flight_mode_counts": {},
        "fault_type_counts": {},
        "fault_id_numeric_counts": {},
    }
    for key in ("flight_mode", "fault_type_canonical", "fault_id_numeric"):
        counts = {}
        for row in rows:
            value = row[key]
            counts[str(value)] = counts.get(str(value), 0) + 1
        stats[
            {
                "flight_mode": "flight_mode_counts",
                "fault_type_canonical": "fault_type_counts",
                "fault_id_numeric": "fault_id_numeric_counts",
            }[key]
        ] = counts

    summary.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE_CSV={output}")
    print(f"WROTE_SUMMARY={summary}")
    print(f"TOTAL_ROWS={len(rows)}")


if __name__ == "__main__":
    main()
