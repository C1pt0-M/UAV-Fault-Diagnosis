import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in NUM_RE.findall(text or "")]


def read_testinfo_xlsx(path: Path) -> dict:
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[0]]
    out = {}
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        key = str(row[0]).strip()
        value = "" if len(row) < 2 or row[1] is None else str(row[1]).strip()
        out[key] = value
    wb.close()
    return out


def _candidate_testinfo_paths(mapping_row: dict) -> list[Path]:
    candidates = []
    testinfo_path = str(mapping_row.get("testinfo_path", "") or "").strip()
    if testinfo_path:
        candidates.append(Path(testinfo_path))
    raw_case_path = str(mapping_row.get("raw_case_path", "") or "").strip()
    if raw_case_path:
        raw_case_dir = Path(raw_case_path)
        candidates.extend(sorted(raw_case_dir.glob("TestInfo*.xlsx")))
    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
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


def build_row(testinfo_path: Path, mapping_row: dict) -> dict:
    case_dir = testinfo_path.parent
    raw = read_testinfo_xlsx(testinfo_path)

    case_id_list = parse_numbers(raw.get("Real Flight case ID", ""))
    fault_id_list = parse_numbers(raw.get("Fault ID", ""))
    fault_params = parse_numbers(raw.get("Fault parameter", ""))
    inject_time = parse_numbers(raw.get("Fault injection time", ""))
    end_time = parse_numbers(raw.get("Test end time", ""))

    processed_case_id = int(mapping_row["processed_case_id"])
    raw_case_index_in_dir = int(mapping_row.get("raw_case_index_in_dir", "0") or 0)

    return {
        "processed_case_id": processed_case_id,
        "processed_csv_path": str(mapping_row.get("processed_csv_path", "") or ""),
        "testinfo_path": str(testinfo_path),
        "build_batch_id": str(mapping_row.get("build_batch_id", "") or ""),
        "case_dir": case_dir.name,
        "case_index_in_dir": raw_case_index_in_dir,
        "bucket_case_num": processed_case_id % 1_000_000,
        "raw_case_index_in_dir": raw_case_index_in_dir,
        "platform": str(mapping_row.get("platform", "") or "Real"),
        "flight_mode": str(mapping_row.get("flight_mode", "") or case_dir.parent.parent.name),
        "fault_type_canonical": str(
            mapping_row.get("fault_type_canonical", "") or case_dir.parent.name
        ),
        "fault_dirname": case_dir.parent.name,
        "flight_dirname": case_dir.parent.parent.name,
        "raw_case_path": str(mapping_row.get("raw_case_path", "") or case_dir),
        "case_id_from_testinfo": int(case_id_list[0]) if case_id_list else None,
        "control_sequence_raw": raw.get("Control Sequence", ""),
        "fault_id_raw": raw.get("Fault ID", ""),
        "fault_id_numeric": int(fault_id_list[0]) if fault_id_list else None,
        "fault_parameter_raw": raw.get("Fault parameter", ""),
        "fault_params_json": json.dumps(fault_params, ensure_ascii=False),
        "fault_injection_time_sec": inject_time[0] if inject_time else None,
        "test_end_time_sec": end_time[0] if end_time else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 Real TestInfo.xlsx 元数据")
    parser.add_argument("--root", default="/home/jack/RflyMAD/Real")
    parser.add_argument("--case-mapping", default="")
    parser.add_argument(
        "--output", default="data/RflyMAD/manifest/real_testinfo_metadata.csv"
    )
    parser.add_argument(
        "--summary",
        default="data/RflyMAD/manifest/real_testinfo_metadata.summary.json",
    )
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
                if candidate.exists():
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
            rows.append(build_row(selected_path, mapping_row))
    else:
        mapping_stats = {}
        rows = []
        for testinfo_path in sorted(root.glob("*/*/*/TestInfo*.xlsx")):
            rows.append(
                {
                    "processed_case_id": None,
                    "processed_csv_path": "",
                    "testinfo_path": str(testinfo_path),
                    "build_batch_id": "",
                    "case_dir": testinfo_path.parent.name,
                    "case_index_in_dir": None,
                    "bucket_case_num": None,
                    "raw_case_index_in_dir": None,
                    "platform": "Real",
                    "flight_mode": testinfo_path.parent.parent.parent.name,
                    "fault_type_canonical": testinfo_path.parent.parent.name,
                    "fault_dirname": testinfo_path.parent.parent.name,
                    "flight_dirname": testinfo_path.parent.parent.parent.name,
                    "raw_case_path": str(testinfo_path.parent),
                    "case_id_from_testinfo": None,
                    "control_sequence_raw": "",
                    "fault_id_raw": "",
                    "fault_id_numeric": None,
                    "fault_parameter_raw": "",
                    "fault_params_json": "[]",
                    "fault_injection_time_sec": None,
                    "test_end_time_sec": None,
                }
            )
        missing_testinfo = []

    if not rows:
        raise RuntimeError(f"未找到 Real TestInfo xlsx: {root}")

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "total_rows": len(rows),
        "platform_counts": {"Real": len(rows)},
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

    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE_CSV={output}")
    print(f"WROTE_SUMMARY={summary}")
    print(f"TOTAL_ROWS={len(rows)}")


if __name__ == "__main__":
    main()
