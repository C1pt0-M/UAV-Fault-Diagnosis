"""Read-only raw RflyMAD inventory and deterministic timeline review.

Never imports native processing scripts, loads models, or changes dataset files.
Run --self-check, then --phase inventory and --phase timelines into a new output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
from openpyxl import load_workbook

KNOWN_IDS = {"motor": 123450, "accelerometer": 123452, "gyroscope": 123453,
             "barometer": 123454, "magnetometer": 123455, "GPS": 123456}
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def numbers(value):
    return [float(v) for v in re.findall(r"[-+]?[0-9]+(?:[.][0-9]+)?", str(value))]


def active_ids(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
        raise ValueError("Invalid fault command IDs")
    return ((values >= 123450) & (values <= 123459)) | ((values >= 123540) & (values <= 123549))


def timeline(times, values, scale):
    times = np.asarray(times, dtype=float) / scale
    values = np.asarray(values)
    if not len(times) or times.shape != values.shape or not np.isfinite(times).all():
        raise ValueError("Empty, nonfinite or mismatched timeline")
    delta = np.diff(times)
    mask = np.asarray(values, dtype=bool)
    hits = np.flatnonzero(mask)
    return {"rows": len(times), "first_s": float(times[0]), "last_s": float(times[-1]),
            "backward_steps": int((delta < 0).sum()), "duplicate_steps": int((delta == 0).sum()),
            "active_rows": int(mask.sum()), "onset_s": float(times[hits[0]]) if len(hits) else None,
            "active_to_inactive": int((mask[:-1] & ~mask[1:]).sum())}


def metadata(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return {row[0].strip(): row[1] for row in csv.reader(f) if len(row) >= 2}
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        return {str(row[0]).strip(): row[1] for row in rows if len(row) >= 2 and row[0] is not None}
    finally:
        wb.close()


def inspect_case(case, domain_root):
    mode, category, name = case.relative_to(domain_root).parts
    rec = {"case_uid": domain_root.name + "/" + case.relative_to(domain_root).as_posix(),
           "domain": domain_root.name, "mode": mode, "category": category,
           "case_name": name, "path": str(case), "issues": []}
    files = sorted(p for p in case.rglob("*") if p.is_file())
    rec["file_counts"] = dict(Counter(p.suffix.lower() for p in files))
    rec["total_bytes"] = sum(p.stat().st_size for p in files)
    kinds = {"metadata": [p for p in files if p.name.startswith("TestInfo") and p.suffix.lower() in (".csv", ".xlsx")],
             "ulog": [p for p in files if p.suffix.lower() == ".ulg"],
             "bag": [p for p in files if p.suffix.lower() == ".bag"],
             "uav_state": [p for p in files if p.name == "UAVState_data.xlsx"],
             "true_state": [p for p in files if p.name == "TrueState_data.xlsx"]}
    rec["sources"] = {k: [str(p) for p in paths] for k, paths in kinds.items()}
    required = ["metadata", "ulog"] + (["bag"] if domain_root.name == "Real" else ["uav_state", "true_state"])
    for key in required:
        if len(kinds[key]) != 1:
            rec["issues"].append(key + "_count_" + str(len(kinds[key])))
    for p in files:
        if p.stat().st_size == 0:
            rec["issues"].append("empty_file:" + p.relative_to(case).as_posix())
    if len(kinds["metadata"]) == 1:
        p = kinds["metadata"][0]
        try:
            meta = metadata(p)
            rec["metadata_sha256"] = digest(p)
            rec["metadata"] = meta
            ids = numbers(meta.get("Fault ID", ""))
            if "Fault ID" not in meta or any(not v.is_integer() for v in ids):
                rec["issues"].append("missing_or_invalid_fault_id")
            rec["fault_ids"] = [int(v) for v in ids]
            if category == "no_fault" and ids:
                rec["issues"].append("normal_metadata_has_fault_id")
            if category != "no_fault" and not ids:
                rec["issues"].append("fault_metadata_has_no_fault_id")
            if category in KNOWN_IDS and ids != [KNOWN_IDS[category]]:
                rec["issues"].append("known_category_id_mismatch")
            caseid = meta.get("Case ID", meta.get("Real Flight case ID"))
            rec["testinfo_case_id"] = str(caseid)
            injection = numbers(meta.get("Fault injection time", ""))
            end = numbers(meta.get("Test end time", ""))
            if len(injection) != 1 or len(end) != 1 or injection[0] < 0 or end[0] < injection[0]:
                rec["issues"].append("invalid_planned_event_times")
            if not meta.get("Control Sequence"):
                rec["issues"].append("missing_control_sequence")
        except Exception as exc:
            rec["issues"].append("metadata_error:" + repr(exc))
    for p in kinds["ulog"]:
        with p.open("rb") as f:
            head = f.read(16)
        if len(head) != 16 or head[:7] != bytes([85, 76, 111, 103, 1, 18, 53]):
            rec["issues"].append("invalid_ulog_header")
    for p in kinds["bag"]:
        with p.open("rb") as f:
            if not f.read(13).startswith(b"#ROSBAG V2.0"):
                rec["issues"].append("invalid_rosbag_header")
    return rec


def inventory(root, out, workers):
    cases = []
    for domain in ["HIL", "SIL", "Real"]:
        domain_root = root / domain
        if not domain_root.is_dir():
            raise ValueError("Missing domain: " + str(domain_root))
        cases.extend((p, domain_root) for mode in sorted(domain_root.iterdir()) if mode.is_dir()
                     for kind in sorted(mode.iterdir()) if kind.is_dir()
                     for p in sorted(kind.iterdir()) if p.is_dir())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda pair: inspect_case(*pair), cases))
    with (out / "case_inventory.jsonl").open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
    summary = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
               "scope": "All case metadata and file signatures; no processed signals, model inference, relabeling or training. Header validity is not full-file integrity.",
               "cases": len(rows), "domains": {}, "issues": dict(Counter(i for r in rows for i in r["issues"]))}
    for domain in ["HIL", "SIL", "Real"]:
        group = [r for r in rows if r["domain"] == domain]
        summary["domains"][domain] = {"cases": len(group), "bytes": sum(r["total_bytes"] for r in group),
            "categories": dict(Counter(r["category"] for r in group)),
            "modes": dict(Counter(r["mode"] for r in group)),
            "fault_ids_by_category": {c: dict(Counter(str(r.get("fault_ids")) for r in group if r["category"] == c)) for c in sorted(set(r["category"] for r in group))},
            "cases_with_issues": sum(bool(r["issues"]) for r in group)}
    (out / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def inspect_timeline(rec):
    from pyulog import ULog
    result = {"case_uid": rec["case_uid"], "domain": rec["domain"], "mode": rec["mode"],
              "category": rec["category"], "issues": [], "warnings": []}
    try:
        paths = rec["sources"]["ulog"]
        if len(paths) != 1:
            raise ValueError("Expected one original ULog")
        p = Path(paths[0])
        before = (p.stat().st_size, p.stat().st_mtime_ns)
        log = ULog(str(p), message_name_filter_list=["rfly_ctrl_lxl", "sensor_combined", "vehicle_local_position", "vehicle_gps_position"])
        result["ulog_sha256"] = digest(p)
        result["ulog_start_s"] = log.start_timestamp / 1e6
        result["ulog_end_s"] = log.last_timestamp / 1e6
        result["dropout_count"] = len(log.dropouts)
        topics = []
        onset = None
        all_ids = set()
        for d in log.data_list:
            if "timestamp" not in d.data:
                continue
            times = d.data["timestamp"]
            mask = active_ids(d.data["id"]) if d.name == "rfly_ctrl_lxl" else np.zeros(len(times), dtype=bool)
            item = {"name": d.name, "instance": int(d.multi_id), **timeline(times, mask, 1e6)}
            if d.name == "rfly_ctrl_lxl":
                ids = sorted(set(int(v) for v in d.data["id"]))
                all_ids.update(ids)
                item["ids"] = ids
                if d.multi_id == 0:
                    onset = item["onset_s"]
            else:
                fields = [k for k, v in d.data.items() if k not in ("timestamp", "timestamp_sample") and np.issubdtype(v.dtype, np.floating)]
                item["nonfinite_field_counts"] = {k: int((~np.isfinite(d.data[k])).sum()) for k in fields if not np.isfinite(d.data[k]).all()}
                if d.name == "sensor_combined" and len(times) > 1:
                    delta = np.diff(np.asarray(times, dtype=float)) / 1e6
                    item["median_dt_s"] = float(np.median(delta))
                    item["max_gap_s"] = float(np.max(delta))
            if item["backward_steps"]:
                result["issues"].append("topic_clock_reversal:" + d.name)
            topics.append(item)
        result["topics"] = topics
        result["fault_command_onset_s"] = onset
        result["control_ids"] = sorted(all_ids)
        if not any(t["name"] == "sensor_combined" for t in topics):
            result["issues"].append("missing_sensor_combined")
        if not any(t["name"] == "rfly_ctrl_lxl" for t in topics):
            result["warnings"].append("no_fault_command_topic_not_proof_of_health")
        if rec["category"] == "no_fault" and onset is not None:
            result["warnings"].append("nominal_normal_has_fault_command")
        if rec["category"] != "no_fault" and onset is None:
            result["warnings"].append("fault_case_no_observed_command")
        expected = rec.get("fault_ids", [])
        if expected and not set(expected).issubset(all_ids):
            result["warnings"].append("metadata_fault_id_not_seen_in_ulog")
        if before != (p.stat().st_size, p.stat().st_mtime_ns):
            result["issues"].append("ulog_changed_during_read")
        if rec["domain"] != "Real":
            paths = rec["sources"]["uav_state"]
            if len(paths) != 1:
                raise ValueError("Expected one GTD workbook")
            p = Path(paths[0])
            result["gtd_sha256"] = digest(p)
            wb = load_workbook(p, read_only=True, data_only=True)
            try:
                rows = wb.active.iter_rows(values_only=True)
                header = next(rows)
                ti, si, ci = (header.index(k) for k in ["uavTime", "fault_state", "cmd"])
                times, states, commands = [], [], []
                for row in rows:
                    if all(v is None for v in row):
                        continue
                    times.append(float(row[ti]))
                    states.append(float(row[si]))
                    commands.append(str(row[ci]))
            finally:
                wb.close()
            if not np.isfinite(states).all():
                raise ValueError("Nonfinite GTD states")
            result["gtd"] = timeline(times, np.asarray(states) > 0, 1000)
            result["gtd_state_values"] = sorted(set(states))
            result["first_gtd_command"] = commands[0]
            if result["gtd"]["backward_steps"]:
                result["issues"].append("gtd_clock_reversal")
            gtd_onset = result["gtd"]["onset_s"]
            if onset is not None and gtd_onset is not None:
                result["gtd_minus_ulog_command_s"] = gtd_onset - onset
                if abs(gtd_onset - onset) > 1:
                    result["warnings"].append("gtd_ulog_onset_difference_over_1s_review_only")
            if (gtd_onset is None) != (onset is None):
                result["warnings"].append("gtd_ulog_activation_presence_disagrees")
            if rec["category"] == "no_fault" and gtd_onset is not None:
                result["warnings"].append("nominal_normal_has_active_gtd")
            start = max(result["gtd"]["first_s"], result["ulog_start_s"])
            end = min(result["gtd"]["last_s"], result["ulog_end_s"])
            result["gtd_ulog_clock_overlap_s"] = end - start
            if end < start:
                result["issues"].append("gtd_ulog_no_clock_overlap")
    except Exception as exc:
        result["issues"].append("read_error:" + repr(exc))
    return result


def timelines(out, workers):
    rows = [json.loads(line) for line in (out / "case_inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["domain"], r["mode"], r["category"])].append(r)
    selected = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r["case_uid"])
        for r in [ordered[0], ordered[-1]]:
            selected[r["case_uid"]] = r
    old = ROOT / "artifacts/dataset_suitability_audit_20261030_r2/raw_audit.jsonl"
    if old.exists():
        known = {json.loads(line)["case_uid"].lower() for line in old.read_text(encoding="utf-8").splitlines()}
        for r in rows:
            if r["case_uid"].lower() in known:
                selected[r["case_uid"]] = r
    for r in rows:
        if r["issues"]:
            selected[r["case_uid"]] = r
    plan = {"method": "First and last case by path in each nonempty domain/mode/category stratum, all 70 previous raw audit cases, and all cases with inventory flags. Deterministic, not random or exhaustive signal review.",
            "count": len(selected), "case_uids": sorted(selected),
            "real_scope": "User-requested raw integrity/metadata inspection includes frozen Real test roles; no model/probability/threshold operations. This is data inspection, not a new blind performance test."}
    (out / "timeline_selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "raw_timelines.jsonl").open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = []
            for i, result in enumerate(pool.map(inspect_timeline, [selected[k] for k in sorted(selected)]), 1):
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + chr(10))
                stream.flush()
                if i % 20 == 0:
                    print("TIMELINES", i, "/", len(selected), flush=True)
    summary = {"sampled_cases": len(results), "domain_counts": dict(Counter(r["domain"] for r in results)),
               "issues": dict(Counter(v for r in results for v in r["issues"])),
               "warnings": dict(Counter(v for r in results for v in r["warnings"])),
               "scope": plan, "completed_at": datetime.now(timezone.utc).isoformat()}
    differences = [r["gtd_minus_ulog_command_s"] for r in results if "gtd_minus_ulog_command_s" in r]
    summary["gtd_ulog_onset_delta_s"] = {"n": len(differences), "min": min(differences) if differences else None,
                                         "max": max(differences) if differences else None,
                                         "median": float(np.median(differences)) if differences else None}
    (out / "timeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in summary.items() if k != "scope"}


def self_check():
    assert numbers("[123450.]") == [123450.0]
    assert numbers("[]") == []
    assert active_ids([0, 1500, 123450, 123549, 123550]).tolist() == [False, False, True, True, False]
    try:
        active_ids([float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite ID not rejected")
    row = timeline([1000, 2000, 2000, 4000], [False, False, True, False], 1000)
    assert row["onset_s"] == 2 and row["duplicate_steps"] == 1 and row["active_to_inactive"] == 1
    assert timeline([2, 1], [False, False], 1)["backward_steps"] == 1
    assert timeline([1, 2], [False, False], 1)["onset_s"] is None
    print("SELF_CHECK_PASS: numeric metadata, fault IDs, nonfinite rejection, timeline units, duplicates, reversal and absent onset")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("D:/RflyMAD"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--phase", choices=["inventory", "timelines"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.output or not args.phase or not 1 <= args.workers <= 8:
        ap.error("Require --output, --phase, and 1..8 workers")
    root, out = args.root.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError("Never write audit outputs beneath raw dataset")
    if args.phase == "inventory":
        out.mkdir(parents=True, exist_ok=False)
        result = inventory(root, out, args.workers)
    else:
        if not (out / "case_inventory.jsonl").exists() or (out / "timeline_selection.json").exists():
            raise ValueError("Require existing inventory and unused timeline outputs")
        result = timelines(out, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
