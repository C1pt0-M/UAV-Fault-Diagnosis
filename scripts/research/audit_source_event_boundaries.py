"""Audit frozen SIL/HIL train/val raw events against existing cropped CSVs.

Read-only dataset access. Diagnostic references are NOT repaired labels or a new
training dataset. Does not load models, final tests, native converters or Real.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pyulog import ULog

from audit_rflymad_raw_sources import active_ids, digest, metadata, numbers

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv")
OLD = ROOT / "artifacts/dataset_suitability_audit_20261030_r2"
LINKS = ROOT / "artifacts/source_training_coverage_audit_20261030_r2"
EXPECTED_MANIFEST = "272a91aeba5b563ebfa23fff8d295ac53372ccac391c835dccf2b5660d9d6ce5"
EXPECTED_LINKS = {"train": "e184e65ee588d051904b5f127ff508c972b1ae626ae8d651963bf5e9bdcc611f",
                  "val": "5963804ae701d9aa6da2eeb3c6c4b491f5bd0e0e0c6863413244f2713f3f7ad0"}


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite timestamps or state")
    return values


def hold(times, values, query):
    times, query = finite(times), finite(query)
    values = np.asarray(values)
    if times.ndim != 1 or not len(times) or values.shape != times.shape or (np.diff(times) < 0).any():
        raise ValueError("Empty, mismatched or reversed raw timeline")
    indices = np.searchsorted(times, query, side="right") - 1
    valid = (query >= times[0]) & (query <= times[-1])
    return values[np.maximum(indices, 0)], valid


def intervals(times, flags):
    times = finite(times)
    flags = np.asarray(flags, dtype=bool)
    if not len(times) or flags.shape != times.shape or (np.diff(times) < 0).any():
        raise ValueError("Invalid event timeline")
    # At a repeated timestamp retain the final recorded state.
    keep = np.r_[times[1:] != times[:-1], True]
    times, flags = times[keep], flags[keep]
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    stops = np.flatnonzero(~flags & np.r_[False, flags[:-1]])
    result = []
    for start in starts:
        later = stops[stops > start]
        result.append({"start_s": float(times[start]),
                       "end_s": float(times[later[0]]) if len(later) else None,
                       "last_observed_s": float(times[-1])})
    return result


def read_gtd(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        header = next(rows)
        ti, si = header.index("uavTime"), header.index("fault_state")
        times, states = [], []
        for row in rows:
            if all(v is None for v in row):
                continue
            times.append(row[ti])
            states.append(row[si])
    finally:
        wb.close()
    times, states = finite(times) / 1000, finite(states)
    if not len(times) or (np.diff(times) < 0).any():
        raise ValueError("Empty or reversed GTD clock")
    if not set(states).issubset({0, 1}):
        raise ValueError("GTD states outside verified binary definition")
    return times, states.astype(bool)


def supported_segments(query, flags, valid):
    # Diagnostic sample runs only. No claim about causality or physical health.
    output = []
    for active, name in [(False, "no_recorded_activation"), (True, "recorded_active")]:
        mask = valid & (flags == active)
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            output.append({"state": name, "start_row": int(start), "end_row_exclusive": int(stop),
                           "first_sample_s": float(query[start]), "last_sample_s": float(query[stop - 1]),
                           "samples": int(stop - start)})
    return sorted(output, key=lambda r: r["start_row"])


def case_audit(task):
    row, old, links = task
    uid = row["case_uid"]
    result = {k: row[k] for k in ["case_uid", "domain", "split", "flight_status", "source_fault_type", "source_path", "final_output_path"]}
    result.update(status="review_required", issues=[], warnings=[])
    assessments = []
    try:
        raw = Path(row["source_path"])
        paths = [raw / "TrueData/UAVState_data.xlsx", raw / "TestInfo.csv", Path(row["final_output_path"])]
        logs = list((raw / "Log").glob("*.ulg"))
        if len(logs) != 1:
            raise ValueError("Expected exactly one original ULog")
        paths += logs
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        gtd_path, meta_path, csv_path, log_path = paths
        csv_bytes = csv_path.read_bytes()
        csv_hash = hashlib.sha256(csv_bytes).hexdigest()
        if csv_hash != old["csv_sha256"]:
            raise ValueError("Processed CSV identity changed since prior audit")
        meta_hash = digest(meta_path)
        if meta_hash != old["testinfo_sha256"]:
            raise ValueError("TestInfo identity changed since prior audit")
        frame = pd.read_csv(io.BytesIO(csv_bytes), usecols=["Timestamp", "UAVState_data_fault_state", "_rfly_ctrl_lxl_0_id"])
        query = finite(frame.Timestamp) / 1e6
        if len(query) != int(row["row_count"]) or len(query) != old["rows"] or not np.allclose(np.diff(query), .02, rtol=0, atol=1e-6):
            raise ValueError("Changed processed grid or row count")
        processed_state = finite(frame.UAVState_data_fault_state) > 0
        gt, gs = read_gtd(gtd_path)
        reference, covered = hold(gt, gs, query)
        gtd_events = intervals(gt, gs)
        log = ULog(str(log_path), message_name_filter_list=["rfly_ctrl_lxl", "sensor_combined"])
        topics = [d for d in log.data_list if d.name == "rfly_ctrl_lxl" and d.multi_id == 0]
        if len(topics) != 1:
            raise ValueError("Missing or ambiguous original fault-command topic")
        command = topics[0].data
        ct = finite(command["timestamp"]) / 1e6
        cf = active_ids(command["id"])
        command_reference, command_covered = hold(ct, cf, query)
        command_events = intervals(ct, cf)
        expected = numbers(metadata(meta_path)["Fault ID"])
        recorded_ids = set(int(v) for v in command["id"])
        if expected and not set(expected).issubset(recorded_ids):
            result["warnings"].append("metadata_id_not_observed_in_command_log")
        if bool(gtd_events) != bool(command_events):
            result["warnings"].append("gtd_and_command_activation_presence_disagree")
        if row["source_fault_type"] == "no_fault" and (gtd_events or command_events):
            result["issues"].append("normal_case_has_recorded_activation")
        if row["source_fault_type"] != "no_fault" and not gtd_events:
            result["warnings"].append("fault_case_without_gtd_activation")
        if not covered.all():
            result["warnings"].append("crop_not_fully_covered_by_gtd")
        if not command_covered.all():
            result["warnings"].append("crop_not_fully_covered_by_command_topic")
        sensors = [d for d in log.data_list if d.name == "sensor_combined" and d.multi_id == 0]
        if len(sensors) != 1:
            raise ValueError("Missing or ambiguous original inertial topic")
        sensor = sensors[0].data
        st = finite(sensor["timestamp"]) / 1e6
        if len(st) < 2 or (np.diff(st) < 0).any():
            raise ValueError("Invalid inertial clock")
        nonfinite_rows = np.zeros(len(st), dtype=bool)
        nonfinite_fields = {}
        for name, values in sensor.items():
            if np.issubdtype(values.dtype, np.floating):
                bad = ~np.isfinite(values)
                if bad.any():
                    nonfinite_fields[name] = int(bad.sum())
                    nonfinite_rows |= bad
        if nonfinite_fields:
            result["warnings"].append("nonfinite_inertial_fields")
        gap_indices = np.flatnonzero(np.diff(st) > .1)
        gap_pairs = np.column_stack([st[gap_indices], st[gap_indices + 1]])
        result.update(
            csv_sha256=csv_hash, testinfo_sha256=meta_hash, raw_gtd_sha256=digest(gtd_path), ulog_sha256=digest(log_path),
            processed_rows=len(query), crop_start_s=float(query[0]), crop_end_s=float(query[-1]),
            gtd_start_s=float(gt[0]), gtd_end_s=float(gt[-1]),
            gtd_duplicate_steps=int((np.diff(gt) == 0).sum()), gtd_max_gap_s=float(np.diff(gt).max()) if len(gt) > 1 else None,
            gtd_duplicate_conflicting_state_steps=int(((np.diff(gt) == 0) & (gs[1:] != gs[:-1])).sum()),
            gtd_events=gtd_events, command_events=command_events, command_ids=sorted(recorded_ids),
            gtd_reference_covered_rows=int(covered.sum()), command_reference_covered_rows=int(command_covered.sum()),
            covered_raw_gtd_active_rows=int((reference & covered).sum()),
            processed_vs_gtd_disagreement_rows=int(((processed_state != reference) & covered).sum()),
            both_reference_covered_rows=int((covered & command_covered).sum()),
            gtd_vs_command_disagreement_rows=int(((reference != command_reference) & covered & command_covered).sum()),
            raw_onset_after_crop=bool(gtd_events and gtd_events[0]["start_s"] > query[-1]),
            raw_onset_before_crop=bool(gtd_events and gtd_events[0]["start_s"] < query[0]),
            gtd_onset_minus_command_s=gtd_events[0]["start_s"] - command_events[0]["start_s"] if gtd_events and command_events else None,
            ulog_start_s=log.start_timestamp / 1e6, ulog_end_s=log.last_timestamp / 1e6,
            sensor_start_s=float(st[0]), sensor_end_s=float(st[-1]), sensor_max_gap_s=float(np.diff(st).max()),
            sensor_gaps_over_0_1s=gap_pairs.tolist(), sensor_nonfinite_fields=nonfinite_fields,
            ulog_dropout_events=len(log.dropouts),
            segments=supported_segments(query, reference, covered),
            comparison_basis="Recorded GTD activation with last duplicate retained; command topic is a distinct cross-check, not physical damage truth.")
        if not result["covered_raw_gtd_active_rows"]:
            result["crop_event_class"] = "no_recorded_activation_in_covered_crop"
        elif int((reference & covered).sum()) == len(query):
            result["crop_event_class"] = "recorded_active_throughout_crop"
        else:
            result["crop_event_class"] = "contains_activation_transition_or_partial_coverage"
        for link in links:
            start, end = int(link["window_start_row"]), int(link["window_end_row"])
            if not 0 <= start < end <= len(query) or end - start != 150:
                raise ValueError("Selected window outside processed crop or unexpected length")
            old_label = int(link["label"])
            if old_label not in (0, 1) or old_label != int(processed_state[end - 1]):
                raise ValueError("Existing window endpoint label disagrees with prior audit")
            endpoint_valid = bool(covered[end - 1])
            value = int(reference[end - 1]) if endpoint_valid else None
            window_start, window_end = float(query[start]), float(query[end - 1])
            intersects_gap = bool(len(gap_pairs) and ((gap_pairs[:, 0] < window_end) & (gap_pairs[:, 1] > window_start)).any())
            bad_sensor = bool(((st >= window_start) & (st <= window_end) & nonfinite_rows).any())
            item = {"case_uid": uid, "domain": row["domain"], "split": row["split"],
                    "cache_index": int(link["cache_index"]), "window_start_row": start, "window_end_row": end,
                    "start_s": window_start, "end_s": window_end, "existing_label": old_label,
                    "gtd_endpoint_reference": value, "gtd_endpoint_covered": endpoint_valid,
                    "gtd_full_window_covered": bool(covered[start:end].all()),
                    "command_endpoint_reference": int(command_reference[end - 1]) if command_covered[end - 1] else None,
                    "endpoint_reference_disagrees": bool(old_label != value) if endpoint_valid else None,
                    "sensor_time_range_covers_window": bool(window_start >= st[0] and window_end <= st[-1]),
                    "sensor_gap_over_0_1s_intersects_window": intersects_gap,
                    "sensor_nonfinite_sample_in_window": bad_sensor}
            assessments.append(item)
        result["selected_windows"] = len(assessments)
        result["endpoint_reference_covered_windows"] = sum(r["gtd_endpoint_covered"] for r in assessments)
        result["full_reference_covered_windows"] = sum(r["gtd_full_window_covered"] for r in assessments)
        result["selected_label_disagreements"] = sum(r["endpoint_reference_disagrees"] is True for r in assessments)
        result["selected_old_fault_raw_normal"] = sum(r["existing_label"] == 1 and r["gtd_endpoint_reference"] == 0 for r in assessments)
        result["selected_old_normal_raw_fault"] = sum(r["existing_label"] == 0 and r["gtd_endpoint_reference"] == 1 for r in assessments)
        result["selected_gap_crossing_windows"] = sum(r["sensor_gap_over_0_1s_intersects_window"] for r in assessments)
        if any(before[str(p)] != (p.stat().st_size, p.stat().st_mtime_ns) for p in paths):
            raise ValueError("Source file size/mtime changed during read")
        result["source_stat_unchanged"] = True
        result["status"] = "diagnostic_complete" if not result["issues"] else "review_required"
    except Exception as exc:
        result["issues"].append(type(exc).__name__ + ": " + str(exc))
        assessments = []
    return result, assessments


def source_tasks():
    if digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("Frozen manifest identity mismatch")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        cases = [r for r in csv.DictReader(f) if r["domain"] in ("SIL", "HIL")
                 and r["split"] in ("train", "val") and r["level2_label"] in ("normal", "motor", "sensor")]
    if Counter(r["split"] for r in cases) != {"train": 2580, "val": 555}:
        raise ValueError("Unexpected source scope")
    old = {r["case_uid"]: r for r in (json.loads(line) for line in (OLD / "case_audit.jsonl").read_text(encoding="utf-8").splitlines())}
    if set(old) != {r["case_uid"] for r in cases}:
        raise ValueError("Prior source audit membership mismatch")
    links = defaultdict(list)
    for split in ("train", "val"):
        p = LINKS / (split + "_source_window_links.csv")
        if digest(p) != EXPECTED_LINKS[split]:
            raise ValueError("Frozen window-link identity mismatch")
        with p.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                links[r["case_uid"]].append(r)
    if sum(map(len, links.values())) != 200640:
        raise ValueError("Unexpected sampled source windows")
    tasks = []
    for r in cases:
        prev = old[r["case_uid"]]
        if any(r[k] != prev[k] for k in ["domain", "split", "source_path", "final_output_path"]):
            raise ValueError("Prior source identity changed")
        if len(links[r["case_uid"]]) != 64:
            raise ValueError("Expected 64 existing selected windows per case")
        tasks.append((r, prev, links[r["case_uid"]]))
    return tasks


def summarize(records):
    good = [r for r in records if r["status"] == "diagnostic_complete"]
    result = {"cases_read": len(records), "diagnostic_complete": len(good),
              "issues": dict(Counter(s for r in records for s in r["issues"])),
              "warnings": dict(Counter(s for r in records for s in r["warnings"])),
              "domains": dict(Counter(r["domain"] for r in records)), "by_split": {}}
    for split in ("train", "val"):
        group = [r for r in good if r["split"] == split]
        result["by_split"][split] = {"cases": len(group),
            "cases_with_label_disagreement": sum(r["selected_label_disagreements"] > 0 for r in group),
            "raw_onset_after_crop_cases": sum(r["raw_onset_after_crop"] for r in group),
            "crop_event_classes": dict(Counter(r["crop_event_class"] for r in group))}
        for key in ["selected_windows", "endpoint_reference_covered_windows", "full_reference_covered_windows",
                    "selected_label_disagreements", "selected_old_fault_raw_normal", "selected_old_normal_raw_fault",
                    "selected_gap_crossing_windows"]:
            result["by_split"][split][key] = sum(r[key] for r in group)
    return result


def self_check():
    values, valid = hold([1, 2, 2, 4], [0, 0, 1, 0], [.5, 1, 2, 3, 4, 5])
    assert values.tolist() == [0, 0, 1, 1, 0, 0]
    assert valid.tolist() == [False, True, True, True, True, False]
    assert intervals([1, 2, 2, 3, 4], [0, 0, 1, 1, 0]) == [{"start_s": 2., "end_s": 4., "last_observed_s": 4.}]
    assert intervals([1, 2], [0, 0]) == []
    assert intervals([1, 2], [0, 1])[0]["end_s"] is None
    segs = supported_segments(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1], dtype=bool), np.array([0, 1, 1, 0], dtype=bool))
    assert [(r["start_row"], r["end_row_exclusive"]) for r in segs] == [(1, 2), (2, 3)]
    for ts in [[2, 1], [], [1, float("nan")]]:
        try:
            hold(ts, np.zeros(len(ts)), [1])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid clock accepted")
    print("SELF_CHECK_PASS: duplicate-last, coverage masks, before/after crop, event intervals, uncovered segments and invalid clocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Optional explicit smoke scope, not a complete audit")
    ap.add_argument("--exclude-completed", type=Path, help="Prior immutable audit directory; continue into a fresh output")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if args.output is None or args.limit < 0:
        ap.error("Require --output and nonnegative --limit")
    out = args.output.resolve()
    for forbidden in [Path("D:/RflyMAD").resolve(), Path("D:/UAV-Data").resolve()]:
        if out == forbidden or forbidden in out.parents:
            raise ValueError("Do not write beneath raw or processed dataset roots")
    tasks = source_tasks()
    continuation = None
    if args.exclude_completed:
        prior = args.exclude_completed.resolve()
        prior_plan = json.loads((prior / "audit_plan.json").read_text(encoding="utf-8"))
        completed_path = prior / "case_assessments.jsonl"
        completed = [json.loads(s) for s in completed_path.read_text(encoding="utf-8").splitlines()]
        done = {r["case_uid"] for r in completed}
        expected = {t[0]["case_uid"] for t in tasks}
        if prior_plan["manifest_sha256"] != EXPECTED_MANIFEST or len(done) != len(completed) or not done.issubset(expected):
            raise ValueError("Continuation membership or manifest identity mismatch")
        continuation = {"prior_directory": str(prior), "prior_case_sha256": digest(completed_path),
                        "prior_plan_sha256": digest(prior / "audit_plan.json"), "completed_cases": len(done)}
        tasks = [t for t in tasks if t[0]["case_uid"] not in done]
    if args.limit:
        tasks = tasks[:args.limit]
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "target_cases": len(tasks),
            "scope": "SIL/HIL source train/val only; diagnostic reference/coverage, not repaired data or clinical/physical fault truth. No Real/final-test/model/probability/native converter access.",
            "manifest_sha256": digest(MANIFEST), "script_sha256": digest(__file__),
            "reference_rule": "GTD last state at/before query timestamp, last duplicate retained; no extrapolation outside observed GTD range. No age-threshold claim.",
            "gap_rule": "Inertial gaps over 0.1 seconds are diagnostic flags, not frozen exclusion thresholds; all other input topics remain unaudited here.",
            "windows": "Existing selected 150-sample endpoint-labeled windows; not a new sampling policy.",
            "continuation": continuation,
            "cases": [t[0]["case_uid"] for t in tasks]}
    (out / "audit_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    with (out / "case_assessments.jsonl").open("x", encoding="utf-8") as cf, (out / "selected_window_assessments.csv").open("x", encoding="utf-8", newline="") as wf:
        writer = None
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, (record, windows) in enumerate(pool.map(case_audit, tasks, chunksize=1), 1):
                records.append(record)
                cf.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + chr(10))
                cf.flush()
                if windows:
                    if writer is None:
                        writer = csv.DictWriter(wf, fieldnames=list(windows[0]))
                        writer.writeheader()
                    writer.writerows(windows)
                    wf.flush()
                if count % 100 == 0:
                    progress = summarize(records)
                    progress.update(target_cases=len(tasks), elapsed_s=time.perf_counter() - started)
                    (out / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                    print("PROGRESS", count, "/", len(tasks), "issues", sum(bool(r["issues"]) for r in records), "elapsed_s", round(time.perf_counter() - started), flush=True)
    summary = summarize(records)
    summary.update(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
                   target_cases=len(tasks), full_source_scope=not bool(args.limit or args.exclude_completed),
                   all_cases_diagnostic_complete=len(records) == len(tasks) and all(r["status"] == "diagnostic_complete" for r in records),
                   meaning="Completion means diagnostic reads succeeded, not data repaired or all windows quality-approved.")
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
