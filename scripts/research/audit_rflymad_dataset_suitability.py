"""Source-only dataset suitability audit; no training, relabeling or final-test reads.

Checks every SIL/HIL train/val CSV against frozen cache windows, compares event
markers and TestInfo commands, then checks a deterministic raw-log subset.
Existing outputs are refused. Native processing tools are invoked read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))
from generate_window_manifest import _infer_fault_mask
from rfly_feature_contract import contract_columns, source_columns_for_domain, window_batch

BASE_SHA = "272a91aeba5b563ebfa23fff8d295ac53372ccac391c835dccf2b5660d9d6ce5"
VERSION_SHA = "5a279decaf8e3dfa8939f0881b8228ed497096e03cc2b462cac6fe8473cec921"
NORM_SHA = "a474fc77c2da1cf7e02b765446919f0d27f5ad64869ef882a678867a2108bf33"
CACHE_SHA = {"train": "f0992b19768575364f19378dd5109bcb640764035ecce4cc9d29132557a88d58",
             "val": "9a6063590b4b313c0d8df0dbc5213d88e3f5e078bb5be13c8054deb816ed2757"}
LINK_SHA = {"train": "e184e65ee588d051904b5f127ff508c972b1ae626ae8d651963bf5e9bdcc611f",
            "val": "5963804ae701d9aa6da2eeb3c6c4b491f5bd0e0e0c6863413244f2713f3f7ad0"}
FAULT_IDS = {"motor": 123450, "accelerometer": 123452, "gyroscope": 123453,
             "barometer": 123454, "magnetometer": 123455, "GPS": 123456}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen identity mismatch: {path}")


def numeric(values):
    output = np.asarray(values, dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite source values; audit never zero-fills")
    return output


def active_id(values):
    values = numeric(values)
    if not np.equal(values, np.round(values)).all():
        raise ValueError("Noninteger fault control ID")
    return ((values >= 123450) & (values <= 123459)) | ((values >= 123540) & (values <= 123549))


def first_time(times, mask):
    hits = np.flatnonzero(mask)
    return float(times[hits[0]]) if len(hits) else None


def last_observed(times, values, query):
    """Causal reference, retaining the last record at duplicate timestamps."""
    times, query = numeric(times), numeric(query)
    values = np.asarray(values)
    if times.ndim != 1 or not len(times) or values.shape != times.shape or (np.diff(times) < 0).any():
        raise ValueError("Invalid or out-of-order raw timeline")
    indices = np.searchsorted(times, query, side="right") - 1
    valid = (query >= times[0]) & (query <= times[-1])
    return values[np.maximum(indices, 0)], valid


def feature_matrix(frame, domain, contract):
    mapping = source_columns_for_domain(contract, domain)
    columns = contract_columns(contract)
    data = {name: numeric(frame[source]) for name, source in mapping.items()}
    data["acc_norm"] = np.sqrt(sum(data[name] * data[name] for name in ("accel_x", "accel_y", "accel_z")))
    return np.column_stack([data[name] for name in columns])


def motion_signals(windows, columns):
    index = {name: columns.index(name) for name in columns}
    velocity = windows[:, [index[name] for name in ("vel_x", "vel_y", "vel_z")]].astype(np.float64)
    return {"vertical_speed_rms": np.sqrt(np.mean(velocity[:, 2] ** 2, axis=1)),
            "horizontal_speed_rms": np.sqrt(np.mean(np.sum(velocity[:, :2] ** 2, axis=1), axis=1)),
            "thrust_mean": windows[:, index["actuator_ctrl_4"]].mean(1, dtype=np.float64),
            "baro_relative_mean": windows[:, index["baro_alt_relative"]].mean(1, dtype=np.float64)}


def read_testinfo(path):
    content = path.read_bytes()
    with io.StringIO(content.decode("utf-8-sig"), newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or any(len(row) != 2 for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Malformed TestInfo: {path}")
    data = dict(rows)
    commands = re.findall(r"'([^']*)'", data["Control Sequence"])
    if not commands:
        raise ValueError(f"Unrecognized control sequence: {path}")
    ids = [int(float(value)) for value in re.findall(r"\d+(?:\.\d*)?", data["Fault ID"])]
    return data, commands, ids, hashlib.sha256(content).hexdigest()


def audit_case(row, contract, versions, links, payloads):
    path = Path(row["final_output_path"])
    content = path.read_bytes()
    expected = versions.get(row["case_uid"], row["output_sha256"])
    current = hashlib.sha256(content).hexdigest()
    if current != expected:
        raise ValueError(f"CSV version mismatch: {row['case_uid']}")
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    if len(frame) != int(row["row_count"]):
        raise ValueError("Case length mismatch")
    stamp = numeric(frame.Timestamp) / 1e6
    if not np.allclose(np.diff(stamp), 0.02, rtol=0, atol=1e-6):
        raise ValueError("Unexpected processed time grid")
    state = _infer_fault_mask(frame)
    command = active_id(frame["_rfly_ctrl_lxl_0_id"])
    state_onset, command_onset = first_time(stamp, state), first_time(stamp, command)
    meta, commands, ids, meta_sha = read_testinfo(Path(row["source_path"]) / "TestInfo.csv")
    expected_ids = [] if row["source_fault_type"] == "no_fault" else [FAULT_IDS[row["source_fault_type"]]]
    selected = links[row["split"]][row["case_uid"]]
    indices = selected.cache_index.to_numpy(dtype=int)
    ends = selected.window_end_row.to_numpy(dtype=int) - 1
    if not np.array_equal(state[ends].astype(int), selected.label.to_numpy()):
        raise ValueError("Selected manifest labels disagree with actual CSV state")
    matrix = feature_matrix(frame, row["domain"], contract)
    columns = contract_columns(contract)
    windows = window_batch(matrix, columns, list(zip(selected.window_start_row, selected.window_end_row)))
    cache = payloads[row["split"]]
    if not np.array_equal(windows, cache["x"][indices].numpy()) or not np.array_equal(state[ends], cache["y"][indices].numpy()):
        raise ValueError(f"Full window/cache mismatch: {row['case_uid']}")
    record = {key: row[key] for key in ("case_uid", "domain", "split", "flight_status", "source_fault_type", "source_path", "final_output_path")}
    record.update(csv_sha256=current, testinfo_sha256=meta_sha, rows=len(frame),
                  duration_s=float(stamp[-1] - stamp[0]), state_active=bool(state.any()), command_active=bool(command.any()),
                  state_onset_s=state_onset, command_onset_s=command_onset,
                  state_minus_command_s=state_onset - command_onset if state_onset is not None and command_onset is not None else None,
                  row_marker_disagreements=int(np.sum(state != command)), sampled_marker_disagreements=int(np.sum(state[ends] != command[ends])),
                  sampled_windows=len(selected), testinfo_ids=ids, testinfo_ids_match=ids == expected_ids,
                  first_command=commands[0], control_sequence=meta["Control Sequence"],
                  fault_parameter=meta.get("Fault Parameter"), injection_time=meta.get("Fault injection time"),
                  commands=commands, processed_state_values=np.unique(frame["UAVState_data_fault_state"]).tolist(),
                  full_cache_features_and_labels_equal=True)
    return record, row["split"], indices, motion_signals(windows, columns), windows.mean(2, dtype=np.float64)


def raw_case(record, native, selected):
    from pyulog import ULog
    raw = Path(record["source_path"])
    raw_path = raw / "TrueData/UAVState_data.xlsx"
    ground = pd.read_excel(raw_path, usecols=["uavTime", "cmd", "fault_state"])
    ground = ground[["uavTime", "cmd", "fault_state"]]
    times = numeric(ground.uavTime) / 1000
    if (np.diff(times) < 0).any():
        raise ValueError("Raw GTD clock reverses")
    state = numeric(ground.fault_state) > 0
    processed = pd.read_csv(record["final_output_path"], usecols=["Timestamp", "UAVState_data_fault_state", "UAVState_data_cmd"])
    query = numeric(processed.Timestamp) / 1e6
    held, valid = last_observed(times, state, query)
    if not valid.any():
        raise ValueError("Raw/processed timelines do not overlap")
    array = np.array(ground.values.tolist())
    generated = native.CSVFile_extractor(str(raw)).freq_adjustment(array, query * 1000)
    native_state_equal = bool(np.array_equal(generated[:, 1].astype(float), processed.UAVState_data_fault_state.to_numpy()))
    native_cmd_equal = bool(np.array_equal(generated[:, 0].astype(str), processed.UAVState_data_cmd.astype(str).to_numpy()))
    raw_onset = first_time(times, state)
    endpoints = selected.window_end_row.to_numpy(dtype=int) - 1
    endpoint_valid = valid[endpoints]
    sampled_labels = selected.label.to_numpy(dtype=int)
    held_labels = held[endpoints].astype(int)
    result = {"case_uid": record["case_uid"], "raw_gtd_sha256": sha256(raw_path), "raw_rows": len(ground),
              "processed_start_s": float(query[0]), "processed_end_s": float(query[-1]),
              "raw_onset_after_processed_end": bool(raw_onset is not None and raw_onset > query[-1]),
              "sampled_windows": len(selected), "sampled_raw_reference_covered": int(endpoint_valid.sum()),
              "sampled_original_fault_labels": int(sampled_labels.sum()),
              "sampled_raw_reference_fault_labels": int(held_labels[endpoint_valid].sum()),
              "sampled_label_disagreements_with_raw_reference": int(np.sum(sampled_labels[endpoint_valid] != held_labels[endpoint_valid])),
              "raw_duplicate_timestamps": int(np.sum(np.diff(times) == 0)), "raw_state_active": bool(state.any()),
              "raw_state_onset_s": raw_onset,
              "raw_onset_from_first_record_s": raw_onset - times[0] if raw_onset is not None else None,
              "processed_minus_raw_gtd_onset_s": record["state_onset_s"] - raw_onset if record["state_onset_s"] is not None and raw_onset is not None else None,
              "raw_reference_covered_rows": int(valid.sum()),
              "processed_vs_timestamp_hold_disagreement_rows": int(np.sum((processed.UAVState_data_fault_state.to_numpy() > 0)[valid] != held[valid])),
              "native_array_dtype": str(array.dtype), "native_state_reproduces_processed": native_state_equal,
              "native_command_reproduces_processed": native_cmd_equal}
    files = sorted((raw / "Log").glob("*.ulg"))
    if len(files) != 1:
        raise ValueError("Expected one raw ULog")
    ulog = ULog(str(files[0]), message_name_filter_list=["rfly_ctrl_lxl", "vehicle_gps_position", "vehicle_local_position", "sensor_combined"])
    result["ulog_sha256"] = sha256(files[0])
    result["raw_logged_topics"] = [{"topic": item.name, "instance": int(item.multi_id), "rows": len(item.data["timestamp"])} for item in ulog.data_list]
    topics = [item for item in ulog.data_list if item.name == "rfly_ctrl_lxl" and item.multi_id == 0]
    if topics:
        item = topics[0].data
        onset = first_time(numeric(item["timestamp"]) / 1e6, active_id(item["id"]))
        result.update(raw_ulog_command_active=onset is not None, raw_ulog_command_onset_s=onset,
                      raw_gtd_minus_raw_command_s=raw_onset - onset if raw_onset is not None and onset is not None else None,
                      raw_control_ids=np.unique(item["id"]).astype(int).tolist())
    else:
        result["raw_control_topic_missing"] = True
    return result


def compact(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())} if len(values) else {"n": 0}


def run(args):
    started = time.perf_counter()
    require_hash(args.manifest, BASE_SHA)
    require_hash(args.versions, VERSION_SHA)
    require_hash(args.cache / "norm_stats.json", NORM_SHA)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    columns = contract_columns(contract)
    stats = json.loads((args.cache / "norm_stats.json").read_text())
    if columns != stats["columns"]:
        raise ValueError("Feature contract and cache normalization disagree")
    cases = pd.read_csv(args.manifest, dtype=str).fillna("")
    cases = cases[cases.domain.isin(["SIL", "HIL"]) & cases.split.isin(["train", "val"]) & cases.level2_label.isin(["normal", "motor", "sensor"])]
    if Counter(cases.split) != {"train": 2580, "val": 555}:
        raise ValueError("Unexpected source scope")
    version_records = json.loads(args.versions.read_text())["versions"]["records"]
    versions = {row["case_uid"]: row["sha256"] for row in version_records}
    payloads, links, frames, means, signals = {}, {}, {}, {}, {}
    torch.set_num_threads(4)
    for split in ("train", "val"):
        require_hash(args.cache / (split + ".pt"), CACHE_SHA[split])
        require_hash(args.links / (split + "_source_window_links.csv"), LINK_SHA[split])
        payloads[split] = torch.load(args.cache / (split + ".pt"), map_location="cpu", mmap=True, weights_only=True)
        frame = pd.read_csv(args.links / (split + "_source_window_links.csv"))
        if not np.array_equal(frame.cache_index, np.arange(len(frame))):
            raise ValueError("Noncanonical source links")
        frames[split] = frame
        links[split] = {str(key): group for key, group in frame.groupby("case_uid", sort=False)}
        means[split] = np.empty((len(frame), 26), dtype=np.float64)
        signals[split] = {name: np.empty(len(frame)) for name in ("vertical_speed_rms", "horizontal_speed_rms", "thrust_mean", "baro_relative_mean")}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    records = []
    with (out / "case_audit.jsonl").open("x", encoding="utf-8") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = pool.map(lambda row: audit_case(row, contract, versions, links, payloads), cases.to_dict("records"))
            for record, split, indices, motion, average in tasks:
                records.append(record)
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                means[split][indices] = average
                for name in signals[split]:
                    signals[split][name][indices] = motion[name]
                if len(records) % 250 == 0 or len(records) == len(cases):
                    journal.flush()
                    print(f"Verified {len(records)}/{len(cases)} source CSVs and complete capped windows; {time.perf_counter()-started:.1f}s", flush=True)
    table = pd.DataFrame(records)
    table.drop(columns=["commands", "testinfo_ids", "processed_state_values"]).to_csv(out / "case_audit.csv", index=False)
    first_commands = table.groupby(["split", "domain", "flight_status", "source_fault_type", "first_command"], sort=True).size().reset_index(name="cases")
    first_commands.to_csv(out / "first_command_coverage.csv", index=False)
    by_group = []
    for key, part in table.groupby(["split", "domain", "source_fault_type"], sort=True):
        by_group.append({"split": key[0], "domain": key[1], "fault_type": key[2], "cases": len(part),
                         "state_without_command": int((part.state_active & ~part.command_active).sum()),
                         "command_without_state": int((~part.state_active & part.command_active).sum()),
                         "state_minus_command_s": compact(part.state_minus_command_s.dropna()),
                         "sampled_marker_disagreements": int(part.sampled_marker_disagreements.sum()),
                         "sampled_windows": int(part.sampled_windows.sum())})
    background = {}
    for split in ("train", "val"):
        frame = frames[split]
        quiet = signals[split]["vertical_speed_rms"] <= 0.2
        hover = frame.flight_status.eq("hover").to_numpy()
        background[split] = {"hover_low_vertical_rms_origin_counts": dict(Counter(frame.loc[quiet & hover, "normal_origin"])),
                             "original_normal_by_time": []}
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 120), (120, float("inf"))]:
            mask = frame.normal_origin.eq("original_normal_case") & hover & frame.end_from_log_start_s.ge(lo) & frame.end_from_log_start_s.lt(hi)
            background[split]["original_normal_by_time"].append({"seconds": [lo, hi if np.isfinite(hi) else None], "windows": int(mask.sum()),
                "cases": int(frame.loc[mask, "case_uid"].nunique()), **{name: compact(values[mask]) for name, values in signals[split].items()}})
        np.savez_compressed(out / (split + "_window_diagnostics.npz"), means=means[split], **signals[split])
    train_mask = frames["train"].flight_status.eq("hover") & frames["train"].normal_origin.eq("original_normal_case") & (signals["train"]["vertical_speed_rms"] <= 0.2)
    val_mask = frames["val"].flight_status.eq("hover") & frames["val"].normal_origin.eq("original_normal_case") & (signals["val"]["vertical_speed_rms"] <= 0.2)
    coverage = []
    for domain in ("SIL", "HIL"):
        train = means["train"][train_mask & frames["train"].domain.eq(domain)]
        val = means["val"][val_mask & frames["val"].domain.eq(domain)]
        if not len(train) or not len(val):
            raise ValueError("Missing healthy low-vertical-motion source cohort")
        low, high = np.quantile(train, [0.01, 0.99], axis=0)
        coverage.append({"domain": domain, "train_windows": len(train), "val_windows": len(val),
                         "per_channel_val_fraction_outside_train_p01_p99": dict(zip(columns, ((val < low) | (val > high)).mean(0).tolist()))})
    native_path = args.tools / "fileprocess.py"
    spec = importlib.util.spec_from_file_location("rfly_native_readonly", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    selected = table[table.flight_status.eq("hover")].groupby(["domain", "source_fault_type"], sort=True).head(1)
    extreme = table.dropna(subset=["state_minus_command_s"]).sort_values("state_minus_command_s")
    selected = pd.concat([selected, extreme.groupby("domain", sort=True).head(1), extreme.groupby("domain", sort=True).tail(1),
                          table[table.state_active & ~table.command_active],
                          table[table.flight_status.eq("hover") & table.source_fault_type.eq("no_fault") & table.split.eq("val")]])
    selected = selected.drop_duplicates("case_uid")
    raw_records = []
    with (out / "raw_audit.jsonl").open("x", encoding="utf-8") as journal:
        for record in selected.to_dict("records"):
            result = raw_case(record, native, links[record["split"]][record["case_uid"]])
            raw_records.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            print("Raw checked", len(raw_records), "/", len(selected), record["case_uid"], flush=True)
    report = {"status": "COMPLETED_SOURCE_DATASET_SUITABILITY_AUDIT", "finished_at": datetime.now(timezone.utc).isoformat(),
              "elapsed_s": time.perf_counter()-started, "scope": "SIL/HIL train/val only; no Real, final tests, training, relabeling, cache rebuild or deployment.",
              "manifest_sha256": BASE_SHA, "version_audit_sha256": VERSION_SHA, "contract_sha256": sha256(args.contract),
              "runner_sha256": sha256(__file__), "native_processing_sha256": sha256(native_path),
              "cases": len(records), "source_windows_fully_reconstructed": int(table.sampled_windows.sum()),
              "testinfo_id_mismatches": int((~table.testinfo_ids_match).sum()), "event_marker_groups": by_group,
              "background_definition": "Nominal hover, RMS of window NED vertical velocity <=0.2 m/s. Descriptive proxy, not verified flight phase or a new selection rule.",
              "background": background, "healthy_hover_mean_feature_coverage": coverage,
              "raw_subset_rule": "First manifest hover case per domain/type; min/max processed marker delay per domain; ALL state-without-command cases; all 12 original healthy hover val cases; deduplicate.",
              "raw_cases": len(raw_records), "raw_records": raw_records,
              "observability_boundary": "No classifier fitted. Input mapping and descriptive coverage do not prove a Bayes-error bound or absence of detectable fault signals. Raw/control/GTD markers are not independent physical damage measurements."}
    for split in ("train", "val"):
        require_hash(args.cache / (split + ".pt"), CACHE_SHA[split])
    require_hash(args.cache / "norm_stats.json", NORM_SHA)
    with (out / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("status", "cases", "source_windows_fully_reconstructed", "raw_cases", "elapsed_s")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv"))
    parser.add_argument("--versions", type=Path, default=ROOT / "artifacts/normal_adaptation_audit_20261029/prerequisites_audit_v2.json")
    parser.add_argument("--cache", type=Path, default=Path("D:/UAV-Data/online_cache_v2"))
    parser.add_argument("--contract", type=Path, default=Path("D:/UAV-Data/feature_contract_v1.json"))
    parser.add_argument("--links", type=Path, default=ROOT / "artifacts/source_training_coverage_audit_20261030_r2")
    parser.add_argument("--tools", type=Path, default=Path("D:/RflyMAD/Data_processing_tools"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
