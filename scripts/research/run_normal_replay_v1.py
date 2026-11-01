"""Full Real retrospective replay following the user's cap64 clarification.

Adaptation and sampled window metrics retain cap64. Calibration and events use
all chronological 150-frame windows at stride10. Never updates model parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from run_normal_adaptation_v1 import (MODEL_SHA, MODEL_SOURCE_SHA, NORM_SHA, PROTOCOL_SHA,
                                      load_roles, require_hash, sha256, validate_source, write_json)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_all_roles(protocol, roles):
    import csv
    load_roles(protocol, roles)
    with Path(roles).open(encoding="utf-8-sig", newline="") as stream:
        return sorted(csv.DictReader(stream), key=lambda r: r["case_uid"])


def prepare(args):
    import pandas as pd
    data_scripts = Path(__file__).resolve().parents[1] / "data"
    sys.path.insert(0, str(data_scripts))
    from rfly_feature_contract import contract_columns, read_case_features, source_columns_for_domain
    from generate_window_manifest import _fault_column_priority, _infer_fault_mask

    rows = read_all_roles(args.protocol, args.roles)
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    require_hash(args.norm, NORM_SHA)
    columns = json.loads(args.norm.read_text())["columns"]
    if contract_columns(contract) != columns:
        raise ValueError("contract column mismatch")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    index = {"protocol_sha256": PROTOCOL_SHA, "role_manifest_sha256": sha256(args.roles),
             "contract_sha256": sha256(args.contract), "columns": columns,
             "extractor_sha256": sha256(data_scripts / "rfly_feature_contract.py"),
             "fault_label_source_sha256": sha256(data_scripts / "generate_window_manifest.py"),
             "cases": []}
    mapping = source_columns_for_domain(contract, "REAL")
    for i, row in enumerate(rows):
        path = Path(row["final_output_path"])
        require_hash(path, row["output_sha256"])
        header = pd.read_csv(path, nrows=0).columns
        fault_col = next((col for col in _fault_column_priority() if col in header), None)
        if fault_col is None or "Timestamp" not in header:
            raise ValueError("missing time/fault columns: " + row["case_uid"])
        raw = pd.read_csv(path, usecols=sorted(set(mapping.values()) | {"Timestamp", fault_col}), low_memory=False)
        values = raw.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or len(raw) != int(row["row_count"]):
            raise ValueError("non-finite input or row mismatch: " + row["case_uid"])
        times = raw["Timestamp"].to_numpy(dtype=np.float64) / 1e6
        times -= times[0]
        if len(times) < 150 or not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-5):
            raise ValueError("expected 50 Hz timestamps: " + row["case_uid"])
        mask = _infer_fault_mask(raw)
        if bool(mask.any()) != (row["role"] == "test_fault"):
            raise ValueError("fault-mask/role mismatch: " + row["case_uid"])
        matrix, actual = read_case_features(path, "REAL", contract)
        if actual != columns or not np.isfinite(matrix).all():
            raise ValueError("invalid feature extraction")
        # Keep float64 frames so relative altitude subtraction precedes float32
        # conversion exactly as in the audited cache window_batch implementation.
        filename = f"case_{i:03}.npz"
        np.savez_compressed(out / filename, matrix=matrix, times=times, fault_mask=mask)
        index["cases"].append(dict(row, file=filename, sha256=sha256(out / filename),
                                   fault_column=fault_col, exposure_s=float(times[-1])))
        if (i + 1) % 50 == 0:
            print(f"prepared {i + 1}/{len(rows)} complete Real cases", flush=True)
    write_json(out / "index.json", index)
    print(f"prepared {len(rows)} cases; index SHA256={sha256(out / 'index.json')}", flush=True)


def intervals(times, mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(float(times[start]), float(times[end])) for start, end in zip(starts, ends)]


def event_metrics(events, truth):
    """Count hits by overlap; delay only for post-onset *overlapping* events.

    Report retrospective start latency as frozen in v1, and confirmation/emission
    separately. A later unrelated alarm is not a detection of an earlier interval.
    """
    records = []
    for start, end in truth:
        hits = [ev for ev in events if ev["start_t"] <= end and ev["end_t"] >= start]
        post = [ev for ev in hits if ev["start_t"] >= start]
        records.append({"gt_start": start, "gt_end": end, "hit": bool(hits),
                        "pre_onset_alarm": any(ev["start_t"] < start for ev in hits),
                        "event_start_delay_s": min((ev["start_t"] - start for ev in post), default=None),
                        "confirmation_delay_s": min((ev["meta"]["timing"]["confirmed_t"] - start for ev in hits
                                                     if ev["meta"]["timing"]["confirmed_t"] >= start), default=None),
                        "final_emission_delay_s": min((ev["meta"]["timing"]["emitted_t"] - start for ev in hits
                                                       if ev["meta"]["timing"]["emitted_t"] >= start), default=None)})
    return {"hit": any(rec["hit"] for rec in records), "intervals": records,
            "non_overlapping_alarm_events": sum(not any(ev["start_t"] <= end and ev["end_t"] >= start for start, end in truth) for ev in events),
            "pre_first_onset_events": sum(ev["start_t"] < truth[0][0] for ev in events) if truth else 0}


def distribution(values):
    values = [float(value) for value in values if value is not None]
    return {"n": len(values), "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, 0.9)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if values else None}


def classification(cm):
    cm = np.asarray(cm, dtype=np.int64)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(2), where=denom != 0)
    return {"windows": int(cm.sum()), "confusion_matrix": cm.tolist(),
            "accuracy": float(cm.trace() / cm.sum()) if cm.sum() else None,
            "macro_f1": float(f1.mean()),
            "fault_recall": float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else None,
            "normal_window_fpr": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None}


def summarize(cases):
    fault = [r for r in cases if r["role"] == "test_fault"]
    normals = [r for r in cases if r["role"] == "test_normal"]
    supplemental = [r for r in cases if r["role"] == "supplemental_normal_test"]
    result = {}
    for key, rows in (("main_normal", normals), ("supplemental_normal", supplemental)):
        exposure = sum(r["exposure_s"] for r in rows)
        event_count = sum(len(r["events"]) for r in rows)
        result[key] = {"cases": len(rows), "cases_with_events": sum(bool(r["events"]) for r in rows),
                       "case_event_fpr": sum(bool(r["events"]) for r in rows) / len(rows) if rows else None,
                       "exposure_s": exposure, "observable_exposure_s": sum(r["observable_exposure_s"] for r in rows),
                       "false_alarm_events": event_count,
                       "false_alarm_events_per_hour": event_count * 3600 / exposure if exposure else None,
                       "false_alarm_duration_s": sum(ev["end_t"] - ev["start_t"] for r in rows for ev in r["events"])}
    result["fault"] = {"cases": len(fault), "hit_cases": sum(r["metrics"]["hit"] for r in fault),
                       "case_event_recall": sum(r["metrics"]["hit"] for r in fault) / len(fault) if fault else None,
                       "missed_cases": [r["case_uid"] for r in fault if not r["metrics"]["hit"]],
                       "pre_first_onset_events": sum(r["metrics"]["pre_first_onset_events"] for r in fault),
                       "non_overlapping_alarm_events": sum(r["metrics"]["non_overlapping_alarm_events"] for r in fault)}
    for label in ("motor", "sensor"):
        rows = [r for r in fault if r["label"] == label]
        result["fault"][label] = {"cases": len(rows), "recall": sum(r["metrics"]["hit"] for r in rows) / len(rows) if rows else None}
    records = [interval for row in fault for interval in row["metrics"]["intervals"]]
    result["fault"]["ground_truth_intervals"] = len(records)
    result["fault"]["hit_intervals"] = sum(r["hit"] for r in records)
    result["fault"]["pre_onset_hit_intervals"] = sum(r["pre_onset_alarm"] for r in records)
    for field in ("event_start_delay_s", "confirmation_delay_s", "final_emission_delay_s"):
        result["fault"][field] = distribution(r[field] for r in records)
    # Cache-style cap64 metrics use argmax, not onset threshold; shared by T0/T1.
    for role, rows in (("main", normals + fault), ("supplemental", supplemental)):
        result[f"sampled_window_{role}"] = classification(sum((np.array(r["sampled_cm"]) for r in rows), np.zeros((2, 2), dtype=np.int64)))
    return result


def predict_case(model, matrix, columns, mean, std, device, sampled=False):
    from rfly_feature_contract import window_batch
    full_ends = np.arange(150, len(matrix) + 1, 10, dtype=np.int64)
    sampled_ends = np.arange(150, len(matrix) + 1, 5, dtype=np.int64)
    if len(sampled_ends) > 64:
        sampled_ends = sampled_ends[np.unique(np.linspace(0, len(sampled_ends) - 1, 64).round().astype(int))]
    if not sampled:
        sampled_ends = np.array([], dtype=np.int64)
    ends = np.union1d(full_ends, sampled_ends)
    probabilities = []
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(ends), 512):
            batch_ends = ends[start:start + 512]
            x = torch.from_numpy(window_batch(matrix, columns, [(int(e) - 150, int(e)) for e in batch_ends]))
            x = ((x - mean) / std).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            probs = model(x).softmax(1).cpu().numpy()
            elapsed += time.perf_counter() - tick
            if not np.isfinite(probs).all():
                raise RuntimeError("non-finite predictions")
            probabilities.append(probs)
    probs = np.concatenate(probabilities)
    return full_ends, probs[np.searchsorted(ends, full_ends)], sampled_ends, probs[np.searchsorted(ends, sampled_ends)], elapsed


def read_case(directory, member):
    path = directory / member["file"]
    require_hash(path, member["sha256"])
    with np.load(path, allow_pickle=False) as data:
        matrix, times, mask = data["matrix"], data["times"], data["fault_mask"]
    if matrix.shape != (int(member["row_count"]), 26) or times.shape != (len(matrix),) or mask.shape != times.shape:
        raise ValueError("replay bundle shape mismatch")
    if matrix.dtype != np.float64 or mask.dtype != np.bool_ or not np.isfinite(matrix).all() or not np.isfinite(times).all():
        raise ValueError("invalid replay bundle values")
    if not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-5) or bool(mask.any()) != (member["role"] == "test_fault"):
        raise ValueError("invalid time grid or fault mask")
    return matrix, times, mask


def aggregate(aggregator_class, times, probs, threshold):
    aggregator = aggregator_class(onset_th=threshold, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5)
    events = []
    for t, p in zip(times, probs):
        events.extend(aggregator.update(t=float(t), fault_prob=float(p)))
    events.extend(aggregator.flush())
    return [asdict(event) for event in events]


def run(args):
    rows = read_all_roles(args.protocol, args.roles)
    index = json.loads((args.input / "index.json").read_text())
    if index["protocol_sha256"] != PROTOCOL_SHA or index["role_manifest_sha256"] != sha256(args.roles):
        raise ValueError("input bundle protocol mismatch")
    members = index["cases"]
    if len(members) != 465 or {m["case_uid"] for m in members} != {r["case_uid"] for r in rows}:
        raise ValueError("replay members differ from frozen manifest")
    for member, row in zip(sorted(members, key=lambda r: r["case_uid"]), rows):
        if any(member[k] != value for k, value in row.items()):
            raise ValueError("replay member role/provenance mismatch")
    adapt_report = json.loads((args.adaptation / "report.json").read_text())
    if adapt_report["status"] != "COMPLETED" or adapt_report["source_checkpoint_sha256"] != MODEL_SHA:
        raise ValueError("adaptation stage incomplete or wrong source")
    require_hash(args.adaptation / "best.pt", MODEL_SHA)
    require_hash(args.adaptation / "A1_adapted.pt", adapt_report["adapted_checkpoint_sha256"])
    require_hash(args.adaptation / "lstm.py", MODEL_SOURCE_SHA)
    require_hash(args.adaptation / "norm_stats.json", NORM_SHA)
    stats = json.loads((args.adaptation / "norm_stats.json").read_text())
    if index["columns"] != stats["columns"]:
        raise ValueError("input/normalization columns mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for source in (args.protocol, args.roles, args.event_source, args.adaptation / "lstm.py", Path(__file__),
                   Path(__file__).with_name("run_normal_adaptation_v1.py"), Path(__file__).with_name("rfly_feature_contract.py"), args.clarification):
        shutil.copy2(source, out / source.name)
    event_module = load_module("replay_event_aggregator", out / args.event_source.name)
    model_module = load_module("replay_source_model", out / "lstm.py")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    models = {}
    for arm, filename in (("A0", "best.pt"), ("A1", "A1_adapted.pt")):
        model = model_module.OnlineKGMSTCA(in_channels=26, num_classes=2, columns=stats["columns"]).to(device)
        model.load_state_dict(torch.load(args.adaptation / filename, map_location=device, weights_only=True), strict=True)
        model.eval()
        model.requires_grad_(False)
        models[arm] = model
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "label": "retrospective_streaming_replay_NOT_strict_realtime",
              "protocol_sha256": PROTOCOL_SHA, "clarification_sha256": sha256(args.clarification),
              "input_index_sha256": sha256(args.input / "index.json"),
              "source_checkpoint_sha256": MODEL_SHA, "adapted_checkpoint_sha256": adapt_report["adapted_checkpoint_sha256"],
              "adaptation_report_sha256": sha256(args.adaptation / "report.json"),
              "source_norm_sha256": NORM_SHA, "model_source_sha256": MODEL_SOURCE_SHA,
              "runner_sha256": sha256(__file__), "event_source_sha256": sha256(args.event_source),
              "window_extractor_sha256": sha256(Path(__file__).with_name("rfly_feature_contract.py")),
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": str(torch.__version__), "numpy": np.__version__, "batch_size": 512,
              "cudnn_tf32": True, "matmul_tf32": False, "deterministic_algorithms": True,
              "full_replay_stride_frames": 10, "sampled_metrics_stride_frames": 5, "sampled_metrics_cap": 64,
              "ground_truth_interval_end": "last active sample timestamp (inclusive)",
              "delay_population": "per true interval; post-onset overlapping predictions only; pre-onset hits may have missing start delay",
              "limitations": ["CSV interpolation can depend on future input; no physical causal delay claim.",
                              "Batched inference time is throughput, not single-window real-time latency.",
                              "Real results had previously been inspected under earlier protocols; not pristine blind testing.",
                              "RflySim event/session replay remains a separate pending validation; sampled holdout is window-only.",
                              "No test-driven method, threshold or rule changes in this run."],
              "source_internal_test": {}, "RflySim_sampled_holdout": {}}
    write_json(out / "report.json", report)
    started = time.perf_counter()
    try:
        thresholds = {}
        calibrate = [m for m in members if m["role"] == "calibrate_normal"]
        if len(calibrate) != 5:
            raise ValueError("expected 5 normal calibration cases")
        for arm, model in models.items():
            probabilities = []
            for member in calibrate:
                matrix, times, mask = read_case(args.input, member)
                ends, probs, _, _, _ = predict_case(model, matrix, stats["columns"], mean, std, device)
                probabilities.extend(probs[:, 1].tolist())
                np.savez_compressed(out / f"calibration_{arm}_{Path(member['file']).stem}.npz", ends=ends, fault_prob=probs[:, 1])
            q = float(np.quantile(probabilities, 0.995, method="linear"))
            thresholds[arm] = {"T0": 0.6, "T1": float(np.clip(max(0.6, q), 0.6, 0.95)),
                               "q995": q, "calibration_points": len(probabilities),
                               "calibration_cases": [m["case_uid"] for m in calibrate]}
            print(f"{arm} calibration: {thresholds[arm]}", flush=True)
        # This file is committed BEFORE the first final-test case is loaded.
        write_json(out / "thresholds_frozen.json", thresholds)
        threshold_sha = sha256(out / "thresholds_frozen.json")
        report["thresholds"] = thresholds
        report["thresholds_sha256_before_test"] = threshold_sha
        write_json(out / "report.json", report)
        results = {f"{arm}-{t}": [] for arm in models for t in ("T0", "T1")}
        tests = [m for m in members if m["role"] in ("test_normal", "test_fault", "supplemental_normal_test")]
        timing = {arm: 0.0 for arm in models}
        with (out / "case_results.jsonl").open("x", encoding="utf-8") as stream:
            for i, member in enumerate(tests):
                matrix, times, mask = read_case(args.input, member)
                truth = intervals(times, mask)
                prediction_arrays = {}
                for arm, model in models.items():
                    ends, probs, sample_ends, sample_probs, elapsed = predict_case(model, matrix, stats["columns"], mean, std, device, sampled=True)
                    timing[arm] += elapsed
                    sample_labels = mask[sample_ends - 1].astype(np.int64)
                    cm = np.bincount(sample_labels * 2 + sample_probs.argmax(1), minlength=4).reshape(2, 2)
                    prediction_arrays[f"{arm}_fault_prob"] = probs[:, 1]
                    prediction_arrays[f"{arm}_sampled_probs"] = sample_probs
                    for threshold_arm in ("T0", "T1"):
                        events = aggregate(event_module.FaultEventAggregator, times[ends - 1], probs[:, 1], thresholds[arm][threshold_arm])
                        row = {"case_uid": member["case_uid"], "arm": f"{arm}-{threshold_arm}", "role": member["role"],
                               "label": member["level2_label"], "flight_status": member["flight_status"],
                               "exposure_s": float(times[-1]), "observable_exposure_s": float(times[-1] - times[149]),
                               "inference_points": len(ends), "events": events, "metrics": event_metrics(events, truth),
                               "sampled_cm": cm.tolist()}
                        results[row["arm"]].append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                np.savez_compressed(out / f"predictions_{Path(member['file']).stem}.npz", ends=ends, times=times[ends - 1],
                                    labels=mask[ends - 1], sampled_ends=sample_ends, sampled_labels=mask[sample_ends - 1], **prediction_arrays)
                if (i + 1) % 50 == 0:
                    print(f"completed Real replay {i + 1}/{len(tests)} cases", flush=True)
        require_hash(out / "thresholds_frozen.json", threshold_sha)
        report["real_results"] = {arm: summarize(cases) for arm, cases in results.items()}
        report["batched_forward_total_s"] = timing
        protocol = json.loads(args.protocol.read_text())
        for filename, destination in (("test.pt", "source_internal_test"), ("live_holdout.pt", "RflySim_sampled_holdout")):
            entry = next(e for e in protocol["cache_identity"]["files"] if e["file"] == filename)
            require_hash(args.cache / filename, entry["local"]["sha256"])
            for arm, model in models.items():
                report[destination][arm] = validate_source(model, args.cache / filename, mean, std, device, out / f"{arm}_{filename}_predictions.npz")
        report["status"] = "COMPLETED_REAL_REPLAY_AND_SAMPLED_HOLDOUT"
        report["elapsed_s"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "report.json", report)
        print(f"DONE: {out / 'report.json'}", flush=True)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        write_json(out / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    execute = commands.add_parser("run")
    for child in (prep, execute):
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--roles", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    prep.add_argument("--contract", type=Path, required=True)
    prep.add_argument("--norm", type=Path, required=True)
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--adaptation", type=Path, required=True)
    execute.add_argument("--event-source", type=Path, required=True)
    execute.add_argument("--clarification", type=Path, required=True)
    execute.add_argument("--cache", type=Path, required=True)
    execute.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
