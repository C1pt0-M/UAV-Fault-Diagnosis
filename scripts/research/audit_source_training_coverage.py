"""Read-only source train/val coverage and label-association audit; no model fitting.

Links capped manifest windows to existing cache offsets, checks every source label,
then reports coverage, timing and fixed scalar feature associations. Real reads are
limited to the already frozen 10-case normal adaptation bundle, not any test log.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from run_normal_adaptation_v1 import NORM_SHA, load_roles, require_hash, sha256, validate_bundle, write_json

BASE_SHA = "272a91aeba5b563ebfa23fff8d295ac53372ccac391c835dccf2b5660d9d6ce5"
WINDOW_COLUMNS = ["case_uid", "split", "domain", "flight_status", "case_level2_label", "level2_label", "phase",
                  "window_start_row", "window_end_row", "endpoint_fault", "fault_fraction", "fault_onset_row"]
TIME_BINS = [0, 10, 20, 30, 60, 120, float("inf")]


def directed_auc(values, labels):
    """P(fault scalar > normal scalar), half credit for ties; no fitted direction."""
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    if values.ndim != 1 or labels.shape != values.shape or not np.isfinite(values).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("invalid scalar/label population")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return None
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def compact(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "p10": None, "p90": None}
    if not np.isfinite(values).all():
        raise ValueError("non-finite descriptive values")
    return {"n": len(values), "median": float(np.median(values)), "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9))}


def window_means(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (26, 150):
        raise ValueError("expected [windows,26,150]")
    result = np.empty((len(x), 26), dtype=np.float64)
    for start in range(0, len(x), 512):
        batch = x[start:start + 512].double()
        if not torch.isfinite(batch).all():
            raise ValueError("non-finite cached features")
        result[start:start + len(batch)] = batch.mean(2).numpy()
    return result


def normal_origin(frame):
    result = np.full(len(frame), "fault_endpoint", dtype=object)
    label0 = frame["level2_label"].to_numpy() == "normal"
    pure = frame["case_level2_label"].to_numpy() == "normal"
    result[label0 & pure] = "original_normal_case"
    result[label0 & ~pure] = "normal_part_of_fault_case"
    return result


def count_table(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, observed=True, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, map(str, values)), windows=len(group), cases=int(group["case_uid"].nunique())))
    return rows


def hover_background(frame, means, columns, predictions=None):
    """Exploratory source-hover groups; near-zero *mean* vz is not a flight-state label."""
    required = ["vel_z", "baro_alt_relative", "actuator_ctrl_4", "acc_norm"]
    if means.shape != (len(frame), len(columns)) or not all(name in columns for name in required) or not np.isfinite(means).all():
        raise ValueError("invalid hover feature population")
    if predictions is not None and (np.shape(predictions) != (len(frame),) or not set(np.unique(predictions)).issubset({0, 1})):
        raise ValueError("invalid source predictions")
    work = frame.copy()
    for name in required:
        work[name] = means[:, columns.index(name)]
    hover = work[work.flight_status == "hover"].copy()
    near_zero = hover[hover.vel_z.abs() <= 0.2]
    result = {"definition": "Nominal hover; near-zero vertical-motion proxy is abs(window mean NED vel_z)<=0.2 m/s. It does not exclude oscillation/lateral motion or prove actual flight state.",
              "hover_origin_medians": {str(origin): {name: compact(part[name]) for name in required}
                                       for origin, part in hover.groupby("normal_origin", sort=True)},
              "near_zero_mean_vz_counts": {str(origin): len(part) for origin, part in near_zero.groupby("normal_origin", sort=True)}}
    if predictions is None:
        return result
    work["prediction"] = predictions
    pure = work[(work.flight_status == "hover") & (work.normal_origin == "original_normal_case")].copy()
    pure["time_bin"] = pd.cut(pure.end_from_log_start_s, [0, 10, 20, 30, 40, 60, 120, float("inf")], right=False).astype(str)
    result["pure_normal_hover"] = {"windows": len(pure), "cases": int(pure.case_uid.nunique()),
        "false_windows": int(pure.prediction.sum()),
        "by_case": [{"case_uid": str(uid), "windows": len(part), "false_windows": int(part.prediction.sum())}
                    for uid, part in pure.groupby("case_uid", sort=True)],
        "by_log_time_bin": [{"bin": str(key), "windows": len(part), "false_windows": int(part.prediction.sum()), "fpr": float(part.prediction.mean())}
                            for key, part in pure.groupby("time_bin", sort=True)],
        "by_prediction_feature_values": {str(int(key)): {name: compact(part[name]) for name in required + ["end_from_log_start_s"]}
                                         for key, part in pure.groupby("prediction", sort=True)}}
    selected = pure[pure.vel_z.abs() <= 0.2]
    result["pure_normal_hover"]["near_zero_mean_vz"] = {"windows": len(selected), "false_windows": int(selected.prediction.sum()),
                                                       "fpr": float(selected.prediction.mean()) if len(selected) else None}
    return result


def reconstruct_metadata(case_rows, window_manifest):
    """Select the existing cap64 grid; preserve the builder's first-seen case order."""
    lookup_rows = []
    expected_counts = {}
    for row in case_rows.to_dict(orient="records"):
        n = (int(row["row_count"]) - 150) // 5 + 1
        if n <= 0:
            raise ValueError("source case shorter than one window")
        expected_counts[row["case_uid"]] = n
        picks = np.unique(np.linspace(0, n - 1, 64).round().astype(int)) if n > 64 else np.arange(n)
        lookup_rows.extend((row["case_uid"], int(i) * 5) for i in picks)
    lookup = pd.DataFrame(lookup_rows, columns=["case_uid", "window_start_row"])
    ids = set(expected_counts)
    order, seen, counts, selected = [], set(), Counter(), []
    before_cap = Counter()
    text_columns = ["case_uid", "split", "domain", "flight_status", "case_level2_label", "level2_label", "phase"]
    for chunk in pd.read_csv(window_manifest, usecols=WINDOW_COLUMNS, chunksize=200000,
                             dtype={name: "string" for name in text_columns}, low_memory=False):
        part = chunk[chunk["case_uid"].isin(ids) & chunk["split"].isin(["train", "val"])
                     & chunk["domain"].isin(["SIL", "HIL"]) & chunk["level2_label"].isin(["normal", "motor", "sensor"])]
        if part.empty:
            continue
        for uid in part["case_uid"].drop_duplicates():
            if uid not in seen:
                order.append(uid)
                seen.add(uid)
        counts.update(part.groupby("case_uid", sort=False).size().to_dict())
        before_cap.update(part.groupby(["split", "level2_label"], sort=False).size().to_dict())
        picked = part.merge(lookup, on=["case_uid", "window_start_row"], how="inner", sort=False, validate="many_to_one")
        if not picked.empty:
            selected.append(picked)
    if dict(counts) != expected_counts:
        raise ValueError("source manifest window counts do not match frozen case lengths")
    result = pd.concat(selected, ignore_index=True)
    if result.duplicated(["case_uid", "window_start_row"]).any() or len(result) != len(lookup):
        raise ValueError("duplicate/missing selected source windows")
    by_uid = case_rows.set_index("case_uid")
    for name, manifest_name in (("split", "split"), ("domain", "domain"), ("flight_status", "flight_status"), ("level2_label", "case_level2_label")):
        if not np.array_equal(result["case_uid"].map(by_uid[name]).to_numpy(), result[manifest_name].to_numpy()):
            raise ValueError("case/window metadata mismatch: " + name)
    if not ((result.window_end_row - result.window_start_row) == 150).all():
        raise ValueError("invalid window spans")
    expected_labels = (result.level2_label != "normal").astype(int)
    if not np.array_equal(expected_labels.to_numpy(), result.endpoint_fault.to_numpy()):
        raise ValueError("source endpoint labels inconsistent")
    result["case_order"] = result["case_uid"].map({uid: i for i, uid in enumerate(order)})
    result = result.sort_values(["case_order", "window_start_row"]).reset_index(drop=True)
    result["label"] = (result.level2_label != "normal").astype(int)
    result["normal_origin"] = normal_origin(result)
    result["end_from_log_start_s"] = (result.window_end_row - 1) / 50.0
    lengths = result.case_uid.map(by_uid.row_count.astype(int)).to_numpy()
    result["relative_log_position"] = (result.window_end_row - 1) / (lengths - 1)
    return result, [{"split": split, "window_label": label, "windows": int(n)} for (split, label), n in sorted(before_cap.items())]


def audit(args):
    started = time.perf_counter()
    require_hash(args.case_manifest, BASE_SHA)
    parent, adapt_rows = load_roles(args.v1_run / "normal_adaptation_protocol_v1.json", args.v1_run / "real_case_roles_v1.csv")
    require_hash(args.cache / "norm_stats.json", NORM_SHA)
    prior = json.loads((args.v1_run / "report.json").read_text())
    require_hash(args.adapt_bundle, prior["bundle_sha256"])
    bundle = torch.load(args.adapt_bundle, weights_only=True, map_location="cpu")
    require_hash(args.window_manifest, bundle["window_manifest_sha256"])
    stats = json.loads((args.cache / "norm_stats.json").read_text())
    validate_bundle(bundle, adapt_rows, stats["columns"])
    base = pd.read_csv(args.case_manifest, encoding="utf-8-sig", dtype=str).fillna("")
    cases = base[base.split.isin(["train", "val"]) & base.domain.isin(["SIL", "HIL"]) & base.level2_label.isin(["normal", "motor", "sensor"])].copy()
    if cases.case_uid.duplicated().any() or Counter(cases.split) != {"train": 2580, "val": 555}:
        raise ValueError("unexpected frozen source case membership")
    pair_splits = cases.groupby("simulation_pair_key")["split"].nunique()
    if (cases.simulation_pair_key == "").any() or (pair_splits > 1).any():
        raise ValueError("missing pair key or train/val pair leakage")
    metadata, before_cap = reconstruct_metadata(cases, args.window_manifest)
    summary = json.loads((args.cache / "build_summary.json").read_text())
    if summary["max_windows_per_case"] != 64:
        raise ValueError("unexpected cache sampling")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha256(__file__),
              "case_manifest_sha256": BASE_SHA, "window_manifest_sha256": bundle["window_manifest_sha256"],
              "norm_sha256": NORM_SHA, "adaptation_bundle_sha256": prior["bundle_sha256"],
              "scope": "Source train/val and ten frozen adaptation-normal cases only. No classifier fitting, new inference or final-test feature/probability reads.",
              "linkage_limitation": "Case IDs reconstructed from documented builder order and selected spans; every source cache label checked. This is not a new full source-CSV-to-tensor numeric audit.",
              "before_cap_label_counts": before_cap,
              "case_coverage": [{k: v for k, v in row.items() if k != "windows"} for row in count_table(cases, ["split", "domain", "flight_status", "level2_label"])],
              "sampled_window_coverage": count_table(metadata, ["split", "domain", "flight_status", "level2_label"]),
              "sampled_phase_label_coverage": count_table(metadata, ["split", "phase", "level2_label"]),
              "source_origin_counts": count_table(metadata, ["split", "normal_origin"]),
              "existing_val_predictions_path": str(args.val_predictions), "existing_val_predictions_sha256": sha256(args.val_predictions),
              "train_val_pair_crossings": 0, "splits": {}}
    write_json(out / "audit.json", report)
    focus = ["actuator_ctrl_4", "pwm_mean_raw", "mag_z", "acc_norm"]
    pwm_indices = [stats["columns"].index(f"actuator_out_{i}") for i in range(1, 5)]
    target_means = window_means(bundle["x"])
    target_signals = dict(zip(stats["columns"], target_means.T))
    target_signals["pwm_mean_raw"] = target_means[:, pwm_indices].mean(1)
    report["adapt_normal_focus"] = {name: compact(target_signals[name]) for name in focus}
    for split in ("train", "val"):
        filename = split + ".pt"
        cache_sha = next(row["local"]["sha256"] for row in parent["cache_identity"]["files"] if row["file"] == filename)
        require_hash(args.cache / filename, cache_sha)
        payload = torch.load(args.cache / filename, weights_only=True, map_location="cpu", mmap=True)
        frame = metadata[metadata.split == split].copy().reset_index(drop=True)
        expected_base = summary["splits"][split].get("manifest_windows", summary["splits"][split]["windows"])
        if len(frame) != expected_base or len(payload["y"]) != summary["splits"][split]["windows"] or payload["x"].shape[0] != len(payload["y"]):
            raise ValueError("cache/sample count mismatch")
        if not np.array_equal(frame.label.to_numpy(), payload["y"][:len(frame)].numpy()):
            raise ValueError("reconstructed source labels do not match entire cache prefix")
        frame["cache_index"] = np.arange(len(frame))
        means = window_means(payload["x"])
        signals = dict(zip(stats["columns"], means[:len(frame)].T))
        signals["pwm_mean_raw"] = means[:len(frame), pwm_indices].mean(1)
        groups = {}
        for origin in sorted(frame.normal_origin.unique()):
            selected = frame.normal_origin.to_numpy() == origin
            groups[origin] = {name: compact(signals[name][selected]) for name in focus}
        extra = len(means) - len(frame)
        extra_report = None
        if extra:
            if split != "train" or extra != 3680 or not torch.equal(payload["y"][len(frame):], torch.zeros(extra, dtype=payload["y"].dtype)):
                raise ValueError("unexpected appended training block")
            extra_signals = dict(zip(stats["columns"], means[len(frame):].T))
            extra_signals["pwm_mean_raw"] = means[len(frame):, pwm_indices].mean(1)
            extra_report = {"domain": "RflySim", "normal_windows": extra, "fault_windows": 0,
                            "sessions_declared_in_summary": summary["splits"]["train"]["extra_normal"]["files"],
                            "file_level_cache_ids": False,
                            "focus": {name: compact(extra_signals[name]) for name in focus}}
        timing = frame.assign(log_time_bin=pd.cut(frame.end_from_log_start_s, TIME_BINS, right=False).astype(str))
        associations = {name: {"directed_auc": directed_auc(values, frame.label.to_numpy()),
                               "normal": compact(values[frame.label.to_numpy() == 0]),
                               "fault": compact(values[frame.label.to_numpy() == 1])} for name, values in signals.items()}
        conditional = []
        for (domain, status), part in frame.groupby(["domain", "flight_status"], sort=True):
            idx = part.index.to_numpy()
            conditional.append({"domain": domain, "flight_status": status, "windows": len(idx),
                                "auc": {name: directed_auc(signals[name][idx], part.label.to_numpy()) for name in focus}})
        pair_frame = frame[["case_uid", "case_level2_label", "phase", "label", "fault_fraction"]].copy()
        for name in focus:
            pair_frame[name] = signals[name]
        pre = pair_frame[(pair_frame.label == 0) & (pair_frame.case_level2_label != "normal") & (pair_frame.phase == "pre_fault")].groupby("case_uid")[focus].mean()
        post = pair_frame[(pair_frame.label == 1) & (pair_frame.fault_fraction == 1)].groupby("case_uid")[focus].mean()
        common = pre.index.intersection(post.index)
        paired = post.loc[common] - pre.loc[common]
        split_report = {"cache_sha256": cache_sha, "manifest_windows": len(frame), "cache_windows": len(means),
                        "all_prefix_labels_match": True, "normal_origins_focus": groups, "appended_normal_block": extra_report,
                        "timing_coverage": count_table(timing, ["log_time_bin", "normal_origin"]),
                        "timing_association": {name: directed_auc(frame[name], frame.label.to_numpy()) for name in ["end_from_log_start_s", "relative_log_position"]},
                        "single_window_mean_association": associations, "conditional_association": conditional,
                        "within_fault_case_pre_to_full_fault": {"cases": len(common), "note": "Same-case temporal contrast, not a controlled causal response; flight progression remains confounded.",
                            "features": {name: dict(compact(paired[name]), fraction_positive=float(np.mean(paired[name] > 0)) if len(paired) else None) for name in focus}}}
        if split == "val":
            with np.load(args.val_predictions, allow_pickle=False) as data:
                probs, labels = data["probabilities"], data["labels"]
            if (probs.shape != (len(frame), 2) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                    or not np.allclose(probs.sum(1), 1, rtol=0, atol=2e-7) or not np.array_equal(labels, frame.label.to_numpy())):
                raise ValueError("existing source-validation predictions do not align")
            predicted = probs.argmax(1)
            errors = []
            for key in ["domain", "flight_status", "normal_origin", "phase"]:
                for value, part in frame.groupby(key, sort=True):
                    idx = part.index.to_numpy()
                    actual = part.label.to_numpy()
                    cm = np.bincount(actual * 2 + predicted[idx], minlength=4).reshape(2, 2)
                    errors.append({"group_key": key, "group": str(value), "windows": len(idx), "confusion_matrix": cm.tolist(),
                                   "normal_window_fpr": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None,
                                   "fault_recall": float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else None})
            split_report["existing_source_val_errors"] = errors
            pure = frame[frame.normal_origin == "original_normal_case"].copy()
            pure["predicted_fault"] = predicted[pure.index.to_numpy()]
            pure_cases = pure.groupby("case_uid").predicted_fault.agg(["count", "sum"])
            split_report["original_normal_case_error_summary"] = {
                "cases": len(pure_cases), "cases_with_any_false_window": int((pure_cases["sum"] > 0).sum()),
                "cases_all_sampled_windows_false": int((pure_cases["sum"] == pure_cases["count"]).sum()),
                "note": "Any false window is not an aggregated alarm event."}
            split_report["original_normal_case_errors_by_status"] = [
                {"flight_status": str(status), "windows": len(part), "cases": int(part.case_uid.nunique()),
                 "false_windows": int(part.predicted_fault.sum()), "fpr": float(part.predicted_fault.mean())}
                for status, part in pure.groupby("flight_status", sort=True)]
        split_report["hover_background"] = hover_background(frame, means[:len(frame)], stats["columns"], predicted if split == "val" else None)
        frame.to_csv(out / f"{split}_source_window_links.csv", index=False)
        np.savez_compressed(out / f"{split}_source_window_means.npz", means=means[:len(frame)], labels=frame.label.to_numpy(), columns=np.array(stats["columns"]))
        report["splits"][split] = split_report
        write_json(out / "audit.json", report)
        print(f"{split}: linked {len(frame)} source windows, {extra} appended normals; all labels match", flush=True)
        del payload, means
    report["interpretation_limits"] = ["Directed scalar AUC is descriptive rank association, not a fitted model or final validation result.",
                                        "Log-relative position uses total future log length, unavailable online; it is an audit variable, not an input feature.",
                                        "Flight-status tags and pre_fault are not proof of airborne/steady-flight phase.",
                                        "Covariate/label association is not proof of an illegitimate shortcut; motor/IMU changes can be genuine fault responses.",
                                        "RflySim block has no per-file cache IDs; only declared session count and block labels are verified here."]
    report["status"] = "COMPLETED_SOURCE_DATA_AUDIT"
    report["elapsed_s"] = time.perf_counter() - started
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "audit.json", report)
    print(f"completed: {out / 'audit.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--adapt-bundle", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args)


if __name__ == "__main__":
    main()
