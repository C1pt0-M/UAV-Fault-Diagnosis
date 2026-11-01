"""Fixed CPU diagnostic probes on source val and ten adaptation-normal cases only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from diagnose_normal_adaptation_v1 import feature_summary, probability_summary
from run_normal_adaptation_v1 import (MODEL_SHA, MODEL_SOURCE_SHA, NORM_SHA, load_roles,
                                      require_hash, sha256, validate_bundle, write_json)
from run_normal_replay_v1 import classification

RELATION_NAMES = ["actuator_out_cross_std", "actuator_out_cross_range", "ctrl_out_gap_mean",
                  "ctrl_out_gap_std", "actuator_effort_proxy", "abs_acc_norm_z", "gyro_norm_z",
                  "vertical_consistency_proxy", "attitude_vector_norm_z"]


def explain_batch(model, x):
    """Algebraic decomposition of the *fused* linear head, with its original gate."""
    head = model.head[1]
    if any(layer.training for layer in model.modules()) or not isinstance(head, torch.nn.Linear) or head.out_features != 2:
        raise ValueError("expected the eval-mode binary source linear head")
    raw = model.raw_encoder(x)
    relation_input = model._relation_features(x)
    relation = model.relation_proj(model.relation_encoder(relation_input))
    alpha = torch.sigmoid(model.gate(torch.cat([raw, relation], dim=1)))
    logits = model.head(alpha * raw + (1.0 - alpha) * relation)
    weights = head.weight[1] - head.weight[0]
    bias = head.bias[1] - head.bias[0]
    raw_part = alpha.squeeze(1) * (raw @ weights)
    relation_part = (1.0 - alpha.squeeze(1)) * (relation @ weights)
    margin = logits[:, 1] - logits[:, 0]
    reconstructed = raw_part + relation_part + bias
    torch.testing.assert_close(reconstructed, margin, atol=1e-4, rtol=1e-4)
    return logits, {"gate_alpha": alpha.squeeze(1), "raw_margin": raw_part,
                    "relation_margin": relation_part, "head_bias": bias.expand_as(margin), "margin": margin,
                    "reconstruction_abs_error": (reconstructed - margin).abs(),
                    "relation_input_window_mean": relation_input.mean(2),
                    "acc_norm_denominator_clamp_fraction": (x[:, model._acc_norm_idx, :] <= 0.1).float().mean(1)}


def translate_channels(x, indices, delta, direction):
    if x.ndim != 3 or delta.shape != (x.shape[1],) or direction not in (-1, 1):
        raise ValueError("invalid channel-translation contract")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= x.shape[1] for i in indices):
        raise ValueError("invalid channel indices")
    if not torch.isfinite(delta).all():
        raise ValueError("non-finite translation")
    output = x.clone()
    output[:, indices, :] += direction * delta[indices].view(1, -1, 1)
    return output


def predict(model, x, mean, std, indices, delta, direction, explain=False):
    probabilities, explanations = [], {}
    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = (x[start:start + 512].float() - mean) / std
            if indices:
                batch = translate_channels(batch, indices, delta, direction)
            if not torch.isfinite(batch).all():
                raise ValueError("non-finite input")
            if explain:
                logits, details = explain_batch(model, batch)
                for name, value in details.items():
                    explanations.setdefault(name, []).append(value.cpu().numpy())
            else:
                logits = model(batch)
            probabilities.append(logits.softmax(1).cpu().numpy())
    probs = np.concatenate(probabilities)
    if probs.shape != (len(x), 2) or not np.isfinite(probs).all():
        raise ValueError("invalid probabilities")
    return probs, {key: np.concatenate(value) for key, value in explanations.items()}


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid scalar population")
    return {"n": len(values), "mean": float(values.mean()), "p10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}


def branch_summary(details, select):
    result = {name: scalar_summary(value[select]) for name, value in details.items() if value.ndim == 1}
    result["raw_absolute_margin_larger_fraction"] = float(np.mean(np.abs(details["raw_margin"][select]) > np.abs(details["relation_margin"][select])))
    result["relation_inputs_window_means"] = {name: scalar_summary(details["relation_input_window_mean"][select, i]) for i, name in enumerate(RELATION_NAMES)}
    return result


def run(args):
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (protocol["protocol_name"], protocol["status"]) != ("rflymad_feature_branch_sensitivity_dev_v1", "FROZEN_DIAGNOSTIC_ONLY_BEFORE_EXECUTION"):
        raise ValueError("unexpected diagnostic protocol")
    for key, expected in (("source_checkpoint_sha256", MODEL_SHA), ("source_code_sha256", MODEL_SOURCE_SHA), ("source_norm_sha256", NORM_SHA)):
        if protocol[key] != expected:
            raise ValueError("source protocol identity mismatch")
    if protocol["execution"] != {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 42, "deterministic": True, "batch_size": 512}:
        raise ValueError("unexpected execution contract")
    source = args.run_root / "results/20261030_adaptation_sourceval_r2"
    parent, rows = load_roles(source / "normal_adaptation_protocol_v1.json", source / "real_case_roles_v1.csv")
    require_hash(source / "best.pt", MODEL_SHA)
    require_hash(source / "lstm.py", MODEL_SOURCE_SHA)
    require_hash(source / "norm_stats.json", NORM_SHA)
    prior = json.loads((source / "report.json").read_text())
    require_hash(args.run_root / "adaptation_input/adapt_normal.pt", prior["bundle_sha256"])
    val_sha = next(row["local"]["sha256"] for row in parent["cache_identity"]["files"] if row["file"] == "val.pt")
    require_hash(args.source_cache / "val.pt", val_sha)
    stats = json.loads((source / "norm_stats.json").read_text())
    config = json.loads((source / "config.json").read_text())
    columns = stats["columns"]
    groups = protocol["feature_groups"]
    names = [name for group in groups.values() for name in group]
    arms = protocol["feature_probes"]["arms"]
    if len(set(names)) != 26 or sorted(names) != sorted(columns) or len(names) != 26:
        raise ValueError("feature groups must partition all 26 source columns")
    if arms != ["baseline"] + ["shift_" + name for name in groups] + ["shift_all"]:
        raise ValueError("unexpected probes")
    bundle = torch.load(args.run_root / "adaptation_input/adapt_normal.pt", map_location="cpu", weights_only=True)
    counts = validate_bundle(bundle, rows, columns)
    if len(bundle["x"]) != 640 or dict(counts) != prior["adaptation_cases"]:
        raise ValueError("unexpected target-normal input")
    val = torch.load(args.source_cache / "val.pt", map_location="cpu", weights_only=True, mmap=True)
    labels = val["y"].numpy()
    if len(labels) != 35520 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("unexpected source validation population")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for path in (args.protocol, source / "lstm.py", source / "config.json", source / "norm_stats.json", Path(__file__),
                 Path(__file__).with_name("diagnose_normal_adaptation_v1.py"), Path(__file__).with_name("run_normal_adaptation_v1.py"),
                 Path(__file__).with_name("run_normal_replay_v1.py")):
        shutil.copy2(path, out / path.name)
    protocol_sha = sha256(out / args.protocol.name)
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256_before_predictions": protocol_sha, "source_checkpoint_sha256": MODEL_SHA,
              "source_code_sha256": MODEL_SOURCE_SHA, "source_norm_sha256": NORM_SHA,
              "source_val_sha256": val_sha, "adaptation_bundle_sha256": prior["bundle_sha256"],
              "runner_sha256": sha256(__file__), "device": "cpu", "threads": 4, "seed": 42,
              "torch": str(torch.__version__), "numpy": np.__version__,
              "forbidden_dataset_reads": False, "weight_or_buffer_updates": False,
              "target_normal_interpretation": "640 in-sample adaptation-normal windows, NOT independent Real test or target fault recall",
              "arms": {}}
    write_json(out / "report.json", report)
    started = time.perf_counter()
    try:
        reference = feature_summary(val["x"], np.flatnonzero(labels == 0), mean, std)
        target_reference = feature_summary(bundle["x"], np.arange(640), mean, std)
        delta = torch.tensor(np.array(target_reference["z_mean"]) - np.array(reference["z_mean"]), dtype=torch.float32)
        report["delta_fitted_once_before_predictions"] = dict(zip(columns, delta.tolist()))
        write_json(out / "mean_shift_reference.json", {"source": reference, "target": target_reference, "columns": columns, "delta": delta.tolist()})
        write_json(out / "report.json", report)
        spec = importlib.util.spec_from_file_location("feature_sensitivity_source_snapshot", out / "lstm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.RFLY_SEQ_COLUMNS_V2 != columns:
            raise ValueError("source module columns mismatch")
        model = module.OnlineKGMSTCA(in_channels=26, num_classes=2, hidden_size=config["hidden_size"],
                                    num_layers=config["num_layers"], dropout=config["dropout"], columns=columns)
        original = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(original, strict=True)
        model.eval().requires_grad_(False)
        target_uids = np.array([sample["case_uid"] for sample in bundle["samples"]])
        for arm in arms:
            if arm == "baseline":
                indices = []
            elif arm == "shift_all":
                indices = list(range(26))
            else:
                indices = [columns.index(name) for name in groups[arm.removeprefix("shift_")]]
            print(f"{arm}: source validation and in-sample target normals", flush=True)
            src_probs, source_details = predict(model, val["x"], mean, std, indices, delta, 1, explain=arm == "baseline")
            dst_probs, target_details = predict(model, bundle["x"], mean, std, indices, delta, -1, explain=arm == "baseline")
            src_preds, dst_preds = src_probs.argmax(1), dst_probs.argmax(1)
            cm = np.bincount(labels.astype(np.int64) * 2 + src_preds, minlength=4).reshape(2, 2)
            row = {"source_val": classification(cm), "target_adapt_normal": {"windows": 640,
                   "fault_predictions": int(dst_preds.sum()), "fpr_argmax": float(dst_preds.mean()),
                   "probability": probability_summary(dst_probs[:, 1]),
                   "cases": {uid: {"windows": int(np.sum(target_uids == uid)),
                                   "fault_predictions": int(dst_preds[target_uids == uid].sum())} for uid in counts}},
                   "source_probability": {str(label): probability_summary(src_probs[labels == label, 1]) for label in (0, 1)}}
            np.savez_compressed(out / f"{arm}_predictions.npz", source_probabilities=src_probs, source_labels=labels,
                                target_probabilities=dst_probs, target_case_uids=target_uids)
            if arm == "baseline":
                row["branch_decomposition"] = {"source_normal": branch_summary(source_details, labels == 0),
                                                "source_fault": branch_summary(source_details, labels == 1),
                                                "target_adapt_normal": branch_summary(target_details, np.ones(640, dtype=bool))}
                np.savez_compressed(out / "baseline_source_explanations.npz", **source_details)
                np.savez_compressed(out / "baseline_target_explanations.npz", **target_details)
                previous_cpu = args.run_root.parent / "normal_bn_estimator_dev_v1_20261030/B0_source_val_predictions.npz"
                if previous_cpu.exists():
                    with np.load(previous_cpu, allow_pickle=False) as data:
                        if not np.array_equal(labels, data["labels"]):
                            raise ValueError("previous CPU source labels differ")
                        row["previous_cpu_baseline_comparison"] = {
                            "argmax_disagreements": int(np.sum(src_preds != data["probabilities"].argmax(1))),
                            "max_probability_difference": float(np.max(np.abs(src_probs - data["probabilities"])))}
            report["arms"][arm] = row
            write_json(out / "report.json", report)
            print(json.dumps({"arm": arm, "source_val": row["source_val"], "target_normal_fpr": row["target_adapt_normal"]["fpr_argmax"]}), flush=True)
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in original.items()):
            raise RuntimeError("model parameter or buffer changed")
        require_hash(out / args.protocol.name, protocol_sha)
        report["all_state_tensors_unchanged"] = True
        report["status"] = "COMPLETED_DIAGNOSTIC_ONLY"
        report["elapsed_s"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "report.json", report)
        print(f"completed: {out / 'report.json'}", flush=True)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        write_json(out / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
