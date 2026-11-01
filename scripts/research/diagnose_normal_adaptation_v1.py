"""CPU-only v1 postmortem on adaptation normals, calibration normals and source val.

No Real test probabilities/features are opened, no inference/training is run, and
no thresholds, models, frozen protocols or existing reports are overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from run_normal_adaptation_v1 import MODEL_SHA, NORM_SHA, PROTOCOL_SHA, require_hash, sha256, write_json


def ema_weights(momentum, batches):
    if not 0 < momentum <= 1 or batches < 1:
        raise ValueError("expected positive momentum <= 1 and at least one batch")
    initial = (1.0 - momentum) ** batches
    weights = [momentum * (1.0 - momentum) ** (batches - 1 - i) for i in range(batches)]
    return initial, weights


def probability_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("expected a nonempty finite probability vector in [0,1]")
    return {"points": len(values), "quantiles": {str(q): float(np.quantile(values, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "fraction_at_or_above_onset_0_6": float(np.mean(values >= 0.6)),
            "fraction_below_clear_0_4": float(np.mean(values < 0.4)),
            "fraction_at_or_above_0_95": float(np.mean(values >= 0.95)),
            "fraction_exactly_one_float32": float(np.mean(values == 1.0))}


def feature_summary(x, indices, mean, std):
    if x.ndim != 3 or mean.shape != (1, x.shape[1], 1) or std.shape != mean.shape:
        raise ValueError("invalid feature/normalization shapes")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid source normalization")
    if not len(indices):
        raise ValueError("empty feature population")
    channels = x.shape[1]
    total = torch.zeros(channels, dtype=torch.float64)
    squares, extreme = total.clone(), total.clone()
    count = 0
    for start in range(0, len(indices), 256):
        z = ((x[indices[start:start + 256]].float() - mean) / std).double()
        if not torch.isfinite(z).all():
            raise ValueError("non-finite normalized features")
        total += z.sum(dim=(0, 2))
        squares += z.square().sum(dim=(0, 2))
        extreme += (z.abs() > 3).sum(dim=(0, 2))
        count += z.shape[0] * z.shape[2]
    average = total / count
    variance = (squares / count - average.square()).clamp_min(0)
    return {"windows": len(indices), "frames_per_channel_with_overlap": count,
            "z_mean": average.tolist(), "z_std": variance.sqrt().tolist(),
            "fraction_abs_z_over_3": (extreme / count).tolist()}


def bn_summary(before, after, metadata):
    if set(before) != set(after):
        raise ValueError("checkpoint state keys differ")
    allowed = {f"{name}.{suffix}" for name in metadata for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    if any(not torch.equal(value, after[name]) for name, value in before.items() if name not in allowed):
        raise ValueError("non-BN-statistics tensor changed")
    result = {}
    for name, meta in metadata.items():
        n = int(after[f"{name}.num_batches_tracked"])
        if n != meta["num_batches_tracked"] or meta["momentum"] is None:
            raise ValueError("BN metadata does not describe the checkpoint")
        residual, weights = ema_weights(meta["momentum"], n)
        old, new = before[f"{name}.running_var"].double(), after[f"{name}.running_var"].double()
        if not torch.isfinite(old).all() or not torch.isfinite(new).all() or (old <= 0).any() or (new < residual - 1e-6).any():
            raise ValueError("invalid BN variance or inconsistent reset floor")
        floor_share = residual / new
        ratio = new / old
        result[name] = {"batches": n, "momentum": meta["momentum"],
                        "reset_mean_zero_residual_weight": residual,
                        "reset_variance_one_additive_floor": residual,
                        "observed_batch_total_weight": sum(weights),
                        "new_over_source_variance_median": float(torch.quantile(ratio, 0.5)),
                        "new_over_source_variance_min": float(ratio.min()),
                        "new_over_source_variance_max": float(ratio.max()),
                        "reset_floor_share_of_saved_variance_median": float(torch.quantile(floor_share, 0.5))}
    return result


def run(args):
    root = args.run_root
    source_run = root / "results/20261030_adaptation_sourceval_r2"
    replay_run = root / "results/20261030_full_replay_r2"
    report = json.loads((source_run / "report.json").read_text())
    replay_report = json.loads((replay_run / "report.json").read_text())
    if report["status"] != "COMPLETED" or report["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("expected the completed frozen v1 adaptation")
    if replay_report["status"] != "COMPLETED_REAL_REPLAY_AND_SAMPLED_HOLDOUT":
        raise ValueError("expected completed replay report")
    require_hash(source_run / "best.pt", MODEL_SHA)
    require_hash(source_run / "A1_adapted.pt", report["adapted_checkpoint_sha256"])
    require_hash(source_run / "norm_stats.json", NORM_SHA)
    require_hash(args.source_cache / "val.pt", report["source_validation_sha256"])
    require_hash(root / "adaptation_input/adapt_normal.pt", report["bundle_sha256"])
    require_hash(root / "replay_input/index.json", replay_report["input_index_sha256"])
    require_hash(replay_run / "thresholds_frozen.json", replay_report["thresholds_sha256_before_test"])
    stats = json.loads((source_run / "norm_stats.json").read_text())
    columns = stats["columns"]
    before = torch.load(source_run / "best.pt", map_location="cpu", weights_only=True)
    after = torch.load(source_run / "A1_adapted.pt", map_location="cpu", weights_only=True)
    bundle = torch.load(root / "adaptation_input/adapt_normal.pt", map_location="cpu", weights_only=True)
    if bundle["columns"] != columns or bundle["samples"] != report["adaptation_samples"]:
        raise ValueError("adaptation bundle differs from executed input")
    case_counts = Counter(sample["case_uid"] for sample in bundle["samples"])
    if dict(case_counts) != report["adaptation_cases"] or set(case_counts.values()) != {64}:
        raise ValueError("expected exactly one 64-window batch per recorded adaptation case")
    batch_counts = {meta["num_batches_tracked"] for meta in report["batchnorm"].values()}
    momenta = {meta["momentum"] for meta in report["batchnorm"].values()}
    if batch_counts != {10} or momenta != {0.1}:
        raise ValueError("unexpected executed v1 BN update schedule")
    residual, weights = ema_weights(0.1, 10)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    val = torch.load(args.source_cache / "val.pt", map_location="cpu", weights_only=True, mmap=True)
    labels = val["y"].numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("unexpected source validation labels")
    source_features = feature_summary(val["x"], np.flatnonzero(labels == 0), mean, std)
    target_features = feature_summary(bundle["x"], np.arange(len(bundle["x"])), mean, std)
    features = []
    for i, name in enumerate(columns):
        features.append({"channel": name,
                         "source_val_endpoint_normal_z_mean": source_features["z_mean"][i],
                         "adapt_normal_z_mean": target_features["z_mean"][i],
                         "absolute_z_mean_shift": abs(source_features["z_mean"][i] - target_features["z_mean"][i]),
                         "source_val_endpoint_normal_z_std": source_features["z_std"][i],
                         "adapt_normal_z_std": target_features["z_std"][i],
                         "source_fraction_abs_z_over_3": source_features["fraction_abs_z_over_3"][i],
                         "adapt_fraction_abs_z_over_3": target_features["fraction_abs_z_over_3"][i]})
    result = {"status": "COMPLETED_DESCRIPTIVE_DIAGNOSIS", "runner_sha256": sha256(__file__),
              "source_adaptation_report_sha256": sha256(source_run / "report.json"),
              "protocol_sha256": PROTOCOL_SHA,
              "restrictions": {"cpu_only": True, "new_inference": False, "model_or_threshold_changes": False,
                               "real_test_features_or_probabilities_opened": False,
                               "real_test_metadata_only": "Existing input index maps the five calibration files to identities."},
              "checkpoint_non_bn_statistics_unchanged": True,
              "ema_schedule": {"initialization_residual": residual, "target_batch_total_weight": sum(weights),
                               "batches": [{"case_uid": uid, "weight": weight} for uid, weight in zip(case_counts, weights)],
                               "note": "Residual is reset initialization (mean=0,var=1), NOT retained source statistics."},
              "batchnorm": bn_summary(before, after, report["batchnorm"]),
              "feature_populations": {"source": source_features["windows"], "target": target_features["windows"],
                                      "note": "Source endpoint-normal windows include pre-fault windows. Overlapping frames are not independent samples."},
              "feature_shift_sorted": sorted(features, key=lambda row: row["absolute_z_mean_shift"], reverse=True),
              "source_val_probabilities": {}, "calibration_normals": {}}
    index = json.loads((root / "replay_input/index.json").read_text())
    calibration = sorted((row for row in index["cases"] if row["role"] == "calibrate_normal"), key=lambda r: r["case_uid"])
    thresholds = json.loads((replay_run / "thresholds_frozen.json").read_text())
    for arm in ("A0", "A1"):
        with np.load(source_run / f"{arm}_val_predictions.npz", allow_pickle=False) as data:
            probs, saved_labels = data["probabilities"], data["labels"]
        if not np.array_equal(saved_labels, labels):
            raise ValueError("saved predictions/source cache label mismatch")
        result["source_val_probabilities"][arm] = {
            "normal": probability_summary(probs[labels == 0, 1]),
            "fault": probability_summary(probs[labels == 1, 1]),
            "fault_fraction_at_or_above_frozen_target_onset": float(np.mean(probs[labels == 1, 1] >= thresholds[arm]["T1"]))}
        rows, pooled = [], []
        for member in calibration:
            if member["level2_label"] != "normal":
                raise ValueError("calibration must be normal-only")
            path = replay_run / f"calibration_{arm}_{Path(member['file']).stem}.npz"
            with np.load(path, allow_pickle=False) as data:
                probabilities = data["fault_prob"]
            pooled.extend(probabilities.tolist())
            rows.append({"case_uid": member["case_uid"], "summary": probability_summary(probabilities)})
        q = float(np.quantile(pooled, 0.995, method="linear"))
        if q != thresholds[arm]["q995"] or len(pooled) != thresholds[arm]["calibration_points"]:
            raise ValueError("calibration probabilities do not reproduce frozen threshold")
        for row in rows:
            row["pooled_point_weight"] = row["summary"]["points"] / len(pooled)
        result["calibration_normals"][arm] = {"pooled": probability_summary(pooled), "cases": rows,
                                             "frozen_onset": thresholds[arm]["T1"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "diagnosis.json", result)
    print(json.dumps({"status": result["status"], "ema": result["ema_schedule"],
                      "top_feature_shifts": result["feature_shift_sorted"][:6],
                      "calibration_pooled": {arm: row["pooled"] for arm, row in result["calibration_normals"].items()}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
