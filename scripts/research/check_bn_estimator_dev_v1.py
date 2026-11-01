"""Three fixed CPU development arms; source validation only, no Real test reads."""
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

from diagnose_normal_adaptation_v1 import probability_summary
from run_normal_adaptation_v1 import (MODEL_SHA, MODEL_SOURCE_SHA, NORM_SHA, PROTOCOL_SHA,
                                      load_roles, recalibrate, require_hash, sha256,
                                      validate_bundle, validate_source, write_json)


def recalibrate_cumulative(model, x):
    """Equal average of batch moments, NOT a pooled population variance."""
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.momentum = None
    return recalibrate(model, x, torch.device("cpu"))


def run(args):
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (protocol["protocol_name"], protocol["status"]) != ("rflymad_normal_bn_estimator_dev_v1", "FROZEN_DEVELOPMENT_ONLY_BEFORE_EXECUTION"):
        raise ValueError("unexpected development protocol")
    if protocol["parent_protocol_sha256"] != PROTOCOL_SHA or protocol["source_checkpoint_sha256"] != MODEL_SHA or protocol["norm_sha256"] != NORM_SHA:
        raise ValueError("development protocol source identity mismatch")
    if protocol["adaptation"]["batch_size"] != 64 or protocol["adaptation"]["passes"] != 1 or protocol["evaluation"]["batch_size"] != 512:
        raise ValueError("development protocol batch schedule mismatch")
    if set(protocol["arms"]) != {"B0_source", "B1_reset_ema", "B2_reset_cumulative"} or protocol["evaluation"]["threshold_search"]:
        raise ValueError("unexpected arms or threshold search")
    source_run = args.run_root / "results/20261030_adaptation_sourceval_r2"
    parent, rows = load_roles(source_run / "normal_adaptation_protocol_v1.json", source_run / "real_case_roles_v1.csv")
    old_report = json.loads((source_run / "report.json").read_text())
    if old_report["status"] != "COMPLETED":
        raise ValueError("v1 adaptation was not completed")
    require_hash(source_run / "best.pt", MODEL_SHA)
    require_hash(source_run / "lstm.py", MODEL_SOURCE_SHA)
    require_hash(source_run / "norm_stats.json", NORM_SHA)
    val_sha = next(row["local"]["sha256"] for row in parent["cache_identity"]["files"] if row["file"] == "val.pt")
    require_hash(args.source_cache / "val.pt", val_sha)
    bundle_path = args.run_root / "adaptation_input/adapt_normal.pt"
    require_hash(bundle_path, old_report["bundle_sha256"])
    stats = json.loads((source_run / "norm_stats.json").read_text())
    config = json.loads((source_run / "config.json").read_text())
    bundle = torch.load(bundle_path, weights_only=True, map_location="cpu")
    counts = validate_bundle(bundle, rows, stats["columns"])
    if len(bundle["x"]) != 640 or dict(counts) != old_report["adaptation_cases"] or set(counts.values()) != {64}:
        raise ValueError("expected the original 10-case, 640-window adaptation input")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    normalized = (bundle["x"] - mean) / std
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any() or not torch.isfinite(normalized).all():
        raise ValueError("invalid source normalization")
    state = torch.load(source_run / "best.pt", weights_only=True, map_location="cpu")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for path in (args.protocol, source_run / "lstm.py", source_run / "config.json", source_run / "norm_stats.json",
                 Path(__file__), Path(__file__).with_name("run_normal_adaptation_v1.py"),
                 Path(__file__).with_name("diagnose_normal_adaptation_v1.py")):
        shutil.copy2(path, out / path.name)
    protocol_sha = sha256(out / args.protocol.name)
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "protocol_name": protocol["protocol_name"], "protocol_sha256_before_inference": protocol_sha,
              "source_checkpoint_sha256": MODEL_SHA, "source_code_sha256": MODEL_SOURCE_SHA,
              "norm_sha256": NORM_SHA, "source_val_sha256": val_sha,
              "adaptation_bundle_sha256": old_report["bundle_sha256"],
              "runner_sha256": sha256(__file__),
              "recalibration_and_eval_helper_sha256": sha256(Path(__file__).with_name("run_normal_adaptation_v1.py")),
              "summary_helper_sha256": sha256(Path(__file__).with_name("diagnose_normal_adaptation_v1.py")),
              "device": "cpu", "threads": 4, "seed": 42, "torch": str(torch.__version__), "numpy": np.__version__,
              "new_real_calibration_or_test_predictions": False, "deployment": False,
              "adaptation_cases": dict(counts), "arms": {}}
    write_json(out / "report.json", report)
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("bn_estimator_source_snapshot", out / "lstm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if stats["columns"] != module.RFLY_SEQ_COLUMNS_V2:
            raise ValueError("source snapshot column mismatch")
        for arm in protocol["arms"]:
            print(f"{arm}: starting CPU development check", flush=True)
            model = module.OnlineKGMSTCA(in_channels=26, num_classes=2, hidden_size=config["hidden_size"],
                                        num_layers=config["num_layers"], dropout=config["dropout"], columns=stats["columns"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            arm_result = {"bn": None, "non_bn_statistics_unchanged": True}
            if arm == "B1_reset_ema":
                arm_result["bn"] = recalibrate(model, normalized, torch.device("cpu"))
            elif arm == "B2_reset_cumulative":
                arm_result["bn"] = recalibrate_cumulative(model, normalized)
            if arm_result["bn"] and any(layer["num_batches_tracked"] != 10 for layer in arm_result["bn"].values()):
                raise RuntimeError("incorrect adaptation batch count")
            arm_result["source_val"] = validate_source(model, args.source_cache / "val.pt", mean, std, torch.device("cpu"), out / f"{arm}_val_predictions.npz")
            with np.load(out / f"{arm}_val_predictions.npz", allow_pickle=False) as data:
                probabilities, labels = data["probabilities"], data["labels"]
            arm_result["probabilities"] = {"normal": probability_summary(probabilities[labels == 0, 1]),
                                           "fault": probability_summary(probabilities[labels == 1, 1])}
            if arm != "B2_reset_cumulative":
                old_arm = "A0" if arm == "B0_source" else "A1"
                with np.load(source_run / f"{old_arm}_val_predictions.npz", allow_pickle=False) as data:
                    old_probs, old_labels = data["probabilities"], data["labels"]
                if not np.array_equal(labels, old_labels):
                    raise ValueError("historical source val labels differ")
                arm_result["historical_gpu_numeric_comparison"] = {
                    "note": "CPU and historical GPU probabilities need not be bitwise equal; primary comparison is all-CPU.",
                    "argmax_disagreements": int(np.sum(probabilities.argmax(1) != old_probs.argmax(1))),
                    "maximum_probability_difference": float(np.max(np.abs(probabilities - old_probs)))}
            if arm != "B0_source":
                torch.save(model.state_dict(), out / f"{arm}.pt.tmp")
                (out / f"{arm}.pt.tmp").replace(out / f"{arm}.pt")
                arm_result["checkpoint_sha256"] = sha256(out / f"{arm}.pt")
            report["arms"][arm] = arm_result
            write_json(out / "report.json", report)
            print(json.dumps({"arm": arm, "source_val": arm_result["source_val"]}, indent=2), flush=True)
        require_hash(out / args.protocol.name, protocol_sha)
        report["status"] = "COMPLETED_DEVELOPMENT_ONLY"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_s"] = time.perf_counter() - started
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
