"""Frozen source-only GPU jobs, CPU execution checks, preflight and paired summary.

Never opens a final-test cache. Per-epoch model/history saves are not optimizer/RNG
resume. Use a new output directory when explicitly rerunning an interrupted job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from smoke_normal_source_sampling import (EXTRA, FAULT, ORIGINAL, PREFAULT, FocalLoss, OnlineKGMSTCA,
    RFLY_SEQ_COLUMNS_V2, probability_table, sampling_weights, save_state, stratified_metrics)
from run_normal_adaptation_v1 import require_hash, sha256, write_json
from uav_fault.models.train_online_detector import get_warmup_cosine_lr
from uav_fault import model_contract

PROTOCOL_SHA = "088bbb0c1ce2d831321304322dab35491b2fdab1dab9fb05aef2dae85d1fcf23"
REPO = Path(__file__).resolve().parents[2]


def load_protocol(path):
    require_hash(path, PROTOCOL_SHA)
    return json.loads(path.read_text(encoding="utf-8"))


def environment():
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "pandas": pd.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available())}


def require_formal_gpu(protocol):
    if not torch.cuda.is_available():
        raise RuntimeError("Formal training requires CUDA; CPU fallback is forbidden. Use --cpu-check only for a tiny check.")
    actual = environment()
    for name, expected in protocol["formal_environment"].items():
        if actual[name] != expected:
            raise RuntimeError(f"Formal environment mismatch: {name}={actual[name]!r}, required {expected!r}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != protocol["reproducibility"]["cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG does not match frozen protocol")


def code_identities(protocol):
    paths = {"model_sha256": Path(inspect.getfile(OnlineKGMSTCA)), "loss_sha256": Path(inspect.getfile(FocalLoss)),
             "model_contract_sha256": Path(inspect.getfile(model_contract)),
             "sampling_helper_sha256": Path(__file__).with_name("smoke_normal_source_sampling.py"),
             "lr_helper_sha256": Path(inspect.getfile(get_warmup_cosine_lr))}
    for key, path in paths.items():
        require_hash(path, protocol["source_code"][key])
    return paths


def prepare_inputs(cache, audit_dir, protocol):
    expected = protocol["inputs"]
    require_hash(audit_dir / "audit.json", expected["audit_sha256"])
    audit = json.loads((audit_dir / "audit.json").read_text())
    if audit["status"] != "COMPLETED_SOURCE_DATA_AUDIT" or audit["train_val_pair_crossings"] != 0:
        raise ValueError("source linkage audit incomplete or paired groups cross train/val")
    payloads, frames = {}, {}
    for split in ("train", "val"):
        require_hash(audit_dir / f"{split}_source_window_links.csv", expected[f"{split}_links_sha256"])
        require_hash(cache / f"{split}.pt", expected[f"{split}_cache_sha256"])
        data = torch.load(cache / f"{split}.pt", map_location="cpu", weights_only=True, mmap=True)
        frame = pd.read_csv(audit_dir / f"{split}_source_window_links.csv")
        if data["x"].dtype != torch.float32 or tuple(data["x"].shape) != (expected[f"{split}_windows"], 26, 150):
            raise ValueError("source tensor shape/dtype mismatch")
        if data["y"].dtype != torch.int64 or data["y"].shape != (len(data["x"]),):
            raise ValueError("source label shape/dtype mismatch")
        prefix = expected["train_source_prefix_windows"] if split == "train" else expected["val_windows"]
        if len(frame) != prefix or not np.array_equal(frame.cache_index, np.arange(prefix)):
            raise ValueError("linked cache order/length mismatch")
        if frame.case_uid.isna().any() or not frame.split.eq(split).all() or not frame.domain.isin(["SIL", "HIL"]).all():
            raise ValueError("non-source or invalid linked metadata")
        if not np.array_equal(frame.label, data["y"][:prefix].numpy()):
            raise ValueError("linked labels do not match cache")
        if not frame.groupby("case_uid").size().eq(expected["per_case_cap"]).all():
            raise ValueError("unexpected source per-case cap")
        frames[split], payloads[split] = frame, data
    if set(frames["train"].case_uid) & set(frames["val"].case_uid):
        raise ValueError("train/val case leakage")
    if frames["val"].case_uid.nunique() != expected["validation_cases"]:
        raise ValueError("unexpected validation cases")
    require_hash(cache / "norm_stats.json", expected["norm_sha256"])
    stats = json.loads((cache / "norm_stats.json").read_text())
    if stats["columns"] != RFLY_SEQ_COLUMNS_V2 or stats["window_size"] != 150 or stats["target_hz"] != 50:
        raise ValueError("normalization input contract mismatch")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid normalization")
    extra = expected["rflysim_normal_suffix_windows"]
    if len(payloads["train"]["y"]) - len(frames["train"]) != extra or torch.count_nonzero(payloads["train"]["y"][-extra:]):
        raise ValueError("RflySim normal suffix mismatch")
    origins = np.r_[frames["train"].normal_origin.to_numpy(dtype=str), np.full(extra, EXTRA)]
    labels = payloads["train"]["y"].numpy()
    weights = {arm: sampling_weights(labels, origins, arm == "balanced_source_normal") for arm in protocol["arms"]}
    tables = {arm: probability_table(weight, labels, origins) for arm, weight in weights.items()}
    uniform, balanced = tables["uniform_replacement"], tables["balanced_source_normal"]
    for key in ("total_weight", "normal_probability", "fault_probability"):
        if not np.isclose(uniform[key], balanced[key], rtol=0, atol=1e-9):
            raise ValueError("sampler class probability changed")
    if not np.isclose(uniform["origins"][EXTRA]["probability"], balanced["origins"][EXTRA]["probability"], rtol=0, atol=1e-12):
        raise ValueError("RflySim probability changed")
    if not np.isclose(balanced["origins"][ORIGINAL]["probability"], balanced["origins"][PREFAULT]["probability"], rtol=0, atol=1e-12):
        raise ValueError("source-normal probabilities not equal")
    return {"payloads": payloads, "frames": frames, "origins": origins, "stats": stats,
            "mean": mean, "std": std, "weights": weights, "sampling": tables}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def selection_step(state, score, epoch, rule):
    if not math.isfinite(score):
        raise FloatingPointError("non-finite validation selection score")
    result = dict(state)
    save_best = score > state["best_score"]
    if save_best:
        result["best_score"], result["best_epoch"] = score, epoch
    if score > state["significant_reference"] + rule["significant_delta"]:
        result["significant_reference"], result["bad_epochs"] = score, 0
    elif epoch >= rule["min_epochs"]:
        result["bad_epochs"] += 1
    stop = epoch >= rule["min_epochs"] and result["bad_epochs"] >= rule["patience"]
    return result, save_best, stop


def evaluate_source(model, loader, mean, std, device, metadata):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for raw, y, unused in loader:
            probs = model((raw.to(device, non_blocking=True) - mean) / std).softmax(1).cpu().numpy()
            if not np.isfinite(probs).all():
                raise FloatingPointError("non-finite source-validation predictions")
            probabilities.append(probs)
            labels.append(y.numpy())
    probs, actual = np.concatenate(probabilities), np.concatenate(labels)
    return stratified_metrics(actual, probs.argmax(1), metadata), probs, actual


def run_job(args, protocol):
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    if args.seed not in protocol["seeds"] or args.arm not in protocol["arms"]:
        raise ValueError("unregistered seed or arm")
    bundle_verification = None
    if not args.cpu_check:
        require_formal_gpu(protocol)
        if not (REPO / "SHA256SUMS").is_file():
            raise RuntimeError("Formal training must run from the frozen bundle containing SHA256SUMS")
        bundle_verification = subprocess.run(["sha256sum", "--check", "--strict", "SHA256SUMS"], cwd=REPO,
                                             check=True, capture_output=True, text=True).stdout
    paths = code_identities(protocol)
    data = prepare_inputs(args.cache, args.audit, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    out, cfg = args.output, protocol["training"]
    report = {"status": "RUNNING_CPU_CHECK" if args.cpu_check else "RUNNING_FORMAL_TRAINING",
              "protocol_name": protocol["protocol_name"], "protocol_sha256": PROTOCOL_SHA,
              "seed": args.seed, "arm": args.arm, "cpu_check": args.cpu_check,
              "started_at": datetime.now(timezone.utc).isoformat(), "environment": environment(),
              "runner_sha256": sha256(__file__),
              "code_bundle_manifest_sha256": sha256(REPO / "SHA256SUMS") if (REPO / "SHA256SUMS").is_file() else None,
              "source_code": protocol["source_code"],
              "inputs": protocol["inputs"], "sampling": data["sampling"][args.arm],
              "scope": "Source train/val only; no Real/final test reads or deployment. No optimizer/RNG resume."}
    write_json(out / "report.json", report)
    if bundle_verification is not None:
        (out / "bundle_verification.txt").write_text(bundle_verification, encoding="utf-8")
    started = time.perf_counter()
    try:
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cpu" if args.cpu_check else "cuda:0")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            report["gpu"] = torch.cuda.get_device_name(device)
        model = OnlineKGMSTCA(in_channels=26, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                              num_classes=2, dropout=cfg["dropout"], columns=data["stats"]["columns"]).float()
        if sum(p.numel() for p in model.parameters()) != cfg["parameters"]:
            raise ValueError("model parameter count changed")
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        report["initial_state_digest"] = state_digest(initial)
        save_state(out / "initial.pt", initial)
        model.to(device)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        mean, std = data["mean"].to(device), data["std"].to(device)
        train = data["payloads"]["train"]
        val = data["payloads"]["val"]
        metadata = data["frames"]["val"]
        check = protocol["cpu_check"]
        if args.cpu_check:
            selected = [group.iloc[np.linspace(0, len(group) - 1, check["val_windows_per_case"]).round().astype(int)]
                        for _, group in metadata.groupby("case_uid", sort=False)]
            metadata = pd.concat(selected, ignore_index=True)
        indices = torch.tensor(metadata.cache_index.to_numpy(), dtype=torch.long)
        val_x = val["x"][indices] if args.cpu_check else val["x"]
        val_y = val["y"][indices] if args.cpu_check else val["y"]
        batch_size = check["batch_size"] if args.cpu_check else cfg["batch_size"]
        draws = check["draws_per_epoch"] if args.cpu_check else len(train["y"])
        epochs = check["epochs"] if args.cpu_check else cfg["max_epochs"]
        sampler = WeightedRandomSampler(data["weights"][args.arm], draws, replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed + 11))
        train_loader = DataLoader(TensorDataset(train["x"], train["y"], torch.arange(len(train["y"]))),
                                  batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda",
                                  generator=torch.Generator().manual_seed(args.seed + 101))
        val_loader = DataLoader(TensorDataset(val_x, val_y, indices), batch_size=check["validation_batch_size"] if args.cpu_check else protocol["validation"]["batch_size"],
                                num_workers=0, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed + 101))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = FocalLoss(gamma=cfg["loss_gamma"])
        selection = {"best_score": -float("inf"), "best_epoch": None, "significant_reference": -float("inf"), "bad_epochs": 0}
        history = []
        for path in [args.protocol, Path(__file__), *paths.values(), args.cache / "norm_stats.json"]:
            shutil.copy2(path, out / path.name)
        config = {"model": "kg_mstca", "task": "online_binary", "feature_schema": "rfly_seq_columns_v2",
                  "protocol_name": data["stats"]["protocol_name"], "experiment_protocol_name": protocol["protocol_name"],
                  "window_size": 150, "in_channels": 26, "feature_count": 26, "num_classes": 2,
                  "hidden_size": cfg["hidden_size"], "num_layers": 1, "dropout": cfg["dropout"],
                  "label_names": ["normal", "fault"], "seed": args.seed, "arm": args.arm,
                  "cpu_check": args.cpu_check, "epochs_trained": 0, "best_epoch": None, "best_val_f1": None,
                  "training": cfg, "validation": protocol["validation"]}
        report.update({"device": str(device), "validation_windows": len(metadata), "draws_per_epoch": draws,
                       "batch_size": batch_size, "maximum_epochs_this_execution": epochs,
                       "formal_validation_complete": not args.cpu_check})
        write_json(out / "config.json", config)
        write_json(out / "report.json", report)
        for epoch in range(1, epochs + 1):
            lr = get_warmup_cosine_lr(epoch - 1, cfg["warmup_epochs"], cfg["max_epochs"], cfg["lr"], cfg["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            losses, realized = [], Counter()
            for raw, y, idx in train_loader:
                batch = (raw.to(device, non_blocking=True) - mean) / std
                if not torch.isfinite(batch).all():
                    raise FloatingPointError("non-finite source training batch")
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), y.to(device, non_blocking=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                loss.backward()
                # Native nonfinite rejection covers every gradient without a GPU
                # synchronization for each individual parameter tensor.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                realized.update(data["origins"][idx.numpy()])
            metrics, unused, unused_labels = evaluate_source(model, val_loader, mean, std, device, metadata)
            score = float(metrics["overall"]["macro_f1"])
            selection, save_best, stop = selection_step(selection, score, epoch, protocol["validation"]["early_stop"])
            if save_best:
                save_state(out / "best.pt", model.state_dict())
            save_state(out / "last.pt", model.state_dict())
            history.append({"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses)), "gradient_steps": len(losses),
                            "training_origin_counts": dict(realized), "source_validation": metrics, "selection": dict(selection)})
            config.update({"epochs_trained": epoch, "best_epoch": selection["best_epoch"], "best_val_f1": selection["best_score"]})
            write_json(out / "history.json", history)
            write_json(out / "config.json", config)
            print(f"seed={args.seed} arm={args.arm} epoch={epoch}/{epochs} lr={lr:.3g} loss={np.mean(losses):.5f} val_f1={score:.6f} "
                  f"pure_normal_fpr={metrics['normal_origin'][ORIGINAL]['normal_window_fpr']:.4f} "
                  f"hover_fpr={metrics['original_normal_by_status']['hover']['normal_window_fpr']:.4f}", flush=True)
            if stop:
                break
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True), strict=True)
        metrics, probs, actual = evaluate_source(model, val_loader, mean, std, device, metadata)
        if metrics != history[selection["best_epoch"] - 1]["source_validation"]:
            raise RuntimeError("best checkpoint reload validation differs")
        if not any(not torch.equal(initial[name], param.detach().cpu()) for name, param in model.named_parameters()):
            raise RuntimeError("no trainable parameter changed")
        np.savez_compressed(out / "validation_predictions.npz", probabilities=probs, labels=actual, cache_indices=indices.numpy())
        for split in ("train", "val"):
            require_hash(args.cache / f"{split}.pt", protocol["inputs"][f"{split}_cache_sha256"])
        require_hash(args.cache / "norm_stats.json", protocol["inputs"]["norm_sha256"])
        for split in ("train", "val"):
            require_hash(args.audit / f"{split}_source_window_links.csv", protocol["inputs"][f"{split}_links_sha256"])
        require_hash(args.protocol, PROTOCOL_SHA)
        require_hash(Path(__file__), report["runner_sha256"])
        code_identities(protocol)
        report.update({"status": "COMPLETED_CPU_CHECK_NOT_FORMAL" if args.cpu_check else "COMPLETED_FORMAL_SOURCE_TRAINING",
                       "epochs_trained": len(history), "best_epoch": selection["best_epoch"], "checkpoint_reload_matches": True,
                       "best_checkpoint_sha256": sha256(out / "best.pt"), "source_validation": metrics,
                       "elapsed_s": time.perf_counter() - started, "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(out / "report.json", report)
        return report
    except Exception as error:
        report.update({"status": "FAILED", "error": repr(error), "elapsed_s": time.perf_counter() - started})
        write_json(out / "report.json", report)
        raise


def summarize(root, protocol):
    reports = {}
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            row = json.loads((root / f"seed_{seed}" / arm / "report.json").read_text())
            if (row["status"] != "COMPLETED_FORMAL_SOURCE_TRAINING" or row["cpu_check"] or row["protocol_sha256"] != PROTOCOL_SHA
                or row["seed"] != seed or row["arm"] != arm or not row["formal_validation_complete"]
                or row["source_code"] != protocol["source_code"] or row["inputs"] != protocol["inputs"]):
                raise ValueError("missing, mismatched, CPU-check or incomplete formal job")
            if (row["validation_windows"] != protocol["inputs"]["val_windows"] or not row["checkpoint_reload_matches"]
                or any(row["environment"][key] != value for key, value in protocol["formal_environment"].items())
                or not row["environment"]["cuda_available"] or len(row["initial_state_digest"]) != 64):
                raise ValueError("formal job validation scope, checkpoint or environment is invalid")
            reports[seed, arm] = row
        if reports[seed, protocol["arms"][0]]["initial_state_digest"] != reports[seed, protocol["arms"][1]]["initial_state_digest"]:
            raise ValueError("paired initial states differ")
    if len({row["runner_sha256"] for row in reports.values()}) != 1 or len({row["code_bundle_manifest_sha256"] for row in reports.values()}) != 1:
        raise ValueError("formal jobs did not use the same trainer/code bundle")
    if not all(row["code_bundle_manifest_sha256"] for row in reports.values()):
        raise ValueError("formal jobs lack frozen bundle identity")
    values = {}
    for key in ("macro_f1", "fault_recall", "normal_fpr", "original_normal_fpr", "prefault_normal_fpr", "original_hover_fpr"):
        groups = {}
        for arm in protocol["arms"]:
            points = []
            for seed in protocol["seeds"]:
                metrics = reports[seed, arm]["source_validation"]
                if key in ("macro_f1", "fault_recall"):
                    value = metrics["overall"][key]
                elif key == "normal_fpr":
                    value = metrics["overall"]["normal_window_fpr"]
                elif key == "original_hover_fpr":
                    value = metrics["original_normal_by_status"]["hover"]["normal_window_fpr"]
                else:
                    origin = ORIGINAL if key == "original_normal_fpr" else PREFAULT
                    value = metrics["normal_origin"][origin]["normal_window_fpr"]
                if value is None or not math.isfinite(value):
                    raise ValueError("missing required stratified validation metric")
                points.append(float(value))
            groups[arm] = {"per_seed": points, "mean": float(np.mean(points)), "sample_std": float(np.std(points, ddof=1))}
        difference = np.array(groups[protocol["arms"][1]]["per_seed"]) - np.array(groups[protocol["arms"][0]]["per_seed"])
        groups["paired_balanced_minus_uniform"] = {"per_seed": difference.tolist(), "mean": float(difference.mean()), "sample_std": float(difference.std(ddof=1))}
        values[key] = groups
    return {"status": "COMPLETED_PAIRED_SOURCE_VALIDATION_SUMMARY", "protocol_sha256": PROTOCOL_SHA,
            "seeds": protocol["seeds"], "jobs": 6, "metrics": values, "final_test_or_deployment_claim": False}


def new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    preflight = commands.add_parser("preflight")
    summary = commands.add_parser("summarize")
    for child in (run, preflight, summary):
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    for child in (run, preflight):
        child.add_argument("--cache", type=Path, required=True)
        child.add_argument("--audit", type=Path, required=True)
    run.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    run.add_argument("--arm", choices=["uniform_replacement", "balanced_source_normal"], required=True)
    run.add_argument("--cpu-check", action="store_true")
    summary.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "run":
        run_job(args, protocol)
    elif args.command == "preflight":
        code = code_identities(protocol)
        data = prepare_inputs(args.cache, args.audit, protocol)
        new_json(args.output, {"status": "PASSED_INPUT_AND_CODE_PREFLIGHT_NOT_GPU_TRAINING", "protocol_sha256": PROTOCOL_SHA,
                 "environment": environment(), "sampling": data["sampling"], "source_code_verified": {k: str(p) for k, p in code.items()},
                 "inputs": protocol["inputs"], "cuda_formal_execution_not_verified": True})
    else:
        new_json(args.output, summarize(args.runs, protocol))


if __name__ == "__main__":
    main()
