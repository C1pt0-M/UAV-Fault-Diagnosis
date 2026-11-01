"""CPU pipeline check only: paired normal-origin samplers, source train/val, no tests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages/uav_fault"))
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.cnn.lstm import OnlineKGMSTCA, RFLY_SEQ_COLUMNS_V2
from run_normal_adaptation_v1 import require_hash, sha256, write_json
from run_normal_replay_v1 import classification

ORIGINAL = "original_normal_case"
PREFAULT = "normal_part_of_fault_case"
FAULT = "fault_endpoint"
EXTRA = "rflysim_normal"


def sampling_weights(labels, origins, balanced):
    labels, origins = np.asarray(labels), np.asarray(origins)
    if labels.ndim != 1 or origins.shape != labels.shape or not len(labels) or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("invalid labels/origins")
    if not set(origins).issubset({ORIGINAL, PREFAULT, FAULT, EXTRA}) or not np.array_equal(labels == 1, origins == FAULT):
        raise ValueError("unknown origin or label/origin mismatch")
    counts = Counter(origins)
    if not counts[ORIGINAL] or not counts[PREFAULT] or not counts[FAULT]:
        raise ValueError("both normal origins and fault windows must be present")
    weights = torch.ones(len(labels), dtype=torch.float64)
    if balanced:
        source_normal = counts[ORIGINAL] + counts[PREFAULT]
        for origin in (ORIGINAL, PREFAULT):
            weights[torch.from_numpy(origins == origin)] = source_normal / (2 * counts[origin])
    return weights


def probability_table(weights, labels, origins):
    labels, origins = np.asarray(labels), np.asarray(origins)
    total = float(weights.sum())
    return {"total_weight": total,
            "normal_probability": float(weights[torch.from_numpy(labels == 0)].sum()) / total,
            "fault_probability": float(weights[torch.from_numpy(labels == 1)].sum()) / total,
            "origins": {str(origin): {"windows": int(np.sum(origins == origin)),
                        "probability": float(weights[torch.from_numpy(origins == origin)].sum()) / total}
                        for origin in sorted(set(origins))}}


def stratified_metrics(labels, predictions, metadata):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != (len(metadata),) or predictions.shape != labels.shape:
        raise ValueError("validation metadata/prediction count mismatch")
    if not set(np.unique(labels)).issubset({0, 1}) or not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("invalid binary labels/predictions")
    if not np.array_equal(labels, metadata.label.to_numpy()):
        raise ValueError("validation labels differ from linked metadata")

    def metrics(indices):
        cm = np.bincount(labels[indices].astype(int) * 2 + predictions[indices], minlength=4).reshape(2, 2)
        return classification(cm)

    frame = metadata.reset_index(drop=True)
    normal = frame[frame.label == 0]
    pure = frame[frame.normal_origin == ORIGINAL]
    fault = frame[frame.label == 1]
    return {"overall": metrics(np.arange(len(frame))),
            "normal_origin": {str(key): metrics(part.index.to_numpy()) for key, part in frame.groupby("normal_origin", sort=True)},
            "normal_by_status": {str(key): metrics(part.index.to_numpy()) for key, part in normal.groupby("flight_status", sort=True)},
            "original_normal_by_status": {str(key): metrics(part.index.to_numpy()) for key, part in pure.groupby("flight_status", sort=True)},
            "fault_by_case_class": {str(key): metrics(part.index.to_numpy()) for key, part in fault.groupby("case_level2_label", sort=True)}}


def save_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def predict(model, loader, mean, std):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for raw, y, unused in loader:
            batch = (raw - mean) / std
            probs = model(batch).softmax(1)
            if not torch.isfinite(probs).all():
                raise FloatingPointError("non-finite validation probabilities")
            probabilities.append(probs.numpy())
            labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING_CPU_SMOKE", "performance_claim": False,
              "started_at": datetime.now(timezone.utc).isoformat(), "arms": {}}
    started = time.perf_counter()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if (protocol["protocol_name"], protocol["status"]) != ("normal_source_sampling_cpu_smoke_v1", "FIXED_PIPELINE_CHECK_BEFORE_EXECUTION"):
            raise ValueError("unexpected smoke protocol")
        cfg = protocol["training"]
        if (cfg["device"], cfg["epochs"], cfg["steps_per_epoch"], cfg["batch_size"], protocol["validation"]["batch_size"]) != ("cpu", 2, 4, 32, 128):
            raise ValueError("this entry is restricted to the fixed CPU smoke workload")
        if protocol["sampling"]["arms"] != ["uniform_replacement", "balanced_source_normal"] or not protocol["sampling"]["replacement"]:
            raise ValueError("unexpected sampler comparison")
        if (cfg["model"] != "kg_mstca" or cfg["augmentation"] or cfg["optimizer"] != "AdamW"
            or cfg["loss"] != "FocalLoss gamma=2, alpha=None" or protocol["sampling"]["distribution_probe_draws"] != 100000
            or protocol["inputs"]["cache_files"] != ["train.pt", "val.pt"] or protocol["inputs"]["real_reads"]):
            raise ValueError("unsupported smoke recipe or data scope")
        audit_path = args.audit / "audit.json"
        require_hash(audit_path, protocol["inputs"]["audit_sha256"])
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "COMPLETED_SOURCE_DATA_AUDIT":
            raise ValueError("source linkage audit incomplete")
        inputs, frames, payloads = {}, {}, {}
        for split in ("train", "val"):
            links = args.audit / f"{split}_source_window_links.csv"
            require_hash(links, protocol["inputs"][f"{split}_links_sha256"])
            cache = args.cache / f"{split}.pt"
            require_hash(cache, audit["splits"][split]["cache_sha256"])
            payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
            frame = pd.read_csv(links)
            if not np.array_equal(frame.cache_index, np.arange(len(frame))) or frame.case_uid.isna().any():
                raise ValueError("invalid linked cache order")
            if payload["x"].dtype != torch.float32 or tuple(payload["x"].shape[1:]) != (26, 150) or len(payload["x"]) != len(payload["y"]):
                raise ValueError("invalid source cache contract")
            if not np.array_equal(frame.label, payload["y"][:len(frame)].numpy()):
                raise ValueError("source cache labels do not match linked prefix")
            if not frame.domain.isin(["SIL", "HIL"]).all() or not frame.split.eq(split).all():
                raise ValueError("non-source metadata or incorrect split")
            inputs[cache.name] = sha256(cache)
            frames[split], payloads[split] = frame, payload
        if set(frames["train"].case_uid) & set(frames["val"].case_uid):
            raise ValueError("train/val case leakage")
        norm_path = args.cache / "norm_stats.json"
        require_hash(norm_path, protocol["inputs"]["norm_sha256"])
        stats = json.loads(norm_path.read_text())
        if stats["columns"] != RFLY_SEQ_COLUMNS_V2 or stats["window_size"] != 150:
            raise ValueError("source normalization schema mismatch")
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid source normalization")
        train_data, val_data = payloads["train"], payloads["val"]
        origins = frames["train"].normal_origin.to_numpy(dtype=str)
        extra = len(train_data["y"]) - len(origins)
        if extra != 3680 or torch.count_nonzero(train_data["y"][len(origins):]):
            raise ValueError("appended RflySim normal block mismatch")
        origins = np.r_[origins, np.full(extra, EXTRA)]
        labels = train_data["y"].numpy()
        selected = []
        for _, part in frames["val"].groupby("case_uid", sort=False):
            if len(part) != 64:
                raise ValueError("expected cap64 source validation cases")
            selected.append(part.iloc[np.linspace(0, 63, 4).round().astype(int)])
        val_meta = pd.concat(selected, ignore_index=True)
        if len(val_meta) != 2220 or val_meta.case_uid.nunique() != 555:
            raise ValueError("unexpected smoke validation subset")
        val_indices = torch.tensor(val_meta.cache_index.to_numpy(), dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_data["x"][val_indices], val_data["y"][val_indices], val_indices), batch_size=128)
        train_dataset = TensorDataset(train_data["x"], train_data["y"], torch.arange(len(labels)))
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["seed"])
        model_kwargs = {"in_channels": 26, "num_classes": 2, "hidden_size": cfg["hidden_size"],
                        "num_layers": cfg["num_layers"], "dropout": cfg["dropout"], "columns": stats["columns"]}
        template = OnlineKGMSTCA(**model_kwargs)
        initial = {key: value.detach().clone() for key, value in template.state_dict().items()}
        save_state(out / "shared_initial.pt", initial)
        for path in (args.protocol, Path(__file__), Path(inspect.getfile(OnlineKGMSTCA)), Path(inspect.getfile(FocalLoss)), norm_path):
            shutil.copy2(path, out / path.name)
        protocol_sha = sha256(out / args.protocol.name)
        report.update({"protocol_sha256_before_training": protocol_sha, "runner_sha256": sha256(__file__),
                       "model_sha256": sha256(Path(inspect.getfile(OnlineKGMSTCA))), "loss_sha256": sha256(Path(inspect.getfile(FocalLoss))),
                       "input_cache_sha256": inputs, "norm_sha256": sha256(norm_path), "audit_sha256": sha256(audit_path),
                       "metadata_sha256": {split: sha256(args.audit / f"{split}_source_window_links.csv") for split in ("train", "val")},
                       "shared_initial_sha256": sha256(out / "shared_initial.pt"), "initialization": "shared random state, not historical trained weights",
                       "device": "cpu", "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                       "parameters": sum(p.numel() for p in template.parameters()), "validation_cases": 555, "validation_windows": 2220,
                       "forbidden_dataset_reads": False, "sampler_probability_invariants": {}, "tiny_workload_only": True})
        write_json(out / "report.json", report)
        val_meta.to_csv(out / "smoke_val_links.csv", index=False)
        tables = {}
        for arm in protocol["sampling"]["arms"]:
            arm_dir = out / arm
            arm_dir.mkdir()
            balanced = arm == "balanced_source_normal"
            weights = sampling_weights(labels, origins, balanced)
            table = probability_table(weights, labels, origins)
            tables[arm] = table
            probe = list(WeightedRandomSampler(weights, num_samples=protocol["sampling"]["distribution_probe_draws"], replacement=True,
                                              generator=torch.Generator().manual_seed(cfg["seed"])))
            probe = np.asarray(probe)
            draw_report = {"windows": len(probe), "normal_fraction": float(np.mean(labels[probe] == 0)),
                           "origin_counts": dict(Counter(origins[probe]))}
            # Separate sampler RNG from model/Dropout RNG. Reset both per arm.
            sampler = WeightedRandomSampler(weights, num_samples=cfg["steps_per_epoch"] * cfg["batch_size"], replacement=True,
                                             generator=torch.Generator().manual_seed(protocol["sampling"]["generator_seed"]))
            loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
            model = OnlineKGMSTCA(**model_kwargs)
            model.load_state_dict(initial, strict=True)
            if any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.items()):
                raise RuntimeError("paired initialization differs")
            torch.manual_seed(cfg["seed"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            criterion = FocalLoss(gamma=2.0)
            history, best_score, best_epoch = [], -float("inf"), None
            actual_counts = Counter()
            print(f"{arm}: probability and two-epoch training smoke", flush=True)
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                losses = []
                for raw, y, indices in loader:
                    batch = (raw - mean) / std
                    if not torch.isfinite(batch).all():
                        raise FloatingPointError("non-finite training input")
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(batch), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("non-finite gradient")
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite gradient norm")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    actual_counts.update(origins[indices.numpy()])
                probs, actual = predict(model, val_loader, mean, std)
                metrics = stratified_metrics(actual, probs.argmax(1), val_meta)
                entry = {"epoch": epoch, "steps": len(losses), "train_loss": float(np.mean(losses)), "sampled_validation": metrics}
                history.append(entry)
                score = metrics["overall"]["macro_f1"]
                if score > best_score:
                    save_state(arm_dir / "best.pt", model.state_dict())
                    best_score, best_epoch = score, epoch
                save_state(arm_dir / "last.pt", model.state_dict())
                write_json(arm_dir / "history.json", history)
                write_json(arm_dir / "checkpoint_metadata.json", {"best_epoch": best_epoch, "epochs_completed": epoch, "best_sampled_val_macro_f1": best_score, "smoke_only": True})
            changed = any(not torch.equal(initial[name], parameter.detach()) for name, parameter in model.named_parameters())
            if not changed or len(history) != 2 or sum(row["steps"] for row in history) != 8:
                raise RuntimeError("training smoke did not perform the expected updates")
            model.load_state_dict(torch.load(arm_dir / "best.pt", map_location="cpu", weights_only=True), strict=True)
            probs, actual = predict(model, val_loader, mean, std)
            reloaded = stratified_metrics(actual, probs.argmax(1), val_meta)
            if reloaded != history[best_epoch - 1]["sampled_validation"]:
                raise RuntimeError("reloaded checkpoint does not reproduce selected validation")
            np.savez_compressed(arm_dir / "validation_predictions.npz", probabilities=probs, labels=actual, cache_indices=val_meta.cache_index.to_numpy())
            report["arms"][arm] = {"expected_sampling": table, "distribution_probe": draw_report,
                "actual_training_origin_counts": dict(actual_counts), "training_windows": sum(actual_counts.values()),
                "gradient_steps": 8, "finite_losses_and_gradients": True, "trainable_parameters_changed": changed,
                "shared_initial_state_identical": True, "best_epoch": best_epoch, "checkpoint_reload_matches": True,
                "best_checkpoint_sha256": sha256(arm_dir / "best.pt"), "sampled_validation": reloaded}
            write_json(out / "report.json", report)
        baseline, balanced = tables["uniform_replacement"], tables["balanced_source_normal"]
        for key in ("total_weight", "normal_probability", "fault_probability"):
            if not np.isclose(baseline[key], balanced[key], rtol=0, atol=1e-9):
                raise RuntimeError("sampler class/total probability changed")
        if not np.isclose(baseline["origins"][EXTRA]["probability"], balanced["origins"][EXTRA]["probability"], rtol=0, atol=1e-12):
            raise RuntimeError("RflySim probability changed")
        if not np.isclose(balanced["origins"][ORIGINAL]["probability"], balanced["origins"][PREFAULT]["probability"], rtol=0, atol=1e-12):
            raise RuntimeError("source normal origins are not balanced")
        require_hash(out / args.protocol.name, protocol_sha)
        report["sampler_probability_invariants"] = {"class_probabilities_preserved": True, "rflysim_probability_preserved": True, "source_normal_origins_equal": True}
        report["status"] = "PASSED_CPU_PIPELINE_SMOKE_ONLY"
        report["elapsed_s"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "report.json", report)
        print(f"passed CPU pipeline smoke: {out / 'report.json'}", flush=True)
    except Exception as error:
        report["status"] = "FAILED_CPU_SMOKE"
        report["error"] = repr(error)
        write_json(out / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
