"""Prepare the frozen 10-case adaptation input, then run A0/A1 source validation.

This stage deliberately does NOT read Real calibration/test data or compute event
metrics. The user confirmed full chronological replay for that separate stage.
Outputs must be new directories. No source cache/model is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

PROTOCOL_SHA = "27982a23273175579849e49d7e5f5986a346951d9d36db46279ed7be7db9c85c"
MODEL_SHA = "b24f2da7b18197362f7d903eca813f78b70f6125cd249044ac530d5106ae5837"
MODEL_SOURCE_SHA = "87a928b147c1e62d283910048d19710d55dd9731b963a33d8a5840b430fae1d2"
NORM_SHA = "a474fc77c2da1cf7e02b765446919f0d27f5ad64869ef882a678867a2108bf33"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def load_roles(protocol_path, roles_path):
    require_hash(protocol_path, PROTOCOL_SHA)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    require_hash(roles_path, protocol["role_manifest_sha256"])
    with Path(roles_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["case_uid"] for r in rows}) != 465:
        raise ValueError("expected 465 unique Real cases")
    if dict(Counter(r["role"] for r in rows)) != protocol["role_counts"]:
        raise ValueError("role counts differ from frozen protocol")
    groups = defaultdict(set)
    for row in rows:
        if row["domain"] != "REAL":
            raise ValueError("non-Real role member")
        role = row["role"]
        groups[row["association_group"]].add(role if role in ("adapt_normal", "calibrate_normal") else "test")
    if any(len(roles) != 1 for roles in groups.values()):
        raise ValueError("association group crosses development/test roles")
    adapt = sorted((r for r in rows if r["role"] == "adapt_normal"), key=lambda r: r["case_uid"])
    if any(r["level2_label"] != "normal" or r["source_disposition"] != "retain_normal" for r in adapt):
        raise ValueError("adaptation is restricted to original normal cases")
    return protocol, adapt


def validate_bundle(bundle, rows, columns):
    x, samples = bundle["x"], bundle["samples"]
    if x.dtype != torch.float32 or x.ndim != 3 or tuple(x.shape[1:]) != (26, 150):
        raise ValueError("expected float32 [windows,26,150]")
    if len(x) != len(samples) or not torch.isfinite(x).all():
        raise ValueError("invalid adaptation samples or non-finite features")
    if bundle["columns"] != columns or bundle["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("bundle contract/protocol mismatch")
    by_uid = {row["case_uid"]: row for row in rows}
    counts = Counter(s["case_uid"] for s in samples)
    if set(counts) != set(by_uid) or any(not 1 <= n <= 64 for n in counts.values()):
        raise ValueError("bundle does not contain exactly the 10 allowed cases with cap 64")
    order = [(s["case_uid"], s["end_row"]) for s in samples]
    if order != sorted(set(order)):
        raise ValueError("adaptation windows must be unique and chronologically ordered within case")
    for sample in samples:
        row = by_uid[sample["case_uid"]]
        start, end = sample["start_row"], sample["end_row"]
        if not (0 <= start < end <= int(row["row_count"])) or end - start != 150 or start % 5:
            raise ValueError("invalid manifest window span")
        if sample["source_csv_sha256"] != row["output_sha256"]:
            raise ValueError("bundle source CSV identity mismatch")
    return counts


def prepare(args):
    import pandas as pd
    # Reuse the same contract extraction as the audited cache; reject bad values
    # BEFORE its legacy nan_to_num can conceal them.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from rfly_feature_contract import contract_columns, read_case_features, source_columns_for_domain, window_batch

    protocol, rows = load_roles(args.protocol, args.roles)
    require_hash(args.norm, NORM_SHA)
    stats = json.loads(Path(args.norm).read_text(encoding="utf-8-sig"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8-sig"))
    if contract_columns(contract) != stats["columns"]:
        raise ValueError("feature contract differs from source normalization")
    selected = {r["case_uid"] for r in rows}
    spans = defaultdict(list)
    usecols = ["case_uid", "window_start_row", "window_end_row", "split", "level2_label"]
    for chunk in pd.read_csv(args.window_manifest, usecols=usecols, chunksize=200000):
        for rec in chunk[chunk.case_uid.isin(selected)].to_dict(orient="records"):
            if rec["split"] != "external_test" or rec["level2_label"] != "normal":
                raise ValueError("unexpected adaptation manifest role/label")
            spans[rec["case_uid"]].append((int(rec["window_start_row"]), int(rec["window_end_row"])))
    batches, samples = [], []
    for row in rows:
        uid, path = row["case_uid"], Path(row["final_output_path"])
        require_hash(path, row["output_sha256"])
        mapping = source_columns_for_domain(contract, "REAL")
        raw = pd.read_csv(path, usecols=sorted(set(mapping.values())), low_memory=False)
        values = raw.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if len(raw) != int(row["row_count"]) or not np.isfinite(values).all():
            raise ValueError("invalid source CSV: " + uid)
        matrix, columns = read_case_features(path, "REAL", contract)
        all_spans = sorted(spans[uid])
        expected = [(s, s + 150) for s in range(0, len(matrix) - 150 + 1, 5)]
        if all_spans != expected or not all_spans:
            raise ValueError("manifest does not cover the expected 5-frame grid: " + uid)
        picks = np.unique(np.linspace(0, len(all_spans) - 1, 64).round().astype(int)) if len(all_spans) > 64 else np.arange(len(all_spans))
        chosen = [all_spans[int(i)] for i in picks]
        batches.append(window_batch(matrix, columns, chosen))
        samples.extend({"case_uid": uid, "start_row": s, "end_row": e, "source_csv_sha256": row["output_sha256"]} for s, e in chosen)
        print(f"prepared {uid}: {len(chosen)} windows", flush=True)
    bundle = {"x": torch.from_numpy(np.concatenate(batches)), "samples": samples,
              "columns": stats["columns"], "protocol_sha256": PROTOCOL_SHA,
              "contract_sha256": sha256(args.contract), "window_manifest_sha256": sha256(args.window_manifest),
              "extractor_sha256": sha256(Path(__file__).resolve().parents[1] / "data/rfly_feature_contract.py")}
    validate_bundle(bundle, rows, stats["columns"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(bundle, out / "adapt_normal.pt")
    for source in (args.protocol, args.roles, args.contract, args.norm):
        shutil.copy2(source, out / Path(source).name)
    write_json(out / "bundle_identity.json", {"adapt_normal.pt": sha256(out / "adapt_normal.pt"),
               "cases": len(rows), "windows": len(samples), "test_or_calibration_csvs_read": 0})


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def recalibrate(model, x, device):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bns = {name: layer for name, layer in model.named_modules() if isinstance(layer, torch.nn.BatchNorm1d)}
    if not bns or any(not layer.track_running_stats for layer in bns.values()):
        raise ValueError("A1 needs tracked BatchNorm1d statistics")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for layer in bns.values():
        layer.reset_running_stats()
        layer.train()
    with torch.no_grad():
        for start in range(0, len(x), 64):
            model(x[start:start + 64].to(device))
    model.eval()
    allowed = {f"{name}.{field}" for name in bns for field in ("running_mean", "running_var", "num_batches_tracked")}
    after = model.state_dict()
    for name, original in before.items():
        if name not in allowed and not torch.equal(original, after[name].cpu()):
            raise RuntimeError("A1 changed a non-BatchNorm-statistics tensor: " + name)
    if any(not torch.isfinite(value).all() for value in after.values()):
        raise RuntimeError("non-finite adapted state")
    return {name: {"momentum": layer.momentum, "eps": layer.eps,
                   "num_batches_tracked": int(layer.num_batches_tracked),
                   "running_mean": layer.running_mean.cpu().tolist(),
                   "running_var": layer.running_var.cpu().tolist()} for name, layer in bns.items()}


def validate_source(model, cache, mean, std, device, output):
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    x, y = payload["x"], payload["y"]
    if x.ndim != 3 or tuple(x.shape[1:]) != (26, 150) or y.shape != (len(x),):
        raise ValueError("source validation shape mismatch")
    if not torch.isfinite(x).all() or not set(y.tolist()).issubset({0, 1}):
        raise ValueError("invalid source validation features/labels")
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = (x[start:start + 512].float() - mean) / std
            probabilities.append(model(batch.to(device)).softmax(1).cpu())
    probs = torch.cat(probabilities)
    preds = probs.argmax(1)
    cm = torch.bincount(y.long() * 2 + preds, minlength=4).reshape(2, 2).numpy()
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(2), where=denom != 0)
    np.savez_compressed(output, probabilities=probs.numpy(), labels=y.numpy())
    return {"windows": len(y), "confusion_matrix": cm.tolist(),
            "accuracy": float(cm.trace() / cm.sum()), "macro_f1": float(f1.mean()),
            "fault_recall": float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else None,
            "normal_window_fpr": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None}


def run(args):
    protocol, rows = load_roles(args.protocol, args.roles)
    source = Path(args.source_model)
    require_hash(source / "best.pt", MODEL_SHA)
    require_hash(source / "norm_stats.json", NORM_SHA)
    require_hash(args.model_source, MODEL_SOURCE_SHA)
    expected_val = next(r["local"]["sha256"] for r in protocol["cache_identity"]["files"] if r["file"] == "val.pt")
    val_path = Path(args.cache) / "val.pt"
    require_hash(val_path, expected_val)
    require_hash(Path(args.bundle) / "adapt_normal.pt", json.loads((Path(args.bundle) / "bundle_identity.json").read_text())["adapt_normal.pt"])
    stats = json.loads((source / "norm_stats.json").read_text())
    config = json.loads((source / "config.json").read_text())
    if (config["model"], config["feature_schema"], config["window_size"], config["in_channels"], config["num_classes"]) != ("kg_mstca", "rfly_seq_columns_v2", 150, 26, 2):
        raise ValueError("unexpected source model contract")
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 26, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 26, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid source normalization")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # The source checkpoint's saved validation matrix reproduces with the
    # historical PyTorch default (cuDNN TF32 on), not with TF32 disabled.
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for src in (args.protocol, args.roles, args.model_source, source / "best.pt", source / "config.json", source / "norm_stats.json", __file__):
        shutil.copy2(src, out / Path(src).name)
    report = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "stage": "adaptation_and_source_validation_only", "protocol_sha256": PROTOCOL_SHA,
              "source_checkpoint_sha256": MODEL_SHA, "source_model_code_sha256": MODEL_SOURCE_SHA,
              "runner_sha256": sha256(__file__), "source_norm_sha256": NORM_SHA,
              "source_validation_sha256": expected_val,
              "bundle_sha256": sha256(Path(args.bundle) / "adapt_normal.pt"),
              "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "seed": 42, "gradients": False, "batch_size": 64,
              "deterministic_algorithms": True, "cudnn_tf32": True, "matmul_tf32": False,
              "calibration_or_final_test_predictions": False,
              "next_stage": "Full chronological calibration and event replay, as confirmed by user."}
    write_json(out / "report.json", report)
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("normal_adapt_source_model", out / Path(args.model_source).name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if stats["columns"] != module.RFLY_SEQ_COLUMNS_V2:
            raise ValueError("source model columns differ from normalization")
        model = module.OnlineKGMSTCA(in_channels=26, num_classes=2, hidden_size=config["hidden_size"],
                                    num_layers=config["num_layers"], dropout=config["dropout"], columns=stats["columns"]).to(device)
        model.load_state_dict(torch.load(source / "best.pt", map_location=device, weights_only=True), strict=True)
        print("A0: reproducing source validation", flush=True)
        report["A0_source_val"] = validate_source(model, val_path, mean, std, device, out / "A0_val_predictions.npz")
        historical = json.loads((source / "metrics_val.json").read_text())
        if report["A0_source_val"]["confusion_matrix"] != historical["confusion_matrix"]:
            raise RuntimeError("A0 source validation does not reproduce historical confusion matrix")
        bundle = torch.load(Path(args.bundle) / "adapt_normal.pt", map_location="cpu", weights_only=True)
        counts = validate_bundle(bundle, rows, stats["columns"])
        report["adaptation_cases"] = dict(counts)
        report["adaptation_samples"] = bundle["samples"]
        report["input_provenance"] = {key: value for key, value in bundle.items() if key not in ("x", "samples")}
        normalized = (bundle["x"] - mean) / std
        if not torch.isfinite(normalized).all():
            raise ValueError("non-finite normalized adaptation input")
        print(f"A1: BatchNorm recalibration on {len(normalized)} normal windows, {len(counts)} cases", flush=True)
        report["batchnorm"] = recalibrate(model, normalized, device)
        report["non_bn_statistics_tensors_unchanged"] = True
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, out / "A1_adapted.pt.tmp")
        (out / "A1_adapted.pt.tmp").replace(out / "A1_adapted.pt")
        report["adapted_checkpoint_sha256"] = sha256(out / "A1_adapted.pt")
        write_json(out / "report.json", report)
        print("A1: source validation (diagnostic only, not adaptation selection)", flush=True)
        report["A1_source_val"] = validate_source(model, val_path, mean, std, device, out / "A1_val_predictions.npz")
        report["status"] = "COMPLETED"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out / "report.json", report)
        print(json.dumps({key: report[key] for key in ("status", "A0_source_val", "A1_source_val", "elapsed_seconds")}, indent=2), flush=True)
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
    prep.add_argument("--window-manifest", type=Path, required=True)
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--source-model", type=Path, required=True)
    execute.add_argument("--model-source", type=Path, required=True)
    execute.add_argument("--cache", type=Path, required=True)
    execute.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
