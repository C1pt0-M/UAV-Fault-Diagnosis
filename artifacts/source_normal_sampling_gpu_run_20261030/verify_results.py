"""Read-only result check; optional GPU replay reads only val.pt, never final tests.

Example (from the repository root):
python artifacts/source_normal_sampling_gpu_run_20261030/verify_results.py \
  --runs artifacts/source_normal_sampling_gpu_run_20261030/results/runs/20261030T170404Z_registered_v1 \
  --protocol configs/experiments/normal_source_sampling_train_v1.json \
  --links artifacts/source_training_coverage_audit_20261030_r2/val_source_window_links.csv \
  --output artifacts/source_normal_sampling_gpu_run_20261030/new_verification.json
Add --gpu --bundle FROZEN_BUNDLE --cache SOURCE_CACHE for exact CUDA replay.
Existing output files are refused. No training or model selection is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def classification(labels, predictions):
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    total = tn + fp + fn + tp
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"windows": total, "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tn + tp) / total if total else None, "macro_f1": (f0 + f1) / 2,
            "fault_recall": tp / (tp + fn) if tp + fn else None,
            "normal_window_fpr": fp / (tn + fp) if tn + fp else None}


def equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, float):
        assert math.isfinite(left) and abs(left - right) < 1e-14, (left, right)
    else:
        assert left == right, (left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "protocol", "links", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert sha256(args.protocol) == "088bbb0c1ce2d831321304322dab35491b2fdab1dab9fb05aef2dae85d1fcf23"
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assert sha256(args.links) == protocol["inputs"]["val_links_sha256"]
    frame = pd.read_csv(args.links)
    assert len(frame) == 35520 and frame.case_uid.nunique() == 555
    assert frame.domain.isin(["SIL", "HIL"]).all() and frame.split.eq("val").all()
    labels = frame.label.to_numpy()
    groups = {"overall": {"all": np.ones(len(frame), dtype=bool)}}
    for section, column, eligible in (
        ("normal_origin", "normal_origin", np.ones(len(frame), dtype=bool)),
        ("normal_by_status", "flight_status", labels == 0),
        ("original_normal_by_status", "flight_status", frame.normal_origin.eq("original_normal_case").to_numpy()),
        ("fault_by_case_class", "case_level2_label", labels == 1),
    ):
        groups[section] = {str(key): eligible & frame[column].eq(key).to_numpy()
                           for key in sorted(frame.loc[eligible, column].unique())}
    if args.gpu:
        assert args.bundle and args.cache and torch.cuda.is_available()
        actual = {"torch": str(torch.__version__), "numpy": np.__version__,
                  "pandas": pd.__version__, "torch_cuda": torch.version.cuda}
        assert actual == protocol["formal_environment"], actual
        sys.path.insert(0, str(args.bundle / "packages/uav_fault"))
        from uav_fault.models.cnn.lstm import OnlineKGMSTCA
        assert sha256(Path(inspect.getfile(OnlineKGMSTCA))) == protocol["source_code"]["model_sha256"]
        assert sha256(args.cache / "val.pt") == protocol["inputs"]["val_cache_sha256"]
        payload = torch.load(args.cache / "val.pt", weights_only=True, mmap=True, map_location="cpu")
        assert np.array_equal(payload["y"].numpy(), labels)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    summary = json.loads((args.runs / "summary.json").read_text())
    assert summary["status"] == "COMPLETED_PAIRED_SOURCE_VALIDATION_SUMMARY" and summary["jobs"] == 6
    assert summary["protocol_sha256"] == sha256(args.protocol)
    assert set(summary["metrics"]) == {"macro_f1", "fault_recall", "normal_fpr", "original_normal_fpr", "prefault_normal_fpr", "original_hover_fpr"}
    jobs, states = {}, {}
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            directory = args.runs / f"seed_{seed}" / arm
            report = json.loads((directory / "report.json").read_text())
            config = json.loads((directory / "config.json").read_text())
            history = json.loads((directory / "history.json").read_text())
            assert report["status"] == "COMPLETED_FORMAL_SOURCE_TRAINING" and not report["cpu_check"]
            assert report["inputs"] == protocol["inputs"] and report["source_code"] == protocol["source_code"]
            assert report["seed"] == seed and report["arm"] == arm and report["checkpoint_reload_matches"]
            assert report["protocol_sha256"] == sha256(args.protocol)
            assert report["epochs_trained"] == config["epochs_trained"] == len(history)
            assert report["formal_validation_complete"] and report["validation_windows"] == 35520
            assert all(report["environment"][key] == value for key, value in protocol["formal_environment"].items())
            scores = [row["source_validation"]["overall"]["macro_f1"] for row in history]
            best_epoch = int(np.argmax(scores)) + 1
            assert best_epoch == report["best_epoch"] == config["best_epoch"]
            assert config["best_val_f1"] == max(scores)
            reference, bad, best_score = -float("inf"), 0, -float("inf")
            for epoch, row in enumerate(history, 1):
                score = scores[epoch - 1]
                assert row["epoch"] == epoch and math.isfinite(row["train_loss"])
                assert row["gradient_steps"] == 330 and sum(row["training_origin_counts"].values()) == 168800
                assert row["source_validation"]["overall"]["windows"] == 35520
                if score > reference + 0.002:
                    reference, bad = score, 0
                elif epoch >= 30:
                    bad += 1
                best_score = max(best_score, score)
                equal(row["selection"], {"best_score": best_score,
                      "best_epoch": int(np.argmax(scores[:epoch])) + 1,
                      "significant_reference": reference, "bad_epochs": bad})
                assert bad < 20 or epoch == len(history)
            assert len(history) == 100 or (len(history) >= 30 and bad == 20)
            initial = torch.load(directory / "initial.pt", weights_only=True, map_location="cpu")
            digest = hashlib.sha256()
            for name, value in sorted(initial.items()):
                tensor = value.contiguous()
                digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
                digest.update(tensor.numpy().tobytes())
            assert digest.hexdigest() == report["initial_state_digest"]
            states[seed, arm] = initial
            assert sha256(directory / "best.pt") == report["best_checkpoint_sha256"]
            best = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
            assert all(not t.is_floating_point() or torch.isfinite(t).all() for t in best.values())
            assert any(not torch.equal(value, initial[name]) for name, value in best.items() if name.endswith(("weight", "bias")))
            saved = np.load(directory / "validation_predictions.npz", allow_pickle=False)
            probs = saved["probabilities"]
            assert probs.shape == (35520, 2) and np.isfinite(probs).all()
            assert ((probs >= 0) & (probs <= 1)).all() and np.allclose(probs.sum(1), 1, rtol=0, atol=2e-7)
            assert np.array_equal(saved["labels"], labels)
            assert np.array_equal(saved["cache_indices"], np.arange(35520))
            predictions = probs.argmax(1)
            metrics = {}
            for section, parts in groups.items():
                computed = {key: classification(labels[mask], predictions[mask]) for key, mask in parts.items()}
                metrics[section] = computed["all"] if section == "overall" else computed
            equal(metrics, report["source_validation"])
            equal(metrics, history[best_epoch - 1]["source_validation"])
            result = {"seed": seed, "arm": arm, "epochs": len(history), "best_epoch": best_epoch,
                      "initial_digest": digest.hexdigest(), "metrics": metrics, "best_sha256": report["best_checkpoint_sha256"]}
            if args.gpu:
                norm_file = directory / "norm_stats.json"
                assert sha256(norm_file) == protocol["inputs"]["norm_sha256"]
                norm = json.loads(norm_file.read_text())
                mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, 26, 1).cuda()
                std = torch.tensor(norm["std"], dtype=torch.float32).view(1, 26, 1).cuda()
                model = OnlineKGMSTCA(in_channels=26, hidden_size=64, num_layers=1,
                                     num_classes=2, dropout=0.2, columns=norm["columns"]).float().cuda()
                model.load_state_dict(best, strict=True)
                model.eval()
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(payload["x"]),
                                                     batch_size=512, num_workers=0, pin_memory=True)
                repeated = []
                with torch.no_grad():
                    for (raw,) in loader:
                        repeated.append(model((raw.cuda(non_blocking=True) - mean) / std).softmax(1).cpu().numpy())
                repeated = np.concatenate(repeated)
                result["gpu_probabilities_bitwise_equal"] = bool(np.array_equal(repeated, probs))
                result["gpu_max_abs_probability_difference"] = float(np.max(np.abs(repeated - probs)))
                assert result["gpu_probabilities_bitwise_equal"], result
                del model
            jobs[seed, arm] = result
            print(seed, arm, "verified", len(history), best_epoch, flush=True)
    for seed in protocol["seeds"]:
        left, right = (states[seed, arm] for arm in protocol["arms"])
        assert left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
    recomputed = {}
    for metric in summary["metrics"]:
        by_arm = {}
        for arm in protocol["arms"]:
            points = []
            for seed in protocol["seeds"]:
                metrics = jobs[seed, arm]["metrics"]
                if metric in ("macro_f1", "fault_recall"):
                    value = metrics["overall"][metric]
                elif metric == "normal_fpr":
                    value = metrics["overall"]["normal_window_fpr"]
                elif metric == "original_hover_fpr":
                    value = metrics["original_normal_by_status"]["hover"]["normal_window_fpr"]
                else:
                    origin = "original_normal_case" if metric == "original_normal_fpr" else "normal_part_of_fault_case"
                    value = metrics["normal_origin"][origin]["normal_window_fpr"]
                points.append(value)
            by_arm[arm] = {"per_seed": points, "mean": float(np.mean(points)), "sample_std": float(np.std(points, ddof=1))}
        delta = np.array(by_arm[protocol["arms"][1]]["per_seed"]) - by_arm[protocol["arms"][0]]["per_seed"]
        by_arm["paired_balanced_minus_uniform"] = {"per_seed": delta.tolist(), "mean": float(delta.mean()), "sample_std": float(delta.std(ddof=1))}
        recomputed[metric] = by_arm
    equal(recomputed, summary["metrics"])
    for key in ("runner_sha256", "code_bundle_manifest_sha256"):
        identities = {json.loads((args.runs / f"seed_{seed}" / arm / "report.json").read_text())[key]
                      for seed, arm in jobs}
        assert len(identities) == 1 and all(identities)
    if args.gpu:
        assert sha256(args.cache / "val.pt") == protocol["inputs"]["val_cache_sha256"]
    result = {"status": "PASSED_SIX_JOB_INDEPENDENT_RESULTS_CHECK", "checked_at": datetime.now(timezone.utc).isoformat(),
              "gpu_checkpoint_replay": args.gpu, "jobs": list(jobs.values()), "recomputed_summary_metrics": recomputed,
              "scope": "Source validation only. All saved strata, earliest strict-best selection, stopping, paired initial tensors and summary checked. No final tests, training or deployment.",
              "verifier_sha256": sha256(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"], args.output, flush=True)


if __name__ == "__main__":
    main()
