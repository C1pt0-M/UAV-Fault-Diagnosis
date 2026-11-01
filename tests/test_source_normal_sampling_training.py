"""Local checks for frozen GPU training; no GPU or real dataset needed here."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("source_sampling_formal", ROOT / "scripts/research/train_source_normal_sampling.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PROTOCOL = ROOT / "configs/experiments/normal_source_sampling_train_v1.json"


def test_frozen_protocol_and_source_code_pins(tmp_path):
    protocol = runner.load_protocol(PROTOCOL)
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["training"]["max_epochs"] == 100
    assert protocol["training"]["batch_size"] == 512
    assert protocol["validation"]["early_stop"]["min_epochs"] == 30
    assert len(runner.code_identities(protocol)) == 5
    copy = tmp_path / "modified.json"
    copy.write_bytes(PROTOCOL.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.load_protocol(copy)


def test_true_best_improvement_is_independent_from_early_stop_delta():
    rule = {"min_epochs": 3, "patience": 2, "significant_delta": .002}
    state = {"best_score": -float("inf"), "best_epoch": None, "significant_reference": -float("inf"), "bad_epochs": 0}
    state, best, stop = runner.selection_step(state, .5, 1, rule)
    assert best and not stop
    state, best, stop = runner.selection_step(state, .501, 2, rule)
    assert best and state["best_epoch"] == 2 and state["significant_reference"] == .5
    assert state["bad_epochs"] == 0 and not stop
    state, best, stop = runner.selection_step(state, .5015, 3, rule)
    assert best and state["best_epoch"] == 3 and state["bad_epochs"] == 1 and not stop
    state, best, stop = runner.selection_step(state, .5015, 4, rule)
    assert not best and state["best_epoch"] == 3 and stop  # earliest exact tie
    state, best, stop = runner.selection_step(state, .504, 5, rule)
    assert best and state["bad_epochs"] == 0 and not stop
    with pytest.raises(FloatingPointError):
        runner.selection_step(state, float("nan"), 6, rule)


def test_warmup_and_cosine_use_fixed_formal_schedule():
    p = runner.load_protocol(PROTOCOL)["training"]
    values = [runner.get_warmup_cosine_lr(i, p["warmup_epochs"], p["max_epochs"], p["lr"], p["min_lr"]) for i in range(100)]
    assert values[0] == pytest.approx(6e-5)
    assert values[4] == values[5] == pytest.approx(3e-4)
    assert all(p["min_lr"] <= value <= p["lr"] for value in values)
    assert values[-1] < values[50] < values[5]


def test_initial_digest_covers_values_dtype_shape_and_key_order():
    state = {"z": torch.tensor([1, 2]), "a": torch.ones(2)}
    value = runner.state_digest(state)
    assert value == runner.state_digest(dict(reversed(list(state.items()))))
    assert value != runner.state_digest({"z": state["z"], "a": torch.zeros(2)})
    assert value != runner.state_digest({"z": state["z"], "a": state["a"].double()})
    assert value != runner.state_digest({"z": state["z"], "a": state["a"].view(1, 2)})


def test_formal_mode_refuses_cpu_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = argparse.Namespace(output=tmp_path / "job", seed=42, arm="uniform_replacement", cpu_check=False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        runner.run_job(args, runner.load_protocol(PROTOCOL))
    assert not args.output.exists()


def test_existing_job_and_summary_are_never_overwritten(tmp_path):
    marker = tmp_path / "best.pt"
    marker.write_bytes(b"prior training")
    with pytest.raises(FileExistsError):
        runner.run_job(argparse.Namespace(output=tmp_path), {})
    assert marker.read_bytes() == b"prior training"
    report = tmp_path / "summary.json"
    report.write_text("old")
    with pytest.raises(FileExistsError):
        runner.new_json(report, {"new": True})
    assert report.read_text() == "old"


def fake_formal_jobs(root):
    protocol = runner.load_protocol(PROTOCOL)
    for i, seed in enumerate(protocol["seeds"]):
        for arm in protocol["arms"]:
            balanced = arm == "balanced_source_normal"
            metrics = {"overall": {"macro_f1": .9 + .01 * i + (.01 if balanced else 0), "fault_recall": .95,
                                    "normal_window_fpr": .04},
                       "normal_origin": {runner.ORIGINAL: {"normal_window_fpr": .05 if balanced else .1},
                                         runner.PREFAULT: {"normal_window_fpr": .01}},
                       "original_normal_by_status": {"hover": {"normal_window_fpr": .2 if balanced else .3}}}
            report = {"status": "COMPLETED_FORMAL_SOURCE_TRAINING", "cpu_check": False, "protocol_sha256": runner.PROTOCOL_SHA,
                      "seed": seed, "arm": arm, "formal_validation_complete": True, "validation_windows": 35520,
                      "checkpoint_reload_matches": True, "source_code": protocol["source_code"], "inputs": protocol["inputs"],
                      "environment": dict(protocol["formal_environment"], cuda_available=True), "initial_state_digest": str(i) * 64,
                      "runner_sha256": "a" * 64, "code_bundle_manifest_sha256": "b" * 64, "source_validation": metrics}
            path = root / f"seed_{seed}" / arm / "report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report))
    return protocol


def test_summary_uses_all_three_pairs_and_sample_std(tmp_path):
    protocol = fake_formal_jobs(tmp_path)
    result = runner.summarize(tmp_path, protocol)
    assert result["jobs"] == 6 and result["seeds"] == [42, 43, 44]
    metric = result["metrics"]["macro_f1"]
    assert metric["uniform_replacement"]["mean"] == pytest.approx(.91)
    assert metric["uniform_replacement"]["sample_std"] == pytest.approx(.01)
    assert metric["paired_balanced_minus_uniform"]["per_seed"] == pytest.approx([.01] * 3)
    assert result["metrics"]["original_normal_fpr"]["paired_balanced_minus_uniform"]["mean"] == pytest.approx(-.05)
    assert not result["final_test_or_deployment_claim"]


@pytest.mark.parametrize("failure", ["cpu", "initial", "bundle", "subset"])
def test_summary_rejects_invalid_pairs(tmp_path, failure):
    protocol = fake_formal_jobs(tmp_path)
    path = tmp_path / "seed_42/balanced_source_normal/report.json"
    report = json.loads(path.read_text())
    if failure == "cpu":
        report["status"] = "COMPLETED_CPU_CHECK_NOT_FORMAL"
    elif failure == "initial":
        report["initial_state_digest"] = "f" * 64
    elif failure == "bundle":
        report["code_bundle_manifest_sha256"] = "c" * 64
    else:
        report["validation_windows"] = 2220
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runner.summarize(tmp_path, protocol)
