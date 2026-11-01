"""python -m pytest -q tests/test_normal_source_sampling_smoke.py"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("normal_source_smoke", ROOT / "scripts/research/smoke_normal_source_sampling.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def population():
    origins = np.array([runner.FAULT] * 8 + [runner.PREFAULT] * 6 + [runner.ORIGINAL] * 2 + [runner.EXTRA] * 4)
    labels = (origins == runner.FAULT).astype(int)
    return labels, origins


def test_balancing_changes_only_normal_origin_mass_not_classes_or_extra():
    labels, origins = population()
    uniform = runner.sampling_weights(labels, origins, False)
    balanced = runner.sampling_weights(labels, origins, True)
    assert torch.equal(uniform, torch.ones(20, dtype=torch.float64))
    first = runner.probability_table(uniform, labels, origins)
    second = runner.probability_table(balanced, labels, origins)
    for key in ("total_weight", "normal_probability", "fault_probability"):
        assert first[key] == pytest.approx(second[key])
    assert first["origins"][runner.EXTRA]["probability"] == second["origins"][runner.EXTRA]["probability"]
    assert second["origins"][runner.ORIGINAL]["probability"] == pytest.approx(0.2)
    assert second["origins"][runner.PREFAULT]["probability"] == pytest.approx(0.2)
    draw = lambda: list(WeightedRandomSampler(balanced, num_samples=100, replacement=True, generator=torch.Generator().manual_seed(42)))
    assert draw() == draw()
    assert all(0 <= index < len(labels) for index in draw())


def test_sampling_rejects_unknown_origin_label_mismatch_and_missing_pool():
    labels, origins = population()
    bad = origins.copy()
    bad[0] = runner.ORIGINAL
    with pytest.raises(ValueError, match="mismatch"):
        runner.sampling_weights(labels, bad, True)
    bad[0] = "Real_test"
    with pytest.raises(ValueError, match="unknown"):
        runner.sampling_weights(labels, bad, True)
    select = origins != runner.ORIGINAL
    with pytest.raises(ValueError, match="both normal"):
        runner.sampling_weights(labels[select], origins[select], True)


def test_stratification_separates_normal_origins_and_handles_absent_classes():
    metadata = pd.DataFrame({"label": [0, 0, 0, 1, 1],
        "normal_origin": [runner.ORIGINAL, runner.ORIGINAL, runner.PREFAULT, runner.FAULT, runner.FAULT],
        "flight_status": ["hover", "hover", "hover", "hover", "velocity"],
        "case_level2_label": ["normal", "normal", "motor", "motor", "sensor"]})
    labels = metadata.label.to_numpy()
    result = runner.stratified_metrics(labels, np.array([1, 0, 0, 1, 0]), metadata)
    assert result["overall"]["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["normal_origin"][runner.PREFAULT]["normal_window_fpr"] == 0
    assert result["original_normal_by_status"]["hover"]["normal_window_fpr"] == 0.5
    assert result["original_normal_by_status"]["hover"]["fault_recall"] is None
    assert result["fault_by_case_class"]["sensor"]["normal_window_fpr"] is None
    with pytest.raises(ValueError, match="differ"):
        runner.stratified_metrics(1 - labels, labels, metadata)


def test_checkpoint_failure_keeps_previous_weights(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    runner.save_state(path, {"weight": torch.ones(2)})
    original = path.read_bytes()

    def fail(payload, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk"):
        runner.save_state(path, {"weight": torch.zeros(2)})
    assert path.read_bytes() == original


def test_existing_output_is_rejected_without_touching_artifacts(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"old experiment")
    args = argparse.Namespace(output=tmp_path)
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert (tmp_path / "best.pt").read_bytes() == b"old experiment"
    assert not (tmp_path / "report.json").exists()
