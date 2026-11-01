"""python -m pytest -q tests/test_source_training_coverage.py"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("source_coverage_audit", ROOT / "scripts/research/audit_source_training_coverage.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_directed_auc_ties_reversal_and_missing_class():
    assert audit.directed_auc([0, 1, 2, 3], [0, 0, 1, 1]) == 1
    assert audit.directed_auc([3, 2, 1, 0], [0, 0, 1, 1]) == 0
    assert audit.directed_auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert audit.directed_auc([0, 1, 1, 2], [0, 0, 1, 1]) == 0.875
    assert audit.directed_auc([1, 2], [0, 0]) is None
    for values, labels in [([float("nan")], [0]), ([1], [2]), ([1, 2], [0])]:
        with pytest.raises(ValueError):
            audit.directed_auc(values, labels)


def test_normal_origin_uses_case_and_endpoint_labels_not_just_phase():
    frame = pd.DataFrame({"level2_label": ["normal", "normal", "motor", "normal"],
                          "case_level2_label": ["normal", "motor", "motor", "sensor"]})
    assert audit.normal_origin(frame).tolist() == ["original_normal_case", "normal_part_of_fault_case", "fault_endpoint", "normal_part_of_fault_case"]


def test_window_means_do_not_mutate_cache_and_reject_invalid_values():
    x = torch.arange(2 * 26 * 150).float().view(2, 26, 150)
    original = x.clone()
    means = audit.window_means(x)
    np.testing.assert_array_equal(means, x.double().mean(2).numpy())
    assert torch.equal(x, original)
    x[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        audit.window_means(x)


def test_hover_followup_separates_log_time_and_near_zero_mean_velocity_without_relabeling():
    frame = pd.DataFrame({"case_uid": ["h1", "h2", "f1", "f1"], "flight_status": ["hover"] * 4,
                          "normal_origin": ["original_normal_case", "original_normal_case", "normal_part_of_fault_case", "fault_endpoint"],
                          "end_from_log_start_s": [5.0, 45.0, 35.0, 45.0]})
    columns = ["vel_z", "baro_alt_relative", "actuator_ctrl_4", "acc_norm"]
    means = np.array([[-2, 3, .3, 9.8], [0, 0, .26, 9.8], [0, 0, .27, 9.8], [0, 0, .3, 9.8]])
    original = frame.copy()
    result = audit.hover_background(frame, means, columns, np.array([0, 1, 0, 1]))
    assert result["near_zero_mean_vz_counts"] == {"fault_endpoint": 1, "normal_part_of_fault_case": 1, "original_normal_case": 1}
    pure = result["pure_normal_hover"]
    assert pure["windows"] == pure["cases"] == 2 and pure["false_windows"] == 1
    assert pure["near_zero_mean_vz"] == {"windows": 1, "false_windows": 1, "fpr": 1.0}
    late = next(row for row in pure["by_log_time_bin"] if row["bin"] == "[40.0, 60.0)")
    assert late["fpr"] == 1
    pd.testing.assert_frame_equal(frame, original)
    assert "pure_normal_hover" not in audit.hover_background(frame, means, columns)
    with pytest.raises(ValueError, match="predictions"):
        audit.hover_background(frame, means, columns, np.array([2, 0, 1, 0]))


def fixture_metadata():
    cases = pd.DataFrame([
        {"case_uid": "A", "split": "train", "domain": "SIL", "flight_status": "hover", "level2_label": "motor", "row_count": "1000"},
        {"case_uid": "Z", "split": "val", "domain": "HIL", "flight_status": "hover", "level2_label": "normal", "row_count": "170"},
    ])
    windows = []
    for row in reversed(cases.to_dict(orient="records")):  # manifest ordering differs from case CSV
        fault_case = row["level2_label"] == "motor"
        for start in range(0, int(row["row_count"]) - 149, 5):
            end = start + 150
            positive = fault_case and end - 1 >= 500
            phase = "normal_flight" if not fault_case else ("pre_fault" if end <= 500 else ("transition" if start < 500 else "stable_fault"))
            windows.append({"case_uid": row["case_uid"], "split": row["split"], "domain": row["domain"],
                            "flight_status": row["flight_status"], "case_level2_label": row["level2_label"],
                            "level2_label": "motor" if positive else "normal", "phase": phase,
                            "window_start_row": start, "window_end_row": end, "endpoint_fault": int(positive),
                            "fault_fraction": max(0, min(150, end - 500)) / 150 if fault_case else 0,
                            "fault_onset_row": 500 if fault_case else -1})
    return cases, pd.DataFrame(windows)


def test_manifest_reconstruction_preserves_case_order_and_original_cap(tmp_path):
    cases, windows = fixture_metadata()
    path = tmp_path / "windows.csv"
    windows.to_csv(path, index=False)
    selected, counts = audit.reconstruct_metadata(cases, path)
    assert selected.case_uid.drop_duplicates().tolist() == ["Z", "A"]
    assert len(selected) == 5 + 64
    actual = selected[selected.case_uid == "A"].window_start_row.to_numpy()
    expected = np.unique(np.linspace(0, 170, 64).round().astype(int)) * 5
    np.testing.assert_array_equal(actual, expected)
    assert sum(row["windows"] for row in counts) == len(windows)
    assert selected[selected.case_uid == "Z"].normal_origin.unique().tolist() == ["original_normal_case"]
    assert selected.relative_log_position.between(0, 1).all()


def test_manifest_reconstruction_rejects_missing_windows_and_metadata_mismatch(tmp_path):
    cases, windows = fixture_metadata()
    path = tmp_path / "windows.csv"
    windows.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="counts"):
        audit.reconstruct_metadata(cases, path)
    windows.loc[0, "case_level2_label"] = "sensor"
    windows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="metadata mismatch"):
        audit.reconstruct_metadata(cases, path)
