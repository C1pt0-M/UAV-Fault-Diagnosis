"""Minimal correctness checks for read-only source-event diagnostics."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from openpyxl import Workbook
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("source_boundary_audit_test", SCRIPTS / "audit_source_event_boundaries.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_hold_repeated_time_uses_last_state_and_marks_uncovered_queries():
    values, valid = audit.hold([1, 2, 2, 4], [0, 0, 1, 0], [0, 1, 2, 3, 4, 5])
    assert values.tolist() == [0, 0, 1, 1, 0, 0]
    assert valid.tolist() == [False, True, True, True, True, False]


@pytest.mark.parametrize("times", [[], [2, 1], [1, float("inf")], [float("nan")]])
def test_hold_rejects_empty_reversed_or_nonfinite_time(times):
    with pytest.raises(ValueError):
        audit.hold(times, np.zeros(len(times)), [1])


def test_event_intervals_do_not_retain_superseded_duplicate_state():
    assert audit.intervals([1, 2, 2, 3], [0, 1, 0, 0]) == []
    assert audit.intervals([1, 2, 3, 4], [0, 1, 0, 1]) == [
        {"start_s": 2., "end_s": 3., "last_observed_s": 4.},
        {"start_s": 4., "end_s": None, "last_observed_s": 4.},
    ]


def test_uncovered_segments_are_not_labeled_normal():
    segments = audit.supported_segments(
        np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1], dtype=bool),
        np.array([0, 1, 1, 0], dtype=bool))
    assert [(r["start_row"], r["end_row_exclusive"]) for r in segments] == [(1, 2), (2, 3)]


@pytest.fixture
def cropped_case(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "TrueData").mkdir(parents=True)
    (raw / "Log").mkdir()
    gtd = raw / "TrueData/UAVState_data.xlsx"
    wb = Workbook()
    wb.active.append(["uavTime", "fault_state"])
    for row in [(0, 0), (1000, 0), (4000, 1), (5000, 1)]:
        wb.active.append(row)
    wb.save(gtd)
    meta = raw / "TestInfo.csv"
    meta.write_text("Fault ID,[123450.]" + chr(10), encoding="utf-8")
    log = raw / "Log/sample.ulg"
    log.write_bytes(b"synthetic ulog stand-in")
    csv = tmp_path / "processed.csv"
    old_state = np.r_[np.zeros(75), np.ones(75)]
    pd.DataFrame({"Timestamp": np.arange(150) * 20000,
                  "UAVState_data_fault_state": old_state,
                  "_rfly_ctrl_lxl_0_id": np.zeros(150)}).to_csv(csv, index=False)
    command = SimpleNamespace(name="rfly_ctrl_lxl", multi_id=0, data={
        "timestamp": np.array([0, 1000000, 4100000, 5000000]),
        "id": np.array([1500, 1500, 123450, 123450])})
    sensor = SimpleNamespace(name="sensor_combined", multi_id=0, data={
        "timestamp": np.arange(501) * 10000,
        "accelerometer_m_s2[0]": np.zeros(501)})
    fake_log = SimpleNamespace(data_list=[command, sensor], start_timestamp=0,
                               last_timestamp=5000000, dropouts=[])
    monkeypatch.setattr(audit, "ULog", lambda *args, **kwargs: fake_log)
    row = dict(case_uid="SIL/hover/motor/synthetic", domain="SIL", split="train",
               flight_status="hover", source_fault_type="motor", source_path=str(raw),
               final_output_path=str(csv), row_count="150")
    previous = dict(csv_sha256=audit.digest(csv), testinfo_sha256=audit.digest(meta), rows=150)
    links = [dict(window_start_row="0", window_end_row="150", cache_index="0", label="1")]
    return row, previous, links, [gtd, meta, log, csv]


def test_event_after_crop_does_not_become_fault_and_inputs_are_unchanged(cropped_case):
    row, previous, links, paths = cropped_case
    before = [audit.digest(p) for p in paths]
    record, windows = audit.case_audit((row, previous, links))
    assert record["status"] == "diagnostic_complete"
    assert record["raw_onset_after_crop"] is True
    assert record["covered_raw_gtd_active_rows"] == 0
    assert record["selected_old_fault_raw_normal"] == 1
    assert windows[0]["gtd_endpoint_reference"] == 0
    assert windows[0]["gtd_full_window_covered"] is True
    assert [audit.digest(p) for p in paths] == before


def test_changed_input_is_reported_not_silently_used(cropped_case):
    row, previous, links, paths = cropped_case
    previous["csv_sha256"] = "wrong identity"
    record, windows = audit.case_audit((row, previous, links))
    assert record["status"] == "review_required" and not windows
    assert any("identity changed" in item for item in record["issues"])


def test_continuation_rejects_duplicate_members_before_creating_output(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "audit_plan.json").write_text(
        json.dumps({"manifest_sha256": audit.EXPECTED_MANIFEST}), encoding="utf-8")
    (prior / "case_assessments.jsonl").write_text(
        (json.dumps({"case_uid": "case1"}) + chr(10)) * 2, encoding="utf-8")
    target = tmp_path / "new_output"
    monkeypatch.setattr(audit, "source_tasks", lambda: [({"case_uid": "case1"}, {}, [])])
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(target),
                                     "--exclude-completed", str(prior)])
    with pytest.raises(ValueError, match="Continuation membership"):
        audit.main()
    assert not target.exists()
