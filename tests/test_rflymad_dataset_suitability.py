import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
from audit_rflymad_dataset_suitability import (active_id, feature_matrix, first_time,
                                              last_observed, motion_signals, read_testinfo)
from rfly_feature_contract import window_batch


def test_fault_ids_do_not_treat_normal_control_as_failure():
    assert active_id([0, 1500, 123450, 123456, 123540, 123549, 123550]).tolist() == [False, False, True, True, True, True, False]
    with pytest.raises(ValueError):
        active_id([123450.5])
    with pytest.raises(ValueError):
        active_id([np.nan])


def test_causal_reference_uses_last_duplicate_and_marks_out_of_range():
    sampled, valid = last_observed([1, 2, 2, 4], [0, 0, 1, 1], [0, 1, 1.5, 2, 3, 4, 5])
    assert sampled.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert valid.tolist() == [False, True, True, True, True, True, False]
    assert first_time([1, 2], [False, False]) is None
    assert first_time([1, 2], [False, True]) == 2


@pytest.mark.parametrize("times,values", [([2, 1], [0, 1]), ([], []), ([1, 2], [1]), ([1, np.nan], [0, 1])])
def test_bad_raw_timelines_are_rejected(times, values):
    with pytest.raises(ValueError):
        last_observed(times, values, [1])


def test_window_relative_baro_removes_a_constant_offset_not_other_signals():
    names = ["accel_x", "accel_y", "accel_z", "acc_norm", "baro_alt_relative", "vel_x", "vel_y", "vel_z", "actuator_ctrl_4"]
    contract = {"features": [{"canonical_name": name, "source_columns": {"SIL": "derived" if name == "acc_norm" else name}} for name in names]}
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in names if name != "acc_norm"})
    before = feature_matrix(frame, "SIL", contract)
    changed = frame.copy()
    changed["baro_alt_relative"] += 100
    after = feature_matrix(changed, "SIL", contract)
    left, right = (window_batch(matrix, names, [(0, 3)]) for matrix in (before, after))
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(left[0, names.index("acc_norm")], np.sqrt(3) * np.array([1, 2, 3]))
    signals = motion_signals(left, names)
    assert signals["vertical_speed_rms"][0] == pytest.approx(np.sqrt(14 / 3))
    frame.loc[0, "accel_x"] = np.inf
    with pytest.raises(ValueError):
        feature_matrix(frame, "SIL", contract)


def test_testinfo_is_parsed_without_evaluating_commands(tmp_path):
    path = tmp_path / "TestInfo.csv"
    path.write_text('Control Sequence,"[\'2,1,0.5,1,15\' \'1,1,25\']"\nFault ID,[123450.]\n', encoding="utf-8")
    data, commands, ids, digest = read_testinfo(path)
    assert commands == ["2,1,0.5,1,15", "1,1,25"] and ids == [123450]
    assert len(digest) == 64
    path.write_text("Control Sequence,invalid\nFault ID,[]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_testinfo(path)


def test_native_mixed_string_resampling_distorts_event_time_when_available():
    path = Path("D:/RflyMAD/Data_processing_tools/fileprocess.py")
    if not path.exists():
        pytest.skip("Optional installed RflyMAD processing-tool regression check")
    spec = importlib.util.spec_from_file_location("rfly_native_test_readonly", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    raw = np.array([[0, "a", 0], [1000, "a", 0], [10000, "b", 1], [11000, "b", 1]])
    grid = np.arange(0, 11000, 1000)
    generated = native.CSVFile_extractor("unused").freq_adjustment(raw, grid)[:, 1].astype(float)
    held, valid = last_observed(raw[:, 0].astype(float), raw[:, 2].astype(float), grid)
    assert valid.all() and not np.array_equal(generated, held)
