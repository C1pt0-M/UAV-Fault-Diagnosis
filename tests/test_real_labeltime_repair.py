from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/data'))
from repair_real_labeltime import assign_references, patch_csv, window_rows, finalize_link_labels, ID, FLAGS
from verify_real_label_bags import compare_export
from generate_window_manifest import _infer_fault_mask


def references():
    origin = 1686794382671454381
    query = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    bag = {'sync_ros_ns': np.array([origin, origin + 5000], dtype=np.int64),
           'sync_remote_ns': np.array([100000, 105000], dtype=np.int64),
           'command_ros_ns': np.array([origin, origin + 1000, origin + 1000, origin + 3000], dtype=np.int64),
           'command_id': np.array([1500, 1500, 123450, 1500], dtype=np.int64)}
    arr = assign_references(query, bag, np.array([100, 102, 104], dtype=np.int64),
                            np.array([1500, 123450, 1500], dtype=np.int64))
    arr.update(timestamp_us=query, historical_state=np.array([0, 0, 1, 1, 1], dtype=np.int8))
    return query, bag, arr


def test_last_duplicate_send_state_no_extrapolation_and_disagreement_unknown():
    query, bag, a = references()
    assert a['ros_query_ns'][0] == bag['sync_ros_ns'][0]
    assert a['reference_id'].tolist() == [1500, 123450, 123450, 1500, -1]
    assert a['ros_state'].tolist() == [0, 1, 1, 0, -1]
    assert a['qualified_state'].tolist() == [0, -1, 1, -1, -1]
    assert a['row_label_valid'].tolist() == [True, False, True, False, False]


def test_original_clock_reversal_and_noninteger_query_rejected():
    query, bag, _ = references()
    bag['sync_remote_ns'] = bag['sync_remote_ns'][::-1]
    with pytest.raises(ValueError, match='clock reversal'):
        assign_references(query, bag, np.array([100, 104]), np.array([1500, 1500]))
    _, bag, _ = references()
    with pytest.raises(ValueError, match='Integer microsecond'):
        assign_references(query.astype(float), bag, np.array([100, 104]), np.array([1500, 1500]))


def make_csv(path, times):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', ID, 'continuous', 'command_text'])
        for t in times:
            writer.writerow([str(t) + '.0', '1500.0', '001.23000e-05', 'keep, exact string'])


def test_repaired_csv_changes_only_id_and_adds_explicit_flags(tmp_path):
    times, _, a = references()
    src, dst = tmp_path / 'old.csv', tmp_path / 'new.csv'
    make_csv(src, times)
    before = src.read_bytes()
    result = patch_csv(src, dst, times, a, False)
    assert src.read_bytes() == before
    assert result['rows'] == 5 and result['changed_id_rows'] == 3
    with src.open(newline='', encoding='utf-8') as f:
        old = list(csv.reader(f))
    with dst.open(newline='', encoding='utf-8') as f:
        new = list(csv.reader(f))
    assert new[0] == old[0] + FLAGS
    for o, n in zip(old[1:], new[1:]):
        assert [o[0], *o[2:]] == [n[0], *n[2:4]]
    with pytest.raises(ValueError, match='legacy boolean-only'):
        _infer_fault_mask(pd.read_csv(dst))
    with pytest.raises(FileExistsError):
        patch_csv(src, dst, times, a, False)


def test_quarantine_keeps_all_original_field_strings_and_unknown_state(tmp_path):
    times, _, a = references()
    a['ros_state'][:] = -1
    a['row_label_valid'][:] = False
    src, dst = tmp_path / 'old.csv', tmp_path / 'quarantine.csv'
    make_csv(src, times)
    result = patch_csv(src, dst, times, a, True)
    with src.open(newline='', encoding='utf-8') as f:
        old = list(csv.reader(f))
    with dst.open(newline='', encoding='utf-8') as f:
        new = list(csv.reader(f))
    assert result['changed_id_rows'] == 0
    assert all(n[:-3] == o and n[-3:] == ['-1', '0', '1'] for o, n in zip(old[1:], new[1:]))


def test_window_metadata_preserves_input_grid_and_does_not_label_unknown_normal():
    n = 165
    a = {'historical_state': np.zeros(n, dtype=np.int8), 'ros_state': np.zeros(n, dtype=np.int8),
         'row_label_valid': np.ones(n, dtype=bool), 'ros_covered': np.ones(n, dtype=bool),
         'ulog_covered': np.ones(n, dtype=bool)}
    a['ros_state'][149] = 1
    a['row_label_valid'][149] = False
    rows = window_rows('REAL/case', 'test_fault', n, a, 'cap64', 'VERIFIED_REQUIRED_BAG_FIELDS')
    assert [(r['window_start_row'], r['window_end_row']) for r in rows] == [(0, 150), (5, 155), (10, 160), (15, 165)]
    assert rows[0]['qualified_label'] == -1 and not rows[0]['endpoint_label_valid']
    assert rows[0]['review_reason'] == 'ROS_sent_ULog_received_state_disagree'
    assert rows[1]['qualified_label'] == 0 and rows[1]['endpoint_label_valid']
    assert not rows[1]['whole_window_label_valid']


def test_metadata_distinguishes_historical_family_and_uncertain_label():
    frame = pd.DataFrame({'case_uid': ['REAL/a', 'REAL/a', 'REAL/a'],
        'level2_label': ['motor', 'normal', 'motor'], 'source_csv': ['old.csv'] * 3,
        'qualified_label': [0, 1, -1]})
    new = finalize_link_labels(frame, {'REAL/a': 'motor'})
    assert new.level2_label.tolist() == ['normal', 'motor', 'uncertain']
    assert new.historical_level2_label.tolist() == ['motor', 'normal', 'motor']
    assert new.historical_source_csv.tolist() == ['old.csv'] * 3
    assert frame.level2_label.tolist() == ['motor', 'normal', 'motor']


def test_bag_export_comparison_rejects_truncated_or_changed_integer_fields(tmp_path):
    path = tmp_path / 'export.csv'
    origin = 1686794382671454381
    path.write_text(f'rosbagTimestamp,id\n{origin},1500\n')
    result = compare_export(path, [(origin, 1500)], ['rosbagTimestamp', 'id'])
    assert result['exact']
    result = compare_export(path, [(origin + 1, 1500)], ['rosbagTimestamp', 'id'])
    assert not result['exact'] and result['different_rows'] == 1
    result = compare_export(path, [(origin, 1500), (origin + 1, 123450)], ['rosbagTimestamp', 'id'])
    assert not result['exact'] and result['bag_rows'] == 2 and result['export_rows'] == 1


def test_missing_rows_rejected_instead_of_silent_truncation(tmp_path):
    times, _, a = references()
    src = tmp_path / 'old.csv'
    make_csv(src, times[:-1])
    with pytest.raises(ValueError, match='Incomplete CSV'):
        patch_csv(src, tmp_path / 'bad.csv', times, a, False)
