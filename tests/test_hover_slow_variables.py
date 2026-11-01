from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from check_hover_slow_variables import label_time_counts, paired_channel_changes
from uav_fault.causal_input import canonical_window
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2


def test_label_time_bins_use_last_retained_sample_and_keep_normal_controls():
    frame = pd.DataFrame({'case_uid': ['SIL/hover/motor/a'] * 3 + ['SIL/hover/no_fault/b'],
        'case_level2_label': ['motor'] * 3 + ['normal'],
        'qualified_y': [0, 1, 1, 0], 'window_end_row': [500, 501, 2001, 2001]})
    rows = label_time_counts(frame)
    fault = [r for r in rows if r['origin'] == 'fault_case']
    assert fault[0]['normal'] == 1 and fault[0]['fault'] == 0
    assert fault[1]['normal'] == 0 and fault[1]['fault'] == 1
    assert fault[4]['fault'] == 1
    normal = [r for r in rows if r['origin'] == 'independent_normal']
    assert normal[4]['normal'] == 1


def test_channel_drift_is_paired_within_case_not_pooled():
    frame = pd.DataFrame({'case_uid': ['a', 'a', 'b', 'b'], 'kind': ['normal'] * 4,
        'split': ['train'] * 4, 'domain': ['SIL'] * 4, 'crop_quarter': [0, 3, 0, 3],
        'mean__x': [1., 3., 100., 99.], 'std__x': [.1] * 4})
    rows = paired_channel_changes(frame, ['x'])
    assert [r['late_minus_early_source_std'] for r in rows] == [2., -1.]


def test_window_relative_baro_removes_constant_absolute_altitude_only():
    columns = list(RFLY_SEQ_COLUMNS_V2); n = len(columns); index = columns.index('baro_alt_relative')
    first = np.zeros((150, n)); first[:, index] = 100 + np.linspace(0, 2, 150)
    second = first.copy(); second[:, index] += 500
    first[:, columns.index('actuator_ctrl_4')] = .4
    second[:, columns.index('actuator_ctrl_4')] = .5
    kwargs = dict(columns=columns, row_valid=np.ones(150, dtype=bool), mean=np.zeros(n), std=np.ones(n))
    a = canonical_window(first, **kwargs); b = canonical_window(second, **kwargs)
    assert np.array_equal(a[index], b[index])
    assert a[index, 0] == 0 and a[index, -1] == 2
    # Fixed source z-score does not remove a control/trim level difference.
    assert not np.array_equal(a[columns.index('actuator_ctrl_4')], b[columns.index('actuator_ctrl_4')])
