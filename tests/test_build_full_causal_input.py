from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/data'))
from build_full_causal_input import extract_group, transform_raw
from uav_fault.causal_pipeline import GROUPS, batch_frames


def test_extract_uses_sample_clock_and_never_future_publication():
    data = {'timestamp': np.array([100000, 120000, 160000], dtype=np.int64),
        'timestamp_sample': np.array([99000, 119000, 159000], dtype=np.int64)}
    for i in range(4): data[f'control[{i}]'] = np.array([1., 2., 999.])
    trace, offset = extract_group(data, 'controls', np.array([110000, 130000], dtype=np.int64))
    assert trace.available_us.tolist() == [100000, 120000]
    assert trace.observation_us.tolist() == [99000, 119000]
    assert trace.values[:, 0].tolist() == [1., 2.] and offset == 0


def test_invalid_original_clock_is_not_repaired_by_sorting_or_fallback():
    t = np.array([100000, 120000], dtype=np.int64)
    data = {'timestamp': t, 'timestamp_sample': np.array([99999, 120001], dtype=np.int64)}
    for i in range(4): data[f'control[{i}]'] = np.ones(2)
    with pytest.raises(ValueError, match='sample_clock'):
        extract_group(data, 'controls', t)
    data['timestamp'] = t[::-1]
    with pytest.raises(ValueError, match='publication_clock_reversal'):
        extract_group(data, 'controls', t)


def test_missing_topic_is_unavailable_without_zero_fill():
    from uav_fault.causal_pipeline import TopicTrace
    columns = [n for _, _, fields in GROUPS.values() for n, _ in fields] + ['acc_norm']
    traces = {g: TopicTrace(np.empty(0,dtype=np.int64), np.empty(0,dtype=np.int64), np.empty((0,len(fields))))
              for g, (_, _, fields) in GROUPS.items()}
    matrix, valid, provenance = batch_frames(traces, np.array([100000],dtype=np.int64), columns, {g:40000 for g in GROUPS})
    assert not valid.any() and np.isnan(matrix).all()
    assert all(p['source_index'].tolist() == [-1] for p in provenance.values())


def test_transform_keeps_only_eligible_original_window_positions():
    columns = [n for _, _, fields in GROUPS.values() for n, _ in fields] + ['acc_norm']
    values = np.ones((160,26),dtype=np.float64)
    b = columns.index('baro_alt_relative'); values[:,b] = 100000 + np.arange(160)*.01
    starts = np.array([0,5,10],dtype=np.int64); mask = np.array([True,False,True])
    x = transform_raw(values,starts,mask,columns)
    assert x.shape == (2,26,150) and x.dtype == np.float32
    assert x[0,b,0] == 0 and x[1,b,0] == 0
    assert x[1,b,-1] == np.float32(values[159,b]-values[10,b])
    assert starts.tolist() == [0,5,10] and mask.tolist() == [True,False,True]
