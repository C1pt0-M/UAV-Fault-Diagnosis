import numpy as np
import pytest

from uav_fault.causal_pipeline import (GROUPS, TopicTrace, CausalInputStream, batch_frames, batch_windows, MISSING, STALE, NONFINITE)

COLUMNS = [name for _, _, fields in GROUPS.values() for name, _ in fields] + ['acc_norm']


def trace_fixture():
    times = np.array([100000, 120000, 140000, 160000, 220000], dtype=np.int64)
    traces = {}
    for group, (_, _, fields) in GROUPS.items():
        values = np.tile(np.arange(1, len(fields) + 1, dtype=float), (len(times), 1))
        values += np.arange(len(times))[:, None] / 10
        traces[group] = TopicTrace(times.copy(), times.copy(), values)
    return traces


def run_stream(traces, grid, limits, *, window=2, stride=1):
    stream = CausalInputStream(COLUMNS, limits, origin_us=int(grid[0]), mean=np.zeros(26), std=np.ones(26),
        window_frames=window, stride_frames=stride)
    events = sorted((int(t), order, i, group) for order, (group, trace) in enumerate(traces.items()) for i, t in enumerate(trace.available_us))
    j = 0; output = []
    for t in grid:
        while j < len(events) and events[j][0] <= t:
            a, _, i, group = events[j]; tr = traces[group]
            stream.push(group, int(tr.observation_us[i]), a, tr.values[i]); j += 1
        output.append(stream.emit(int(t)))
    return output


def test_batch_and_stateful_path_match_masks_values_and_each_window():
    traces = trace_fixture(); limits = {g: 20000 for g in GROUPS}
    grid = np.arange(100000, 240001, 20000, dtype=np.int64)
    matrix, valid, provenance = batch_frames(traces, grid, COLUMNS, limits)
    starts = np.arange(len(grid) - 1, dtype=np.int64)
    mask, windows = batch_windows(matrix, valid, starts, COLUMNS, mean=np.zeros(26), std=np.ones(26), window_frames=2)
    streamed = run_stream(traces, grid, limits)
    assert [r['input_valid'] for r in streamed] == valid.tolist()
    assert valid.tolist() == [True, True, True, True, True, False, True, True]
    for i, item in enumerate(streamed):
        assert np.array_equal(item['values'], matrix[i], equal_nan=True)
        for group, p in provenance.items():
            for k, v in item['provenance'][group].items(): assert v == p[k][i]
        if item['window_due']:
            start = item['window_start_row']; assert item['window_input_valid'] == mask[start]
            if mask[start]: assert np.array_equal(item['window'], windows[start])
            else: assert item['window'] is None
    # Recovery requires two valid frames, never removes gap or connects across it.
    assert not streamed[6]['window_input_valid'] and streamed[7]['window_input_valid']


def test_duplicate_latest_nonfinite_and_missing_topic_stay_unknown():
    traces = trace_fixture(); base = traces['gyro']
    times = np.array([100000, 120000, 120000], dtype=np.int64)
    traces['gyro'] = TopicTrace(times, times, np.array([[1, 2, 3], [4, 5, 6], [np.nan, 8, 9]]))
    traces['mag'] = TopicTrace(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 3)))
    grid = np.array([100000, 120000], dtype=np.int64); limits = {g: 40000 for g in GROUPS}
    rows, mask, p = batch_frames(traces, grid, COLUMNS, limits); output = run_stream(traces, grid, limits)
    assert not mask.any()
    assert output[1]['provenance']['gyro']['source_index'] == 2
    assert output[1]['provenance']['gyro']['reason'] == NONFINITE
    assert output[0]['provenance']['mag']['reason'] == MISSING
    assert np.isnan(output[1]['values'][COLUMNS.index('gyro_x')])


def test_actual_arrival_is_used_instead_of_old_sample_timestamp():
    traces = trace_fixture(); t = np.array([100000, 120000], dtype=np.int64)
    traces['gyro'] = TopicTrace(t, np.array([100000, 160000], dtype=np.int64), np.array([[1, 2, 3], [4, 5, 6]]))
    limits = {g: 20000 for g in GROUPS}; grid = np.array([100000, 120000, 140000, 160000], dtype=np.int64)
    output = run_stream(traces, grid, limits)
    assert output[2]['provenance']['gyro']['source_index'] == 0
    assert output[2]['provenance']['gyro']['reason'] == STALE
    # Delayed arrival is still stale by observation age, not magically fresh.
    assert output[3]['provenance']['gyro']['source_index'] == 1
    assert output[3]['provenance']['gyro']['reason'] == STALE


def test_runtime_rejects_retroactive_or_future_clock_and_wrong_grid():
    s = CausalInputStream(COLUMNS, {g: 40000 for g in GROUPS}, origin_us=100000, mean=np.zeros(26), std=np.ones(26))
    with pytest.raises(ValueError, match='availability'):
        s.push('gyro', 110000, 100000, [1, 2, 3])
    s.emit(100000)
    with pytest.raises(ValueError, match='retroactive'):
        s.push('gyro', 90000, 100000, [1, 2, 3])
    with pytest.raises(ValueError, match='grid'):
        s.emit(130000)
    with pytest.raises(ValueError, match='fields'):
        s.push('gyro', 110000, 110000, [1, 2])


def test_pipeline_future_records_never_change_past_rows_or_masks():
    traces = trace_fixture(); limits = {g: 20000 for g in GROUPS}
    grid = np.array([100000, 120000, 140000], dtype=np.int64)
    original = batch_frames(traces, grid, COLUMNS, limits)
    prefix = {g: TopicTrace(t.observation_us[t.available_us <= grid[-1]],
        t.available_us[t.available_us <= grid[-1]], t.values[t.available_us <= grid[-1]]) for g, t in traces.items()}
    restricted = batch_frames(prefix, grid, COLUMNS, limits)
    for matrix in [restricted[0], np.stack([r['values'] for r in run_stream(prefix, grid, limits)])]:
        assert np.array_equal(matrix, original[0], equal_nan=True)
    assert np.array_equal(original[1], restricted[1])
    for g in GROUPS:
        assert np.array_equal(original[2][g]['source_index'], restricted[2][g]['source_index'])


@pytest.mark.parametrize('tick', [100000.0, True, -1])
def test_stream_grid_requires_nonnegative_integer_clock(tick):
    stream = CausalInputStream(COLUMNS, {g: 40000 for g in GROUPS}, origin_us=100000, mean=np.zeros(26), std=np.ones(26))
    with pytest.raises(ValueError, match='integer grid'):
        stream.emit(tick)
    assert stream.rows == 0 and stream.last_grid == -1


def test_absent_timing_limits_and_channel_layout_rejected():
    traces = trace_fixture(); grid = np.array([100000], dtype=np.int64)
    with pytest.raises(ValueError, match='every|Every'):
        batch_frames(traces, grid, COLUMNS, {})
    with pytest.raises(ValueError, match='26-channel'):
        batch_frames(traces, grid, COLUMNS[::-1][1:], {g: 40000 for g in GROUPS})
    with pytest.raises(ValueError, match='Missing/extra'):
        batch_frames({}, grid, COLUMNS, {g: 40000 for g in GROUPS})


def test_grid_starts_not_global_file_start_control_relative_altitude():
    traces = trace_fixture(); limits = {g: 40000 for g in GROUPS}; grid = np.array([100000, 120000, 140000], dtype=np.int64)
    rows, valid, _ = batch_frames(traces, grid, COLUMNS, limits)
    _, windows = batch_windows(rows, valid, np.array([0, 1], dtype=np.int64), COLUMNS, mean=np.zeros(26), std=np.ones(26), window_frames=2)
    b = COLUMNS.index('baro_alt_relative')
    assert windows[0][b, 0] == windows[1][b, 0] == 0
    assert windows[1][b, 1] == np.float32(rows[2, b] - rows[1, b])
