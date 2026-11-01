from pathlib import Path
import json

import numpy as np
import pytest
import torch

from uav_fault.causal_input import sample_available, canonical_window, window_input_valid
from uav_fault.models.train_online_detector import validate_cache_contract, OnlineWindowDataset


def us(values):
    return np.asarray(values, dtype=np.int64)


def test_cannot_use_observation_until_it_arrives():
    result = sample_available(us([1, 4, 8]), us([2, 7, 9]), [[10], [40], [80]], us([1, 2, 6, 7, 9]), max_age_us=10)
    assert result.source_index.tolist() == [-1, 0, 0, 1, 2]
    assert result.valid.tolist() == [False, True, True, True, True]
    assert np.isnan(result.values[0, 0])
    assert result.values[1:, 0].tolist() == [10, 10, 40, 80]
    assert np.all(result.available_us[result.valid] <= us([1, 2, 6, 7, 9])[result.valid])


def test_future_append_or_mutation_cannot_change_past_window():
    t = us([149200667, 151681060]); grid = us([149208661, 149508661])
    a = sample_available(t, t, [[1.0], [999.0]], grid, max_age_us=300000)
    b = sample_available(t[:1], t[:1], [[1.0]], grid, max_age_us=300000)
    c = sample_available(t, t, [[1.0], [-999.0]], grid, max_age_us=300000)
    assert np.array_equal(a.values, b.values, equal_nan=True)
    assert np.array_equal(a.values, c.values, equal_nan=True)
    assert a.source_index.tolist() == [0, 0]
    assert a.valid.tolist() == [True, False]
    assert a.age_us.tolist() == [7994, 307994]
    assert a.observation_us[1] == 149200667


def test_streaming_prefix_and_batch_sampling_match():
    obs, arrival, grid = us([0, 3, 7, 9]), us([1, 5, 7, 10]), us([0, 2, 4, 6, 8, 10])
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    batch = sample_available(obs, arrival, values, grid, max_age_us=2)
    for i, t in enumerate(grid):
        known = arrival <= t
        streamed = sample_available(obs[known], arrival[known], values[known], us([t]), max_age_us=2)
        assert streamed.valid[0] == batch.valid[i]
        assert np.array_equal(streamed.values[0], batch.values[i], equal_nan=True)
        assert streamed.source_index[0] == batch.source_index[i]


def test_latest_invalid_value_does_not_fall_back_to_older_good_sample():
    result = sample_available(us([1, 2, 2]), us([1, 2, 2]), [[1.0], [2.0], [np.nan]], us([1, 2, 3]), max_age_us=3)
    assert result.source_index.tolist() == [0, 2, 2]
    assert result.valid.tolist() == [True, False, False]
    assert np.isnan(result.values[1:]).all()


@pytest.mark.parametrize('age', [None, -1, True, 1.0])
def test_no_implicit_age_threshold(age):
    with pytest.raises(ValueError, match='Explicit'):
        sample_available(us([1]), us([1]), [[0.0]], us([1]), max_age_us=age)


def test_reversed_or_inconsistent_clocks_fail_instead_of_sorting():
    for obs, arrival in [(us([2, 1]), us([2, 3])), (us([1, 2]), us([3, 2])), (us([1, 3]), us([1, 2]))]:
        with pytest.raises(ValueError):
            sample_available(obs, arrival, [[1.0], [2.0]], us([1, 2]), max_age_us=1)
    with pytest.raises(ValueError, match='int64'):
        sample_available([1.0], us([1]), [[1.0]], us([1]), max_age_us=1)


def test_gap_invalidity_keeps_original_window_ids():
    row_valid = np.array([True, True, False, True, True])
    starts = us([0, 1, 2, 3])
    assert window_input_valid(row_valid, starts, window_frames=2).tolist() == [True, False, False, True]
    assert starts.tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='outside'):
        window_input_valid(row_valid, us([4]), window_frames=2)


def test_shared_window_derivation_precision_and_normalization():
    columns = ['accel_x', 'accel_y', 'accel_z', 'baro_alt_relative', 'acc_norm']
    source = np.asarray([[3, 4, 0, 100000.123456, 999], [0, 0, 2, 100000.123556, 999]], dtype=np.float64)
    original = source.copy()
    out = canonical_window(source, columns, np.ones(2, dtype=bool), mean=[0, 0, 0, 0, 1], std=[1, 1, 1, 1, 2])
    assert out.dtype == np.float32 and out.shape == (5, 2)
    assert out[4].tolist() == [2.0, 0.5]
    assert out[3, 0] == 0 and out[3, 1] == np.float32(source[1, 3] - source[0, 3])
    assert np.array_equal(source, original)
    with pytest.raises(ValueError, match='Unavailable'):
        canonical_window(source, columns, np.array([True, False]), mean=[0]*5, std=[1]*5)
    source[0, 3] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        canonical_window(source, columns, np.ones(2, dtype=bool), mean=[0]*5, std=[1]*5)


def test_empty_topic_retains_unknown_values_and_original_grid():
    sampled = sample_available(us([]), us([]), np.empty((0, 2)), us([1, 2, 3]), max_age_us=0)
    assert sampled.values.shape == (3, 2) and np.isnan(sampled.values).all()
    assert not sampled.valid.any() and sampled.source_index.tolist() == [-1, -1, -1]


def test_canonical_transform_rejects_bad_stats_and_float32_overflow():
    names = ['accel_x', 'accel_y', 'accel_z', 'baro_alt_relative', 'acc_norm']
    data = np.zeros((1, 5)); valid = np.ones(1, dtype=bool)
    with pytest.raises(ValueError, match='normalization'):
        canonical_window(data, names, valid, mean=[0]*5, std=[0]*5)
    data[0, 0] = 1e100
    with pytest.raises(ValueError):
        canonical_window(data, names, valid, mean=[0]*5, std=[1]*5)


def test_ordinary_legacy_cache_validation_still_works_without_training(tmp_path):
    from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V2
    names = list(RFLY_SEQ_COLUMNS_V2)
    stats = {'columns': names, 'mean': [0.0]*len(names), 'std': [1.0]*len(names),
        'window_size': 150, 'target_hz': 50.0, 'stride': 5, 'feature_schema': 'rfly_seq_columns_v2'}
    (tmp_path / 'norm_stats.json').write_text(json.dumps(stats))
    for split in ['train', 'val', 'test']:
        torch.save({'x': torch.zeros(2, len(names), 150), 'y': torch.tensor([0, 1])}, tmp_path / (split + '.pt'))
    assert validate_cache_contract(tmp_path) == stats
    dataset = OnlineWindowDataset(str(tmp_path / 'train.pt'), str(tmp_path / 'norm_stats.json'))
    assert len(dataset) == 2 and dataset[1][1].item() == 1


def test_legacy_entry_rejects_versioned_repair_before_loading_or_forward(tmp_path):
    (tmp_path / 'repair_manifest.json').write_text(json.dumps({'version': 'repaired'}))
    with pytest.raises(ValueError, match='explicit version'):
        validate_cache_contract(tmp_path)
    with pytest.raises(ValueError, match='explicit version'):
        OnlineWindowDataset(str(tmp_path / 'train.pt'), str(tmp_path / 'norm_stats.json'))


def test_legacy_dataset_rejects_disguised_evidence_and_unknown_y(tmp_path):
    p = tmp_path / 'disguised.pt'
    torch.save({'x': torch.zeros(2, 1, 1), 'y': torch.tensor([0, 1]), 'role_id': torch.tensor([0, 3])}, p)
    with pytest.raises(ValueError, match='legacy x/y reader refuses'):
        OnlineWindowDataset(str(p), str(tmp_path / 'norm_stats.json'))
    torch.save({'x': torch.zeros(2, 1, 1), 'y': torch.tensor([0, -1])}, p)
    with pytest.raises(ValueError, match='unknown -1'):
        OnlineWindowDataset(str(p), str(tmp_path / 'norm_stats.json'))
