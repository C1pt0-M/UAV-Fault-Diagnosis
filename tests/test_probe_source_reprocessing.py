"""Checks for candidate reprocessing semantics, not model quality."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from probe_source_reprocessing import coalesce, feature_sources, flight_bounds, sample_published


def test_last_duplicate_is_used_without_changing_source():
    source = np.array([0, 1, 2, 3])
    times, values = coalesce([0, 10, 10, 20], source)
    assert times.tolist() == [0, 10, 20]
    assert values.tolist() == [0, 2, 3]
    assert source.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize('times', [[], [10, 9], [0, np.nan], [0, 1.5]])
def test_original_clock_errors_are_rejected_not_sorted(times):
    with pytest.raises(ValueError):
        coalesce(times, np.zeros(len(times)))


def test_latest_published_value_no_future_extrapolation_or_stale_fill():
    values, valid, age = sample_published(
        [10, 20, 100], [1, 2, 9], np.array([5, 10, 15, 40, 41, 100, 101]), 20)
    assert valid.tolist() == [False, True, True, True, False, True, False]
    assert values[valid, 0].tolist() == [1, 1, 2, 9]
    assert np.isnan(values[~valid]).all()
    assert age[valid].tolist() == [0, 5, 20, 0]


def test_nonfinite_latest_sample_is_not_replaced_by_an_older_finite_one():
    values, valid, _ = sample_published([0, 10, 20], [1, np.nan, 3], np.array([10, 15, 20]), 50)
    assert valid.tolist() == [False, False, True]
    assert np.isnan(values[:2]).all()


def test_inserting_a_future_observation_does_not_change_past_sample():
    a, av, _ = sample_published([0, 10], [1, 9], np.array([5]), 10)
    b, bv, _ = sample_published([0, 6, 10], [1, 1000, 9], np.array([5]), 10)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(av, bv)


def test_transient_landed_flag_does_not_end_the_episode():
    times = np.arange(9) * 1000000
    flags = [1, 0, 0, 1, 0, 1, 1, 1, 1]
    result = flight_bounds(times, flags, flags)
    assert result['start_us'] == 1000000
    assert result['end_us'] == 5000000
    assert result['landing_confirmed'] is True
    assert result['left_censored'] is False
    assert result['right_censored'] is False


def test_land_clock_gap_cannot_confirm_persistent_landing():
    result = flight_bounds([0, 1000000, 2000000, 8000000, 9000000],
                           [1, 0, 1, 1, 1], [1, 0, 1, 1, 1])
    assert result['right_censored'] is True
    assert result['end_us'] == 9000000


def test_existing_airborne_at_start_is_reported_as_left_censored():
    result = flight_bounds([100, 1000100, 2000100], [0, 0, 0], [0, 0, 0])
    assert result['left_censored'] is True
    assert result['right_censored'] is True


def test_no_airborne_or_nonbinary_landing_flags_rejected():
    with pytest.raises(ValueError, match='No observed airborne'):
        flight_bounds([0, 1], [1, 1], [1, 1])
    with pytest.raises(ValueError, match='Nonbinary'):
        flight_bounds([0, 1], [0, 2], [0, 1])


def test_raw_contract_mapping_does_not_admit_label_features():
    contract = {'features': [
        {'canonical_name': 'gyro_x', 'source_columns': {'SIL': '_sensor_combined_0_gyro_rad[0]'}},
        {'canonical_name': 'acc_norm', 'source_columns': {'SIL': 'derived'}},
    ]}
    assert feature_sources(contract, 'SIL') == [('sensor_combined', 'gyro_rad[0]'), None]
    contract['features'][0]['source_columns']['SIL'] = '_rfly_ctrl_lxl_0_id'
    with pytest.raises(ValueError, match='Unmapped raw source'):
        feature_sources(contract, 'SIL')
