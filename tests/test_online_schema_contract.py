"""Contract checks for the two online feature schemas.

The online path is migrating from the 31-channel rfly_seq_columns_v1 runtime schema
to the 26-channel rfly_seq_columns_v2 set that matches the offline feature contract.
Both must keep working while the deployed v1 models are still in service.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from uav_fault.data.build_sequence_cache import SEQ_COLUMNS
from uav_fault.models.cnn.lstm import (
    RFLY_SEQ_COLUMNS_V1,
    RFLY_SEQ_COLUMNS_V2,
    RFLY_SEQ_SCHEMAS,
    OnlineKGMSTCA,
    resolve_seq_index,
)
from uav_fault.models.train_online_detector import validate_cache_contract

# Channel order of D:\UAV-Data\feature_contract_v1.json.
OFFLINE_CONTRACT_COLUMNS = [
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_alt_relative",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "acc_norm",
]


def test_v1_matches_the_sequence_cache_column_list():
    """The runtime and cache copies of the 31-channel schema must not drift apart."""
    assert list(RFLY_SEQ_COLUMNS_V1) == list(SEQ_COLUMNS)
    assert len(RFLY_SEQ_COLUMNS_V1) == 31


def test_v2_matches_the_offline_feature_contract():
    assert list(RFLY_SEQ_COLUMNS_V2) == OFFLINE_CONTRACT_COLUMNS
    assert len(RFLY_SEQ_COLUMNS_V2) == 26


def test_v2_drops_the_environment_dependent_channels():
    dropped = set(RFLY_SEQ_COLUMNS_V1) - set(RFLY_SEQ_COLUMNS_V2)
    assert dropped == {
        "baro_alt",
        "baro_temp",
        "baro_pressure",
        "actuator_mean",
        "actuator_std",
        "actuator_effort",
    }


def test_resolve_seq_index_infers_schema_from_channel_count():
    assert resolve_seq_index(None, 31) == {
        name: idx for idx, name in enumerate(RFLY_SEQ_COLUMNS_V1)
    }
    index = resolve_seq_index(None, 26)
    assert index["baro_alt_relative"] == 14
    assert "baro_alt" not in index


def test_resolve_seq_index_rejects_unknown_channel_count():
    with pytest.raises(ValueError):
        resolve_seq_index(None, 17)


def test_resolve_seq_index_rejects_mismatched_explicit_columns():
    with pytest.raises(ValueError):
        resolve_seq_index(RFLY_SEQ_COLUMNS_V2, 31)


@pytest.mark.parametrize(
    ("columns", "expected_altitude"),
    [
        (RFLY_SEQ_COLUMNS_V1, "baro_alt"),
        (RFLY_SEQ_COLUMNS_V2, "baro_alt_relative"),
    ],
)
def test_kg_mstca_runs_on_both_schemas(columns, expected_altitude):
    torch.manual_seed(0)
    model = OnlineKGMSTCA(in_channels=len(columns), columns=list(columns)).eval()
    assert model.altitude_column == expected_altitude
    with torch.no_grad():
        logits = model(torch.randn(3, len(columns), 150))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_kg_mstca_infers_schema_without_explicit_columns():
    assert OnlineKGMSTCA(in_channels=31).altitude_column == "baro_alt"
    assert OnlineKGMSTCA(in_channels=26).altitude_column == "baro_alt_relative"


def test_kg_mstca_derives_effort_when_the_column_is_absent():
    v1 = OnlineKGMSTCA(in_channels=31, columns=list(RFLY_SEQ_COLUMNS_V1))
    v2 = OnlineKGMSTCA(in_channels=26, columns=list(RFLY_SEQ_COLUMNS_V2))
    assert v1._effort_idx is not None
    assert v2._effort_idx is None
    # both branches still emit 9 relation channels
    with torch.no_grad():
        assert v1._relation_features(torch.randn(2, 31, 150)).shape == (2, 9, 150)
        assert v2._relation_features(torch.randn(2, 26, 150)).shape == (2, 9, 150)


def test_kg_mstca_rejects_a_schema_missing_required_channels():
    with pytest.raises(ValueError, match="needs channels"):
        OnlineKGMSTCA(in_channels=3, columns=["gyro_x", "gyro_y", "gyro_z"])


def _write_cache(cache_dir: Path, columns: list[str], feature_schema: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "columns": list(columns),
        "mean": [0.0] * len(columns),
        "std": [1.0] * len(columns),
        "window_size": 150,
        "target_hz": 50.0,
        "stride": 10,
        "feature_schema": feature_schema,
    }
    (cache_dir / "norm_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    for name, labels in (("train", [0, 1]), ("val", [0]), ("test", [1])):
        torch.save(
            {
                "x": torch.zeros((len(labels), len(columns), 150), dtype=torch.float32),
                "y": torch.tensor(labels, dtype=torch.int64),
            },
            cache_dir / f"{name}.pt",
        )


def test_cache_contract_accepts_both_schemas(tmp_path: Path):
    v1_dir = tmp_path / "v1"
    _write_cache(v1_dir, RFLY_SEQ_COLUMNS_V1, "rfly_seq_columns_v1")
    assert validate_cache_contract(v1_dir)["columns"] == list(RFLY_SEQ_COLUMNS_V1)

    v2_dir = tmp_path / "v2"
    _write_cache(v2_dir, RFLY_SEQ_COLUMNS_V2, "rfly_seq_columns_v2")
    assert validate_cache_contract(v2_dir)["columns"] == list(RFLY_SEQ_COLUMNS_V2)


def test_cache_contract_rejects_columns_that_do_not_match_the_declared_schema(tmp_path: Path):
    # Declares v2 but ships v1 columns: a length check alone would let this through.
    cache_dir = tmp_path / "mismatch"
    _write_cache(cache_dir, RFLY_SEQ_COLUMNS_V1, "rfly_seq_columns_v2")
    with pytest.raises(ValueError, match="do not match feature_schema"):
        validate_cache_contract(cache_dir)


def test_cache_contract_rejects_unknown_schema(tmp_path: Path):
    cache_dir = tmp_path / "unknown"
    _write_cache(cache_dir, RFLY_SEQ_COLUMNS_V2, "rfly_seq_columns_v9")
    with pytest.raises(ValueError, match="unknown feature_schema"):
        validate_cache_contract(cache_dir)


def test_schema_registry_covers_both_known_schemas():
    assert set(RFLY_SEQ_SCHEMAS) == {"rfly_seq_columns_v1", "rfly_seq_columns_v2"}
    assert [len(v) for v in RFLY_SEQ_SCHEMAS.values()] == [31, 26]
