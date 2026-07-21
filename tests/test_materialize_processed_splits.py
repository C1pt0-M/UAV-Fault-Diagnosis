from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


from scripts.data.materialize_processed_splits import materialize_processed_splits


def _row(t: float = 1.0) -> dict:
    return {
        "Index": int(t * 100),
        "Timestamp": int(t * 1_000_000),
        "trueTime": t,
        "_sensor_combined_0_gyro_rad[0]": 0.1,
        "_sensor_combined_0_gyro_rad[1]": 0.2,
        "_sensor_combined_0_gyro_rad[2]": 0.3,
        "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
        "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
        "_sensor_combined_0_accelerometer_m_s2[2]": -9.8,
        "_vehicle_air_data_0_baro_alt_meter": 10.0,
        "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
        "_vehicle_attitude_0_q[0]": 1.0,
        "_vehicle_attitude_0_q[1]": 0.0,
        "_vehicle_attitude_0_q[2]": 0.0,
        "_vehicle_attitude_0_q[3]": 0.0,
        "_vehicle_local_position_0_vx": 0.0,
        "_vehicle_local_position_0_vy": 0.0,
        "_vehicle_local_position_0_vz": 0.0,
        "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
        "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
        "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
        "TrueState_data_fault_state": 0,
    }


def _write_case(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_row(1.0), _row(1.02)]).to_csv(path, index=False)


def test_materialize_processed_splits_preserves_existing_split_tree(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_RflyMAD"
    _write_case(processed_root / "train" / "Case_1010000000.csv")
    _write_case(processed_root / "val" / "Case_1000000000.csv")
    _write_case(processed_root / "test" / "Case_2000000000.csv")
    _write_case(processed_root / "external_test" / "Case_3005000000.csv")
    output_dir = tmp_path / "training_full"

    summary = materialize_processed_splits(
        processed_root=processed_root,
        output_dir=output_dir,
    )

    splits = json.loads((output_dir / "splits_from_processed_splits.json").read_text())
    manifest = pd.read_csv(output_dir / "manifest_from_processed_splits.csv")
    written_summary = json.loads((output_dir / "processed_splits.summary.json").read_text())

    assert splits == {
        "train": ["Case_1010000000.csv"],
        "val": ["Case_1000000000.csv"],
        "test": ["Case_2000000000.csv"],
        "external_test": ["Case_3005000000.csv"],
    }
    assert summary["split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
        "external_test": 1,
    }
    assert written_summary["quality_counts"] == {"ok": 4}
    assert written_summary["has_required_online_columns_counts"] == {"1": 4}
    assert set(manifest["filename"]) == {
        "Case_1010000000.csv",
        "Case_1000000000.csv",
        "Case_2000000000.csv",
        "Case_3005000000.csv",
    }
    assert manifest["source_csv_path"].map(lambda value: Path(value).exists()).all()
