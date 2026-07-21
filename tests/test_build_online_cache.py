from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


from uav_fault.data.build_online_cache import build_cache


def _write_case(csv_path: Path, *, fault: bool) -> None:
    rows = 180
    df = pd.DataFrame(
        {
            "Timestamp": [i * 20000 for i in range(rows)],
            "_actuator_controls_0_0_control[0]": [0.1] * rows,
            "_actuator_controls_0_0_control[1]": [0.1] * rows,
            "_actuator_controls_0_0_control[2]": [0.1] * rows,
            "_actuator_controls_0_0_control[3]": [0.1] * rows,
            "_actuator_outputs_0_output[0]": [1200.0] * rows,
            "_actuator_outputs_0_output[1]": [1200.0] * rows,
            "_actuator_outputs_0_output[2]": [1200.0] * rows,
            "_actuator_outputs_0_output[3]": [1200.0] * rows,
            "_sensor_combined_0_accelerometer_m_s2[2]": [-9.8] * rows,
            "_vehicle_air_data_0_baro_alt_meter": [10.0] * rows,
            "_vehicle_air_data_0_baro_temp_celcius": [25.0] * rows,
            "_vehicle_air_data_0_baro_pressure_pa": [101325.0] * rows,
            "_vehicle_attitude_0_q[0]": [1.0] * rows,
            "_vehicle_attitude_0_q[1]": [0.0] * rows,
            "_vehicle_attitude_0_q[2]": [0.0] * rows,
            "_vehicle_attitude_0_q[3]": [0.0] * rows,
            "_vehicle_local_position_0_vx": [0.0] * rows,
            "_vehicle_local_position_0_vy": [0.0] * rows,
            "_vehicle_local_position_0_vz": [0.0] * rows,
            "_vehicle_magnetometer_0_magnetometer_ga[0]": [0.0] * rows,
            "_vehicle_magnetometer_0_magnetometer_ga[1]": [0.0] * rows,
            "_vehicle_magnetometer_0_magnetometer_ga[2]": [0.0] * rows,
            "flight_phase": ["fault" if fault and i >= 30 else "normal" for i in range(rows)],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def test_build_online_cache_accepts_explicit_manifest_v2_path(tmp_path: Path) -> None:
    csv_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifest"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "cache"

    _write_case(csv_dir / "Case_1.csv", fault=False)
    _write_case(csv_dir / "Case_2.csv", fault=True)
    _write_case(csv_dir / "Case_3.csv", fault=True)

    manifest_path = manifest_dir / "manifest_v2.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"filename": "Case_1.csv", "label": "normal"},
            {"filename": "Case_2.csv", "label": "motor"},
            {"filename": "Case_3.csv", "label": "motor"},
        ]
    ).to_csv(manifest_path, index=False)

    splits_path = splits_dir / "splits_v2.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(
            {
                "train": ["Case_1.csv"],
                "val": ["Case_2.csv"],
                "test": ["Case_3.csv"],
            }
        ),
        encoding="utf-8",
    )

    build_cache(
        str(csv_dir),
        str(splits_path),
        str(output_dir),
        window_size=150,
        stride=10,
        manifest_path=str(manifest_path),
        protocol_name="rflymad_online_real3_binary_v1",
        label_space="real3",
        source_fault_classes=["motor", "sensor"],
    )

    train_pt = torch.load(output_dir / "train.pt", weights_only=True)
    assert train_pt["x"].ndim == 3
    assert (output_dir / "norm_stats.json").exists()
    norm_stats = json.loads((output_dir / "norm_stats.json").read_text(encoding="utf-8"))
    assert norm_stats["window_size"] == 150
    assert norm_stats["stride"] == 10
    assert norm_stats["protocol_name"] == "rflymad_online_real3_binary_v1"
    assert norm_stats["label_space"] == "real3"
    assert norm_stats["source_fault_classes"] == ["motor", "sensor"]


def test_build_online_cache_uses_source_csv_path_from_manifest_v2(tmp_path: Path) -> None:
    csv_dir = tmp_path / "empty_processed_root"
    manifest_dir = tmp_path / "manifest"
    splits_dir = tmp_path / "splits"
    source_root = tmp_path / "separate_sources"
    output_dir = tmp_path / "cache"

    sil_csv = source_root / "processed_SIL_50hz" / "Case_10.csv"
    hil_csv = source_root / "processed_HIL_50hz" / "Case_20.csv"
    real_csv = source_root / "processed_Real_50hz" / "Case_30.csv"
    _write_case(sil_csv, fault=False)
    _write_case(hil_csv, fault=True)
    _write_case(real_csv, fault=True)

    manifest_path = manifest_dir / "manifest_v2.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"filename": "Case_10.csv", "label_binary": "normal", "source_csv_path": str(sil_csv)},
            {"filename": "Case_20.csv", "label_binary": "fault", "source_csv_path": str(hil_csv)},
            {"filename": "Case_30.csv", "label_binary": "fault", "source_csv_path": str(real_csv)},
        ]
    ).to_csv(manifest_path, index=False)

    splits_path = splits_dir / "splits_v2.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(
            {
                "train": ["Case_10.csv"],
                "val": ["Case_20.csv"],
                "test": ["Case_30.csv"],
            }
        ),
        encoding="utf-8",
    )

    build_cache(
        str(csv_dir),
        str(splits_path),
        str(output_dir),
        window_size=150,
        stride=10,
        manifest_path=str(manifest_path),
    )

    train_pt = torch.load(output_dir / "train.pt", weights_only=True)
    val_pt = torch.load(output_dir / "val.pt", weights_only=True)
    test_pt = torch.load(output_dir / "test.pt", weights_only=True)
    assert train_pt["x"].shape[0] > 0
    assert val_pt["x"].shape[0] > 0
    assert test_pt["x"].shape[0] > 0
    norm_stats = json.loads((output_dir / "norm_stats.json").read_text(encoding="utf-8"))
    assert norm_stats["source_hz_est"] is not None
