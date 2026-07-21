from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


from uav_fault.data.build_online_real3_splits import build_online_real3_splits


def _row(
    *,
    idx: int,
    filename: str,
    platform: str,
    label: str,
    flight_mode: str = "hover",
    online: int = 1,
    quality: str = "ok",
) -> dict:
    return {
        "sample_id": f"sample_{idx}",
        "filename": filename,
        "source_csv_path": f"D:/fake/{filename}",
        "platform": platform,
        "flight_mode": flight_mode,
        "label_binary": "normal" if label == "normal" else "fault",
        "label_binary_id": 0 if label == "normal" else 1,
        "label_main_6class": label,
        "is_in_main_6class": 1,
        "has_required_online_columns": online,
        "quality_flag": quality,
    }


def _write_manifest(path: Path) -> None:
    rows = []
    idx = 0
    for platform in ("SIL", "HIL"):
        for label in ("normal", "motor", "sensor"):
            for repeat in range(4):
                rows.append(
                    _row(
                        idx=idx,
                        filename=f"{platform}_{label}_{repeat}.csv",
                        platform=platform,
                        label=label,
                        flight_mode="hover" if repeat % 2 == 0 else "cruise",
                    )
                )
                idx += 1

    for label in ("normal", "motor", "sensor"):
        for repeat in range(6):
            rows.append(
                _row(
                    idx=idx,
                    filename=f"Real_{label}_{repeat}.csv",
                    platform="Real",
                    label=label,
                    flight_mode="hover" if repeat < 3 else "cruise",
                )
            )
            idx += 1

    for label in ("propeller", "wind", "load", "low_voltage"):
        rows.append(
            _row(
                idx=idx,
                filename=f"SIL_{label}.csv",
                platform="SIL",
                label=label,
            )
        )
        idx += 1
    rows.append(
        _row(
            idx=idx,
            filename="SIL_motor_missing_online.csv",
            platform="SIL",
            label="motor",
            online=0,
        )
    )
    idx += 1
    rows.append(
        _row(
            idx=idx,
            filename="Real_sensor_bad.csv",
            platform="Real",
            label="sensor",
            quality="bad",
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_online_real3_splits_excludes_non_real3_classes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest_v2.csv"
    output_dir = tmp_path / "online_real3"
    _write_manifest(manifest_path)

    build_online_real3_splits(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        seed=42,
    )

    splits = json.loads(
        (output_dir / "splits_online_real3_binary_v1.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (output_dir / "protocol_online_real3_binary_v1.json").read_text(encoding="utf-8")
    )

    all_split_files = {
        filename
        for filenames in splits.values()
        for filename in filenames
    }
    assert all("propeller" not in filename for filename in all_split_files)
    assert all("wind" not in filename for filename in all_split_files)
    assert all("load" not in filename for filename in all_split_files)
    assert all("low_voltage" not in filename for filename in all_split_files)
    assert "SIL_motor_missing_online.csv" not in all_split_files
    assert "Real_sensor_bad.csv" not in all_split_files

    assert protocol["protocol_name"] == "rflymad_online_real3_binary_v1"
    assert protocol["task"] == "online_binary"
    assert protocol["label_space"] == "real3"
    assert protocol["source_fault_classes"] == ["motor", "sensor"]
    assert protocol["label_counts"]["train"]
    assert protocol["source_fault_type_counts"]["real_holdout_test"]

    train_csv = pd.read_csv(output_dir / "train.csv")
    assert set(train_csv["label_main_3class"].unique()).issubset({"normal", "motor", "sensor"})
    assert set(train_csv["label_binary"].unique()).issubset({"normal", "fault"})
