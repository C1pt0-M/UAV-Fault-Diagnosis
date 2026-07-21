from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


from uav_fault.data.build_online_a_layer_splits import build_online_a_layer_splits


def _write_manifest(path: Path) -> None:
    rows = []
    for idx, name in enumerate(["dev_train_a.csv", "dev_val_a.csv", "dev_test_a.csv"]):
        rows.append(
            {
                "sample_id": f"dev_{idx}",
                "filename": name,
                "source_csv_path": f"D:/fake/{name}",
                "platform": "SIL" if idx != 1 else "HIL",
                "flight_mode": "hover",
                "label_binary": "fault" if idx != 1 else "normal",
                "label_binary_id": 1 if idx != 1 else 0,
                "label_main_6class": "motor" if idx != 1 else "normal",
                "is_in_main_6class": 1,
                "has_required_online_columns": 1,
                "quality_flag": "ok",
            }
        )

    for idx in range(10):
        rows.append(
            {
                "sample_id": f"real_{idx}",
                "filename": f"real_{idx}.csv",
                "source_csv_path": f"D:/real/real_{idx}.csv",
                "platform": "Real",
                "flight_mode": "hover" if idx < 5 else "cruise",
                "label_binary": "fault" if idx < 8 else "normal",
                "label_binary_id": 1 if idx < 8 else 0,
                "label_main_6class": "motor" if idx < 8 else "normal",
                "is_in_main_6class": 1,
                "has_required_online_columns": 1,
                "quality_flag": "ok",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_online_a_layer_splits_writes_baseline_and_fewshot_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest_v2.csv"
    dev_splits_path = tmp_path / "splits_v2.json"
    output_dir = tmp_path / "online_a_splits"

    _write_manifest(manifest_path)
    dev_splits_path.write_text(
        json.dumps(
            {
                "train": ["dev_train_a.csv"],
                "val": ["dev_val_a.csv"],
                "test": ["dev_test_a.csv"],
            }
        ),
        encoding="utf-8",
    )

    build_online_a_layer_splits(
        manifest_path=str(manifest_path),
        dev_splits_path=str(dev_splits_path),
        output_dir=str(output_dir),
        seed=42,
        real_train_ratio=0.2,
        real_val_ratio=0.1,
        real_holdout_ratio=0.7,
    )

    baseline = json.loads((output_dir / "splits_online_a_baseline_v1.json").read_text(encoding="utf-8"))
    fewshot = json.loads((output_dir / "splits_online_a_fewshot_v1.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "protocol_online_a_v1.json").read_text(encoding="utf-8"))

    assert baseline["train"] == ["dev_train_a.csv"]
    assert baseline["val"] == ["dev_val_a.csv"]
    assert baseline["test"] == ["dev_test_a.csv"]
    assert "external_test" in baseline

    assert fewshot["test"] == ["dev_test_a.csv"]
    assert "external_test" in fewshot
    assert len(fewshot["train"]) > len(baseline["train"])
    assert len(fewshot["val"]) > len(baseline["val"])

    assert summary["real_counts"]["real_adapt_train"] == 2
    assert summary["real_counts"]["real_adapt_val"] == 1
    assert summary["real_counts"]["real_holdout_test"] == 7
