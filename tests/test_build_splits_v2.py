from pathlib import Path
import json

import pandas as pd


from uav_fault.data.build_splits_v2 import build_splits_v2


def test_build_splits_v2_creates_main_and_external_protocol(tmp_path: Path):
    manifest_path = tmp_path / "manifest_v2.csv"
    output_dir = tmp_path / "splits"

    rows = []
    # SIL + HIL: 12 dev samples, balanced over 6 classes
    classes = [
        ("normal", 0),
        ("motor", 1),
        ("propeller", 2),
        ("sensor", 3),
        ("wind", 4),
        ("load", 5),
    ]
    idx = 0
    for platform in ["SIL", "HIL"]:
        for label_name, label_id in classes:
            rows.append(
                {
                    "sample_id": f"{platform.lower()}-{label_name}-a",
                    "filename": f"Case_dev_{idx:03d}.csv",
                    "platform": platform,
                    "label_main_6class": label_name,
                    "label_main_6class_id": label_id,
                    "is_in_main_6class": 1,
                    "label_binary": "normal" if label_name == "normal" else "fault",
                    "label_binary_id": 0 if label_name == "normal" else 1,
                    "has_required_online_columns": 1,
                    "quality_flag": "ok",
                }
            )
            idx += 1

    # Real: 4 external samples
    for i, (label_name, label_id) in enumerate(classes[:4]):
        rows.append(
            {
                "sample_id": f"real-{label_name}-{i}",
                "filename": f"Case_real_{i:03d}.csv",
                "platform": "Real",
                "label_main_6class": label_name,
                "label_main_6class_id": label_id,
                "is_in_main_6class": 1,
                "label_binary": "normal" if label_name == "normal" else "fault",
                "label_binary_id": 0 if label_name == "normal" else 1,
                "has_required_online_columns": 1,
                "quality_flag": "ok",
            }
        )

    # one low_voltage sample kept in manifest but excluded from main_6class
    rows.append(
        {
            "sample_id": "sil-low-voltage",
            "filename": "Case_low_voltage.csv",
            "platform": "SIL",
            "label_main_6class": None,
            "label_main_6class_id": None,
            "is_in_main_6class": 0,
            "label_binary": "fault",
            "label_binary_id": 1,
            "has_required_online_columns": 1,
            "quality_flag": "ok",
        }
    )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    build_splits_v2(manifest_path=manifest_path, output_dir=output_dir, seed=7)

    protocol = json.loads((output_dir / "protocol_main_v1.json").read_text())
    assert protocol["protocol_name"] == "rflymad_main_v1"
    assert protocol["split_basis"]["external_test_domains"] == ["Real"]
    assert set(protocol["tasks"].keys()) == {"main_6class", "online_binary"}

    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "val.csv")
    test_df = pd.read_csv(output_dir / "test.csv")
    external_df = pd.read_csv(output_dir / "external_test.csv")

    assert set(train_df["platform"]).issubset({"SIL", "HIL"})
    assert set(val_df["platform"]).issubset({"SIL", "HIL"})
    assert set(test_df["platform"]).issubset({"SIL", "HIL"})
    assert set(external_df["platform"]) == {"Real"}

    dev_ids = set(train_df["sample_id"]) | set(val_df["sample_id"]) | set(test_df["sample_id"])
    assert "sil-low-voltage" not in dev_ids
    assert set(train_df["sample_id"]).isdisjoint(set(val_df["sample_id"]))
    assert set(train_df["sample_id"]).isdisjoint(set(test_df["sample_id"]))
    assert set(val_df["sample_id"]).isdisjoint(set(test_df["sample_id"]))

    splits_json = json.loads((output_dir / "splits_v2.json").read_text())
    assert set(splits_json.keys()) == {"train", "val", "test", "external_test"}


def test_build_splits_v2_requires_manifest_columns(tmp_path: Path):
    manifest_path = tmp_path / "manifest_v2.csv"
    output_dir = tmp_path / "splits"
    pd.DataFrame([{"sample_id": "x"}]).to_csv(manifest_path, index=False)

    try:
        build_splits_v2(manifest_path=manifest_path, output_dir=output_dir, seed=42)
    except ValueError as exc:
        assert "manifest missing required columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing manifest columns")
