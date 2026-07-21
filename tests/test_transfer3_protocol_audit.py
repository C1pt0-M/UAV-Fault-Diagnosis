from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


from scripts.eval.transfer3_protocol_audit import (
    build_comparison_tables,
    validate_cache_summary,
    validate_transfer3_split_integrity,
)


def _write_split_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metrics(
    model_dir: Path,
    *,
    test_metrics: dict,
    external_metrics: dict,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics_test.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (model_dir / "metrics_external_test.json").write_text(
        json.dumps(external_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _metrics_payload(
    *,
    accuracy: float,
    balanced_accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    normal: tuple[float, float, float, int],
    motor: tuple[float, float, float, int],
    sensor: tuple[float, float, float, int],
) -> dict:
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "label_names": ["normal", "motor", "sensor"],
        "classification_report": {
            "normal": {
                "precision": normal[0],
                "recall": normal[1],
                "f1-score": normal[2],
                "support": float(normal[3]),
            },
            "motor": {
                "precision": motor[0],
                "recall": motor[1],
                "f1-score": motor[2],
                "support": float(motor[3]),
            },
            "sensor": {
                "precision": sensor[0],
                "recall": sensor[1],
                "f1-score": sensor[2],
                "support": float(sensor[3]),
            },
            "accuracy": accuracy,
            "macro avg": {
                "precision": (normal[0] + motor[0] + sensor[0]) / 3.0,
                "recall": (normal[1] + motor[1] + sensor[1]) / 3.0,
                "f1-score": macro_f1,
                "support": float(normal[3] + motor[3] + sensor[3]),
            },
            "weighted avg": {
                "precision": weighted_f1,
                "recall": weighted_f1,
                "f1-score": weighted_f1,
                "support": float(normal[3] + motor[3] + sensor[3]),
            },
        },
    }


def test_validate_transfer3_split_integrity_and_cache_summary(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    _write_split_csv(
        split_dir / "train.csv",
        [
            {"sample_id": "sil-train-1", "platform": "SIL", "label_main_3class": "motor"},
            {"sample_id": "hil-train-2", "platform": "HIL", "label_main_3class": "sensor"},
        ],
    )
    _write_split_csv(
        split_dir / "val.csv",
        [{"sample_id": "sil-val-1", "platform": "SIL", "label_main_3class": "normal"}],
    )
    _write_split_csv(
        split_dir / "test.csv",
        [{"sample_id": "hil-test-1", "platform": "HIL", "label_main_3class": "motor"}],
    )
    _write_split_csv(
        split_dir / "real_adapt_train.csv",
        [{"sample_id": "real-adapt-train-1", "platform": "Real", "label_main_3class": "sensor"}],
    )
    _write_split_csv(
        split_dir / "real_adapt_val.csv",
        [{"sample_id": "real-adapt-val-1", "platform": "Real", "label_main_3class": "motor"}],
    )
    real_holdout_rows = [
        {"sample_id": "real-holdout-1", "platform": "Real", "label_main_3class": "normal"},
        {"sample_id": "real-holdout-2", "platform": "Real", "label_main_3class": "sensor"},
    ]
    _write_split_csv(split_dir / "real_holdout_test.csv", real_holdout_rows)
    _write_split_csv(split_dir / "external_test.csv", real_holdout_rows)

    integrity = validate_transfer3_split_integrity(
        split_dir,
        expected_counts={
            "train": 2,
            "val": 1,
            "test": 1,
            "real_adapt_train": 1,
            "real_adapt_val": 1,
            "real_holdout_test": 2,
        },
    )

    assert integrity["counts"]["train"] == 2
    assert integrity["counts"]["real_holdout_test"] == 2
    assert integrity["alias_checks"]["external_test_matches_real_holdout_test"] is True
    assert integrity["overlaps"] == {}

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "build_summary.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"saved": 2, "skipped": 0},
                    "real_adapt_train": {"saved": 1, "skipped": 0},
                },
                "train_splits_for_norm": ["train"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cache_summary = validate_cache_summary(cache_dir, expected_train_splits=("train",))
    assert cache_summary["train_splits_for_norm"] == ["train"]
    assert cache_summary["saved_counts"]["train"] == 2


def test_validate_transfer3_split_integrity_rejects_overlap(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    shared_row = {"sample_id": "dup-1", "platform": "SIL", "label_main_3class": "motor"}
    _write_split_csv(split_dir / "train.csv", [shared_row])
    _write_split_csv(split_dir / "val.csv", [shared_row])
    _write_split_csv(
        split_dir / "test.csv",
        [{"sample_id": "test-1", "platform": "HIL", "label_main_3class": "sensor"}],
    )
    _write_split_csv(
        split_dir / "real_adapt_train.csv",
        [{"sample_id": "real-train-1", "platform": "Real", "label_main_3class": "sensor"}],
    )
    _write_split_csv(
        split_dir / "real_adapt_val.csv",
        [{"sample_id": "real-val-1", "platform": "Real", "label_main_3class": "motor"}],
    )
    _write_split_csv(
        split_dir / "real_holdout_test.csv",
        [{"sample_id": "real-test-1", "platform": "Real", "label_main_3class": "normal"}],
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_transfer3_split_integrity(split_dir)


def test_build_comparison_tables_calculates_metric_deltas(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    audit_root = tmp_path / "audit"

    _write_metrics(
        legacy_root / "transfer3_cnnlstm_baseline_v1",
        test_metrics=_metrics_payload(
            accuracy=0.99,
            balanced_accuracy=0.98,
            macro_f1=0.97,
            weighted_f1=0.98,
            normal=(1.0, 0.9, 0.95, 10),
            motor=(0.98, 1.0, 0.99, 20),
            sensor=(0.96, 1.0, 0.98, 15),
        ),
        external_metrics=_metrics_payload(
            accuracy=0.41,
            balanced_accuracy=0.34,
            macro_f1=0.24,
            weighted_f1=0.31,
            normal=(0.0, 0.0, 0.0, 6),
            motor=(0.48, 0.10, 0.17, 12),
            sensor=(0.40, 0.91, 0.56, 9),
        ),
    )
    _write_metrics(
        audit_root / "transfer3_cnnlstm_baseline_v1",
        test_metrics=_metrics_payload(
            accuracy=0.98,
            balanced_accuracy=0.97,
            macro_f1=0.96,
            weighted_f1=0.97,
            normal=(0.95, 0.9, 0.92, 10),
            motor=(0.97, 0.98, 0.97, 20),
            sensor=(0.94, 0.98, 0.96, 15),
        ),
        external_metrics=_metrics_payload(
            accuracy=0.35,
            balanced_accuracy=0.30,
            macro_f1=0.20,
            weighted_f1=0.28,
            normal=(0.0, 0.0, 0.0, 6),
            motor=(0.40, 0.08, 0.13, 12),
            sensor=(0.37, 0.82, 0.51, 9),
        ),
    )

    model_df, class_df = build_comparison_tables(legacy_root, audit_root)

    baseline_row = model_df.loc[
        model_df["experiment_name"] == "transfer3_cnnlstm_baseline_v1"
    ].iloc[0]
    assert baseline_row["test_accuracy_legacy"] == pytest.approx(0.99)
    assert baseline_row["test_accuracy_audit"] == pytest.approx(0.98)
    assert baseline_row["test_accuracy_delta"] == pytest.approx(-0.01)
    assert baseline_row["real_holdout_test_macro_f1_delta"] == pytest.approx(-0.04)

    normal_real_row = class_df.loc[
        (class_df["experiment_name"] == "transfer3_cnnlstm_baseline_v1")
        & (class_df["split"] == "real_holdout_test")
        & (class_df["class_name"] == "motor")
    ].iloc[0]
    assert normal_real_row["legacy_recall"] == pytest.approx(0.10)
    assert normal_real_row["audit_recall"] == pytest.approx(0.08)
    assert normal_real_row["delta_recall"] == pytest.approx(-0.02)
