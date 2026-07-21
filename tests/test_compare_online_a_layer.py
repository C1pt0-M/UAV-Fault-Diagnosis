from __future__ import annotations

from pathlib import Path

import pandas as pd


from scripts.eval.compare_online_a_layer import write_outputs


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_result_dir(
    root: Path,
    dirname: str,
    *,
    mode: str,
    test_accuracy: float,
    test_macro_f1: float,
    test_f1_fault: float,
    real_accuracy: float,
    real_macro_f1: float,
    real_f1_fault: float,
    normal_recall: float,
    fault_recall: float,
) -> None:
    exp_dir = root / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        exp_dir / "config.json",
        {
            "model": "cnn1d",
            "mode": mode,
            "label_names": ["normal", "fault"],
            "best_val_f1": 0.95,
            "epochs_trained": 40,
        },
    )
    _write_json(
        exp_dir / "metrics_test.json",
        {
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1,
            "f1_fault": test_f1_fault,
        },
    )
    _write_json(
        exp_dir / "metrics_external_test.json",
        {
            "accuracy": real_accuracy,
            "macro_f1": real_macro_f1,
            "f1_fault": real_f1_fault,
            "classification_report": {
                "normal": {"recall": normal_recall},
                "fault": {"recall": fault_recall},
            },
        },
    )


def test_compare_online_a_layer_writes_summary_and_conclusion(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "summary"

    _write_result_dir(
        results_root,
        "A1_online_cnn1d_baseline_v1",
        mode="baseline",
        test_accuracy=0.96,
        test_macro_f1=0.96,
        test_f1_fault=0.96,
        real_accuracy=0.81,
        real_macro_f1=0.79,
        real_f1_fault=0.82,
        normal_recall=0.60,
        fault_recall=0.84,
    )
    _write_result_dir(
        results_root,
        "A2_online_cnn1d_fewshot_v1",
        mode="fewshot",
        test_accuracy=0.97,
        test_macro_f1=0.97,
        test_f1_fault=0.97,
        real_accuracy=0.88,
        real_macro_f1=0.87,
        real_f1_fault=0.89,
        normal_recall=0.76,
        fault_recall=0.90,
    )

    summary_df = pd.DataFrame(
        [
            {
                "experiment_id": "A1_online_cnn1d_baseline_v1",
                "model": "cnn1d",
                "protocol": "baseline",
                "experiment_dir": "A1_online_cnn1d_baseline_v1",
                "epochs_trained": 40,
                "best_val_f1": 0.95,
                "test_accuracy": 0.96,
                "test_macro_f1": 0.96,
                "test_f1_fault": 0.96,
                "real_accuracy": 0.81,
                "real_macro_f1": 0.79,
                "real_f1_fault": 0.82,
                "normal_recall": 0.60,
                "fault_recall": 0.84,
            },
            {
                "experiment_id": "A2_online_cnn1d_fewshot_v1",
                "model": "cnn1d",
                "protocol": "fewshot",
                "experiment_dir": "A2_online_cnn1d_fewshot_v1",
                "epochs_trained": 40,
                "best_val_f1": 0.95,
                "test_accuracy": 0.97,
                "test_macro_f1": 0.97,
                "test_f1_fault": 0.97,
                "real_accuracy": 0.88,
                "real_macro_f1": 0.87,
                "real_f1_fault": 0.89,
                "normal_recall": 0.76,
                "fault_recall": 0.90,
            },
        ]
    )

    outputs = write_outputs(summary_df, output_dir)
    assert outputs["summary_csv"].exists()
    assert outputs["summary_md"].exists()
    assert outputs["manifest_csv"].exists()

    markdown = outputs["summary_md"].read_text(encoding="utf-8")
    assert "few-shot Real 成立" in markdown
    assert "A2_online_cnn1d_fewshot_v1" in markdown
