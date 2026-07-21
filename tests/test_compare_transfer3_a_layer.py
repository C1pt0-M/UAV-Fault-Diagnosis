from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.eval.compare_transfer3_a_layer import write_outputs


def _write_result_dir(
    root: Path,
    dirname: str,
    *,
    mode: str,
    test_accuracy: float,
    test_balanced_accuracy: float,
    test_macro_f1: float,
    real_accuracy: float,
    real_balanced_accuracy: float,
    real_macro_f1: float,
    normal_recall: float,
    motor_recall: float,
    sensor_recall: float,
) -> None:
    exp_dir = root / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(
        (
            "{\n"
            '  "model": "lstm",\n'
            f'  "mode": "{mode}",\n'
            '  "label_names": ["normal", "motor", "sensor"],\n'
            '  "best_val_f1": 0.95,\n'
            '  "epochs_trained": 50\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (exp_dir / "metrics_test.json").write_text(
        (
            "{\n"
            f'  "accuracy": {test_accuracy},\n'
            f'  "balanced_accuracy": {test_balanced_accuracy},\n'
            f'  "macro_f1": {test_macro_f1}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (exp_dir / "metrics_external_test.json").write_text(
        (
            "{\n"
            f'  "accuracy": {real_accuracy},\n'
            f'  "balanced_accuracy": {real_balanced_accuracy},\n'
            f'  "macro_f1": {real_macro_f1},\n'
            '  "classification_report": {\n'
            f'    "normal": {{"recall": {normal_recall}}},\n'
            f'    "motor": {{"recall": {motor_recall}}},\n'
            f'    "sensor": {{"recall": {sensor_recall}}}\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_compare_transfer3_a_layer_writes_summary_and_conclusion(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "summary"

    _write_result_dir(
        results_root,
        "A1_bilstm_focal_baseline_v1",
        mode="baseline",
        test_accuracy=0.94,
        test_balanced_accuracy=0.93,
        test_macro_f1=0.92,
        real_accuracy=0.41,
        real_balanced_accuracy=0.34,
        real_macro_f1=0.24,
        normal_recall=0.10,
        motor_recall=0.52,
        sensor_recall=0.40,
    )
    _write_result_dir(
        results_root,
        "A2_bilstm_focal_fewshot_v1",
        mode="fewshot",
        test_accuracy=0.95,
        test_balanced_accuracy=0.94,
        test_macro_f1=0.93,
        real_accuracy=0.79,
        real_balanced_accuracy=0.71,
        real_macro_f1=0.70,
        normal_recall=0.48,
        motor_recall=0.85,
        sensor_recall=0.80,
    )

    script_path = PROJECT_ROOT / "scripts" / "eval" / "compare_transfer3_a_layer.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    csv_path = output_dir / "a_layer_summary.csv"
    md_path = output_dir / "a_layer_summary.md"
    manifest_path = output_dir / "experiment_manifest.csv"
    assert csv_path.exists()
    assert md_path.exists()
    assert manifest_path.exists()

    df = pd.read_csv(csv_path)
    assert df["experiment_id"].tolist() == [
        "A1_bilstm_focal_baseline_v1",
        "A2_bilstm_focal_fewshot_v1",
    ]
    assert df.loc[df["experiment_id"] == "A2_bilstm_focal_fewshot_v1", "real_macro_f1_delta_vs_a1"].iloc[0] == pytest.approx(0.46)
    assert df.loc[df["experiment_id"] == "A2_bilstm_focal_fewshot_v1", "normal_recall_delta_vs_a1"].iloc[0] == pytest.approx(0.38)

    markdown = md_path.read_text(encoding="utf-8")
    assert "few-shot Real 成立" in markdown
    assert "A2_bilstm_focal_fewshot_v1" in markdown


def test_compare_transfer3_a_layer_falls_back_without_tabulate(monkeypatch, tmp_path: Path) -> None:
    summary_df = pd.DataFrame(
        [
            {
                "experiment_id": "A1_bilstm_focal_baseline_v1",
                "model": "lstm",
                "protocol": "baseline",
                "experiment_dir": "A1_bilstm_focal_baseline_v1",
                "epochs_trained": 40,
                "best_val_f1": 0.9,
                "test_accuracy": 0.9,
                "test_balanced_accuracy": 0.88,
                "test_macro_f1": 0.87,
                "real_accuracy": 0.4,
                "real_balanced_accuracy": 0.35,
                "real_macro_f1": 0.3,
                "normal_recall": 0.1,
                "motor_recall": 0.4,
                "sensor_recall": 0.5,
                "real_accuracy_delta_vs_a1": 0.0,
                "real_balanced_accuracy_delta_vs_a1": 0.0,
                "real_macro_f1_delta_vs_a1": 0.0,
                "normal_recall_delta_vs_a1": 0.0,
                "motor_recall_delta_vs_a1": 0.0,
                "sensor_recall_delta_vs_a1": 0.0,
            }
            ,
            {
                "experiment_id": "A2_bilstm_focal_fewshot_v1",
                "model": "lstm",
                "protocol": "fewshot",
                "experiment_dir": "A2_bilstm_focal_fewshot_v1",
                "epochs_trained": 45,
                "best_val_f1": 0.92,
                "test_accuracy": 0.93,
                "test_balanced_accuracy": 0.91,
                "test_macro_f1": 0.9,
                "real_accuracy": 0.75,
                "real_balanced_accuracy": 0.7,
                "real_macro_f1": 0.68,
                "normal_recall": 0.45,
                "motor_recall": 0.82,
                "sensor_recall": 0.74,
                "real_accuracy_delta_vs_a1": 0.35,
                "real_balanced_accuracy_delta_vs_a1": 0.35,
                "real_macro_f1_delta_vs_a1": 0.38,
                "normal_recall_delta_vs_a1": 0.35,
                "motor_recall_delta_vs_a1": 0.42,
                "sensor_recall_delta_vs_a1": 0.24,
            }
        ]
    )

    def _raise_import_error(*args, **kwargs):
        raise ImportError("Missing optional dependency 'tabulate'")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", _raise_import_error)

    outputs = write_outputs(summary_df, tmp_path)
    content = outputs["summary_md"].read_text(encoding="utf-8")
    assert "transfer3 A层汇总" in content
    assert "A1_bilstm_focal_baseline_v1" in content
