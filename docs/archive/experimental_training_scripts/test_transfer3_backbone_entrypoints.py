from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.eval.compare_transfer3_backbones import write_outputs


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _find_add_argument_choices(tree: ast.Module, option_name: str) -> list[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != option_name:
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return list(ast.literal_eval(kw.value))
    raise AssertionError(f"choices for {option_name} not found")


def test_train_transfer3_supports_lstm_backbone() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "train_transfer3.py"
    tree = _parse_module(module_path)

    model_choices = _find_add_argument_choices(tree, "--model")
    assert "lstm" in model_choices

    build_model = _find_function(tree, "build_model")
    constants = {node.value for node in ast.walk(build_model) if isinstance(node, ast.Constant)}
    names = {node.id for node in ast.walk(build_model) if isinstance(node, ast.Name)}

    assert "lstm" in constants
    assert "BiLSTMClassifier" in names


def test_compare_transfer3_backbones_writes_expected_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "results_transfer3_compare"
    output_dir = tmp_path / "summary"
    results_root.mkdir()
    (results_root / "transfer3_textcnn_results").mkdir()
    (results_root / "transfer3_lstm_results").mkdir()
    (results_root / "transfer3_cnnlstm_results").mkdir()

    experiments = {
        "single_cnn_focal": ("transfer3_textcnn_results", "transfer3_textcnn_fewshot_v1", 0.70, 0.79),
        "single_lstm_focal": ("transfer3_lstm_results", "transfer3_lstm_fewshot_v1", 0.68, 0.76),
        "cnn_lstm_focal": ("transfer3_cnnlstm_results", "transfer3_cnnlstm_fewshot_v1", 0.75, 0.84),
    }

    for label, (group_dir, dirname, ext_macro_f1, ext_acc) in experiments.items():
        exp_dir = results_root / group_dir / dirname
        exp_dir.mkdir()
        (exp_dir / "config.json").write_text(
            (
                "{\n"
                f'  "model": "{label}",\n'
                '  "mode": "fewshot",\n'
                '  "label_names": ["normal", "motor", "sensor"],\n'
                '  "best_val_f1": 0.95,\n'
                '  "epochs_trained": 60\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        (exp_dir / "metrics_test.json").write_text(
            (
                "{\n"
                '  "accuracy": 0.99,\n'
                '  "balanced_accuracy": 0.98,\n'
                '  "macro_f1": 0.98,\n'
                '  "weighted_f1": 0.99\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        (exp_dir / "metrics_external_test.json").write_text(
            (
                "{\n"
                f'  "accuracy": {ext_acc},\n'
                '  "balanced_accuracy": 0.70,\n'
                f'  "macro_f1": {ext_macro_f1},\n'
                '  "weighted_f1": 0.78\n'
                "}\n"
            ),
            encoding="utf-8",
        )

    script_path = PROJECT_ROOT / "scripts" / "eval" / "compare_transfer3_backbones.py"
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

    summary_csv = output_dir / "transfer3_backbone_summary.csv"
    ranking_md = output_dir / "transfer3_backbone_ranking.md"
    assert summary_csv.exists()
    assert ranking_md.exists()

    df = pd.read_csv(summary_csv)
    assert df["comparison_label"].tolist() == [
        "cnn_lstm_focal",
        "single_cnn_focal",
        "single_lstm_focal",
    ]
    assert df.iloc[0]["experiment_dir"] == "transfer3_cnnlstm_fewshot_v1"
    assert df.iloc[0]["real_macro_f1"] == 0.75


def test_compare_transfer3_backbones_falls_back_without_tabulate(monkeypatch, tmp_path: Path) -> None:
    summary_df = pd.DataFrame(
        [
            {
                "comparison_label": "cnn_lstm_focal",
                "experiment_dir": "exp",
                "model": "cnn_lstm",
                "mode": "fewshot",
                "epochs_trained": 10,
                "best_val_f1": 0.9,
                "test_accuracy": 0.95,
                "test_balanced_accuracy": 0.94,
                "test_macro_f1": 0.93,
                "real_accuracy": 0.8,
                "real_balanced_accuracy": 0.78,
                "real_macro_f1": 0.77,
            }
        ]
    )

    def _raise_import_error(*args, **kwargs):
        raise ImportError("Missing optional dependency 'tabulate'")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", _raise_import_error)

    outputs = write_outputs(summary_df, tmp_path)
    content = outputs["markdown"].read_text(encoding="utf-8")
    assert "comparison_label" in content
    assert "cnn_lstm_focal" in content
