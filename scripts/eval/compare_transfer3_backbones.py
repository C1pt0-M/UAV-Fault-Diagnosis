#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_EXPERIMENTS = {
    "single_cnn_focal": "transfer3_textcnn_fewshot_v1",
    "single_lstm_focal": "transfer3_lstm_fewshot_v1",
    "cnn_lstm_focal": "transfer3_cnnlstm_fewshot_v1",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _parse_experiment_mapping(raw_items: list[str] | None) -> dict[str, str]:
    if not raw_items:
        return dict(DEFAULT_EXPERIMENTS)
    mapping: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"invalid experiment mapping, expected label=dir: {item}")
        label, dirname = item.split("=", 1)
        label = label.strip()
        dirname = dirname.strip()
        if not label or not dirname:
            raise ValueError(f"invalid experiment mapping, expected label=dir: {item}")
        mapping[label] = dirname
    return mapping


def _resolve_experiment_dir(results_root: Path, dirname: str) -> Path:
    direct = results_root / dirname
    if direct.exists():
        return direct
    matches = sorted(
        path for path in results_root.rglob(dirname) if path.is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"experiment dir not found under {results_root}: {dirname}")
    if len(matches) > 1:
        raise ValueError(f"multiple experiment dirs matched {dirname}: {matches}")
    return matches[0]


def collect_backbone_summary(results_root: Path, experiments: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison_label, dirname in experiments.items():
        exp_dir = _resolve_experiment_dir(results_root, dirname)
        config = _read_json(exp_dir / "config.json")
        metrics_test = _read_json(exp_dir / "metrics_test.json")
        metrics_external = _read_json(exp_dir / "metrics_external_test.json")
        rows.append(
            {
                "comparison_label": comparison_label,
                "experiment_dir": dirname,
                "model": config.get("model", ""),
                "mode": config.get("mode", ""),
                "epochs_trained": config.get("epochs_trained"),
                "best_val_f1": config.get("best_val_f1"),
                "test_accuracy": metrics_test.get("accuracy"),
                "test_balanced_accuracy": metrics_test.get("balanced_accuracy"),
                "test_macro_f1": metrics_test.get("macro_f1"),
                "real_accuracy": metrics_external.get("accuracy"),
                "real_balanced_accuracy": metrics_external.get("balanced_accuracy"),
                "real_macro_f1": metrics_external.get("macro_f1"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["real_macro_f1", "real_accuracy", "test_macro_f1"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def write_outputs(summary_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transfer3_backbone_summary.csv"
    md_path = output_dir / "transfer3_backbone_ranking.md"
    summary_df.to_csv(csv_path, index=False)

    lines = [
        "# transfer3 Backbone Comparison",
        "",
        "按 `Real Macro F1 -> Real Accuracy -> Test Macro F1` 排序。",
        "",
        _render_table(summary_df),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 transfer3 backbone 结果")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment",
        action="append",
        help="label=dirname，默认比较 single_cnn_focal/single_lstm_focal/cnn_lstm_focal",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    experiments = _parse_experiment_mapping(args.experiment)
    summary_df = collect_backbone_summary(results_root, experiments)
    outputs = write_outputs(summary_df, output_dir)
    print(f"[OK] wrote: {outputs['csv']}")
    print(f"[OK] wrote: {outputs['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
