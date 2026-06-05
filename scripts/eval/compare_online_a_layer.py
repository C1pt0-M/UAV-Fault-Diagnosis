#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_EXPERIMENTS = [
    "A1_online_cnn1d_baseline_v1",
    "A2_online_cnn1d_fewshot_v1",
]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _extract_recall(metrics: dict, class_name: str) -> float | None:
    report = metrics.get("classification_report", {})
    class_report = report.get(class_name, {}) if isinstance(report, dict) else {}
    if not isinstance(class_report, dict):
        return None
    value = class_report.get("recall")
    return float(value) if value is not None else None


def _resolve_experiment_dir(results_root: Path, experiment_id: str) -> Path:
    direct = results_root / experiment_id
    if direct.exists():
        return direct
    matches = sorted(path for path in results_root.rglob(experiment_id) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"experiment dir not found under {results_root}: {experiment_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple experiment dirs matched {experiment_id}: {matches}")
    return matches[0]


def collect_online_a_layer_summary(results_root: Path, experiment_ids: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment_id in experiment_ids:
        exp_dir = _resolve_experiment_dir(results_root, experiment_id)
        config = _read_json(exp_dir / "config.json")
        metrics_test = _read_json(exp_dir / "metrics_test.json")
        metrics_external = _read_json(exp_dir / "metrics_external_test.json")
        rows.append(
            {
                "experiment_id": experiment_id,
                "model": config.get("model", ""),
                "protocol": config.get("mode", ""),
                "experiment_dir": exp_dir.name,
                "epochs_trained": config.get("epochs_trained"),
                "best_val_f1": config.get("best_val_f1"),
                "test_accuracy": metrics_test.get("accuracy"),
                "test_macro_f1": metrics_test.get("macro_f1"),
                "test_f1_fault": metrics_test.get("f1_fault"),
                "real_accuracy": metrics_external.get("accuracy"),
                "real_macro_f1": metrics_external.get("macro_f1"),
                "real_f1_fault": metrics_external.get("f1_fault"),
                "normal_recall": _extract_recall(metrics_external, "normal"),
                "fault_recall": _extract_recall(metrics_external, "fault"),
            }
        )

    df = pd.DataFrame(rows)
    df["experiment_order"] = df["experiment_id"].map({name: idx for idx, name in enumerate(experiment_ids)})
    df = df.sort_values("experiment_order", kind="stable").drop(columns=["experiment_order"]).reset_index(drop=True)

    baseline = df.loc[df["protocol"] == "baseline"]
    if baseline.empty:
        raise ValueError("baseline experiment missing from online A-layer summary")
    baseline_row = baseline.iloc[0]
    for metric in ("real_accuracy", "real_macro_f1", "real_f1_fault", "normal_recall", "fault_recall"):
        df[f"{metric}_delta_vs_a1"] = df[metric] - baseline_row[metric]
    return df


def _build_manifest(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        ["experiment_id", "model", "protocol", "experiment_dir", "epochs_trained", "best_val_f1"]
    ].copy()


def _conclusion(summary_df: pd.DataFrame) -> str:
    baseline = summary_df.loc[summary_df["protocol"] == "baseline"].iloc[0]
    fewshot = summary_df.loc[summary_df["protocol"] == "fewshot"].iloc[0]
    if (
        fewshot["real_macro_f1"] > baseline["real_macro_f1"]
        and fewshot["real_f1_fault"] > baseline["real_f1_fault"]
        and fewshot["normal_recall"] > baseline["normal_recall"]
    ):
        return "few-shot Real 成立"
    return "few-shot Real 暂不成立"


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def write_outputs(summary_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "online_a_layer_summary.csv"
    summary_md = output_dir / "online_a_layer_summary.md"
    manifest_csv = output_dir / "online_a_layer_manifest.csv"

    summary_df.to_csv(summary_csv, index=False)
    _build_manifest(summary_df).to_csv(manifest_csv, index=False)

    conclusion = _conclusion(summary_df)
    lines = [
        "# online A层汇总",
        "",
        f"- 结论：{conclusion}",
        "",
        _render_table(summary_df),
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {"summary_csv": summary_csv, "summary_md": summary_md, "manifest_csv": manifest_csv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize online A-layer experiments")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", action="append", help="Explicit experiment ids, default A1/A2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_ids = args.experiment if args.experiment else list(DEFAULT_EXPERIMENTS)
    summary_df = collect_online_a_layer_summary(Path(args.results_root), experiment_ids)
    outputs = write_outputs(summary_df, Path(args.output_dir))
    print(f"[OK] wrote: {outputs['summary_csv']}")
    print(f"[OK] wrote: {outputs['summary_md']}")
    print(f"[OK] wrote: {outputs['manifest_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
