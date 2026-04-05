#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS = {
    "cnn1d_baseline": "online_detector_cnn1d_rfly",
    "bilstm_candidate": "online_detector_bilstm",
    "transformer_candidate": "online_detector_transformer",
    "mstca_candidate": "online_detector_mstca",
    "kg_mstca_candidate": "online_detector_kg_mstca_rfly",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    matches = sorted(path for path in results_root.rglob(dirname) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"experiment dir not found under {results_root}: {dirname}")
    if len(matches) > 1:
        raise ValueError(f"multiple experiment dirs matched {dirname}: {matches}")
    return matches[0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _event_metrics_from_csv(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "file_recall": None,
            "hit_rate": None,
            "false_alarm_rate_normal": None,
            "false_alarm_count": None,
        }

    rows = _read_csv_rows(path)
    labeled = [row for row in rows if str(row.get("gt_has_labels", "")).strip() == "1"]
    if not labeled:
        return {
            "file_recall": None,
            "hit_rate": None,
            "false_alarm_rate_normal": None,
            "false_alarm_count": None,
        }

    positives = [row for row in labeled if str(row.get("gt_fault", "")).strip() == "1"]
    negatives = [row for row in labeled if str(row.get("gt_fault", "")).strip() == "0"]

    tp = sum(1 for row in positives if int(float(row.get("pred_fault_by_event", 0) or 0)) == 1)
    hits = sum(1 for row in positives if int(float(row.get("hit", 0) or 0)) == 1)
    false_alarms = sum(1 for row in negatives if int(float(row.get("false_alarm", 0) or 0)) == 1)

    file_recall = tp / len(positives) if positives else None
    hit_rate = hits / len(positives) if positives else None
    false_alarm_rate_normal = false_alarms / len(negatives) if negatives else None
    return {
        "file_recall": file_recall,
        "hit_rate": hit_rate,
        "false_alarm_rate_normal": false_alarm_rate_normal,
        "false_alarm_count": int(false_alarms),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    f1_fault = _safe_float(row.get("f1_fault")) or -1.0
    macro_f1 = _safe_float(row.get("macro_f1")) or -1.0
    accuracy = _safe_float(row.get("accuracy")) or -1.0
    file_recall = _safe_float(row.get("file_recall")) or -1.0
    false_alarm = _safe_float(row.get("false_alarm_rate_normal"))
    false_alarm_key = false_alarm if false_alarm is not None else 999.0
    return (-f1_fault, -macro_f1, -accuracy, -file_recall, false_alarm_key)


def collect_online_summary(results_root: Path, experiments: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_label, dirname in experiments.items():
        exp_dir = _resolve_experiment_dir(results_root, dirname)
        config = _read_json(exp_dir / "config.json")
        metrics_test = _read_json(exp_dir / "metrics_test.json")
        event_metrics = _event_metrics_from_csv(exp_dir / "telemetry_live_eval.csv")
        rows.append(
            {
                "comparison_label": comparison_label,
                "experiment_dir": dirname,
                "model": config.get("model", ""),
                "epochs_trained": config.get("epochs_trained"),
                "best_val_f1": config.get("best_val_f1"),
                "accuracy": metrics_test.get("accuracy"),
                "macro_f1": metrics_test.get("macro_f1"),
                "weighted_f1": metrics_test.get("weighted_f1"),
                "precision": metrics_test.get("precision"),
                "recall": metrics_test.get("recall"),
                "f1_fault": metrics_test.get("f1_fault"),
                **event_metrics,
            }
        )
    return sorted(rows, key=_rank_key)


def _format_value(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    return f"{numeric:.4f}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, (int, float)) or _safe_float(value) is not None:
                values.append(_format_value(value))
            else:
                values.append(str(value or ""))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_outputs(summary_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "online_backbone_summary.csv"
    md_path = output_dir / "online_backbone_ranking.md"

    columns = [
        "comparison_label",
        "experiment_dir",
        "model",
        "epochs_trained",
        "best_val_f1",
        "accuracy",
        "macro_f1",
        "f1_fault",
        "file_recall",
        "hit_rate",
        "false_alarm_rate_normal",
        "false_alarm_count",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in columns})

    lines = [
        "# Online Backbone Comparison",
        "",
        "Ranked by `f1_fault -> macro_f1 -> accuracy -> file_recall -> false_alarm_rate_normal`.",
        "",
        _markdown_table(summary_rows, columns),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare online backbone results")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment",
        action="append",
        help="label=dirname mapping. Defaults to common online model dirs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    experiments = _parse_experiment_mapping(args.experiment)
    summary_rows = collect_online_summary(results_root, experiments)
    outputs = write_outputs(summary_rows, output_dir)
    print(f"[OK] wrote: {outputs['csv']}")
    print(f"[OK] wrote: {outputs['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
