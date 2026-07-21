from __future__ import annotations

import csv
from pathlib import Path


from scripts.eval.compare_online_backbones import collect_online_summary, write_outputs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_eval_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_experiment(
    root: Path,
    dirname: str,
    *,
    model: str,
    macro_f1: float,
    f1_fault: float,
    accuracy: float,
    file_recall: float,
    false_alarm_rate_normal: float,
) -> None:
    exp_dir = root / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        exp_dir / "config.json",
        {
            "model": model,
            "task": "online_binary",
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "best_val_f1": macro_f1 - 0.01,
            "epochs_trained": 40,
        },
    )
    _write_json(
        exp_dir / "metrics_test.json",
        {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1,
            "precision": f1_fault,
            "recall": f1_fault,
            "f1_fault": f1_fault,
        },
    )
    _write_eval_csv(
        exp_dir / "telemetry_live_eval.csv",
        [
            {
                "file": "fault_a.csv",
                "gt_has_labels": 1,
                "gt_fault": 1,
                "pred_event_count": 1,
                "hit": 1 if file_recall > 0.5 else 0,
                "miss": 0 if file_recall > 0.5 else 1,
                "false_alarm": 0,
                "pred_fault_by_event": 1 if file_recall > 0.5 else 0,
            },
            {
                "file": "normal_a.csv",
                "gt_has_labels": 1,
                "gt_fault": 0,
                "pred_event_count": 1 if false_alarm_rate_normal > 0 else 0,
                "hit": 0,
                "miss": 0,
                "false_alarm": 1 if false_alarm_rate_normal > 0 else 0,
                "pred_fault_by_event": 1 if false_alarm_rate_normal > 0 else 0,
            },
        ],
    )


def test_compare_online_backbones_writes_ranked_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "summary"

    _write_experiment(
        results_root,
        "online_detector_cnn1d_rfly",
        model="cnn1d",
        macro_f1=0.973,
        f1_fault=0.973,
        accuracy=0.973,
        file_recall=1.0,
        false_alarm_rate_normal=1.0,
    )
    _write_experiment(
        results_root,
        "online_detector_kg_mstca_rfly",
        model="kg_mstca",
        macro_f1=0.978,
        f1_fault=0.979,
        accuracy=0.978,
        file_recall=1.0,
        false_alarm_rate_normal=0.0,
    )

    summary = collect_online_summary(
        results_root,
        {
            "cnn1d_baseline": "online_detector_cnn1d_rfly",
            "kg_mstca_candidate": "online_detector_kg_mstca_rfly",
        },
    )

    assert [row["comparison_label"] for row in summary] == [
        "kg_mstca_candidate",
        "cnn1d_baseline",
    ]
    assert summary[0]["file_recall"] == 1.0
    assert summary[0]["false_alarm_rate_normal"] == 0.0

    outputs = write_outputs(summary, output_dir)
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "kg_mstca_candidate" in markdown
    assert "false_alarm_rate_normal" in markdown
