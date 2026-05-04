#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "competition" / "2026-software-app"
METRICS_DIR = OUTPUT_ROOT / "metrics"


OFFLINE_MODELS = {
    "cnn_lstm": REPO_ROOT / "assets" / "models" / "cnn_lstm",
    "textcnn": REPO_ROOT / "assets" / "models" / "textcnn",
    "transformer": REPO_ROOT / "assets" / "models" / "transformer",
}

ONLINE_MODELS = {
    "cnn_legacy": REPO_ROOT / "assets" / "models" / "online_detector_CNN",
    "cnn1d_rfly": REPO_ROOT / "assets" / "models" / "online_detector_cnn1d_rfly",
}

SELECTED_EVENT_LOGS = {
    "normal_baseline": REPO_ROOT
    / "logs"
    / "telemetry_live_eval_rfly_cnn1d_normal_full_20260320.csv",
    "external_fault_test": REPO_ROOT
    / "logs"
    / "telemetry_live_eval_rfly_cnn1d_external_test_fixed.csv",
}


@dataclass(frozen=True)
class EventDatasetSummary:
    dataset: str
    csv_name: str
    rows_ok: int
    actual_fault_files: int
    actual_normal_files: int
    tp: int
    fp: int
    tn: int
    fn: int
    file_accuracy: float
    file_precision: float | None
    file_recall: float | None
    file_f1: float | None
    false_alarm_rate_all: float
    false_alarm_rate_normal: float | None
    hit_rate: float | None
    hit_count: int
    miss_count: int
    false_alarm_count: int
    alarm_offset_mean_s: float | None
    alarm_offset_median_s: float | None
    alarm_offset_p95_s: float | None
    early_warning_rate: float | None
    early_warning_count: int
    early_warning_lead_mean_s: float | None
    early_warning_lead_median_s: float | None
    post_onset_delay_count: int
    post_onset_delay_mean_s: float | None
    post_onset_delay_median_s: float | None
    post_onset_delay_p95_s: float | None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _offline_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    class_rows: list[dict] = []
    for model_name, model_dir in OFFLINE_MODELS.items():
        for split in ("val", "test", "external_test"):
            path = model_dir / f"metrics_{split}.json"
            if not path.exists():
                continue
            data = _load_json(path)
            report = data.get("classification_report", {})
            model_rows.append(
                {
                    "task": "offline_multiclass",
                    "model": model_name,
                    "split": split,
                    "accuracy": _round(_safe_float(data.get("accuracy"))),
                    "macro_f1": _round(_safe_float(data.get("macro_f1"))),
                    "weighted_f1": _round(_safe_float(data.get("weighted_f1"))),
                    "label_names": ",".join(data.get("label_names", [])),
                    "available_classes": ",".join(
                        [
                            key
                            for key, value in report.items()
                            if isinstance(value, dict)
                            and key not in {"macro avg", "weighted avg"}
                        ]
                    ),
                    "source_file": str(path.relative_to(REPO_ROOT)),
                }
            )
            for class_name, metrics in report.items():
                if not isinstance(metrics, dict):
                    continue
                if class_name in {"macro avg", "weighted avg"}:
                    continue
                class_rows.append(
                    {
                        "task": "offline_multiclass",
                        "model": model_name,
                        "split": split,
                        "class_name": class_name,
                        "precision": _round(_safe_float(metrics.get("precision"))),
                        "recall": _round(_safe_float(metrics.get("recall"))),
                        "f1_score": _round(_safe_float(metrics.get("f1-score"))),
                        "support": int(metrics.get("support", 0)),
                    }
                )
    return pd.DataFrame(model_rows), pd.DataFrame(class_rows)


def _online_window_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    class_rows: list[dict] = []
    for model_name, model_dir in ONLINE_MODELS.items():
        for split in ("val", "test"):
            path = model_dir / f"metrics_{split}.json"
            if not path.exists():
                continue
            data = _load_json(path)
            report = data.get("classification_report", {})
            model_rows.append(
                {
                    "task": "online_binary_window",
                    "model": model_name,
                    "split": split,
                    "accuracy": _round(_safe_float(data.get("accuracy"))),
                    "macro_f1": _round(_safe_float(data.get("macro_f1"))),
                    "weighted_f1": _round(_safe_float(data.get("weighted_f1"))),
                    "precision": _round(_safe_float(data.get("precision"))),
                    "recall": _round(_safe_float(data.get("recall"))),
                    "f1_fault": _round(_safe_float(data.get("f1_fault"))),
                    "source_file": str(path.relative_to(REPO_ROOT)),
                }
            )
            for class_name, metrics in report.items():
                if not isinstance(metrics, dict):
                    continue
                if class_name in {"macro avg", "weighted avg"}:
                    continue
                class_rows.append(
                    {
                        "task": "online_binary_window",
                        "model": model_name,
                        "split": split,
                        "class_name": class_name,
                        "precision": _round(_safe_float(metrics.get("precision"))),
                        "recall": _round(_safe_float(metrics.get("recall"))),
                        "f1_score": _round(_safe_float(metrics.get("f1-score"))),
                        "support": int(metrics.get("support", 0)),
                    }
                )
    return pd.DataFrame(model_rows), pd.DataFrame(class_rows)


def _event_dataset_summary(dataset: str, csv_path: Path) -> EventDatasetSummary:
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    pred = df["pred_fault_by_event"].fillna(0).astype(int)
    actual = df["gt_fault"].fillna(0).astype(int)

    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())

    total = int(len(df))
    positives = int((actual == 1).sum())
    negatives = int((actual == 0).sum())

    file_accuracy = (tp + tn) / total if total else 0.0
    file_precision = tp / (tp + fp) if (tp + fp) else None
    file_recall = tp / (tp + fn) if (tp + fn) else None
    file_f1 = (
        2 * file_precision * file_recall / (file_precision + file_recall)
        if file_precision is not None
        and file_recall is not None
        and (file_precision + file_recall) > 0
        else None
    )
    false_alarm_rate_all = fp / total if total else 0.0
    false_alarm_rate_normal = fp / negatives if negatives else None

    hit_count = int(df["hit"].fillna(0).astype(int).sum())
    miss_count = int(df["miss"].fillna(0).astype(int).sum())
    false_alarm_count = int(df["false_alarm"].fillna(0).astype(int).sum())
    hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) else None

    latencies = df["detection_latency_s"].dropna().astype(float)
    alarm_offset_mean_s = _safe_float(latencies.mean()) if not latencies.empty else None
    alarm_offset_median_s = (
        _safe_float(latencies.median()) if not latencies.empty else None
    )
    alarm_offset_p95_s = (
        _safe_float(latencies.quantile(0.95)) if not latencies.empty else None
    )

    early = latencies[latencies < 0]
    late = latencies[latencies >= 0]
    early_warning_count = int(len(early))
    early_warning_rate = (
        early_warning_count / len(latencies) if len(latencies) else None
    )
    early_warning_lead_mean_s = _safe_float((-early).mean()) if len(early) else None
    early_warning_lead_median_s = (
        _safe_float((-early).median()) if len(early) else None
    )

    post_onset_delay_count = int(len(late))
    post_onset_delay_mean_s = _safe_float(late.mean()) if len(late) else None
    post_onset_delay_median_s = _safe_float(late.median()) if len(late) else None
    post_onset_delay_p95_s = _safe_float(late.quantile(0.95)) if len(late) else None

    return EventDatasetSummary(
        dataset=dataset,
        csv_name=csv_path.name,
        rows_ok=total,
        actual_fault_files=positives,
        actual_normal_files=negatives,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        file_accuracy=_round(file_accuracy),
        file_precision=_round(file_precision),
        file_recall=_round(file_recall),
        file_f1=_round(file_f1),
        false_alarm_rate_all=_round(false_alarm_rate_all),
        false_alarm_rate_normal=_round(false_alarm_rate_normal),
        hit_rate=_round(hit_rate),
        hit_count=hit_count,
        miss_count=miss_count,
        false_alarm_count=false_alarm_count,
        alarm_offset_mean_s=_round(alarm_offset_mean_s),
        alarm_offset_median_s=_round(alarm_offset_median_s),
        alarm_offset_p95_s=_round(alarm_offset_p95_s),
        early_warning_rate=_round(early_warning_rate),
        early_warning_count=early_warning_count,
        early_warning_lead_mean_s=_round(early_warning_lead_mean_s),
        early_warning_lead_median_s=_round(early_warning_lead_median_s),
        post_onset_delay_count=post_onset_delay_count,
        post_onset_delay_mean_s=_round(post_onset_delay_mean_s),
        post_onset_delay_median_s=_round(post_onset_delay_median_s),
        post_onset_delay_p95_s=_round(post_onset_delay_p95_s),
    )


def _event_rows() -> pd.DataFrame:
    rows = [_event_dataset_summary(name, path).__dict__ for name, path in SELECTED_EVENT_LOGS.items()]
    return pd.DataFrame(rows)


def _build_summary_json(
    offline_model_df: pd.DataFrame,
    online_window_df: pd.DataFrame,
    online_event_df: pd.DataFrame,
) -> dict:
    offline_test = offline_model_df[offline_model_df["split"] == "test"].copy()
    offline_external = offline_model_df[
        offline_model_df["split"] == "external_test"
    ].copy()
    offline_merged = offline_test.merge(
        offline_external,
        on="model",
        suffixes=("_test", "_external"),
    )
    offline_merged["accuracy_gap"] = (
        offline_merged["accuracy_test"] - offline_merged["accuracy_external"]
    ).round(4)
    offline_merged["macro_f1_gap"] = (
        offline_merged["macro_f1_test"] - offline_merged["macro_f1_external"]
    ).round(4)

    online_test = online_window_df[online_window_df["split"] == "test"].copy()
    legacy = online_test[online_test["model"] == "cnn_legacy"].iloc[0].to_dict()
    current = online_test[online_test["model"] == "cnn1d_rfly"].iloc[0].to_dict()

    normal_event = online_event_df[online_event_df["dataset"] == "normal_baseline"].iloc[
        0
    ].to_dict()
    fault_event = online_event_df[
        online_event_df["dataset"] == "external_fault_test"
    ].iloc[0].to_dict()

    return {
        "competition_track": "软件应用与开发-算法设计与应用",
        "evidence_generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "selected_sources": {
            "offline_models": {
                model: str(path.relative_to(REPO_ROOT))
                for model, path in OFFLINE_MODELS.items()
            },
            "online_models": {
                model: str(path.relative_to(REPO_ROOT))
                for model, path in ONLINE_MODELS.items()
            },
            "event_logs": {
                name: str(path.relative_to(REPO_ROOT))
                for name, path in SELECTED_EVENT_LOGS.items()
            },
        },
        "headline_metrics": {
            "offline_same_distribution_best_test_accuracy": _round(
                _safe_float(offline_test["accuracy"].max())
            ),
            "offline_same_distribution_best_test_macro_f1": _round(
                _safe_float(offline_test["macro_f1"].max())
            ),
            "offline_external_best_accuracy": _round(
                _safe_float(offline_external["accuracy"].max())
            ),
            "offline_external_best_macro_f1": _round(
                _safe_float(offline_external["macro_f1"].max())
            ),
            "online_window_accuracy_legacy_test": legacy["accuracy"],
            "online_window_accuracy_cnn1d_rfly_test": current["accuracy"],
            "online_window_f1_fault_legacy_test": legacy["f1_fault"],
            "online_window_f1_fault_cnn1d_rfly_test": current["f1_fault"],
            "normal_set_false_alarm_rate": normal_event["false_alarm_rate_normal"],
            "external_fault_file_recall": fault_event["file_recall"],
            "external_fault_hit_rate": fault_event["hit_rate"],
            "external_fault_early_warning_rate": fault_event["early_warning_rate"],
            "external_fault_early_warning_lead_median_s": fault_event[
                "early_warning_lead_median_s"
            ],
            "external_fault_post_onset_delay_median_s": fault_event[
                "post_onset_delay_median_s"
            ],
        },
        "key_findings": [
            "离线四分类模型在同分布 val/test 上 accuracy 与 macro_f1 均接近或超过 0.98，说明在已见分布内识别能力较强。",
            "离线四分类在 external_test 上出现显著下降，其中 transformer 的 external accuracy=0.6122，说明跨场景泛化仍是当前主要短板。",
            f"在线窗口级评测中，cnn1d_rfly 相比旧 cnn_legacy，test accuracy 从 {legacy['accuracy']} 提升到 {current['accuracy']}，fault 类 F1 从 {legacy['f1_fault']} 提升到 {current['f1_fault']}。",
            f"在线事件级评测中，正常集 normal_baseline 的文件级误报率为 {normal_event['false_alarm_rate_normal']}，外部故障集 external_fault_test 的文件级 recall 为 {fault_event['file_recall']}，hit_rate 为 {fault_event['hit_rate']}。",
            f"external_fault_test 中 {round(float(fault_event['early_warning_rate']) * 100, 2)}% 的命中样本在故障 onset 前已触发告警，提前预警提前量中位数为 {fault_event['early_warning_lead_median_s']}s；未提前的样本故障后检测时延中位数为 {fault_event['post_onset_delay_median_s']}s。",
        ],
        "offline_generalization_table": offline_merged[
            [
                "model",
                "accuracy_test",
                "accuracy_external",
                "accuracy_gap",
                "macro_f1_test",
                "macro_f1_external",
                "macro_f1_gap",
            ]
        ].to_dict(orient="records"),
        "notes": [
            "online event-level 指标基于现成 replay CSV 日志汇总，口径与 scripts/eval_telemetry_live.py 一致。",
            "detection_latency_s 为首次检测时间减去 GT onset 时间，负值表示故障 onset 前已预警，因此在比赛材料中应解释为 alarm offset，而非单纯 delay。",
            "本轮未做 blind test，也未重新训练模型，重点是复用现有证据包装可答辩的量化结果。",
        ],
    }


def _write_markdown(summary: dict, out_path: Path) -> None:
    headline = summary["headline_metrics"]
    lines = [
        "# 量化评测证据汇总",
        "",
        "## 可直接用于答辩的结论",
        "",
        f"- 离线四分类同分布最佳 test accuracy: {headline['offline_same_distribution_best_test_accuracy']}",
        f"- 离线四分类外部分布最佳 accuracy: {headline['offline_external_best_accuracy']}",
        f"- 在线窗口级新模型 test accuracy / f1_fault: {headline['online_window_accuracy_cnn1d_rfly_test']} / {headline['online_window_f1_fault_cnn1d_rfly_test']}",
        f"- 在线窗口级旧模型 test accuracy / f1_fault: {headline['online_window_accuracy_legacy_test']} / {headline['online_window_f1_fault_legacy_test']}",
        f"- 正常集误报率: {headline['normal_set_false_alarm_rate']}",
        f"- 外部故障集文件级 recall / hit_rate: {headline['external_fault_file_recall']} / {headline['external_fault_hit_rate']}",
        f"- 提前预警率: {headline['external_fault_early_warning_rate']}",
        f"- 提前预警中位提前量: {headline['external_fault_early_warning_lead_median_s']} s",
        f"- 故障后检测中位时延: {headline['external_fault_post_onset_delay_median_s']} s",
        "",
        "## 解释建议",
        "",
        "- 离线模型要强调“已见场景识别强，跨场景泛化还在持续优化”。",
        "- 在线模型要强调“窗口级准确率高，同时事件级具备提前预警能力”。",
        "- 对于负 latency，不要表述成延迟异常，应解释为在故障 onset 前已触发预警。",
        "",
        "## 关键发现",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["key_findings"])
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    offline_model_df, offline_class_df = _offline_rows()
    online_window_df, online_window_class_df = _online_window_rows()
    online_event_df = _event_rows()

    offline_model_df.to_csv(METRICS_DIR / "offline_model_summary.csv", index=False)
    offline_class_df.to_csv(METRICS_DIR / "offline_class_summary.csv", index=False)
    online_window_df.to_csv(METRICS_DIR / "online_window_summary.csv", index=False)
    online_window_class_df.to_csv(
        METRICS_DIR / "online_window_class_summary.csv", index=False
    )
    online_event_df.to_csv(METRICS_DIR / "online_event_summary.csv", index=False)

    summary = _build_summary_json(offline_model_df, online_window_df, online_event_df)
    (METRICS_DIR / "competition_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, OUTPUT_ROOT / "competition_evidence.md")

    print(f"[OK] wrote metrics to: {METRICS_DIR}")
    print(f"[OK] wrote brief to: {OUTPUT_ROOT / 'competition_evidence.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
