#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "assets" / "models"
MANIFEST_PATH = MODEL_ROOT / "manifest.json"
OUTPUT_ROOT = REPO_ROOT / "competition" / "2026-software-app"
METRICS_DIR = OUTPUT_ROOT / "metrics"

SPLITS = ("val", "test", "external_test")


def _safe_float(value: Any) -> float | None:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_entries() -> list[dict[str, Any]]:
    manifest = _load_json(MANIFEST_PATH)
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise ValueError(f"manifest models must be a list: {MANIFEST_PATH}")

    resolved: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        raw_path = str(entry.get("path") or name).strip()
        if not name or not raw_path:
            continue
        model_dir = Path(raw_path)
        if not model_dir.is_absolute():
            model_dir = MODEL_ROOT / model_dir
        if not model_dir.is_dir():
            raise FileNotFoundError(f"manifest model directory does not exist: {model_dir}")
        resolved.append({**entry, "name": name, "path": model_dir})
    return resolved


def _infer_runtime(entry: dict[str, Any], config: dict[str, Any]) -> str:
    runtime = str(entry.get("runtime") or "").strip().lower()
    if runtime in {"offline", "online"}:
        return runtime
    task = str(config.get("task") or "").strip().lower()
    if task == "online_binary" or int(config.get("num_classes") or 0) == 2:
        return "online"
    return "offline"


def _label_names(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    labels = metrics.get("label_names") or config.get("label_names") or []
    if isinstance(labels, list):
        return ",".join(str(item).strip() for item in labels if str(item).strip())
    return ""


def _available_classes(report: dict[str, Any]) -> str:
    return ",".join(
        key
        for key, value in report.items()
        if isinstance(value, dict) and key not in {"macro avg", "weighted avg"}
    )


def _metric_row(
    *,
    entry: dict[str, Any],
    config: dict[str, Any],
    runtime: str,
    split: str,
    metrics_path: Path,
) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    data = _load_json(metrics_path)
    report = data.get("classification_report", {})
    if not isinstance(report, dict):
        report = {}

    task = "online_binary_window" if runtime == "online" else "offline_multiclass"
    return {
        "task": task,
        "runtime": runtime,
        "model": entry["name"],
        "split": split,
        "architecture": str(config.get("architecture") or config.get("model") or ""),
        "num_classes": int(config.get("num_classes") or 0),
        "label_names": _label_names(config, data),
        "available_classes": _available_classes(report),
        "accuracy": _round(_safe_float(data.get("accuracy"))),
        "balanced_accuracy": _round(_safe_float(data.get("balanced_accuracy"))),
        "macro_f1": _round(_safe_float(data.get("macro_f1"))),
        "weighted_f1": _round(_safe_float(data.get("weighted_f1"))),
        "precision": _round(_safe_float(data.get("precision"))),
        "recall": _round(_safe_float(data.get("recall"))),
        "f1_fault": _round(_safe_float(data.get("f1_fault"))),
        "source_file": str(metrics_path.relative_to(REPO_ROOT)),
    }


def _class_rows(
    *,
    entry: dict[str, Any],
    runtime: str,
    split: str,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    data = _load_json(metrics_path)
    report = data.get("classification_report", {})
    if not isinstance(report, dict):
        return []

    task = "online_binary_window" if runtime == "online" else "offline_multiclass"
    rows: list[dict[str, Any]] = []
    for class_name, metrics in report.items():
        if not isinstance(metrics, dict) or class_name in {"macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "task": task,
                "runtime": runtime,
                "model": entry["name"],
                "split": split,
                "class_name": class_name,
                "precision": _round(_safe_float(metrics.get("precision"))),
                "recall": _round(_safe_float(metrics.get("recall"))),
                "f1_score": _round(_safe_float(metrics.get("f1-score"))),
                "support": int(metrics.get("support", 0)),
            }
        )
    return rows


def _metric_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    offline_rows: list[dict[str, Any]] = []
    offline_class_rows: list[dict[str, Any]] = []
    online_rows: list[dict[str, Any]] = []
    online_class_rows: list[dict[str, Any]] = []

    for entry in _manifest_entries():
        model_dir: Path = entry["path"]
        config = _load_json(model_dir / "config.json")
        runtime = _infer_runtime(entry, config)
        for split in SPLITS:
            metrics_path = model_dir / f"metrics_{split}.json"
            row = _metric_row(
                entry=entry,
                config=config,
                runtime=runtime,
                split=split,
                metrics_path=metrics_path,
            )
            if row is None:
                continue
            class_rows = _class_rows(
                entry=entry,
                runtime=runtime,
                split=split,
                metrics_path=metrics_path,
            )
            if runtime == "online":
                online_rows.append(row)
                online_class_rows.extend(class_rows)
            else:
                offline_rows.append(row)
                offline_class_rows.extend(class_rows)

    return (
        pd.DataFrame(offline_rows),
        pd.DataFrame(offline_class_rows),
        pd.DataFrame(online_rows),
        pd.DataFrame(online_class_rows),
    )


def _best_row(df: pd.DataFrame, *, split: str, metric: str) -> dict[str, Any] | None:
    if df.empty or metric not in df.columns:
        return None
    scoped = df[df["split"] == split].copy()
    if scoped.empty:
        return None
    scoped[metric] = pd.to_numeric(scoped[metric], errors="coerce")
    scoped = scoped.dropna(subset=[metric])
    if scoped.empty:
        return None
    return scoped.sort_values(metric, ascending=False).iloc[0].to_dict()


def _split_value(df: pd.DataFrame, *, split: str, metric: str) -> float | None:
    best = _best_row(df, split=split, metric=metric)
    return _safe_float(best.get(metric)) if best else None


def _offline_generalization_rows(offline_df: pd.DataFrame) -> list[dict[str, Any]]:
    if offline_df.empty:
        return []
    test = offline_df[offline_df["split"] == "test"].copy()
    external = offline_df[offline_df["split"] == "external_test"].copy()
    if test.empty or external.empty:
        return []
    merged = test.merge(external, on="model", suffixes=("_test", "_external"))
    merged["accuracy_gap"] = (
        pd.to_numeric(merged["accuracy_test"], errors="coerce")
        - pd.to_numeric(merged["accuracy_external"], errors="coerce")
    ).round(4)
    merged["macro_f1_gap"] = (
        pd.to_numeric(merged["macro_f1_test"], errors="coerce")
        - pd.to_numeric(merged["macro_f1_external"], errors="coerce")
    ).round(4)
    return merged[
        [
            "model",
            "accuracy_test",
            "accuracy_external",
            "accuracy_gap",
            "macro_f1_test",
            "macro_f1_external",
            "macro_f1_gap",
        ]
    ].to_dict(orient="records")


def _selected_sources() -> dict[str, dict[str, str]]:
    sources = {"offline_models": {}, "online_models": {}}
    for entry in _manifest_entries():
        model_dir: Path = entry["path"]
        config = _load_json(model_dir / "config.json")
        runtime = _infer_runtime(entry, config)
        bucket = "online_models" if runtime == "online" else "offline_models"
        sources[bucket][entry["name"]] = str(model_dir.relative_to(REPO_ROOT))
    return sources


def _build_summary_json(
    offline_df: pd.DataFrame,
    online_df: pd.DataFrame,
) -> dict[str, Any]:
    best_offline_test = _best_row(offline_df, split="test", metric="macro_f1")
    best_offline_external = _best_row(offline_df, split="external_test", metric="macro_f1")
    best_online_test = _best_row(online_df, split="test", metric="f1_fault")
    best_online_external = _best_row(online_df, split="external_test", metric="f1_fault")

    headline = {
        "offline_3class_best_test_accuracy": _round(
            _split_value(offline_df, split="test", metric="accuracy")
        ),
        "offline_3class_best_test_macro_f1": _round(
            _split_value(offline_df, split="test", metric="macro_f1")
        ),
        "offline_3class_external_accuracy": _round(
            _split_value(offline_df, split="external_test", metric="accuracy")
        ),
        "offline_3class_external_macro_f1": _round(
            _split_value(offline_df, split="external_test", metric="macro_f1")
        ),
        "online_best_test_model": best_online_test.get("model") if best_online_test else None,
        "online_best_test_accuracy": _round(
            _safe_float(best_online_test.get("accuracy")) if best_online_test else None
        ),
        "online_best_test_f1_fault": _round(
            _safe_float(best_online_test.get("f1_fault")) if best_online_test else None
        ),
        "online_best_external_model": best_online_external.get("model")
        if best_online_external
        else None,
        "online_best_external_accuracy": _round(
            _safe_float(best_online_external.get("accuracy"))
            if best_online_external
            else None
        ),
        "online_best_external_f1_fault": _round(
            _safe_float(best_online_external.get("f1_fault"))
            if best_online_external
            else None
        ),
    }

    offline_model = best_offline_test.get("model") if best_offline_test else "暂无"
    online_model = headline["online_best_test_model"] or "暂无"
    online_external_model = headline["online_best_external_model"] or "暂无"

    return {
        "competition_track": "软件应用与开发-算法设计与应用",
        "evidence_generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "selected_sources": _selected_sources(),
        "headline_metrics": headline,
        "key_findings": [
            (
                f"当前离线运行模型为三分类模型 {offline_model}，"
                f"test accuracy={headline['offline_3class_best_test_accuracy']}，"
                f"macro_f1={headline['offline_3class_best_test_macro_f1']}。"
            ),
            (
                "离线三分类 external_test 指标为 "
                f"accuracy={headline['offline_3class_external_accuracy']}，"
                f"macro_f1={headline['offline_3class_external_macro_f1']}；"
                "跨场景泛化仍是后续优化重点。"
            ),
            (
                f"当前在线运行模型以 manifest 中的 3 个 Rfly 模型为准，"
                f"test 集 fault 类 F1 最优模型为 {online_model}，"
                f"f1_fault={headline['online_best_test_f1_fault']}，"
                f"accuracy={headline['online_best_test_accuracy']}。"
            ),
            (
                f"online external_test 的 fault 类 F1 最优模型为 {online_external_model}，"
                f"f1_fault={headline['online_best_external_f1_fault']}，"
                f"accuracy={headline['online_best_external_accuracy']}。"
            ),
        ],
        "offline_generalization_table": _offline_generalization_rows(offline_df),
        "notes": [
            "本脚本只读取 assets/models/manifest.json 中登记的当前运行模型。",
            "未登记在 manifest 中的历史实验目录不再作为默认评估输入。",
            "当前默认输出为模型级与类别级指标汇总；事件级 replay 评估应由 scripts/eval/eval_telemetry_live.py 单独生成。",
        ],
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    headline = summary["headline_metrics"]
    lines = [
        "# 量化评测证据汇总",
        "",
        "## 当前运行模型",
        "",
        f"- 离线三分类 test accuracy / macro_f1: {headline['offline_3class_best_test_accuracy']} / {headline['offline_3class_best_test_macro_f1']}",
        f"- 离线三分类 external accuracy / macro_f1: {headline['offline_3class_external_accuracy']} / {headline['offline_3class_external_macro_f1']}",
        f"- 在线 test 最优模型: {headline['online_best_test_model']}",
        f"- 在线 test accuracy / f1_fault: {headline['online_best_test_accuracy']} / {headline['online_best_test_f1_fault']}",
        f"- 在线 external 最优模型: {headline['online_best_external_model']}",
        f"- 在线 external accuracy / f1_fault: {headline['online_best_external_accuracy']} / {headline['online_best_external_f1_fault']}",
        "",
        "## 解释建议",
        "",
        "- 离线模型按当前三分类口径说明，不再写四分类或六分类结果。",
        "- 在线模型按 manifest 中的 Rfly 模型说明，不再和历史实验模型对比。",
        "- external_test 指标用于说明跨场景泛化情况，后续优化重点仍是 Real external 泛化能力。",
        "",
        "## 关键发现",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["key_findings"])
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    offline_df, offline_class_df, online_df, online_class_df = _metric_tables()

    offline_df.to_csv(METRICS_DIR / "offline_model_summary.csv", index=False)
    offline_class_df.to_csv(METRICS_DIR / "offline_class_summary.csv", index=False)
    online_df.to_csv(METRICS_DIR / "online_window_summary.csv", index=False)
    online_class_df.to_csv(
        METRICS_DIR / "online_window_class_summary.csv", index=False
    )

    summary = _build_summary_json(offline_df, online_df)
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
