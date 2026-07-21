#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(r"D:\UAV-Data")
DEFAULT_SPLIT_DIR = DEFAULT_DATA_ROOT / "splits_transfer3_v1"
DEFAULT_LEGACY_RESULTS_ROOT = DEFAULT_DATA_ROOT / "results_transfer3"
DEFAULT_AUDIT_ROOT = DEFAULT_DATA_ROOT / "results_transfer3_audit_v1"
DEFAULT_BASELINE_CACHE_DIR = DEFAULT_DATA_ROOT / "cache_transfer3_baseline_v1"
DEFAULT_FEWSHOT_CACHE_DIR = DEFAULT_DATA_ROOT / "cache_transfer3_fewshot_v1"
DEFAULT_REPORT_DIR = DEFAULT_AUDIT_ROOT / "audit_report"

REQUIRED_SPLITS = (
    "train",
    "val",
    "test",
    "real_adapt_train",
    "real_adapt_val",
    "real_holdout_test",
)
ALIAS_SPLITS = {"external_test": "real_holdout_test"}
DEV_SPLITS = {"train", "val", "test"}
REAL_SPLITS = {"real_adapt_train", "real_adapt_val", "real_holdout_test"}
EXPECTED_TRANSFER3_COUNTS = {
    "train": 2368,
    "val": 592,
    "test": 741,
    "real_adapt_train": 98,
    "real_adapt_val": 49,
    "real_holdout_test": 345,
}
CLASS_NAMES = ("normal", "motor", "sensor")
SPLIT_LABELS = {"test": "test", "external_test": "real_holdout_test"}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: str
    mode: str
    cache_kind: str
    adabn: bool = False
    checkpoint_experiment: str | None = None


EXPERIMENT_SPECS = (
    ExperimentSpec(
        name="transfer3_transformer_baseline_v1",
        model="transformer",
        mode="baseline",
        cache_kind="baseline",
    ),
    ExperimentSpec(
        name="transfer3_transformer_fewshot_v1",
        model="transformer",
        mode="fewshot",
        cache_kind="fewshot",
    ),
    ExperimentSpec(
        name="transfer3_cnnlstm_baseline_v1",
        model="cnn_lstm",
        mode="baseline",
        cache_kind="baseline",
    ),
    ExperimentSpec(
        name="transfer3_cnnlstm_fewshot_v1",
        model="cnn_lstm",
        mode="fewshot",
        cache_kind="fewshot",
    ),
    ExperimentSpec(
        name="transfer3_cnnlstm_baseline_adabn_v1",
        model="cnn_lstm",
        mode="baseline",
        cache_kind="baseline",
        adabn=True,
        checkpoint_experiment="transfer3_cnnlstm_baseline_v1",
    ),
    ExperimentSpec(
        name="transfer3_cnnlstm_fewshot_adabn_v1",
        model="cnn_lstm",
        mode="fewshot",
        cache_kind="fewshot",
        adabn=True,
        checkpoint_experiment="transfer3_cnnlstm_fewshot_v1",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_remove(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"path already exists, pass --force to overwrite: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _load_split_frames(split_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split_name in REQUIRED_SPLITS:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing split file: {csv_path}")
        frames[split_name] = pd.read_csv(csv_path)
    alias_path = split_dir / "external_test.csv"
    if alias_path.exists():
        frames["external_test"] = pd.read_csv(alias_path)
    return frames


def validate_transfer3_split_integrity(
    split_dir: str | Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    split_path = Path(split_dir)
    frames = _load_split_frames(split_path)
    expected = expected_counts or {}

    counts = {name: int(len(frame)) for name, frame in frames.items() if name in REQUIRED_SPLITS}
    for split_name, expected_count in expected.items():
        actual = counts.get(split_name)
        if actual != expected_count:
            raise ValueError(
                f"count mismatch for {split_name}: expected {expected_count}, got {actual}"
            )

    platform_counts: dict[str, dict[str, int]] = {}
    sample_sets: dict[str, set[str]] = {}
    for split_name, frame in frames.items():
        required_columns = {"sample_id", "platform"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"split {split_name} missing columns: {sorted(missing)}")
        platform_counts[split_name] = {
            str(k): int(v)
            for k, v in frame["platform"].value_counts().sort_index().to_dict().items()
        }
        sample_sets[split_name] = {
            str(item).strip()
            for item in frame["sample_id"].astype(str).tolist()
            if str(item).strip()
        }

    for split_name in DEV_SPLITS:
        unexpected = set(platform_counts.get(split_name, {})) - {"SIL", "HIL"}
        if unexpected:
            raise ValueError(f"split {split_name} contains non-dev platforms: {sorted(unexpected)}")
    for split_name in REAL_SPLITS:
        unexpected = set(platform_counts.get(split_name, {})) - {"Real"}
        if unexpected:
            raise ValueError(f"split {split_name} contains non-real platforms: {sorted(unexpected)}")

    overlaps: dict[str, list[str]] = {}
    canonical_splits = list(REQUIRED_SPLITS)
    for idx, left in enumerate(canonical_splits):
        for right in canonical_splits[idx + 1 :]:
            intersection = sorted(sample_sets[left] & sample_sets[right])
            if intersection:
                overlaps[f"{left}__{right}"] = intersection
    if overlaps:
        raise ValueError(f"split overlap detected: {overlaps}")

    alias_checks: dict[str, bool] = {}
    for alias_name, canonical_name in ALIAS_SPLITS.items():
        if alias_name in sample_sets:
            matches = sample_sets[alias_name] == sample_sets[canonical_name]
            alias_checks[f"{alias_name}_matches_{canonical_name}"] = matches
            if not matches:
                raise ValueError(
                    f"alias split {alias_name} does not match canonical split {canonical_name}"
                )

    return {
        "split_dir": str(split_path),
        "counts": counts,
        "platform_counts": {
            name: platform_counts[name]
            for name in REQUIRED_SPLITS
            if name in platform_counts
        },
        "alias_checks": alias_checks,
        "overlaps": {},
    }


def validate_cache_summary(
    cache_dir: str | Path,
    expected_train_splits: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    build_summary_path = cache_path / "build_summary.json"
    if not build_summary_path.exists():
        raise FileNotFoundError(f"missing cache build summary: {build_summary_path}")

    build_summary = _read_json(build_summary_path)
    train_splits = [str(item) for item in build_summary.get("train_splits_for_norm", [])]
    expected = [str(item) for item in expected_train_splits]
    if train_splits != expected:
        raise ValueError(
            f"train_splits_for_norm mismatch for {cache_path}: expected {expected}, got {train_splits}"
        )

    saved_counts = {
        split_name: int((split_meta or {}).get("saved", 0))
        for split_name, split_meta in (build_summary.get("splits") or {}).items()
    }
    return {
        "cache_dir": str(cache_path),
        "train_splits_for_norm": train_splits,
        "saved_counts": saved_counts,
        "seq_len": build_summary.get("seq_len"),
        "label_column": build_summary.get("label_column"),
    }


def _extract_class_metrics(metrics: dict[str, Any], class_name: str) -> dict[str, float | None]:
    report = metrics.get("classification_report", {})
    class_report = report.get(class_name, {}) if isinstance(report, dict) else {}
    if not isinstance(class_report, dict):
        class_report = {}
    return {
        "precision": _safe_float(class_report.get("precision")),
        "recall": _safe_float(class_report.get("recall")),
        "f1": _safe_float(class_report.get("f1-score")),
        "support": _safe_float(class_report.get("support")),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _discover_experiments(root: Path) -> set[str]:
    if not root.exists():
        return set()
    discovered = set()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "metrics_test.json").exists() and (child / "metrics_external_test.json").exists():
            discovered.add(child.name)
    return discovered


def build_comparison_tables(
    legacy_root: str | Path,
    audit_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_path = Path(legacy_root)
    audit_path = Path(audit_root)
    experiments = sorted(_discover_experiments(legacy_path) & _discover_experiments(audit_path))
    if not experiments:
        raise ValueError("no common experiments found between legacy and audit roots")

    model_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for experiment_name in experiments:
        legacy_dir = legacy_path / experiment_name
        audit_dir = audit_path / experiment_name
        metrics_pairs = {
            split_label: (
                _read_json(legacy_dir / f"metrics_{metrics_name}.json"),
                _read_json(audit_dir / f"metrics_{metrics_name}.json"),
            )
            for metrics_name, split_label in SPLIT_LABELS.items()
        }

        model_row: dict[str, Any] = {"experiment_name": experiment_name}
        for split_label, (legacy_metrics, audit_metrics) in metrics_pairs.items():
            for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
                legacy_value = _safe_float(legacy_metrics.get(metric_name))
                audit_value = _safe_float(audit_metrics.get(metric_name))
                model_row[f"{split_label}_{metric_name}_legacy"] = legacy_value
                model_row[f"{split_label}_{metric_name}_audit"] = audit_value
                if legacy_value is None or audit_value is None:
                    model_row[f"{split_label}_{metric_name}_delta"] = None
                else:
                    model_row[f"{split_label}_{metric_name}_delta"] = audit_value - legacy_value
        model_rows.append(model_row)

        for split_label, (legacy_metrics, audit_metrics) in metrics_pairs.items():
            for class_name in CLASS_NAMES:
                legacy_class = _extract_class_metrics(legacy_metrics, class_name)
                audit_class = _extract_class_metrics(audit_metrics, class_name)
                class_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "split": split_label,
                        "class_name": class_name,
                        "legacy_precision": legacy_class["precision"],
                        "audit_precision": audit_class["precision"],
                        "delta_precision": _delta(audit_class["precision"], legacy_class["precision"]),
                        "legacy_recall": legacy_class["recall"],
                        "audit_recall": audit_class["recall"],
                        "delta_recall": _delta(audit_class["recall"], legacy_class["recall"]),
                        "legacy_f1": legacy_class["f1"],
                        "audit_f1": audit_class["f1"],
                        "delta_f1": _delta(audit_class["f1"], legacy_class["f1"]),
                        "legacy_support": legacy_class["support"],
                        "audit_support": audit_class["support"],
                    }
                )

    model_df = pd.DataFrame(model_rows).sort_values("experiment_name").reset_index(drop=True)
    class_df = pd.DataFrame(class_rows).sort_values(
        ["experiment_name", "split", "class_name"]
    ).reset_index(drop=True)
    return model_df, class_df


def _delta(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None:
        return None
    return new_value - old_value


def _ranking_by_real_holdout_macro_f1(model_df: pd.DataFrame, suffix: str) -> list[str]:
    return (
        model_df.sort_values(
            by=f"real_holdout_test_macro_f1_{suffix}",
            ascending=False,
            kind="stable",
        )["experiment_name"]
        .tolist()
    )


def build_audit_summary_payload(
    split_summary: dict[str, Any],
    baseline_cache_summary: dict[str, Any],
    fewshot_cache_summary: dict[str, Any],
    model_df: pd.DataFrame,
) -> dict[str, Any]:
    legacy_ranking = _ranking_by_real_holdout_macro_f1(model_df, "legacy")
    audit_ranking = _ranking_by_real_holdout_macro_f1(model_df, "audit")
    best_audit = model_df.sort_values(
        by="real_holdout_test_macro_f1_audit", ascending=False, kind="stable"
    ).iloc[0]
    return {
        "split_summary": split_summary,
        "baseline_cache_summary": baseline_cache_summary,
        "fewshot_cache_summary": fewshot_cache_summary,
        "legacy_ranking_by_real_holdout_macro_f1": legacy_ranking,
        "audit_ranking_by_real_holdout_macro_f1": audit_ranking,
        "ranking_changed": legacy_ranking != audit_ranking,
        "best_audit_experiment": {
            "experiment_name": best_audit["experiment_name"],
            "real_holdout_test_macro_f1": best_audit["real_holdout_test_macro_f1_audit"],
            "real_holdout_test_accuracy": best_audit["real_holdout_test_accuracy_audit"],
            "test_macro_f1": best_audit["test_macro_f1_audit"],
        },
    }


def _render_summary_markdown(summary_payload: dict[str, Any], model_df: pd.DataFrame) -> str:
    best = summary_payload["best_audit_experiment"]
    lines = [
        "# transfer3 协议审计结果",
        "",
        "## 审计结论",
        "",
        f"- ranking_changed: {summary_payload['ranking_changed']}",
        f"- best_audit_experiment: `{best['experiment_name']}`",
        f"- best_real_holdout_macro_f1: {best['real_holdout_test_macro_f1']:.4f}",
        f"- best_real_holdout_accuracy: {best['real_holdout_test_accuracy']:.4f}",
        f"- best_test_macro_f1: {best['test_macro_f1']:.4f}",
        "",
        "## split / cache 核对",
        "",
        f"- split_counts: `{json.dumps(summary_payload['split_summary']['counts'], ensure_ascii=False)}`",
        f"- baseline_train_splits_for_norm: `{summary_payload['baseline_cache_summary']['train_splits_for_norm']}`",
        f"- fewshot_train_splits_for_norm: `{summary_payload['fewshot_cache_summary']['train_splits_for_norm']}`",
        "",
        "## Real holdout 排名",
        "",
        f"- legacy: `{summary_payload['legacy_ranking_by_real_holdout_macro_f1']}`",
        f"- audit: `{summary_payload['audit_ranking_by_real_holdout_macro_f1']}`",
        "",
        "## 关键对照",
        "",
    ]
    key_cols = [
        "experiment_name",
        "test_macro_f1_legacy",
        "test_macro_f1_audit",
        "test_macro_f1_delta",
        "real_holdout_test_macro_f1_legacy",
        "real_holdout_test_macro_f1_audit",
        "real_holdout_test_macro_f1_delta",
    ]
    lines.append(model_df[key_cols].to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def write_comparison_outputs(
    legacy_root: str | Path,
    audit_root: str | Path,
    output_dir: str | Path,
    *,
    split_summary: dict[str, Any],
    baseline_cache_summary: dict[str, Any],
    fewshot_cache_summary: dict[str, Any],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_df, class_df = build_comparison_tables(legacy_root, audit_root)
    model_csv = out_dir / "transfer3_model_audit_summary.csv"
    class_csv = out_dir / "transfer3_class_audit_summary.csv"
    model_df.to_csv(model_csv, index=False)
    class_df.to_csv(class_csv, index=False)

    summary_payload = build_audit_summary_payload(
        split_summary=split_summary,
        baseline_cache_summary=baseline_cache_summary,
        fewshot_cache_summary=fewshot_cache_summary,
        model_df=model_df,
    )
    summary_json = out_dir / "transfer3_protocol_audit_summary.json"
    _write_json(summary_json, summary_payload)

    summary_md = out_dir / "transfer3_protocol_audit_summary.md"
    summary_md.write_text(
        _render_summary_markdown(summary_payload, model_df),
        encoding="utf-8",
    )
    return {
        "model_csv": model_csv,
        "class_csv": class_csv,
        "summary_json": summary_json,
        "summary_md": summary_md,
    }


def _python_has_torch(python_exe: Path) -> bool:
    proc = subprocess.run(
        [str(python_exe), "-c", "import torch"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("[RUN]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _cache_dir_for_kind(kind: str, baseline_cache_dir: Path, fewshot_cache_dir: Path) -> Path:
    if kind == "baseline":
        return baseline_cache_dir
    if kind == "fewshot":
        return fewshot_cache_dir
    raise ValueError(f"unknown cache kind: {kind}")


def run_transfer3_protocol_audit(args: argparse.Namespace) -> dict[str, Any]:
    python_exe = Path(args.python_exe or sys.executable)
    if not _python_has_torch(python_exe):
        raise RuntimeError(
            f"python executable does not have torch installed: {python_exe}. "
            "Pass --python-exe with a training environment."
        )

    split_dir = Path(args.split_dir)
    baseline_cache_dir = Path(args.baseline_cache_dir)
    fewshot_cache_dir = Path(args.fewshot_cache_dir)
    legacy_results_root = Path(args.legacy_results_root)
    audit_results_root = Path(args.audit_results_root)
    report_dir = Path(args.report_dir)

    split_summary = validate_transfer3_split_integrity(
        split_dir,
        expected_counts=EXPECTED_TRANSFER3_COUNTS,
    )

    if args.rebuild_cache:
        _maybe_remove(baseline_cache_dir, force=args.force)
        _maybe_remove(fewshot_cache_dir, force=args.force)
        _run_command(
            [
                str(python_exe),
                str(PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "data" / "build_sequence_cache_3class.py"),
                "--split-dir",
                str(split_dir),
                "--output",
                str(baseline_cache_dir),
                "--train-splits",
                "train",
                "--label-column",
                "label_main_3class",
                "--seq-len",
                "2048",
            ],
            dry_run=args.dry_run,
        )
        _run_command(
            [
                str(python_exe),
                str(PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "data" / "build_sequence_cache_3class.py"),
                "--split-dir",
                str(split_dir),
                "--output",
                str(fewshot_cache_dir),
                "--train-splits",
                "train",
                "real_adapt_train",
                "--label-column",
                "label_main_3class",
                "--seq-len",
                "2048",
            ],
            dry_run=args.dry_run,
        )

    baseline_cache_summary = validate_cache_summary(
        baseline_cache_dir, expected_train_splits=("train",)
    )
    fewshot_cache_summary = validate_cache_summary(
        fewshot_cache_dir, expected_train_splits=("train", "real_adapt_train")
    )

    if args.run_experiments:
        if args.force:
            _maybe_remove(audit_results_root, force=True)
        audit_results_root.mkdir(parents=True, exist_ok=True)
        for spec in EXPERIMENT_SPECS:
            cache_dir = _cache_dir_for_kind(spec.cache_kind, baseline_cache_dir, fewshot_cache_dir)
            output_dir = audit_results_root / spec.name
            if spec.adabn:
                checkpoint_dir = audit_results_root / str(spec.checkpoint_experiment)
                checkpoint_path = checkpoint_dir / "best.pt"
                if not checkpoint_path.exists() and not args.dry_run:
                    raise FileNotFoundError(f"checkpoint missing for AdaBN experiment: {checkpoint_path}")
                _run_command(
                    [
                        str(python_exe),
                        str(PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "eval_transfer3_adabn.py"),
                        "--model",
                        spec.model,
                        "--cache-dir",
                        str(cache_dir),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--output-dir",
                        str(output_dir),
                        "--adapt-splits",
                        "real_adapt_train",
                        "real_adapt_val",
                        "--eval-split",
                        "real_holdout_test",
                    ],
                    dry_run=args.dry_run,
                )
            else:
                _run_command(
                    [
                        str(python_exe),
                        str(PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "train_transfer3.py"),
                        "--mode",
                        spec.mode,
                        "--model",
                        spec.model,
                        "--cache-dir",
                        str(cache_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    dry_run=args.dry_run,
                )

    outputs = write_comparison_outputs(
        legacy_results_root,
        audit_results_root,
        report_dir,
        split_summary=split_summary,
        baseline_cache_summary=baseline_cache_summary,
        fewshot_cache_summary=fewshot_cache_summary,
    )

    return {
        "split_summary": split_summary,
        "baseline_cache_summary": baseline_cache_summary,
        "fewshot_cache_summary": fewshot_cache_summary,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="transfer3 协议核实与基线固化脚本")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--baseline-cache-dir", default=str(DEFAULT_BASELINE_CACHE_DIR))
    parser.add_argument("--fewshot-cache-dir", default=str(DEFAULT_FEWSHOT_CACHE_DIR))
    parser.add_argument("--legacy-results-root", default=str(DEFAULT_LEGACY_RESULTS_ROOT))
    parser.add_argument("--audit-results-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--no-rebuild-cache", dest="rebuild_cache", action="store_false")
    parser.add_argument("--no-run-experiments", dest="run_experiments", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(rebuild_cache=True, run_experiments=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_transfer3_protocol_audit(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
