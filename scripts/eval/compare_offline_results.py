#!/usr/bin/env python3
"""
Compare offline (5-class) model results under assets/models/*/metrics_test.json.

Default behavior:
- reads every assets/models/<dir>/metrics_test.json
- excludes online detector directories (online_detector_*) and ablation/
- prints a sorted table to stdout
- writes CSV + 1-2 comparison plots into assets/models/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_offline_model_dir(model_dir: Path) -> bool:
    name = model_dir.name
    if not model_dir.is_dir():
        return False
    if name.startswith("online_detector_"):
        return False
    if name in {"ablation"}:
        return False
    return (model_dir / "metrics_test.json").exists()


def _load_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_class_f1(classification_report: dict) -> dict[str, float]:
    # sklearn classification_report(output_dict=True) uses label keys plus
    # 'accuracy', 'macro avg', 'weighted avg'.
    ignore = {"accuracy", "macro avg", "weighted avg"}
    out: dict[str, float] = {}
    for k, v in classification_report.items():
        if k in ignore:
            continue
        if isinstance(v, dict) and "f1-score" in v:
            out[k] = float(v["f1-score"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare offline model metrics under assets/models/*.")
    parser.add_argument("--models-dir", default="assets/models", help="Models root directory.")
    parser.add_argument(
        "--output-prefix",
        default="offline_model",
        help="Output prefix written under models-dir (CSV/PNG).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate PNG plots (requires a working matplotlib installation).",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"[ERROR] models dir not found: {models_dir}")

    rows = []
    per_class = {}

    for model_dir in sorted(models_dir.iterdir()):
        if not _is_offline_model_dir(model_dir):
            continue
        metrics_path = model_dir / "metrics_test.json"
        metrics = _load_metrics(metrics_path)

        rows.append(
            {
                "model": model_dir.name,
                "accuracy": float(metrics.get("accuracy", float("nan"))),
                "macro_f1": float(metrics.get("macro_f1", float("nan"))),
                "weighted_f1": float(metrics.get("weighted_f1", float("nan"))),
                "has_results_png": (model_dir / "results.png").exists(),
            }
        )

        cr = metrics.get("classification_report")
        if isinstance(cr, dict):
            per_class[model_dir.name] = _extract_class_f1(cr)

    if not rows:
        print("[WARN] no offline models found (missing assets/models/*/metrics_test.json).")
        return 0

    import pandas as pd  # local import to keep script lightweight if unused
    import numpy as np

    df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

    # Console table (compact)
    display = df.copy()
    for c in ["accuracy", "macro_f1", "weighted_f1"]:
        display[c] = display[c].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "nan")
    print(display.to_string(index=False))

    # CSV output
    out_csv = models_dir / f"{args.output_prefix}_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote: {out_csv}")

    if not args.plots:
        print("[INFO] plots skipped (pass --plots to generate PNGs).")
        return 0

    # Plot: macro_f1 + accuracy (top-to-bottom = best-to-worst)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib/numpy ABI issues can raise non-ImportError exceptions
        print(f"[WARN] failed to import matplotlib; skipped plots: {e}")
        return 0

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.4 * len(df) + 1)))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["macro_f1"].values, height=0.4, label="Macro F1")
    ax.barh(y + 0.2, df["accuracy"].values, height=0.4, label="Accuracy")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Offline 5-Class Model Comparison (Test)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_png = models_dir / f"{args.output_prefix}_comparison.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[OK] wrote: {out_png}")

    # Plot: per-class F1 heatmap (if available)
    if per_class:
        # Build a stable class list across models.
        all_classes = sorted({c for m in per_class.values() for c in m.keys()})
        mat = np.full((len(df), len(all_classes)), np.nan, dtype=np.float32)
        for i, model_name in enumerate(df["model"].tolist()):
            f1s = per_class.get(model_name, {})
            for j, cls in enumerate(all_classes):
                if cls in f1s:
                    mat[i, j] = f1s[cls]

        fig2, ax2 = plt.subplots(figsize=(max(8, 1.2 * len(all_classes)), max(3.5, 0.45 * len(df) + 1)))
        im = ax2.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
        ax2.set_yticks(np.arange(len(df)))
        ax2.set_yticklabels(df["model"].tolist())
        ax2.set_xticks(np.arange(len(all_classes)))
        ax2.set_xticklabels(all_classes, rotation=30, ha="right")
        ax2.set_title("Per-Class F1 (Test)")
        fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        fig2.tight_layout()

        out_png2 = models_dir / f"{args.output_prefix}_per_class_f1.png"
        fig2.savefig(out_png2, dpi=160)
        plt.close(fig2)
        print(f"[OK] wrote: {out_png2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
