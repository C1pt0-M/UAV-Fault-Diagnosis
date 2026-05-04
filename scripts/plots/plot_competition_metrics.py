#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "competition" / "2026-software-app"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"


def _plot_offline_generalization() -> None:
    df = pd.read_csv(METRICS_DIR / "offline_model_summary.csv")
    test_df = df[df["split"] == "test"].set_index("model")
    ext_df = df[df["split"] == "external_test"].set_index("model")
    merged = test_df.join(ext_df, lsuffix="_test", rsuffix="_external")

    models = list(merged.index)
    x = range(len(models))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([i - width / 2 for i in x], merged["accuracy_test"], width, label="test")
    axes[0].bar(
        [i + width / 2 for i in x], merged["accuracy_external"], width, label="external"
    )
    axes[0].set_title("Offline Accuracy")
    axes[0].set_xticks(list(x), models)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar([i - width / 2 for i in x], merged["macro_f1_test"], width, label="test")
    axes[1].bar(
        [i + width / 2 for i in x],
        merged["macro_f1_external"],
        width,
        label="external",
    )
    axes[1].set_title("Offline Macro F1")
    axes[1].set_xticks(list(x), models)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Offline Multiclass: Same-distribution vs External")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "offline_generalization.png", dpi=180)
    plt.close(fig)


def _plot_online_window_comparison() -> None:
    df = pd.read_csv(METRICS_DIR / "online_window_summary.csv")
    df = df[df["split"] == "test"].copy().set_index("model")

    metrics = ["accuracy", "macro_f1", "f1_fault"]
    display_names = ["Accuracy", "Macro F1", "Fault F1"]
    models = list(df.index)
    x = range(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, model in enumerate(models):
        offset = (-width / 2) if idx == 0 else (width / 2)
        ax.bar(
            [i + offset for i in x],
            [df.loc[model, metric] for metric in metrics],
            width,
            label=model,
        )

    ax.set_title("Online Window-level Comparison")
    ax.set_xticks(list(x), display_names)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "online_window_comparison.png", dpi=180)
    plt.close(fig)


def _plot_online_event_evidence() -> None:
    df = pd.read_csv(METRICS_DIR / "online_event_summary.csv").set_index("dataset")
    normal = df.loc["normal_baseline"]
    external = df.loc["external_fault_test"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        ["normal false alarm", "fault recall", "fault hit rate"],
        [
            normal["false_alarm_rate_normal"],
            external["file_recall"],
            external["hit_rate"],
        ],
        color=["#c0392b", "#1f77b4", "#2ca02c"],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Event-level Quality")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        ["early warn rate", "lead median(s)", "post delay median(s)"],
        [
            external["early_warning_rate"],
            external["early_warning_lead_median_s"],
            external["post_onset_delay_median_s"],
        ],
        color=["#ff7f0e", "#9467bd", "#8c564b"],
    )
    axes[1].set_title("Event-level Timing")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Online Event-level Evidence")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "online_event_evidence.png", dpi=180)
    plt.close(fig)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_offline_generalization()
    _plot_online_window_comparison()
    _plot_online_event_evidence()
    print(f"[OK] wrote figures to: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
