#!/usr/bin/env python3
"""
Dataset audit for UAV Fault Dataset (CSV + metadata JSON).

Focus: sample-check 3-4 flights per class (5 classes) and generate:
- per-sample summaries + quick plots
- dataset-level distribution stats from metadata

Notes:
- Uses the csv-data-summarizer skill's summarize_csv() via a dynamic import.
- Avoids seaborn dependency by injecting a tiny stub module into sys.modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SKILL_ANALYZE_PY = Path("/home/jack/.cc-switch/skills/csv-data-summarizer/analyze.py")


@dataclass(frozen=True)
class Sample:
    label: str  # normal / motor_failure / ... (metadata fault_type mapping)
    csv_path: Path
    meta_path: Path
    seed: int | None
    sim_time: float | None
    fault_injection: dict | None

    @property
    def stem(self) -> str:
        return self.csv_path.stem


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_samples(csv_dir: Path, meta_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        meta_path = meta_dir / f"{csv_path.stem}.json"
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        ft = meta.get("fault_type", "none")
        label = "normal" if ft in ("none", None, "") else str(ft)
        samples.append(
            Sample(
                label=label,
                csv_path=csv_path,
                meta_path=meta_path,
                seed=meta.get("seed"),
                sim_time=meta.get("sim_time"),
                fault_injection=meta.get("fault_injection"),
            )
        )
    return samples


def _choose_closest(items: list[Sample], key_fn, target) -> Sample:
    def dist(s: Sample) -> float:
        v = key_fn(s)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("inf")
        return abs(float(v) - float(target))

    return min(items, key=dist)


def _pick_samples(all_samples: list[Sample], n_per_class: int) -> dict[str, list[Sample]]:
    # Hard-to-detect settings (closest-to-normal) based on metadata fields:
    # - motor_failure: effectiveness max (0.8)
    # - thrust_loss: scale max (0.9)
    # - battery_failure: drain_time max (55)
    # - prop_imbalance: amplitude min (0.1)
    target = {
        "motor_failure": {"field": ("fault_injection", "effectiveness"), "value": 0.8},
        "thrust_loss": {"field": ("fault_injection", "scale"), "value": 0.9},
        "battery_failure": {"field": ("fault_injection", "drain_time"), "value": 55.0},
        "prop_imbalance": {"field": ("fault_injection", "amplitude"), "value": 0.1},
    }

    by_label: dict[str, list[Sample]] = defaultdict(list)
    for s in all_samples:
        by_label[s.label].append(s)

    picked: dict[str, list[Sample]] = {}

    # Normal: pick diverse sim_time (short/median/long + another)
    normals = sorted(by_label.get("normal", []), key=lambda s: (s.sim_time or 0.0, s.stem))
    if normals:
        idxs = sorted(
            {
                0,
                len(normals) // 2,
                max(0, len(normals) - 1),
                len(normals) // 3,
            }
        )
        picked["normal"] = [normals[i] for i in idxs[:n_per_class]]
    else:
        picked["normal"] = []

    # Faults: enforce target setting + diversify by fault_ratio buckets (early/mid/late injection)
    ratio_targets = [0.28, 0.35, 0.45, 0.52]

    def get_fault_ratio(s: Sample) -> float | None:
        fi = s.fault_injection or {}
        v = fi.get("fault_ratio")
        return float(v) if isinstance(v, (int, float)) else None

    def get_field(s: Sample, path: tuple[str, str]) -> float | None:
        root, leaf = path
        if root != "fault_injection":
            return None
        fi = s.fault_injection or {}
        v = fi.get(leaf)
        return float(v) if isinstance(v, (int, float)) else None

    for lbl, cfg in target.items():
        candidates = [s for s in by_label.get(lbl, []) if get_field(s, cfg["field"]) == cfg["value"]]
        if not candidates:
            picked[lbl] = []
            continue

        selected: list[Sample] = []
        used = set()
        for rt in ratio_targets[:n_per_class]:
            s = _choose_closest([c for c in candidates if c.stem not in used], get_fault_ratio, rt)
            selected.append(s)
            used.add(s.stem)
        picked[lbl] = selected

    # Motor failure: if we have room, ensure motor_index diversity (1..4) for interpretability.
    mf = picked.get("motor_failure", [])
    if mf:
        need = {}
        for i in (1, 2, 3, 4):
            need[i] = True
        for s in mf:
            mi = int((s.fault_injection or {}).get("motor_index") or 0)
            if mi in need:
                need[mi] = False
        missing = [i for i, ok in need.items() if ok]
        if missing:
            candidates = [
                s
                for s in by_label.get("motor_failure", [])
                if (s.fault_injection or {}).get("effectiveness") == 0.8
            ]
            # Replace from the end (least important) to cover missing motor indices.
            for mi in missing:
                rep = next((c for c in candidates if (c.fault_injection or {}).get("motor_index") == mi), None)
                if rep is None:
                    continue
                mf[-1] = rep
        picked["motor_failure"] = mf[:n_per_class]

    # Prop imbalance: encourage frequency diversity (10/15/20).
    pi = picked.get("prop_imbalance", [])
    if pi:
        have_freq = {float((s.fault_injection or {}).get("frequency_hz") or -1) for s in pi}
        want = [10.0, 15.0, 20.0]
        missing = [f for f in want if f not in have_freq]
        if missing:
            candidates = [
                s
                for s in by_label.get("prop_imbalance", [])
                if (s.fault_injection or {}).get("amplitude") == 0.1
            ]
            for f in missing:
                rep = next((c for c in candidates if (c.fault_injection or {}).get("frequency_hz") == f), None)
                if rep is None:
                    continue
                pi[-1] = rep
        picked["prop_imbalance"] = pi[:n_per_class]

    return picked


def _make_seaborn_stub():
    # analyze.py only uses seaborn.heatmap; provide a minimal replacement.
    import types
    import numpy as np
    import matplotlib.pyplot as plt

    sns = types.ModuleType("seaborn")

    def heatmap(data, annot=True, cmap="coolwarm", center=0, square=True, linewidths=1, **kwargs):
        arr = np.asarray(data, dtype=float)
        ax = plt.gca()

        # If the matrix is large, annotations are unreadable and very slow.
        n = arr.shape[0]
        do_annot = bool(annot) and n <= 40

        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        vmax = max(vmax, 1e-6)
        vmin, vmax = (-vmax, vmax) if center == 0 else (np.nanmin(arr), np.nanmax(arr))

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

        # Tick labels if pandas DataFrame.
        try:
            cols = list(getattr(data, "columns", []))
            idx = list(getattr(data, "index", []))
        except Exception:
            cols, idx = [], []
        if cols:
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90, fontsize=6)
        if idx:
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(idx, fontsize=6)

        if linewidths:
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)

        if do_annot:
            for i in range(n):
                for j in range(n):
                    v = arr[i, j]
                    if not np.isfinite(v):
                        continue
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

        # Keep API similar to seaborn: return the Axes.
        return ax

    sns.heatmap = heatmap
    return sns


def _import_skill_summarizer():
    if not SKILL_ANALYZE_PY.exists():
        raise FileNotFoundError(f"Skill analyze.py not found: {SKILL_ANALYZE_PY}")

    import importlib.util

    # Inject seaborn stub before loading the skill module.
    sys.modules.setdefault("seaborn", _make_seaborn_stub())

    spec = importlib.util.spec_from_file_location("csv_data_summarizer_skill", str(SKILL_ANALYZE_PY))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "summarize_csv"):
        raise AttributeError("Skill module does not expose summarize_csv(file_path)")
    return mod.summarize_csv


def _select_columns(df, wanted: list[str]) -> list[str]:
    cols = [c for c in wanted if c in df.columns]
    # Keep a few additional useful columns if present.
    for extra in ["battery_current_a", "airspeed_m_s", "dist_bottom", "hover_thrust"]:
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    return cols


def _compute_basic_stats(df) -> dict:
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    missing_pct = float(missing_cells) / float(max(1, n_rows * n_cols)) * 100.0
    out = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
    }

    if "time_s" in df.columns:
        t = df["time_s"].astype(float)
        out["time_s_min"] = float(t.min())
        out["time_s_max"] = float(t.max())
        dt = t.diff().dropna()
        if len(dt) > 0:
            out["dt_median_s"] = float(dt.median())
            out["hz_est"] = float(1.0 / out["dt_median_s"]) if out["dt_median_s"] > 0 else None
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _plot_key_signals(df, meta: dict, out_png: Path) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "time_s" not in df.columns:
        return

    t = df["time_s"].astype(float).values

    def get(col: str):
        if col not in df.columns:
            return None
        return df[col].astype(float).values

    motors = [get(f"motor{i}") for i in range(1, 5)]
    roll = get("roll")
    pitch = get("pitch")
    pos_z = get("pos_z")
    if pos_z is None:
        pos_z = get("alt_local_m")
    batt_v = get("battery_voltage_v")
    batt_rem = get("battery_remaining")

    ax = get("accel_x")
    ay = get("accel_y")
    az = get("accel_z")
    gx = get("gyro_x")
    gy = get("gyro_y")
    gz = get("gyro_z")
    acc_norm = None
    gyro_norm = None
    if ax is not None and ay is not None and az is not None:
        acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if gx is not None and gy is not None and gz is not None:
        gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    fault_t = None
    fi = meta.get("fault_injection") or {}
    if isinstance(fi, dict) and isinstance(fi.get("fault_time_rel"), (int, float)):
        fault_t = float(fi["fault_time_rel"])

    is_fault_active = None
    if "is_fault_active" in df.columns:
        try:
            is_fault_active = df["is_fault_active"].fillna(0).astype(float).values
        except Exception:
            is_fault_active = None

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1) Motors
    ax0 = axes[0]
    for i, m in enumerate(motors, start=1):
        if m is None:
            continue
        ax0.plot(t, m, label=f"motor{i}", linewidth=1)
    ax0.set_ylabel("motor")
    ax0.grid(True, alpha=0.25)
    ax0.legend(ncol=4, fontsize=8, loc="upper right")

    # 2) Attitude
    ax1 = axes[1]
    if roll is not None:
        ax1.plot(t, roll, label="roll", linewidth=1)
    if pitch is not None:
        ax1.plot(t, pitch, label="pitch", linewidth=1)
    ax1.set_ylabel("attitude")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="upper right")

    # 3) Position + battery
    ax2 = axes[2]
    if pos_z is not None:
        ax2.plot(t, pos_z, label="pos_z/alt", linewidth=1)
    if batt_v is not None:
        ax2.plot(t, batt_v, label="battery_voltage_v", linewidth=1)
    if batt_rem is not None:
        ax2.plot(t, batt_rem, label="battery_remaining", linewidth=1)
    ax2.set_ylabel("pos/batt")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, loc="upper right")

    # 4) Acc/Gyro norms + vibration if present
    ax3 = axes[3]
    if acc_norm is not None:
        ax3.plot(t, acc_norm, label="acc_norm", linewidth=1)
    if gyro_norm is not None:
        ax3.plot(t, gyro_norm, label="gyro_norm", linewidth=1)
    for vib in ["accel_vibration_metric", "gyro_vibration_metric"]:
        v = get(vib)
        if v is not None:
            ax3.plot(t, v, label=vib, linewidth=1)
    ax3.set_ylabel("imu/vib")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlabel("time_s")

    # Mark fault start.
    if fault_t is not None:
        for a in axes:
            a.axvline(fault_t, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes[0].text(fault_t, axes[0].get_ylim()[1], "fault_start", color="red", fontsize=8, va="top", ha="left")

    # Shade fault-active area if present.
    if is_fault_active is not None:
        active = is_fault_active > 0.5
        if active.any():
            # Find contiguous active spans.
            idx = np.where(active)[0]
            start = idx[0]
            prev = idx[0]
            spans = []
            for i in idx[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                spans.append((start, prev))
                start = prev = i
            spans.append((start, prev))
            for s, e in spans:
                t0, t1 = t[s], t[e]
                for a in axes:
                    a.axvspan(t0, t1, color="red", alpha=0.08)

    title = meta.get("scene_id") or out_png.parent.name
    ft = meta.get("fault_type")
    fig.suptitle(f"{title} ({ft})", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit dataset samples and generate summaries/plots.")
    parser.add_argument("--csv-dir", default="data/csv_raw", help="CSV directory (may be a symlink).")
    parser.add_argument("--meta-dir", default="data/metadata", help="Metadata JSON directory.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to logs/dataset_audit_<ts>.")
    parser.add_argument("--n-per-class", type=int, default=4, help="Samples per class (3-4 recommended).")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    meta_dir = Path(args.meta_dir)
    if not csv_dir.exists():
        raise SystemExit(f"[ERROR] csv dir not found: {csv_dir}")
    if not meta_dir.exists():
        raise SystemExit(f"[ERROR] meta dir not found: {meta_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.out_dir else Path("logs") / f"dataset_audit_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_samples = _iter_samples(csv_dir, meta_dir)
    picked = _pick_samples(all_samples, n_per_class=max(1, min(4, args.n_per_class)))

    # Dataset-level distribution from metadata (fast, no CSV parsing).
    dist = {
        "counts": Counter(s.label for s in all_samples),
        "picked": {k: [s.stem for s in v] for k, v in picked.items()},
    }
    _write_json(out_root / "dataset_overview.json", {
        "counts": {k: int(v) for k, v in dist["counts"].items()},
        "picked": dist["picked"],
    })

    summarize_csv = _import_skill_summarizer()

    import pandas as pd

    wanted_cols = [
        "time_s",
        "flight_phase",
        "is_fault_active",
        "motor1",
        "motor2",
        "motor3",
        "motor4",
        "roll",
        "pitch",
        "yaw",
        "rollspeed",
        "pitchspeed",
        "yawspeed",
        "pos_z",
        "alt_local_m",
        "vx",
        "vy",
        "vz",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "battery_voltage_v",
        "battery_remaining",
        "rates_sp_roll",
        "rates_sp_pitch",
        "rates_sp_yaw",
        "rate_err_roll",
        "rate_err_pitch",
        "rate_err_yaw",
        "thrust_z",
        "accel_vibration_metric",
        "gyro_vibration_metric",
    ]

    sample_rows = []
    for label, items in picked.items():
        for s in items:
            meta = _load_json(s.meta_path)
            out_dir = out_root / label / s.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_json(out_dir / "metadata.json", meta)

            df = pd.read_csv(s.csv_path)
            stats = _compute_basic_stats(df)

            # Per-column missingness (top 20).
            miss = df.isna().mean().sort_values(ascending=False)
            miss.head(20).to_csv(out_dir / "missingness_top20.csv", header=["missing_frac"])

            # Trimmed CSV for the skill (avoid huge correlation plots).
            cols = _select_columns(df, wanted_cols)
            df_trim = df[cols].copy()
            # Work around a bug in the skill's time-series block:
            # it treats any column containing "time" as datetime, then tries to sum it.
            # Renaming keeps the skill usable for telemetry CSVs.
            df_skill = df_trim.rename(columns={"time_s": "t_s"}) if "time_s" in df_trim.columns else df_trim
            trimmed_csv = out_dir / "trimmed.csv"
            df_skill.to_csv(trimmed_csv, index=False)

            # Run the skill analysis inside the sample directory to keep plots separate.
            cwd = os.getcwd()
            os.chdir(out_dir)
            try:
                skill_summary = summarize_csv("trimmed.csv")
                Path("skill_summary.txt").write_text(skill_summary, encoding="utf-8")
            finally:
                os.chdir(cwd)

            # Domain-specific plot on trimmed DF (signals over time).
            _plot_key_signals(df_trim, meta, out_dir / "signals.png")

            fi = meta.get("fault_injection") or {}
            sample_rows.append(
                {
                    "label": label,
                    "file": s.csv_path.name,
                    "seed": meta.get("seed"),
                    "sim_time": meta.get("sim_time"),
                    "fault_ratio": fi.get("fault_ratio") if isinstance(fi, dict) else None,
                    "inject_mode": fi.get("inject_mode") if isinstance(fi, dict) else None,
                    "severity": (
                        fi.get("effectiveness")
                        or fi.get("scale")
                        or fi.get("drain_time")
                        or fi.get("amplitude")
                    ),
                    **stats,
                    "out_dir": str(out_dir),
                }
            )

    # Write a compact index CSV so you can quickly jump to artifacts.
    import pandas as pd

    idx_df = pd.DataFrame(sample_rows)
    idx_df.to_csv(out_root / "picked_samples_index.csv", index=False)

    # Minimal markdown report (high-level; details live in per-sample folders).
    lines = []
    lines.append(f"# Dataset Audit ({ts})")
    lines.append("")
    lines.append("## Class Counts (from metadata)")
    for k, v in sorted(dist["counts"].items(), key=lambda kv: kv[0]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Picked Samples")
    lines.append(f"Index: `{out_root / 'picked_samples_index.csv'}`")
    lines.append("")
    for label in ["normal", "motor_failure", "battery_failure", "prop_imbalance", "thrust_loss"]:
        items = picked.get(label, [])
        if not items:
            continue
        lines.append(f"### {label}")
        for s in items:
            lines.append(f"- `{(out_root / label / s.stem).as_posix()}`")
        lines.append("")
    (out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote audit to: {out_root}")
    print(f"[OK] report: {out_root / 'REPORT.md'}")
    print(f"[OK] index:  {out_root / 'picked_samples_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
