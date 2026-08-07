#!/usr/bin/env python3
"""
Plot misclassified thrust_loss samples from historical offline multiclass models.

This helps answer: are thrust_loss->normal confusions caused by "mild" faults
that barely change telemetry under the current flight profile?

Outputs:
  logs/thrust_loss_misclassified_<ts>/
    index.csv            # misclassified list (scene_id, scale, fault_ratio, ...)
    plots_index.csv      # per-sample plot paths
    <scene_id>/signals.png
    <scene_id>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LABEL_NAMES = ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss"]


def _make_model(model_name: str, in_channels: int, num_classes: int):
    from uav_fault.models.cnn.cnn_lstm import CNNLSTMClassifier
    from uav_fault.models.cnn.lstm import BiLSTMClassifier
    from uav_fault.models.cnn.textcnn import TextCNN1d
    from uav_fault.models.cnn.transformer import TransformerClassifier

    if model_name == "cnn_lstm":
        return CNNLSTMClassifier(
            in_channels=in_channels,
            cnn_channels=(64, 128, 128),
            lstm_hidden=128,
            lstm_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        )
    if model_name == "lstm":
        return BiLSTMClassifier(
            in_channels=in_channels,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        )
    if model_name == "transformer":
        return TransformerClassifier(
            in_channels=in_channels,
            d_model=128,
            nhead=4,
            num_layers=4,
            num_classes=num_classes,
            patch_size=8,
            dropout=0.2,
        )
    if model_name == "textcnn":
        return TextCNN1d(
            in_channels=in_channels,
            num_classes=num_classes,
            num_filters=128,
            kernel_sizes=(3, 5, 7, 11, 21),
            dropout=0.3,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _plot_one(csv_path: Path, meta: dict, out_png: Path, *, pre_s: float, post_s: float) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # Reduce clutter: keep only normal/fault if available.
    if "flight_phase" in df.columns:
        df = df[df["flight_phase"].isin(["normal", "fault"])].reset_index(drop=True)

    if "time_s" not in df.columns:
        return {"error": "missing time_s"}

    t = pd.to_numeric(df["time_s"], errors="coerce")
    df = df[t.notna()].copy()
    df["time_s"] = df["time_s"].astype(float)

    fault_start = None
    if "is_fault_active" in df.columns:
        gt = pd.to_numeric(df["is_fault_active"], errors="coerce").fillna(0).astype(int)
        idx = np.where(gt.values == 1)[0]
        if idx.size:
            fault_start = float(df.loc[idx[0], "time_s"])

    center = fault_start if fault_start is not None else float(df["time_s"].quantile(0.4))
    t0, t1 = center - pre_s, center + post_s
    w = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].copy()
    if w.empty:
        w = df.copy()

    def col(name: str):
        return pd.to_numeric(w[name], errors="coerce") if name in w.columns else None

    tt = w["time_s"].values
    motors = [col(f"motor{i}") for i in range(1, 5)]
    thrust_z = col("thrust_z")
    unalloc = col("unalloc_thrust_z")
    pos_z = col("pos_z")
    alt = col("alt_local_m")
    vz = col("vz")
    batt_v = col("battery_voltage_v")
    batt_rem = col("battery_remaining")

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # 1) motors
    ax = axes[0]
    for i, m in enumerate(motors, start=1):
        if m is None:
            continue
        ax.plot(tt, m.values, label=f"motor{i}", lw=1)
    ax.set_ylabel("motor")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=8, loc="upper right")

    # 2) thrust + motor mean
    ax = axes[1]
    if thrust_z is not None:
        ax.plot(tt, thrust_z.values, label="thrust_z", lw=1)
    if all(m is not None for m in motors):
        mm = pd.concat(motors, axis=1).mean(axis=1)
        ax.plot(tt, mm.values, label="motor_mean", lw=1, alpha=0.8)
    ax.set_ylabel("thrust")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # 3) unallocated thrust
    ax = axes[2]
    if unalloc is not None:
        ax.plot(tt, unalloc.values, label="unalloc_thrust_z", lw=1)
        ax.plot(tt, unalloc.abs().values, label="|unalloc|", lw=1, alpha=0.7)
    ax.set_ylabel("unalloc")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # 4) altitude + vz
    ax = axes[3]
    z = pos_z if pos_z is not None else alt
    if z is not None:
        ax.plot(tt, z.values, label="pos_z/alt", lw=1)
    if vz is not None:
        ax2 = ax.twinx()
        ax2.plot(tt, vz.values, label="vz", lw=1, color="tab:orange", alpha=0.85)
        ax2.set_ylabel("vz")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("alt")
    ax.grid(True, alpha=0.25)

    # 5) battery
    ax = axes[4]
    if batt_v is not None:
        ax.plot(tt, batt_v.values, label="battery_voltage_v", lw=1)
    if batt_rem is not None:
        ax.plot(tt, batt_rem.values, label="battery_remaining", lw=1)
    ax.set_ylabel("battery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("time_s")

    # Fault markers
    if fault_start is not None:
        for a in axes:
            a.axvline(fault_start, color="red", ls="--", lw=1, alpha=0.7)

    if "is_fault_active" in w.columns:
        gt = pd.to_numeric(w["is_fault_active"], errors="coerce").fillna(0).astype(float).values
        active = gt > 0.5
        if active.any():
            idx = np.where(active)[0]
            spans = []
            s0 = idx[0]
            prev = idx[0]
            for i in idx[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                spans.append((s0, prev))
                s0 = prev = i
            spans.append((s0, prev))
            for s0, e0 in spans:
                a0, a1 = tt[s0], tt[e0]
                for a in axes:
                    a.axvspan(a0, a1, color="red", alpha=0.08)

    fi = meta.get("fault_injection") if isinstance(meta.get("fault_injection"), dict) else {}
    title = f"{csv_path.stem} | scale={fi.get('scale')} fault_ratio={fi.get('fault_ratio')} | pred=normal (misclassified)"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    return {
        "fault_start_s": fault_start,
        "t_window_start": float(tt[0]) if len(tt) else None,
        "t_window_end": float(tt[-1]) if len(tt) else None,
        "rows_window": int(len(w)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot misclassified thrust_loss samples for an offline seq model.")
    parser.add_argument("--model-name", default="cnn_lstm", choices=["cnn_lstm", "lstm", "transformer", "textcnn"])
    parser.add_argument("--model-dir", default="assets/models/cnn_lstm", help="Directory containing best.pt.")
    parser.add_argument("--cache-test-dir", default="data/processed/seq_cache/test", help="Test split cache (.pt).")
    parser.add_argument("--norm-stats", default="data/processed/seq_cache/norm_stats.json")
    parser.add_argument("--csv-dir", default="data/csv_raw")
    parser.add_argument("--meta-dir", default="data/metadata")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pre-s", type=float, default=20.0)
    parser.add_argument("--post-s", type=float, default=40.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of plots (0=all).")
    args = parser.parse_args()

    repo = Path.cwd()
    sys.path.insert(0, str(repo))

    cache_test = Path(args.cache_test_dir)
    model_dir = Path(args.model_dir)
    csv_dir = Path(args.csv_dir)
    meta_dir = Path(args.meta_dir)
    norm_stats_path = Path(args.norm_stats)

    stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean"], dtype=torch.float32).unsqueeze(1)
    std = torch.tensor(stats["std"], dtype=torch.float32).unsqueeze(1)

    model = _make_model(args.model_name, in_channels=len(stats["columns"]), num_classes=len(LABEL_NAMES))
    state = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Find misclassified thrust_loss
    mis = []
    with torch.no_grad():
        for p in sorted(cache_test.glob("*.pt")):
            s = torch.load(p, map_location="cpu")
            y = int(s["y"])
            if y != 4:
                continue

            x = s["x"].float()
            x = (x - mean) / std
            pred = int(model(x.unsqueeze(0)).argmax(dim=1).item())
            if pred == y:
                continue

            fname = s.get("filename") or (p.stem + ".csv")
            scene_id = Path(fname).stem

            meta_path = meta_dir / f"{scene_id}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            fi = meta.get("fault_injection") if isinstance(meta.get("fault_injection"), dict) else {}
            fi = fi or {}

            mis.append(
                {
                    "scene_id": scene_id,
                    "filename": fname,
                    "pred": LABEL_NAMES[pred],
                    "scale": fi.get("scale"),
                    "fault_ratio": fi.get("fault_ratio"),
                    "sim_time": meta.get("sim_time"),
                }
            )

    mis_df = pd.DataFrame(mis)
    if mis_df.empty:
        print("[OK] No misclassified thrust_loss found.")
        return 0

    mis_df["scale"] = pd.to_numeric(mis_df["scale"], errors="coerce")
    mis_df["fault_ratio"] = pd.to_numeric(mis_df["fault_ratio"], errors="coerce")
    mis_df = mis_df.sort_values(["scale", "fault_ratio", "scene_id"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.out_dir else (Path("logs") / f"thrust_loss_misclassified_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    mis_df.to_csv(out_root / "index.csv", index=False)

    plot_rows = []
    n = len(mis_df) if not args.limit else min(len(mis_df), args.limit)
    for i, r in enumerate(mis_df.to_dict(orient="records")[:n], start=1):
        scene_id = r["scene_id"]
        csv_path = csv_dir / f"{scene_id}.csv"
        meta_path = meta_dir / f"{scene_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        out_dir = out_root / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        info = _plot_one(csv_path, meta, out_dir / "signals.png", pre_s=args.pre_s, post_s=args.post_s)
        plot_rows.append({**r, **info, "out_dir": str(out_dir)})
        if i % 5 == 0 or i == n:
            print(f"[plot] {i}/{n} done")

    pd.DataFrame(plot_rows).to_csv(out_root / "plots_index.csv", index=False)

    print(f"[OK] wrote: {out_root}")
    print(f"[OK] index: {out_root / 'plots_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
