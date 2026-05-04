#!/usr/bin/env python3
"""
Deep audit for the Fault Dataset CSVs produced by /home/jack/Fault Dataset.

Goals:
- Validate label alignment (metadata fault_time_rel vs CSV is_fault_active transition).
- Quantify signal changes pre/post fault for key channels.
- Check whether "hard" severities still show consistent physical signatures.
- Surface potential label noise / injection failures as outliers.

Outputs:
  logs/dataset_deep_audit_<ts>/
    overview.json
    per_flight_stats.csv
    flagged_outliers.csv
    plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_ORDER = ["normal", "motor_failure", "battery_failure", "prop_imbalance", "thrust_loss"]


@dataclass(frozen=True)
class MetaRow:
    scene_id: str
    label: str
    seed: int | None
    sim_time: float | None
    fault_ratio: float | None
    inject_mode: str | None
    ramp_duration: float | None
    curve: str | None
    motor_index: int | None
    effectiveness: float | None
    scale: float | None
    drain_time: float | None
    amplitude: float | None
    frequency_hz: float | None
    fault_time_rel: float | None


def _safe_float(x) -> float | None:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def _safe_int(x) -> int | None:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return None


def _load_meta(meta_path: Path) -> MetaRow:
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    scene_id = str(m.get("scene_id") or meta_path.stem)
    fault_type = m.get("fault_type")
    label = "normal" if fault_type in ("none", None, "", "normal_flight") else str(fault_type)
    seed = _safe_int(m.get("seed"))
    sim_time = _safe_float(m.get("sim_time"))

    fi = m.get("fault_injection") if isinstance(m.get("fault_injection"), dict) else {}
    fi = fi or {}

    return MetaRow(
        scene_id=scene_id,
        label=label,
        seed=seed,
        sim_time=sim_time,
        fault_ratio=_safe_float(fi.get("fault_ratio")),
        inject_mode=str(fi.get("inject_mode")) if fi.get("inject_mode") is not None else None,
        ramp_duration=_safe_float(fi.get("ramp_duration")),
        curve=str(fi.get("curve")) if fi.get("curve") is not None else None,
        motor_index=_safe_int(fi.get("motor_index")),
        effectiveness=_safe_float(fi.get("effectiveness")),
        scale=_safe_float(fi.get("scale")),
        drain_time=_safe_float(fi.get("drain_time")),
        amplitude=_safe_float(fi.get("amplitude")),
        frequency_hz=_safe_float(fi.get("frequency_hz")),
        fault_time_rel=_safe_float(fi.get("fault_time_rel")),
    )


def _robust_median_dt(time_s: pd.Series) -> float | None:
    if time_s is None or len(time_s) < 3:
        return None
    dt = pd.to_numeric(time_s, errors="coerce").diff()
    dt = dt[(dt > 0) & dt.notna()]
    if dt.empty:
        return None
    return float(dt.median())


def _lin_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if len(x2) < 20:
        return None
    # simple least squares slope
    vx = x2 - x2.mean()
    denom = float(np.dot(vx, vx))
    if denom <= 0:
        return None
    slope = float(np.dot(vx, y2 - y2.mean()) / denom)
    return slope


def _mean(arr: np.ndarray) -> float | None:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _rel_change(pre: float | None, post: float | None, eps: float = 1e-9) -> float | None:
    if pre is None or post is None:
        return None
    denom = abs(pre) if abs(pre) > eps else eps
    return float((post - pre) / denom)


def _pick_cols(desired: list[str]):
    desired_set = set(desired)
    return lambda c: c in desired_set


def _compute_per_flight_features(df: pd.DataFrame, label: str) -> dict:
    out: dict[str, float | int | None] = {}

    # Basic shape/time stats.
    out["rows"] = int(len(df))
    out["cols_present"] = int(df.shape[1])

    if "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce")
        out["time_s_min"] = _safe_float(t.min())
        out["time_s_max"] = _safe_float(t.max())
        dt_med = _robust_median_dt(t)
        out["dt_median_s"] = dt_med
        out["hz_est"] = (1.0 / dt_med) if (dt_med is not None and dt_med > 0) else None

    # Offboard alignment (nav_state==14)
    if "nav_state" in df.columns and "time_s" in df.columns:
        nav = pd.to_numeric(df["nav_state"], errors="coerce")
        off = df.loc[nav == 14, "time_s"]
        out["offboard_rows"] = int(off.shape[0])
        out["offboard_start_s"] = _safe_float(off.iloc[0]) if len(off) else None
    else:
        out["offboard_rows"] = None
        out["offboard_start_s"] = None

    # Phase stats
    if "flight_phase" in df.columns:
        vc = df["flight_phase"].astype(str).value_counts(dropna=False)
        out["phase_normal_rows"] = int(vc.get("normal", 0))
        out["phase_fault_rows"] = int(vc.get("fault", 0))
        out["phase_crashed_rows"] = int(vc.get("crashed", 0))

    # Missingness for key channels (mean missing fraction across selected columns).
    key_cols = [
        "motor1",
        "motor2",
        "motor3",
        "motor4",
        "battery_voltage_v",
        "battery_remaining",
        "thrust_z",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_vibration_metric",
        "gyro_vibration_metric",
        "hover_thrust",
        "accel_innov",
        "accel_innov_test_ratio",
        "esc_rpm_1",
        "esc_rpm_2",
        "esc_rpm_3",
        "esc_rpm_4",
    ]
    present = [c for c in key_cols if c in df.columns]
    if present:
        miss = df[present].isna().mean()
        out["missing_key_mean_frac"] = float(miss.mean())
        out["missing_vibration_frac"] = float(
            miss[[c for c in ["accel_vibration_metric", "gyro_vibration_metric"] if c in miss.index]].mean()
        ) if any(c in miss.index for c in ["accel_vibration_metric", "gyro_vibration_metric"]) else None

    # Derived signals for pre/post comparisons.
    if not {"motor1", "motor2", "motor3", "motor4"}.issubset(df.columns):
        return out

    motors = df[["motor1", "motor2", "motor3", "motor4"]].astype(float).to_numpy()
    motor_mean = np.nanmean(motors, axis=1)
    motor_std = np.nanstd(motors, axis=1)
    motor_range = np.nanmax(motors, axis=1) - np.nanmin(motors, axis=1)

    acc_norm = None
    if {"accel_x", "accel_y", "accel_z"}.issubset(df.columns):
        ax = df["accel_x"].astype(float).to_numpy()
        ay = df["accel_y"].astype(float).to_numpy()
        az = df["accel_z"].astype(float).to_numpy()
        acc_norm = np.sqrt(ax * ax + ay * ay + az * az)
        # controller effort proxy
        motor_effort = motor_mean / np.clip(np.abs(az), 0.1, None)
    else:
        motor_effort = None

    gyro_norm = None
    if {"gyro_x", "gyro_y", "gyro_z"}.issubset(df.columns):
        gx = df["gyro_x"].astype(float).to_numpy()
        gy = df["gyro_y"].astype(float).to_numpy()
        gz = df["gyro_z"].astype(float).to_numpy()
        gyro_norm = np.sqrt(gx * gx + gy * gy + gz * gz)

    esc_mean = None
    esc_std = None
    if {"esc_rpm_1", "esc_rpm_2", "esc_rpm_3", "esc_rpm_4"}.issubset(df.columns):
        esc = df[["esc_rpm_1", "esc_rpm_2", "esc_rpm_3", "esc_rpm_4"]].astype(float).to_numpy()
        esc_mean = np.nanmean(esc, axis=1)
        esc_std = np.nanstd(esc, axis=1)

    def seg_stats(mask: np.ndarray) -> dict[str, float | None]:
        idx = np.where(mask)[0]
        if idx.size < 200:  # too short → unstable stats
            return {}

        out2: dict[str, float | None] = {
            "motor_mean": _mean(motor_mean[idx]),
            "motor_std": _mean(motor_std[idx]),
            "motor_range": _mean(motor_range[idx]),
        }
        if motor_effort is not None:
            out2["motor_effort"] = _mean(motor_effort[idx])
        if acc_norm is not None:
            out2["acc_norm"] = _mean(acc_norm[idx])
        if gyro_norm is not None:
            out2["gyro_norm"] = _mean(gyro_norm[idx])
        if "battery_voltage_v" in df.columns:
            out2["battery_voltage_v"] = _mean(df["battery_voltage_v"].astype(float).to_numpy()[idx])
        if "battery_remaining" in df.columns:
            out2["battery_remaining"] = _mean(df["battery_remaining"].astype(float).to_numpy()[idx])
        if "thrust_z" in df.columns:
            out2["thrust_z"] = _mean(df["thrust_z"].astype(float).to_numpy()[idx])
        if "unalloc_thrust_z" in df.columns:
            u = df["unalloc_thrust_z"].astype(float).to_numpy()[idx]
            out2["unalloc_thrust_z"] = _mean(u)
            out2["unalloc_thrust_z_abs"] = _mean(np.abs(u))
        for c in ["unalloc_torque_roll", "unalloc_torque_pitch"]:
            if c in df.columns:
                u = df[c].astype(float).to_numpy()[idx]
                out2[c] = _mean(u)
                out2[f"{c}_abs"] = _mean(np.abs(u))
        if "imbalanced_prop_metric" in df.columns:
            out2["imbalanced_prop_metric"] = _mean(df["imbalanced_prop_metric"].astype(float).to_numpy()[idx])
        if esc_mean is not None:
            out2["esc_mean"] = _mean(esc_mean[idx])
        if esc_std is not None:
            out2["esc_std"] = _mean(esc_std[idx])

        # slopes (battery drift is informative)
        if "time_s" in df.columns:
            t = df["time_s"].astype(float).to_numpy()[idx]
            if "battery_voltage_v" in df.columns:
                v = df["battery_voltage_v"].astype(float).to_numpy()[idx]
                out2["battery_voltage_slope"] = _lin_slope(t, v)
            if "battery_remaining" in df.columns:
                rem = df["battery_remaining"].astype(float).to_numpy()[idx]
                out2["battery_remaining_slope"] = _lin_slope(t, rem)

        return out2

    # Define split masks (windowed around fault start to reduce landing/tail confounders).
    if "time_s" in df.columns:
        t_all = df["time_s"].astype(float).to_numpy()
    else:
        t_all = None

    center_s: float | None = None
    if label != "normal" and "is_fault_active" in df.columns and t_all is not None:
        gt = pd.to_numeric(df["is_fault_active"], errors="coerce").fillna(0).astype(int).to_numpy()
        if (gt == 1).any():
            center_s = float(t_all[np.where(gt == 1)[0][0]])
            out["fault_start_s"] = center_s

    if center_s is None and t_all is not None and np.isfinite(t_all).any():
        # Normal flights (or missing GT): pick a stable interior time point.
        center_s = float(np.nanpercentile(t_all, 40))

    window_s = 10.0
    if center_s is not None and t_all is not None:
        pre_mask = (t_all >= (center_s - window_s)) & (t_all < center_s)
        post_mask = (t_all >= center_s) & (t_all < (center_s + window_s))
    else:
        pre_mask = np.zeros(len(df), dtype=bool)
        post_mask = np.zeros(len(df), dtype=bool)

    # Filter out landed/idle frames using motor activity.
    active = motor_mean > 0.05
    pre_mask &= active
    post_mask &= active

    pre = seg_stats(pre_mask)
    post = seg_stats(post_mask)

    # Store raw means and relative changes.
    for k, v in pre.items():
        out[f"pre_{k}"] = v
    for k, v in post.items():
        out[f"post_{k}"] = v
        out[f"{k}_rel"] = _rel_change(pre.get(k), v)

    # Key heuristic scores for outlier detection.
    out["score_motor_imbalance"] = out.get("motor_std_rel")
    # slope magnitude is more stable than signed rel-change (negative slopes).
    bv0 = pre.get("battery_voltage_slope")
    bv1 = post.get("battery_voltage_slope")
    out["battery_voltage_slope_mag_rel"] = (
        (abs(bv1) - abs(bv0)) / abs(bv0)
        if (bv0 is not None and bv1 is not None and abs(bv0) > 1e-12)
        else None
    )
    out["score_battery_drop"] = out.get("battery_voltage_slope_mag_rel")
    out["score_gyro_change"] = out.get("gyro_norm_rel")
    out["score_thrust_comp"] = out.get("unalloc_thrust_z_abs_rel") or out.get("motor_effort_rel")

    return out


def _save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) dt distribution
    if "dt_median_s" in df.columns:
        x = pd.to_numeric(df["dt_median_s"], errors="coerce").dropna()
        if not x.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(x, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_title("Median dt (time_s diff) Distribution")
            ax.set_xlabel("dt_median_s")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "dt_median_hist.png", dpi=160)
            plt.close(fig)

    # 2) fault label alignment diff
    if "fault_start_diff_s" in df.columns:
        x = pd.to_numeric(df["fault_start_diff_s"], errors="coerce").dropna()
        if not x.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(x, bins=60, color="darkorange", edgecolor="black", alpha=0.85)
            ax.set_title("Fault Start Alignment: observed - expected (s)")
            ax.set_xlabel("fault_start_diff_s")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "fault_start_alignment_hist.png", dpi=160)
            plt.close(fig)

    # 3) per-class boxplots for key relative changes
    key = [
        ("motor_std_rel", "Motor Std Rel Change"),
        ("motor_mean_rel", "Motor Mean Rel Change"),
        ("gyro_norm_rel", "Gyro Norm Rel Change"),
        ("battery_voltage_slope_mag_rel", "Battery Voltage Slope |slope| Rel Change"),
    ]
    for col, title in key:
        if col not in df.columns:
            continue
        sub = df[df["label"].isin(LABEL_ORDER)].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        if sub[col].notna().sum() < 10:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        data = [sub.loc[sub["label"] == lbl, col].dropna().values for lbl in LABEL_ORDER if lbl in sub["label"].values]
        labels = [lbl for lbl in LABEL_ORDER if lbl in sub["label"].values]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"box_{col}.png", dpi=160)
        plt.close(fig)

    # 4) severity vs effect for each fault type (hard cases are at extremes)
    scatter_specs = [
        ("motor_failure", "effectiveness", "motor_std_rel"),
        ("thrust_loss", "scale", "unalloc_thrust_z_abs_rel"),
        ("battery_failure", "drain_time", "battery_voltage_slope_mag_rel"),
        ("prop_imbalance", "amplitude", "gyro_norm_rel"),
    ]
    for fault_label, sev_col, y_col in scatter_specs:
        if sev_col not in df.columns or y_col not in df.columns:
            continue
        g = df[df["label"] == fault_label].copy()
        g[sev_col] = pd.to_numeric(g[sev_col], errors="coerce")
        g[y_col] = pd.to_numeric(g[y_col], errors="coerce")
        g = g.dropna(subset=[sev_col, y_col])
        if len(g) < 20:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(g[sev_col].values, g[y_col].values, s=10, alpha=0.6, color="tab:blue")
        ax.set_title(f"{fault_label}: {y_col} vs {sev_col}")
        ax.set_xlabel(sev_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_{fault_label}_{sev_col}_{y_col}.png", dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep audit of the Fault Dataset CSVs + metadata.")
    parser.add_argument("--csv-dir", default="data/csv_raw", help="CSV directory (symlink ok).")
    parser.add_argument("--meta-dir", default="data/metadata", help="Metadata JSON directory (symlink ok).")
    parser.add_argument("--out-dir", default=None, help="Output directory (default logs/dataset_deep_audit_<ts>).")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0=all).")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    meta_dir = Path(args.meta_dir)
    if not csv_dir.exists():
        raise SystemExit(f"[ERROR] csv dir not found: {csv_dir}")
    if not meta_dir.exists():
        raise SystemExit(f"[ERROR] meta dir not found: {meta_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs") / f"dataset_deep_audit_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(meta_dir.glob("*.json"))
    if args.limit and args.limit > 0:
        meta_paths = meta_paths[: args.limit]

    meta_rows: list[MetaRow] = []
    for p in meta_paths:
        try:
            meta_rows.append(_load_meta(p))
        except Exception:
            continue

    meta_df = pd.DataFrame([m.__dict__ for m in meta_rows])
    # Keep a stable label order for later grouping.
    meta_df["label"] = pd.Categorical(meta_df["label"], categories=LABEL_ORDER, ordered=True)

    # Per-flight CSV audit
    desired_cols = [
        "time_s",
        "nav_state",
        "flight_phase",
        "is_fault_active",
        "motor1",
        "motor2",
        "motor3",
        "motor4",
        "esc_rpm_1",
        "esc_rpm_2",
        "esc_rpm_3",
        "esc_rpm_4",
        "battery_voltage_v",
        "battery_remaining",
        "thrust_z",
        "unalloc_thrust_z",
        "unalloc_torque_roll",
        "unalloc_torque_pitch",
        "imbalanced_prop_metric",
        "hover_thrust",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_vibration_metric",
        "gyro_vibration_metric",
        "accel_innov",
        "accel_innov_test_ratio",
    ]

    per_rows: list[dict] = []
    n = len(meta_df)
    for i, r in meta_df.iterrows():
        scene_id = r["scene_id"]
        label = str(r["label"])
        csv_path = csv_dir / f"{scene_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, usecols=_pick_cols(desired_cols))
        feats = _compute_per_flight_features(df, label)
        feats["scene_id"] = scene_id
        feats["label"] = label

        # Label alignment: compare observed is_fault_active start vs expected (offboard_start + meta fault_time_rel)
        if label != "normal" and "is_fault_active" in df.columns and "time_s" in df.columns:
            gt = pd.to_numeric(df["is_fault_active"], errors="coerce").fillna(0).astype(int).to_numpy()
            t = df["time_s"].astype(float).to_numpy()
            if (gt == 1).any():
                fault_start = float(t[np.where(gt == 1)[0][0]])
                feats["fault_start_s"] = fault_start
            else:
                feats["fault_start_s"] = None

            off_start = feats.get("offboard_start_s")
            ft_rel = r.get("fault_time_rel")
            if off_start is not None and isinstance(ft_rel, (int, float)) and math.isfinite(float(ft_rel)):
                expected = float(off_start) + float(ft_rel)
                feats["fault_start_expected_s"] = expected
                if feats.get("fault_start_s") is not None:
                    feats["fault_start_diff_s"] = float(feats["fault_start_s"]) - expected
                else:
                    feats["fault_start_diff_s"] = None
            else:
                feats["fault_start_expected_s"] = None
                feats["fault_start_diff_s"] = None

        # Attach metadata fields for grouping.
        for c in [
            "seed",
            "sim_time",
            "fault_ratio",
            "inject_mode",
            "ramp_duration",
            "curve",
            "motor_index",
            "effectiveness",
            "scale",
            "drain_time",
            "amplitude",
            "frequency_hz",
            "fault_time_rel",
        ]:
            feats[c] = r.get(c)

        per_rows.append(feats)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"[audit] {i + 1}/{n} processed")

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(out_dir / "per_flight_stats.csv", index=False)

    # Outlier heuristics: flag potential "no-effect" injections.
    flags = []
    if not per_df.empty:
        df2 = per_df.copy()
        df2["motor_std_rel"] = pd.to_numeric(df2.get("motor_std_rel"), errors="coerce")
        df2["gyro_norm_rel"] = pd.to_numeric(df2.get("gyro_norm_rel"), errors="coerce")
        df2["battery_voltage_slope_mag_rel"] = pd.to_numeric(df2.get("battery_voltage_slope_mag_rel"), errors="coerce")
        df2["unalloc_thrust_z_abs_rel"] = pd.to_numeric(df2.get("unalloc_thrust_z_abs_rel"), errors="coerce")
        df2["motor_effort_rel"] = pd.to_numeric(df2.get("motor_effort_rel"), errors="coerce")

        # motor_failure: expect imbalance increase; severe faults should be big.
        mf = df2[df2["label"] == "motor_failure"].copy()
        if not mf.empty:
            suspicious = mf[mf["motor_std_rel"].notna() & (mf["motor_std_rel"] < 0.3)]
            for _, row in suspicious.iterrows():
                flags.append({"scene_id": row["scene_id"], "label": row["label"], "reason": "motor_std_rel_low"})

        # battery_failure: expect voltage slope more negative after fault.
        bf = df2[df2["label"] == "battery_failure"].copy()
        if not bf.empty:
            # More drain => larger |slope| after fault (mag_rel > 0).
            suspicious = bf[bf["battery_voltage_slope_mag_rel"].notna() & (bf["battery_voltage_slope_mag_rel"] < 0.05)]
            for _, row in suspicious.iterrows():
                flags.append({"scene_id": row["scene_id"], "label": row["label"], "reason": "battery_slope_mag_change_small"})

        # prop_imbalance: expect gyro_norm increase.
        pi = df2[df2["label"] == "prop_imbalance"].copy()
        if not pi.empty:
            suspicious = pi[pi["gyro_norm_rel"].notna() & (pi["gyro_norm_rel"] < 0.1)]
            for _, row in suspicious.iterrows():
                flags.append({"scene_id": row["scene_id"], "label": row["label"], "reason": "gyro_norm_rel_low"})

        # thrust_loss: expect motor_effort increase (more controller effort).
        tf = df2[df2["label"] == "thrust_loss"].copy()
        if not tf.empty:
            suspicious = tf[tf["unalloc_thrust_z_abs_rel"].notna() & (tf["unalloc_thrust_z_abs_rel"] < 0.05)]
            for _, row in suspicious.iterrows():
                flags.append({"scene_id": row["scene_id"], "label": row["label"], "reason": "unalloc_thrust_abs_rel_low"})

    flagged_df = pd.DataFrame(flags).drop_duplicates() if flags else pd.DataFrame(columns=["scene_id", "label", "reason"])
    flagged_df.to_csv(out_dir / "flagged_outliers.csv", index=False)

    # Overview JSON
    counts = per_df["label"].value_counts(dropna=False).to_dict() if "label" in per_df.columns else {}
    overview = {
        "ts": ts,
        "n_flights": int(len(per_df)),
        "counts": {k: int(v) for k, v in counts.items()},
        "outliers": {
            "n_flagged": int(len(flagged_df)),
            "by_reason": {k: int(v) for k, v in flagged_df["reason"].value_counts().to_dict().items()} if not flagged_df.empty else {},
        },
    }
    (out_dir / "overview.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Plots
    _save_plots(per_df, out_dir)

    print(f"[OK] wrote: {out_dir / 'per_flight_stats.csv'}")
    print(f"[OK] wrote: {out_dir / 'flagged_outliers.csv'}")
    print(f"[OK] wrote: {out_dir / 'overview.json'}")
    print(f"[OK] plots: {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
