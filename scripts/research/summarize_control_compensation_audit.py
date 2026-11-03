"""Build compact tables and native-clock figures from the fixed audit sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("artifacts/control_compensation_feasibility_audit_sampled_v1")


def read_csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name, low_memory=False, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.results, args.output
    out.mkdir(parents=True, exist_ok=False)

    inventory = read_csv(root, "raw_signal_inventory.csv")
    metadata = read_csv(root, "case_fault_metadata.csv")
    coverage = read_csv(root, "frozen_motor_normal_coverage.csv")
    tlog = read_csv(root, "sampled_TLog_internal_fields.csv")
    timelines = read_csv(root, "within_clock_timeline_summary.csv")

    # One row per source case: preserve independent logs as the unit of count.
    source_rows = []
    for uid, meta in metadata.set_index("case_uid").iterrows():
        q = inventory[inventory.case_uid.eq(uid)]
        row = meta.to_dict()
        row["case_uid"] = uid
        for kind, time_field in [("TrueState", "trueTime"), ("UAVState", "uavTime")]:
            t = q[(q.topic == kind) & (q.field == time_field)]
            row[f"{kind}_rows"] = int(float(t.iloc[0].records)) if len(t) else 0
            row[f"{kind}_nominal_hz_sampled"] = 50.0
        for topic in ["actuator_controls_0", "actuator_outputs", "rfly_ctrl_lxl", "sensor_combined", "vehicle_attitude", "vehicle_local_position", "test_motor"]:
            qtopic = q[q.topic.eq(topic)]
            row[f"{topic}_present"] = bool(len(qtopic))
            row[f"{topic}_records"] = float(pd.to_numeric(qtopic.n, errors="coerce").median()) if len(qtopic) else np.nan
            row[f"{topic}_median_hz"] = float(pd.to_numeric(qtopic.approx_hz, errors="coerce").median()) if len(qtopic) else np.nan
            if topic == "actuator_controls_0" and len(qtopic):
                row["controls_four_fields_complete"] = bool(qtopic.channel_complete.fillna(False).astype(str).str.lower().eq("true").all())
                row["controls_all_finite"] = bool(qtopic.channels_finite_all.fillna(False).astype(str).str.lower().eq("true").all())
            if topic == "actuator_outputs" and len(qtopic):
                row["outputs_first4_fields_finite"] = bool(qtopic.first4_finite_all.fillna(False).astype(str).str.lower().eq("true").all())
                row["outputs_noutputs_values"] = ";".join(sorted(set(qtopic.noutputs_values.astype(str))))
        rpm = q[(q.topic == "TrueState") & q.field.astype(str).isin([f"motorRPMs[{i}]" for i in range(1, 5)])]
        row["TrueState_RPM_records_min"] = int(pd.to_numeric(rpm.records, errors="coerce").min()) if len(rpm) else 0
        row["TrueState_RPM_nonfinite_max"] = int(pd.to_numeric(rpm.missing_nonfinite, errors="coerce").max()) if len(rpm) else np.nan
        row["TrueState_RPM_zero_fraction_mean"] = float(pd.to_numeric(rpm.zero_fraction, errors="coerce").mean()) if len(rpm) else np.nan
        source_rows.append(row)
    case_table = pd.DataFrame(source_rows)
    case_table.to_csv(out / "case_signal_summary.csv", index=False)

    topic_summary = case_table.groupby(["split", "domain", "fault_type"], dropna=False).agg(
        cases=("case_uid", "nunique"),
        controls_present=("actuator_controls_0_present", "sum"),
        outputs_present=("actuator_outputs_present", "sum"),
        true_rpm_records_median=("TrueState_RPM_records_min", "median"),
        control_rate_hz_median=("actuator_controls_0_median_hz", "median"),
        output_rate_hz_median=("actuator_outputs_median_hz", "median"),
    ).reset_index()
    topic_summary.to_csv(out / "signal_coverage_by_stratum.csv", index=False)

    # Summarize TLog ESC fields without coercing missing values into zeros.
    rpm_cols = [c for c in tlog.columns if c.startswith("escStatus.rpm") and c.endswith("_finite")]
    current_cols = [c for c in tlog.columns if c.startswith("escStatus.current") and c.endswith("_finite")]
    tlog_summary = []
    for label, cols in [("ESC_RPM", rpm_cols), ("ESC_current", current_cols)]:
        values = tlog[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=tlog.index)
        finite_cols = [c.replace("_finite", "_nonzero") for c in cols]
        nonzero = tlog[finite_cols].apply(pd.to_numeric, errors="coerce") if finite_cols else pd.DataFrame(index=tlog.index)
        tlog_summary.append({
            "signal_group": label,
            "cases_with_tlog": int(tlog.file_count.fillna(0).astype(float).gt(0).sum()),
            "case_channel_cells_finite": int(values.fillna(0).to_numpy().sum()) if len(values.columns) else 0,
            "case_channel_cells_nonzero": int(nonzero.fillna(0).to_numpy().sum()) if len(nonzero.columns) else 0,
            "interpretation": "finite zero and missing are reported separately; TLog is low-rate telemetry, not early-response evidence",
        })
    pd.DataFrame(tlog_summary).to_csv(out / "tlog_telemetry_summary.csv", index=False)
    coverage.to_csv(out / "frozen_ledger_coverage.csv", index=False)

    # Descriptive, case-level pre/post deltas. Windows/samples are not treated
    # as independent replicates; ULog and TrueState remain separate clocks.
    effects = read_csv(root, "within_case_pre_post_effects.csv")
    for column in ["delta", "pre_median", "post_median", "pre_n", "post_n"]:
        effects[column] = pd.to_numeric(effects[column], errors="coerce")
    effects.to_csv(out / "within_case_effects.csv", index=False)
    effects.groupby(["clock", "topic", "field", "fault_type", "domain"], dropna=False).agg(
        cases=("case_uid", "nunique"), median_case_delta=("delta", "median"),
        q10_case_delta=("delta", lambda x: x.quantile(.1)),
        q90_case_delta=("delta", lambda x: x.quantile(.9)),
    ).reset_index().to_csv(out / "effect_summary_descriptive.csv", index=False)

    # Two intentionally separate time axes. Start-anchored overlays use the
    # upstream toolkit convention and are explicitly exploratory, not verified alignment.
    cases = [
        "HIL/hover/motor/TestCase_2_2000000001",
        "HIL/hover/no_fault/TestCase_2_2019000001",
        "HIL/velocity/motor/TestCase_2_2200000001",
        "HIL/velocity/no_fault/TestCase_123_2219000122",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for col, flight in enumerate(["hover", "velocity"]):
        for row, fault in enumerate(["motor", "no_fault"]):
            uid = next(x for x in cases if f"/{flight}/{fault}/" in x)
            case = Path("D:/RflyMAD") / Path(*uid.split("/"))
            state = pd.read_excel(case / "TrueData" / "TrueState_data.xlsx", sheet_name="truedataset")
            t = pd.to_numeric(state.trueTime, errors="coerce")
            for i in range(1, 5):
                axes[row, col].plot(t, state[f"motorRPMs[{i}]"], color=colors[i - 1], lw=.8, label=f"TrueState motorRPMs[{i}]")
            if fault == "motor":
                active = pd.to_numeric(state.fault_state, errors="coerce").fillna(0).gt(0)
                if active.any():
                    axes[row, col].axvline(t[active].iloc[0], color="#333333", ls="--", lw=1, label="TrueState fault_state first 1")
            axes[row, col].set_title(f"{flight.title()} / {fault.replace('_', ' ')}")
            axes[row, col].set_ylabel("TrueState motorRPMs (source units)")
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=7, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("TrueState trueTime (s)")
    fig.suptitle("TrueState RPM: native workbook clock only")
    fig.tight_layout()
    fig.savefig(out / "true_state_rpm_examples.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, flight in enumerate(["hover", "velocity"]):
        for row, fault in enumerate(["motor", "no_fault"]):
            uid = next(x for x in cases if f"/{flight}/{fault}/" in x)
            case = Path("D:/RflyMAD") / Path(*uid.split("/"))
            matches = sorted((case / "Log").glob("*.ulg"))
            if not matches:
                continue
            from pyulog import ULog
            log = ULog(str(matches[0]))
            ds = {d.name: d.data for d in log.data_list if d.name in {"actuator_outputs", "rfly_ctrl_lxl"} and d.multi_id == 0}
            output = ds.get("actuator_outputs")
            if output is not None:
                x = (output["timestamp"] - output["timestamp"][0]) / 1e6
                for i in range(4):
                    if f"output[{i}]" in output:
                        axes[row, col].plot(x, output[f"output[{i}]"], color=colors[i], lw=.7, label=f"output[{i}]")
            event = ds.get("rfly_ctrl_lxl")
            if fault == "motor" and event is not None and "id" in event:
                ix = np.flatnonzero((event["id"] >= 123450) & (event["id"] <= 123459))
                if len(ix):
                    anchor = (event["timestamp"][ix[0]] - output["timestamp"][0]) / 1e6 if output is not None else np.nan
                    if np.isfinite(anchor): axes[row, col].axvline(anchor, color="#333333", ls="--", lw=1, label="rfly_ctrl_lxl fault event")
            axes[row, col].set_title(f"{flight.title()} / {fault.replace('_', ' ')}")
            axes[row, col].set_ylabel("actuator_outputs.output[i] (raw units)")
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=7, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("ULog timestamp relative to actuator_outputs start (s)")
    fig.suptitle("ULog actuator_outputs: native publication clock only")
    fig.tight_layout()
    fig.savefig(out / "ulog_output_examples.png", dpi=180)
    plt.close(fig)

    align = timelines[timelines.clock.eq("clock_alignment_audit")].copy()
    align.to_csv(out / "clock_alignment_examples.csv", index=False)
    print(f"Wrote audit tables and figures to {out}")


if __name__ == "__main__":
    main()
