"""Read-only audit of current full-flight CSVs, labels and feature availability.

Writes only research artifacts inside the repository. Never changes source data,
splits, labels or model assets. Durations describe processed logs, not proof of
airborne time or causal raw acquisition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("D:/UAV-Data")
MANIFEST = DATA / "splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv"
CONTRACT = DATA / "feature_contract_v1.json"
OUT = ROOT / "docs/research/2026-10-08/data_feasibility"
REAL_FAULT_IDS = {123450, 123452, 123453, 123454, 123455, 123456}


def segments(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def audit_case(row, features):
    keep = ["domain", "case_uid", "flight_status", "source_fault_type", "source_disposition",
        "level1_label", "level2_label", "level3_label", "split", "simulation_pair_key",
        "source_path", "final_output_path", "window_label_source", "open_set_fault_type"]
    result = {k: row.get(k, "") for k in keep}
    path = Path(row["final_output_path"])
    result["source_directory_exists"] = Path(row["source_path"]).is_dir()
    try:
        data = path.read_bytes()
        result["file_bytes"] = len(data)
        result["sha256"] = hashlib.sha256(data).hexdigest()
        result["manifest_hash_matches"] = result["sha256"] == row["output_sha256"]
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
        result["rows"] = n = len(df)
        result["columns"] = len(df.columns)
        result["manifest_rows_match"] = n == int(row["row_count"])
        result["manifest_columns_match"] = len(df.columns) == int(row["column_count"])
        if n < 2 or "Timestamp" not in df:
            raise ValueError("CSV lacks at least two rows and Timestamp")
        t = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(dtype=float) / 1e6
        dt = np.diff(t)
        finite_dt = dt[np.isfinite(dt)]
        step = float(np.median(finite_dt)) if len(finite_dt) else 0.02
        result.update(timestamp_nonfinite=int((~np.isfinite(t)).sum()),
            nonincreasing_steps=int((dt <= 0).sum()), median_step_s=step,
            steps_off_50hz=int((np.abs(dt - 0.02) > 1e-5).sum()),
            max_step_s=float(np.max(finite_dt)) if len(finite_dt) else None,
            duration_s=float(t[-1] - t[0]), timestamp_start_us=float(t[0]*1e6),
            timestamp_end_us=float(t[-1]*1e6))

        label_col = row["window_label_source"]
        result["label_column_present"] = label_col in df
        if label_col not in df:
            raise ValueError(f"Declared label column missing: {label_col}")
        values = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
        result["label_nonfinite"] = int((~np.isfinite(values)).sum())
        result["label_noninteger"] = int((np.isfinite(values) & (np.abs(values-np.round(values)) > 1e-5)).sum())
        result["observed_label_values"] = np.unique(values[np.isfinite(values)]).tolist()
        if row["domain"] == "REAL":
            # Decode exact known IDs; do not treat the normal control ID 1500 as a fault.
            mask = np.isin(values, list(REAL_FAULT_IDS))
            result["unknown_label_rows"] = int((np.isfinite(values) & ~np.isin(values, [0,1500,*REAL_FAULT_IDS])).sum())
        else:
            mask = values > 0
            result["unknown_label_rows"] = 0
        runs = segments(mask)
        normal_runs = segments(~mask)
        result["fault_rows"] = int(mask.sum())
        result["normal_rows"] = int((~mask).sum())
        result["fault_event_count"] = len(runs)
        result["fault_active_s"] = float(mask.sum()*step)
        result["normal_exposure_s"] = float((~mask).sum()*step)
        result["normal_longest_run_s"] = max((b-a)*step for a,b in normal_runs) if normal_runs else 0.0
        result["first_fault_s"] = float(t[runs[0][0]]-t[0]) if runs else None
        result["fault_event_intervals_s"] = [[float(t[a]-t[0]), float(t[b-1]-t[0]+step)] for a,b in runs]
        result["first_event_duration_s"] = float((runs[0][1]-runs[0][0])*step) if runs else None
        result["case_label_consistent"] = bool(mask.any()) == (row["level1_label"] == "fault")
        # Main risk gate: never silently relabel contract-normal cases for experiments.
        result["contract_normal_with_active_rows"] = row["level1_label"] == "normal" and bool(mask.any())

        source_domain = "Real" if row["domain"] == "REAL" else row["domain"]
        stats = {}
        for f in features:
            col = f["source_columns"].get(source_domain)
            name = f["canonical_name"]
            if col == "derived" and name == "acc_norm":
                axes = [next(x for x in features if x["canonical_name"] == axis)["source_columns"][source_domain]
                        for axis in ("accel_x", "accel_y", "accel_z")]
                if not all(axis in df for axis in axes):
                    stats[name] = {"source_column":col, "present":False}
                    continue
                v = np.linalg.norm(df[axes].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), axis=1)
            elif not col or col not in df:
                stats[name] = {"source_column":col, "present":False}
                continue
            else:
                v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(v)
            normal = v[valid & ~mask]
            stats[name] = {"source_column":col,"present":True,
                "nonfinite_rows":int((~valid).sum()),
                "constant":bool(np.ptp(v[valid]) <= 1e-10) if valid.any() else None,
                "normal_median":float(np.median(normal)) if len(normal) else None,
                "normal_q05":float(np.quantile(normal,.05)) if len(normal) else None,
                "normal_q95":float(np.quantile(normal,.95)) if len(normal) else None}
        result["missing_feature_count"] = sum(not s["present"] for s in stats.values())
        result["nonfinite_feature_cells"] = sum(s.get("nonfinite_rows",0) for s in stats.values())
        result["feature_stats"] = stats
        # Real logs can retain explicit flight-state streams; unique values alone
        # are not proof that the aircraft performed verified within-flight changes.
        phase_cols = [c for c in df if c.endswith("send_data_flight_status") or c == "UAVState_data_cmd"]
        result["phase_columns"] = {c: pd.to_numeric(df[c],errors="coerce").dropna().unique()[:30].tolist() for c in phase_cols}
        result["status"] = "ok"
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    features = json.loads(CONTRACT.read_text(encoding="utf-8"))["features"]
    rows = manifest.to_dict("records")
    if args.limit:
        # A deterministic spread through the complete manifest, not just hover/motor.
        indices = np.linspace(0,len(rows)-1,min(args.limit,len(rows)),dtype=int)
        rows = [rows[i] for i in indices]
    name = "pilot" if args.limit else "full"
    start = time.monotonic()
    records = []
    with (OUT/f"{name}_case_audit.jsonl").open("w",encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(audit_case,row,features) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                records.append(result)
                output.write(json.dumps(result,ensure_ascii=False)+"\n")
                if len(records)%250 == 0 or len(records)==len(rows):
                    output.flush()
                    print(f"Audited {len(records)}/{len(rows)}; errors={sum(r['status']!='ok' for r in records)}; elapsed={time.monotonic()-start:.0f}s",flush=True)
    flat = [{k:v for k,v in r.items() if k not in {"feature_stats","phase_columns","fault_event_intervals_s","observed_label_values"}} for r in records]
    df = pd.DataFrame(flat).sort_values(["domain","case_uid"])
    df.to_csv(OUT/f"{name}_case_audit.csv",index=False,encoding="utf-8-sig")
    summary = {"audited_at":datetime.now(timezone.utc).isoformat(),"source_manifest":str(MANIFEST),
        "source_contract":str(CONTRACT),"manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "contract_sha256":hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "rows_expected":len(rows),"read_ok":int((df.status=="ok").sum()),"read_errors":int((df.status!="ok").sum()),
        "manifest_duplicate_case_ids":int(manifest.case_uid.duplicated().sum()),
        "manifest_duplicate_paths":int(manifest.final_output_path.duplicated().sum()),
        "simulation_groups_crossing_splits":int((manifest[manifest.domain!="REAL"].groupby("simulation_pair_key").split.nunique()>1).sum()),
        "elapsed_s":time.monotonic()-start}
    (OUT/f"{name}_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
