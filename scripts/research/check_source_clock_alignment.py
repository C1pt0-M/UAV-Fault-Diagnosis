"""Fixed-crop source clock checks; no relabeling, converter calls or models."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import csv

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pyulog import ULog

from audit_source_event_boundaries import digest, hold, read_gtd, intervals
from audit_rflymad_raw_sources import active_ids, metadata

ROOT = Path(__file__).resolve().parents[2]


def first_book_time(path, field):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book.active.iter_rows(values_only=True)
        header = next(rows)
        index = header.index(field)
        for row in rows:
            if row[index] is not None:
                value = float(row[index])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite clock origin')
                return value
        raise ValueError('Empty original clock')
    finally:
        book.close()


def bridge_errors(px4_us, simulator_s, uav_first_ms, true_first_s):
    px4 = np.asarray(px4_us, dtype=float)
    simulator = np.asarray(simulator_s, dtype=float)
    if px4.shape != simulator.shape or len(px4) < 2 or not np.isfinite(px4).all() or not np.isfinite(simulator).all():
        raise ValueError('Invalid clock columns')
    if (np.diff(px4) <= 0).any() or (np.diff(simulator) <= 0).any():
        raise ValueError('Nonincreasing processed clock')
    expected = true_first_s + (px4 - uav_first_ms * 1000) / 1e6
    return {'max_bridge_error_s': float(np.max(np.abs(simulator - expected))),
            'max_step_difference_s': float(np.max(np.abs(np.diff(simulator) - np.diff(px4) / 1e6))),
            'observed_sim_minus_px4_s': float(np.median(simulator - px4 / 1e6)),
            'origin_sim_minus_px4_s': float(true_first_s - uav_first_ms / 1000)}


def summary(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())} if len(a) else {'n': 0}


def load_records():
    result = []
    for folder in ['source_event_boundaries_full_20261031', 'source_event_boundaries_remaining_20261031']:
        p = ROOT / 'artifacts' / folder / 'case_assessments.jsonl'
        result.extend(json.loads(s) for s in p.read_text(encoding='utf-8').splitlines())
    extra = json.loads((ROOT / 'artifacts/source_event_boundaries_full_20261031/raw_clock_exception_events.json').read_text(encoding='utf-8'))
    replacements = {r['case_uid']: r for r in extra['cases']}
    for r in result:
        if r['case_uid'] in replacements:
            for k, v in replacements[r['case_uid']].items():
                if k not in ['status', 'issues']:
                    r[k] = v
    if len(result) != 3135 or len({r['case_uid'] for r in result}) != 3135:
        raise ValueError('Unexpected source membership')
    return result


def selected_uids(records, old):
    chosen = {'HIL/circling/motor/TestCase_10_2300000009'}
    for domain in ['SIL', 'HIL']:
        events = [r for r in records if r['domain'] == domain and r.get('gtd_events') and r.get('command_events')]
        key = lambda r: r['gtd_events'][0]['start_s'] - r['command_events'][0]['start_s']
        chosen.add(min(events, key=key)['case_uid'])
        chosen.add(max(events, key=key)['case_uid'])
        for kind in ['no_fault', 'motor', 'barometer']:
            chosen.add(next(r['case_uid'] for r in records if r['domain'] == domain and r['source_fault_type'] == kind))
        measured = [r for r in events if old[r['case_uid']]['state_onset_s'] is not None]
        delay = lambda r: old[r['case_uid']]['state_onset_s'] - r['gtd_events'][0]['start_s']
        chosen.add(min(measured, key=delay)['case_uid'])
        chosen.add(max(measured, key=delay)['case_uid'])
    return sorted(chosen)


def inspect(record, old):
    raw = Path(record['source_path'])
    gtd = raw / 'TrueData/UAVState_data.xlsx'
    true = raw / 'TrueData/TrueState_data.xlsx'
    logfile = list((raw / 'Log').glob('*.ulg'))
    if len(logfile) != 1:
        raise ValueError('Expected one original ULog')
    logfile = logfile[0]
    if digest(gtd) != record['raw_gtd_sha256'] or digest(logfile) != record['ulog_sha256']:
        raise ValueError('Original identity differs from full audit')
    if digest(Path(record['final_output_path'])) != record['csv_sha256']:
        raise ValueError('Processed identity differs from audit')
    frame = pd.read_csv(record['final_output_path'], usecols=['Timestamp', 'trueTime', 'UAVState_data_fault_state', '_rfly_ctrl_lxl_0_id'])
    gt, gs = read_gtd(gtd)
    origin_ms = first_book_time(gtd, 'uavTime')
    origin_true = first_book_time(true, 'trueTime')
    query = frame.Timestamp.to_numpy(dtype=float) / 1e6
    bridge = bridge_errors(frame.Timestamp, frame.trueTime, origin_ms, origin_true)
    log = ULog(str(logfile), message_name_filter_list=['rfly_ctrl_lxl'])
    data = next(d.data for d in log.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0)
    ct = data['timestamp'].astype(float) / 1e6
    cf = active_ids(data['id'])
    ge, ce = intervals(gt, gs), intervals(ct, cf)
    reference, valid = hold(gt, gs, query)
    command_reference, command_valid = hold(ct, cf, query)
    meta = metadata(raw / 'TestInfo.csv')
    return {'case_uid': record['case_uid'], 'domain': record['domain'], 'split': record['split'],
            'crop_start_s': float(query[0]), 'crop_end_s': float(query[-1]),
            'uav_first_ms': origin_ms, 'true_first_s': origin_true, 'bridge': bridge,
            'raw_gtd_activation_px4_s': ge[0]['start_s'] if ge else None,
            'raw_ulog_command_px4_s': ce[0]['start_s'] if ce else None,
            'processed_fault_state_px4_s': old['state_onset_s'],
            'testinfo_planned_injection_value': meta.get('Fault injection time'),
            'gtd_activation_since_uav_first_s': ge[0]['start_s'] - origin_ms / 1000 if ge else None,
            'gtd_activation_in_simulator_clock_s': origin_true + ge[0]['start_s'] - origin_ms / 1000 if ge else None,
            'csv_full_gtd_coverage': bool(valid.all()),
            'csv_gtd_reference_active_rows': int((reference & valid).sum()),
            'csv_processed_active_rows': int((frame.UAVState_data_fault_state.to_numpy() > 0).sum()),
            'csv_command_full_coverage': bool(command_valid.all()),
            'gtd_command_disagreement_rows': int(((reference != command_reference) & valid & command_valid).sum()),
            'raw_gtd_sha256': record['raw_gtd_sha256'], 'ulog_sha256': record['ulog_sha256'],
            'original_true_state_sha256': digest(true)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    for root in [Path('D:/RflyMAD').resolve(), Path('D:/UAV-Data').resolve()]:
        if out == root or root in out.parents:
            raise ValueError('Output must be outside source dataset roots')
    records = load_records()
    old = {r['case_uid']: r for r in (json.loads(s) for s in (ROOT / 'artifacts/dataset_suitability_audit_20261030_r2/case_audit.jsonl').read_text(encoding='utf-8').splitlines())}
    selected = selected_uids(records, old)
    out.mkdir(parents=True, exist_ok=False)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__),
            'selection': 'Fixed illustrative after-crop case; per-domain min/max original onset delta, first healthy/motor/barometer case, min/max processed-minus-raw activation. Deterministic nonrandom anchor checks.',
            'raw_clock_anchor_cases': selected,
            'scope': 'Source train/val only, original crop fixed; no converter, environment installation, Real, final tests, relabeling, training or model access.'}
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    anchors = []
    for uid in selected:
        record = next(r for r in records if r['case_uid'] == uid)
        anchors.append(inspect(record, old[uid]))
    with (out / 'raw_clock_anchors.json').open('x', encoding='utf-8') as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2, allow_nan=False)
    global_rows = []
    for r in records:
        ge, ce = r.get('gtd_events', []), r.get('command_events', [])
        raw = ge[0]['start_s'] if ge else None
        command = ce[0]['start_s'] if ce else None
        processed = old[r['case_uid']]['state_onset_s']
        global_rows.append({'case_uid': r['case_uid'], 'domain': r['domain'], 'split': r['split'],
                            'crop_start_s': r['crop_start_s'], 'crop_end_s': r['crop_end_s'],
                            'raw_gtd_activation_s': raw, 'raw_ulog_command_s': command,
                            'processed_state_activation_s': processed,
                            'gtd_minus_command_s': raw - command if raw is not None and command is not None else None,
                            'processed_minus_gtd_s': processed - raw if processed is not None and raw is not None else None})
    pd.DataFrame(global_rows).to_csv(out / 'source_event_time_inventory.csv', index=False)
    report = {'cases': len(global_rows), 'raw_anchor_cases': len(anchors),
              'all_anchor_bridge_max_error_s': max(r['bridge']['max_bridge_error_s'] for r in anchors),
              'all_anchor_step_max_error_s': max(r['bridge']['max_step_difference_s'] for r in anchors),
              'anchors_with_original_gtd_ulog_presence_match': sum((r['raw_gtd_activation_px4_s'] is None) == (r['raw_ulog_command_px4_s'] is None) for r in anchors),
              'by_domain': {}, 'scope': plan['scope'],
              'limits': '3135-row inventory summarizes existing full raw audit. Only selected anchors re-read original TrueState clocks; not full 3135 TrueState clock validation. Affine first-record alignment is official tool definition, not independent physical clock synchronization proof. No global offset correction fitted.'}
    for domain in ['SIL', 'HIL']:
        group = [r for r in global_rows if r['domain'] == domain]
        source = [r['gtd_minus_command_s'] for r in group if r['gtd_minus_command_s'] is not None]
        delta = [r['processed_minus_gtd_s'] for r in group if r['processed_minus_gtd_s'] is not None]
        report['by_domain'][domain] = {'cases': len(group), 'gtd_minus_ulog_command_s': summary(source),
                                      'processed_minus_gtd_s': summary(delta),
                                      'processed_earlier_over_1s': sum(v < -1 for v in delta),
                                      'processed_later_over_1s': sum(v > 1 for v in delta),
                                      'note': 'One-second split is descriptive only, not an approved tolerance.'}
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
