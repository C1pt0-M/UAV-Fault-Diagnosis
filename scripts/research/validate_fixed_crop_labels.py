"""Diagnostic label-only references, not a repaired training-cache builder.

Retains existing source train/val crop, CSVs, feature values and selected windows.
Reconstructs saved raw event intervals on integer microseconds, then independently
checks selected original GTD/ULog files. Never loads models, Real or final tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog

from audit_source_event_boundaries import source_tasks, read_gtd, hold, digest
from audit_rflymad_raw_sources import active_ids

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'artifacts/source_event_boundaries_review_20261031'


def microseconds(seconds):
    value = np.asarray(seconds, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite event clock')
    scaled = value * 1e6
    if np.any(np.abs(scaled - np.rint(scaled)) > .001):
        raise ValueError('Event clock not representable on recorded microseconds')
    return np.rint(scaled).astype(np.int64)


def interval_states(query_us, events, coverage_start_us, coverage_end_us):
    q = np.asarray(query_us)
    if q.ndim != 1 or not len(q) or not np.issubdtype(q.dtype, np.integer) or (np.diff(q) <= 0).any():
        raise ValueError('Invalid unchanged query grid')
    if coverage_end_us < coverage_start_us:
        raise ValueError('Reversed original coverage')
    state = np.zeros(len(q), dtype=np.int8)
    previous_end = None
    for i, event in enumerate(events):
        start = int(microseconds(event['start_s']))
        end = int(microseconds(event['end_s'])) if event['end_s'] is not None else None
        if end is not None and end <= start:
            raise ValueError('Nonpositive active interval')
        if i and (previous_end is None or start < previous_end):
            raise ValueError('Overlapping or unordered active intervals')
        state[(q >= start) & ((q < end) if end is not None else True)] = 1
        previous_end = end
    covered = (q >= coverage_start_us) & (q <= coverage_end_us)
    state[~covered] = -1
    return state, covered


def raw_check(task, saved, query_us, gtd_reference, command_reference):
    row, _, _ = task
    path = Path(row['source_path'])
    gtd_path = path / 'TrueData/UAVState_data.xlsx'
    logs = list((path / 'Log').glob('*.ulg'))
    if len(logs) != 1:
        raise ValueError('Expected exactly one original ULog')
    log_path = logs[0]
    if digest(gtd_path) != saved['raw_gtd_sha256'] or digest(log_path) != saved['ulog_sha256']:
        raise ValueError('Original identity changed since raw audit')
    gt, gs = read_gtd(gtd_path)
    gi = microseconds(gt)
    gv, gc = hold(gi, gs, query_us)
    expected = np.where(gc, gv.astype(np.int8), -1)
    if not np.array_equal(expected, gtd_reference):
        raise ValueError('Fresh original GTD differs from interval-derived reference')
    log = ULog(str(log_path), message_name_filter_list=['rfly_ctrl_lxl'])
    topics = [d.data for d in log.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0]
    if len(topics) != 1:
        raise ValueError('Missing/ambiguous original fault command topic')
    data = topics[0]
    cv, cc = hold(data['timestamp'], active_ids(data['id']), query_us)
    command = np.where(cc, cv.astype(np.int8), -1)
    if not np.array_equal(command, command_reference):
        raise ValueError('Fresh ULog differs from interval-derived reference')
    return {'case_uid': row['case_uid'], 'original_gtd_sha256': saved['raw_gtd_sha256'],
            'original_ulog_sha256': saved['ulog_sha256'], 'rows_checked': len(query_us),
            'gtd_direct_vs_interval_equal': True, 'command_direct_vs_interval_equal': True,
            'gtd_first_us': int(gi[0]), 'gtd_last_us': int(gi[-1])}


def load_raw_records():
    result = {}
    for folder in ['source_event_boundaries_full_20261031', 'source_event_boundaries_remaining_20261031']:
        p = ROOT / 'artifacts' / folder / 'case_assessments.jsonl'
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['case_uid'] in result:
                raise ValueError('Repeated raw audit member')
            result[r['case_uid']] = r
    extra = json.loads((ROOT / 'artifacts/source_event_boundaries_full_20261031/raw_clock_exception_events.json').read_text(encoding='utf-8'))
    for r in extra['cases']:
        result[r['case_uid']].update(r)
    if len(result) != 3135:
        raise ValueError('Unexpected source event audit scope')
    return result


def select_anchors(tasks, saved):
    chosen = {uid for uid, r in saved.items() if r.get('raw_onset_after_crop')}
    chosen.update(uid for uid, r in saved.items() if r.get('clock_reversals'))
    clock = json.loads((ROOT / 'artifacts/source_clock_alignment_20261031/plan.json').read_text(encoding='utf-8'))
    chosen.update(clock['raw_clock_anchor_cases'])
    for domain in ['SIL', 'HIL']:
        for mode in ['hover', 'waypoint', 'circling', 'velocity', 'acce', 'dece']:
            matches = [t[0]['case_uid'] for t in tasks if t[0]['domain'] == domain
                       and t[0]['flight_status'] == mode and t[0]['source_fault_type'] == 'no_fault']
            if matches:
                chosen.add(sorted(matches)[0])
    return chosen


def validate_case(task, saved, out, recheck):
    row, old, links = task
    uid = row['case_uid']
    csv_path = Path(row['final_output_path'])
    source_bytes = csv_path.read_bytes()
    before_hash = hashlib.sha256(source_bytes).hexdigest()
    if before_hash != old['csv_sha256']:
        raise ValueError('Existing CSV changed: ' + uid)
    before_stat = (csv_path.stat().st_size, csv_path.stat().st_mtime_ns)
    frame = pd.read_csv(io.BytesIO(source_bytes), usecols=['Timestamp', 'UAVState_data_fault_state'])
    stamps = pd.to_numeric(frame.Timestamp, errors='raise').to_numpy()
    if not np.isfinite(stamps).all() or not np.equal(stamps, np.rint(stamps)).all():
        raise ValueError('Noninteger original processed timestamp')
    q = stamps.astype(np.int64)
    if len(q) != old['rows'] or not np.all(np.diff(q) == 20000):
        raise ValueError('Original 50Hz crop changed')
    original = pd.to_numeric(frame.UAVState_data_fault_state, errors='raise').to_numpy()
    if not np.isfinite(original).all() or not np.isin(original, [0, 1]).all():
        raise ValueError('Invalid historical binary labels')
    original = original.astype(np.int8)
    # Prior raw audit proved complete GTD coverage even for the clock-exception
    # cases. For those three, freshly read GTD is also required below.
    if 'gtd_start_s' in saved:
        gs, ge = int(microseconds(saved['gtd_start_s'])), int(microseconds(saved['gtd_end_s']))
    else:
        if not recheck or saved.get('gtd_covered_rows') != len(q):
            raise ValueError('Missing verified raw coverage')
        gt, _ = read_gtd(Path(row['source_path']) / 'TrueData/UAVState_data.xlsx')
        gs, ge = int(microseconds(gt[0])), int(microseconds(gt[-1]))
    reference, covered = interval_states(q, saved['gtd_events'], gs, ge)
    command_events = saved['command_events']
    if not command_events:
        command_end = int(microseconds(saved.get('ulog_end_s', saved.get('crop_end_s', q[-1] / 1e6))))
    else:
        command_end = int(microseconds(command_events[-1]['last_observed_s']))
    # Start of command-topic coverage is not stored in prior event-only files.
    # The full audit verified command coverage for these exact windows/crop;
    # use crop start solely for reconstructing that already-verified overlap.
    if 'command_reference_covered_rows' in saved and saved['command_reference_covered_rows'] != len(q):
        raise ValueError('Command crop coverage not complete')
    command, command_covered = interval_states(q, command_events, int(q[0]), command_end)
    if not covered.all() or not command_covered.all():
        raise ValueError('Fixed crop outside verified raw reference coverage')
    fresh = raw_check(task, saved, q, reference, command) if recheck else None
    agree = covered & command_covered & (reference == command)
    records = []
    for link in links:
        start, end = int(link['window_start_row']), int(link['window_end_row'])
        idx = end - 1
        if not 0 <= start < end <= len(q) or end - start != 150 or int(link['label']) != int(original[idx]):
            raise ValueError('Changed selected window or old label')
        records.append({'case_uid': uid, 'domain': row['domain'], 'split': row['split'],
                        'cache_index': int(link['cache_index']), 'window_start_row': start, 'window_end_row': end,
                        'endpoint_timestamp_us': int(q[idx]), 'existing_label': int(original[idx]),
                        'gtd_state_reference': int(reference[idx]), 'command_state_reference': int(command[idx]),
                        'source_references_agree': bool(agree[idx]),
                        'label_changed_in_diagnostic_reference': bool(original[idx] != reference[idx]),
                        'input_clock_review_required': bool(saved.get('clock_reversals'))})
    sidecar = out / 'label_references' / (uid.replace('/', '__') + '.npz')
    np.savez_compressed(sidecar, timestamp_us=q, existing_state=original,
                        gtd_state_reference=reference, command_state_reference=command,
                        gtd_covered=covered, command_covered=command_covered, source_references_agree=agree)
    if before_stat != (csv_path.stat().st_size, csv_path.stat().st_mtime_ns):
        raise ValueError('Source file changed during diagnostic read')
    rec = {k: row[k] for k in ['case_uid', 'domain', 'split', 'flight_status', 'source_fault_type']}
    rec.update(processed_csv_sha256=before_hash, original_gtd_sha256=saved['raw_gtd_sha256'],
               original_ulog_sha256=saved['ulog_sha256'], sidecar_sha256=digest(sidecar),
               source_rows=len(q), changed_rows=int((original != reference).sum()),
               crop_start_us=int(q[0]), crop_end_us=int(q[-1]),
               selected_windows=len(records), changed_windows=sum(r['label_changed_in_diagnostic_reference'] for r in records),
               changed_windows_sources_agree=sum(r['label_changed_in_diagnostic_reference'] and r['source_references_agree'] for r in records),
               source_disagreement_windows=sum(not r['source_references_agree'] for r in records),
               old_fault_reference_inactive=sum(r['existing_label'] == 1 and r['gtd_state_reference'] == 0 for r in records),
               old_normal_reference_active=sum(r['existing_label'] == 0 and r['gtd_state_reference'] == 1 for r in records),
               all_crop_reference_inactive=bool(not reference.any()),
               after_crop_activation=bool(saved.get('raw_onset_after_crop')),
               input_clock_review_required=bool(saved.get('clock_reversals')),
               raw_anchor_reread=recheck, source_identity_checked=True)
    return rec, records, fresh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    args = ap.parse_args()
    out = args.output.resolve()
    for root in [Path('D:/RflyMAD').resolve(), Path('D:/UAV-Data').resolve()]:
        if out == root or root in out.parents:
            raise ValueError('Never write beneath dataset roots')
    tasks = source_tasks()
    saved = load_raw_records()
    if {t[0]['case_uid'] for t in tasks} != set(saved):
        raise ValueError('Source membership mismatch')
    anchors = select_anchors(tasks, saved)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'label_references').mkdir()
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__),
            'cases': [t[0]['case_uid'] for t in tasks], 'original_raw_reread_anchors': sorted(anchors),
            'scope': '3135 source train/val cases only; official crop, original CSV and features, selected cap64 windows unchanged. Sidecars are diagnostic references, not authoritative repair labels or training cache. No Real/final-test/model/fitting access.',
            'reference_rule': 'Last GTD state at/before timestamp, final duplicate retained, integer microseconds, outside coverage=-1; retain command disagreement explicitly. No offset fit or forced agreement.',
            'coverage_scope': 'All CSV hashes and full crop time grids re-read. All source event intervals reused from full raw audit. Only listed raw anchors freshly re-read. No input-quality approval, crop replacement or automatic member deletion.'}
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    records, windows, fresh = [], [], []
    with (out / 'case_results.jsonl').open('x', encoding='utf-8') as cf:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def work(task):
                uid = task[0]['case_uid']
                return validate_case(task, saved[uid], out, uid in anchors)
            for i, (record, result, check) in enumerate(pool.map(work, tasks), 1):
                records.append(record); windows.extend(result)
                if check:
                    fresh.append(check)
                cf.write(json.dumps(record, ensure_ascii=False) + chr(10)); cf.flush()
                if i % 500 == 0:
                    print('VALIDATED', i, '/', len(tasks), flush=True)
    pd.DataFrame(windows).to_csv(out / 'selected_window_comparison.csv', index=False)
    (out / 'fresh_raw_anchor_checks.json').write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {'status': 'diagnostic_complete_not_approved_for_training', 'cases': len(records), 'existing_windows': len(windows),
              'fresh_raw_anchor_cases': len(fresh), 'source_csv_identities_checked': len(records),
              'by_split': {}, 'normal_case_checks': {}, 'scope': plan['scope']}
    for split in ['train', 'val']:
        group = [r for r in records if r['split'] == split]
        result['by_split'][split] = {'cases': len(group), 'cases_with_changed_windows': sum(r['changed_windows'] > 0 for r in group)}
        for field in ['selected_windows', 'changed_windows', 'changed_windows_sources_agree', 'source_disagreement_windows',
                      'old_fault_reference_inactive', 'old_normal_reference_active']:
            result['by_split'][split][field] = sum(r[field] for r in group)
        result['by_split'][split]['clock_review_cases'] = sum(r['input_clock_review_required'] for r in group)
    normal = [r for r in records if r['source_fault_type'] == 'no_fault']
    result['normal_case_checks'] = {'cases': len(normal), 'selected_windows': sum(r['selected_windows'] for r in normal),
                                    'changed_windows': sum(r['changed_windows'] for r in normal),
                                    'all_no_recorded_activation': all(r['all_crop_reference_inactive'] for r in normal)}
    after = [r for r in records if r['after_crop_activation']]
    result['after_crop_activation_checks'] = {'cases': len(after), 'old_fault_reference_inactive': sum(r['old_fault_reference_inactive'] for r in after),
                                             'all_fixed_crops_reference_inactive': all(r['all_crop_reference_inactive'] for r in after)}
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
