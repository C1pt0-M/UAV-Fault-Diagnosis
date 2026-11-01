"""Materialize Real sent-command label repair without changing crops or roles.

Separate from SIL/HIL GTD repair. Preserve every original X row. Unknown labels
are not normal, and this evidence bundle deliberately has no generic y key.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
from audit_real_labeltime import hold_integer, selected_ends
from audit_rflymad_raw_sources import digest, active_ids
from rfly_feature_contract import read_case_features, window_batch, source_columns_for_domain

ART = ROOT / 'artifacts/real_labeltime_repair_v1_20261031'
AUDIT = ROOT / 'artifacts/real_labeltime_audit_20261031'
BAGS = ART / 'bag_verification'
ROLES = ROOT / 'artifacts/normal_adaptation_audit_20261029/real_case_roles_v1.csv'
OLD = Path('D:/UAV-Data/online_cache_v2')
MANIFEST = Path('D:/UAV-Data/window_manifest_v1/window_manifest.csv')
CONTRACT = Path('D:/UAV-Data/feature_contract_v1.json')
VERSION = 'rflymad_real_labeltime_v1'
ID = 'rfly_ctrl_lxl_slash_send_data_id'
FLAGS = ['labeltime_state_reference', 'labeltime_supervision_valid', 'labeltime_input_clock_review']
ROLE_IDS = {'adapt_normal': 0, 'calibrate_normal': 1, 'test_normal': 2, 'test_fault': 3, 'supplemental_normal_test': 4}
ROLE_COUNTS = {'adapt_normal': 10, 'calibrate_normal': 5, 'test_normal': 33, 'test_fault': 392, 'supplemental_normal_test': 25}


def json_write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def checked_digest(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError('Input identity changed: ' + str(path))
    return actual


def assign_references(query_us, bag, ulog_times, ulog_ids):
    """Conservative endpoints: source agreement, no coverage extrapolation."""
    query_us = np.asarray(query_us)
    sync_t, sync_remote = bag['sync_ros_ns'], bag['sync_remote_ns']
    if not len(sync_t) or len(sync_t) != len(sync_remote) or int(sync_remote[0]) <= 0:
        raise ValueError('Invalid original bridge anchor')
    if (np.diff(sync_t) < 0).any() or (np.diff(sync_remote) < 0).any():
        raise ValueError('Unresolved original clock reversal')
    px0, ros0 = int(sync_remote[0]) // 1000, int(sync_t[0])
    if query_us.dtype != np.int64:
        raise ValueError('Integer microsecond query required')
    ros_query = ros0 + (query_us - px0) * 1000
    command_t, command_id = bag['command_ros_ns'], bag['command_id']
    state, covered = hold_integer(command_t, active_ids(command_id), ros_query)
    received, received_covered = hold_integer(ulog_times, active_ids(ulog_ids), query_us)
    valid = covered & received_covered & (state == received)
    ids = np.full(len(query_us), -1, dtype=np.int64)
    positions = np.searchsorted(command_t, ros_query, side='right') - 1
    ids[covered] = command_id[positions[covered]]
    qualified = np.where(valid, state, -1).astype(np.int8)
    return {'ros_query_ns': ros_query, 'reference_id': ids, 'ros_state': state,
            'ros_covered': covered, 'ulog_state': received, 'ulog_covered': received_covered,
            'row_label_valid': valid, 'qualified_state': qualified}


def patch_csv(original, target, timestamps, arrays, quarantined):
    if target.exists():
        raise FileExistsError('Refuse overwrite: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256()
    count = changed_ids = changed_state = 0
    with Path(original).open(encoding='utf-8-sig', newline='') as source, target.open('x', encoding='utf-8', newline='') as output:
        reader, writer = csv.reader(source), csv.writer(output)
        header = next(reader)
        if len(header) != len(set(header)) or any(f in header for f in FLAGS):
            raise ValueError('Unexpected CSV schema')
        ii, ti = header.index(ID), header.index('Timestamp')
        writer.writerow(header + FLAGS)
        for i, row in enumerate(reader):
            if i >= len(timestamps) or len(row) != len(header) or float(row[ti]) != int(timestamps[i]):
                raise ValueError('CSV row shape/time mismatch')
            immutable = row if quarantined else row[:ii] + row[ii + 1:]
            before.update(json.dumps(immutable, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
            oldid = int(float(row[ii]))
            if not quarantined:
                new = int(arrays['reference_id'][i])
                changed_ids += int(oldid != new)
                if arrays['ros_covered'][i]:
                    changed_state += int(bool(active_ids(np.asarray([oldid]))[0]) != bool(arrays['ros_state'][i]))
                row[ii] = str(new)
            writer.writerow(row + [str(int(arrays['ros_state'][i])),
                str(int(arrays['row_label_valid'][i])), str(int(quarantined))])
            count += 1
    if count != len(timestamps):
        raise ValueError('Incomplete CSV')
    after = hashlib.sha256()
    with target.open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != header + FLAGS:
            raise ValueError('Output schema mismatch')
        checked = 0
        for i, row in enumerate(reader):
            if len(row) != len(header) + len(FLAGS):
                raise ValueError('Output CSV width mismatch')
            base = row[:len(header)]
            immutable = base if quarantined else base[:ii] + base[ii + 1:]
            after.update(json.dumps(immutable, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
            if row[-3:] != [str(int(arrays['ros_state'][i])), str(int(arrays['row_label_valid'][i])), str(int(quarantined))]:
                raise ValueError('Reference flags not persisted')
            if not quarantined and int(base[ii]) != int(arrays['reference_id'][i]):
                raise ValueError('Repaired ID not persisted')
            checked += 1
    if checked != count or before.digest() != after.digest():
        raise ValueError('Immutable original fields changed')
    return {'rows': count, 'changed_id_rows': changed_ids, 'changed_state_rows': changed_state,
            'immutable_field_strings_sha256': after.hexdigest(), 'new_csv_sha256': digest(target)}


def window_rows(uid, role, n, arrays, mode, case_status):
    ends = selected_ends(n, mode)
    old, ref, valid = arrays['historical_state'], arrays['ros_state'], arrays['row_label_valid']
    onset_rows = np.flatnonzero(ref == 1)
    onset = int(onset_rows[0]) if len(onset_rows) else -1
    records = []
    for j, endpoint in enumerate(ends):
        start, end = int(endpoint - 149), int(endpoint + 1)
        sequence = ref[start:end]
        coverage = bool(arrays['ros_covered'][start:end].all())
        endpoint_valid = bool(valid[endpoint])
        if case_status == 'QUARANTINE':
            reason = 'original_bag_required_record_integrity_unresolved'
        elif not arrays['ros_covered'][endpoint] or not arrays['ulog_covered'][endpoint]:
            reason = 'reference_not_covered'
        elif not endpoint_valid:
            reason = 'ROS_sent_ULog_received_state_disagree'
        else:
            reason = ''
        if not coverage:
            phase = 'unknown_reference'
        elif np.all(sequence == 0):
            phase = 'normal_flight' if onset < 0 else ('pre_fault' if end <= onset else 'post_fault_normal')
        elif np.all(sequence == 1):
            phase = 'stable_fault'
        else:
            phase = 'transition'
        records.append({'case_uid': uid, 'role': role, 'role_id': ROLE_IDS[role], 'sampling': mode,
            'case_sample_index': j, 'window_start_row': start, 'window_end_row': end,
            'historical_label': int(old[endpoint]), 'ros_reference_label': int(ref[endpoint]),
            'qualified_label': int(ref[endpoint]) if endpoint_valid else -1,
            'endpoint_label_valid': endpoint_valid, 'whole_window_reference_covered': coverage,
            'whole_window_label_valid': bool(valid[start:end].all()), 'review_reason': reason,
            'case_reference_status': case_status, 'fault_onset_row': onset,
            'fault_fraction': float((sequence == 1).mean()) if coverage else None, 'phase': phase})
    return records


def finalize_link_labels(meta, family_by_uid):
    meta = meta.rename(columns={'level2_label': 'historical_level2_label',
                                'source_csv': 'historical_source_csv'}).copy()
    families = meta.case_uid.map(family_by_uid)
    if families.isna().any():
        raise ValueError('Missing frozen case family')
    meta['case_level2_label'] = families
    meta['level2_label'] = np.where(meta.qualified_label < 0, 'uncertain',
        np.where(meta.qualified_label == 0, 'normal', families))
    if not meta.loc[meta.qualified_label == 1, 'level2_label'].isin(['motor', 'sensor']).all():
        raise ValueError('Qualified fault contradicts frozen role family')
    return meta


def load_inputs():
    roles = list(csv.DictReader(ROLES.open(encoding='utf-8-sig', newline='')))
    if Counter(r['role'] for r in roles) != ROLE_COUNTS or len({r['case_uid'] for r in roles}) != 465:
        raise ValueError('Frozen Real role scope changed')
    audits = {r['case_uid']: r for r in read_jsonl(AUDIT / 'case_results.jsonl')}
    bags = {r['case_uid']: r for r in read_jsonl(BAGS / 'case_results.jsonl')}
    if set(audits) != set(bags) or set(audits) != {r['case_uid'] for r in roles}:
        raise ValueError('Audit/bag/role membership mismatch')
    if digest(ROLES) != json.loads((AUDIT / 'plan.json').read_text(encoding='utf-8'))['role_manifest_sha256']:
        raise ValueError('Frozen role identity changed')
    return roles, audits, bags


def prepare_csv(output, resume=False):
    if output.exists() and not resume:
        raise FileExistsError('Independent new output required')
    if resume and (not (output / 'RUNNING.json').exists() or (output / 'CSV_COMPLETE.json').exists()):
        raise ValueError('Resume requires an incomplete recorded CSV build')
    for old in [Path('D:/RflyMAD'), OLD, Path('D:/UAV-Data/online_cache_v2_labeltime_v1')]:
        if output == old.resolve() or output.is_relative_to(old.resolve()):
            raise ValueError('Never overwrite old assets')
    roles, audits, bags = load_inputs()
    policy = json.loads((ART / 'output_contract.json').read_text(encoding='utf-8'))
    identities = {str(p): digest(p) for p in [ROLES, AUDIT / 'case_results.jsonl', BAGS / 'case_results.jsonl',
        ART / 'output_contract.json', CONTRACT, OLD / 'norm_stats.json', OLD / 'real_holdout_test.pt']}
    cap, replay, cases = [], [], []
    pending = roles
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'references').mkdir()
        json_write(output / 'RUNNING.json', {'version': VERSION, 'script_sha256': digest(__file__),
            'started_at': datetime.now(timezone.utc).isoformat(), 'input_identities': identities, 'policy': policy})
        shutil.copyfile(ROLES, output / 'historical_real_roles.csv')
        shutil.copyfile(OLD / 'norm_stats.json', output / 'norm_stats.json')
    else:
        previous = json.loads((output / 'RUNNING.json').read_text(encoding='utf-8'))
        if previous['version'] != VERSION or previous['input_identities'] != identities:
            raise ValueError('Interrupted build input identities differ')
        cases = read_jsonl(output / 'case_repair_results.jsonl')
        done = {r['case_uid']: r for r in cases}
        rolemap = {r['case_uid']: r for r in roles}
        if len(done) != len(cases) or not set(done).issubset(rolemap):
            raise ValueError('Interrupted case identity mismatch')
        for r in cases:
            if r['role'] != rolemap[r['case_uid']]['role'] or r['output_sha256'] != rolemap[r['case_uid']]['output_sha256']:
                raise ValueError('Interrupted role or original hash changed')
            checked_digest(Path(r['final_output_path']), r['new_csv_sha256'])
            checked_digest(Path(r['reference_file']), r['reference_sha256'])
            checked_digest(Path(r['historical_final_output_path']), r['output_sha256'])
            with np.load(r['reference_file'], allow_pickle=False) as d:
                arrays = {k: d[k] for k in d.files}
            cap.extend(window_rows(r['case_uid'], r['role'], r['rows'], arrays, 'cap64', r['reference_status']))
            replay.extend(window_rows(r['case_uid'], r['role'], r['rows'], arrays, 'stride10', r['reference_status']))
        pending = [r for r in roles if r['case_uid'] not in done]
        preserved = []
        for role in pending:
            uid = role['case_uid']
            candidates = [output / section / Path(*uid.split('/')[1:]).with_suffix('.csv') for section in ['cases', 'quarantine']]
            candidates.append(output / 'references' / (uid.replace('/', '__') + '.npz'))
            for leftover in candidates:
                if leftover.exists():
                    saved = output / 'interrupted_files' / leftover.relative_to(output)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    if saved.exists():
                        raise FileExistsError('Interrupted snapshot already exists')
                    leftover.replace(saved); preserved.append(str(saved))
        json_write(output / 'RESUME.json', {'script_sha256': digest(__file__), 'completed_cases_verified': len(cases),
            'remaining_cases': len(pending), 'preserved_incomplete_outputs': preserved})

    def patch(role):
        uid = role['case_uid']
        audit, bag = audits[uid], bags[uid]
        original = Path(role['final_output_path'])
        checked_digest(original, role['output_sha256'])
        checked_digest(Path(bag['bag_path']), bag['bag_sha256'])
        frame = pd.read_csv(original, usecols=['Timestamp', ID])
        t = frame.Timestamp.to_numpy(dtype=np.float64)
        if not np.isfinite(t).all() or not np.equal(t, np.rint(t)).all():
            raise ValueError('Noninteger CSV timestamps')
        times = t.astype(np.int64)
        if len(times) != int(role['row_count']) or not np.all(np.diff(times) == 20000):
            raise ValueError('Original crop/grid changed')
        oldid = frame[ID].to_numpy(dtype=float)
        if not np.isfinite(oldid).all() or not np.equal(oldid, np.rint(oldid)).all():
            raise ValueError('Noninteger original IDs require separate review')
        old = active_ids(oldid.astype(np.int64)).astype(np.int8)
        quarantined = bag['status'] != 'VERIFIED_REQUIRED_BAG_FIELDS'
        if quarantined:
            arrays = {k: np.full(len(times), -1, dtype=np.int8) for k in ['ros_state', 'ulog_state', 'qualified_state']}
            arrays.update({k: np.zeros(len(times), dtype=bool) for k in ['ros_covered', 'ulog_covered', 'row_label_valid']})
            arrays['reference_id'] = np.full(len(times), -1, dtype=np.int64)
            arrays['ros_query_ns'] = np.full(len(times), -1, dtype=np.int64)
        else:
            if bag['errors'] or not all(v['exact'] for v in bag['export_differences'].values()):
                raise ValueError('Verified status contains unresolved bag errors')
            source = BAGS / bag['reference_file']
            checked_digest(source, bag['reference_sha256'])
            raw = Path('D:/RflyMAD/Real').joinpath(*uid.split('/')[1:])
            logs = list(raw.rglob('*.ulg'))
            if len(logs) != 1:
                raise ValueError('Expected original ULog')
            checked_digest(logs[0], audit['raw_ulog_sha256'])
            log = ULog(str(logs[0]), message_name_filter_list=['rfly_ctrl_lxl'])
            topics = [d.data for d in log.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0]
            if len(topics) != 1:
                raise ValueError('Expected ULog injection topic')
            with np.load(source, allow_pickle=False) as data:
                arrays = assign_references(times, data, topics[0]['timestamp'].astype(np.int64), topics[0]['id'])
                if int(data['sync_ros_ns'][0]) != audit['original_ros_ns'] or int(data['sync_remote_ns'][0]) // 1000 != audit['original_px4_us']:
                    raise ValueError('Historical first anchor differs from original bag')
            if int(((old != arrays['ros_state']) & arrays['ros_covered']).sum()) != audit['old_vs_ros_row_disagreements']:
                raise ValueError('Bag-derived changes differ from previous diagnostic')
        arrays.update(timestamp_us=times, historical_state=old)
        target = output / ('quarantine' if quarantined else 'cases') / Path(*uid.split('/')[1:]).with_suffix('.csv')
        info = patch_csv(original, target, times, arrays, quarantined)
        sidecar = output / 'references' / (uid.replace('/', '__') + '.npz')
        np.savez_compressed(sidecar, **arrays)
        record = dict(role, historical_final_output_path=str(original), final_output_path=str(target),
            reference_status=bag['status'], reference_file=str(sidecar), reference_sha256=digest(sidecar),
            bag_sha256=bag['bag_sha256'], bag_path=bag['bag_path'], quarantined=quarantined,
            row_label_valid=int(arrays['row_label_valid'].sum()), **info)
        return record, window_rows(uid, role['role'], len(times), arrays, 'cap64', bag['status']), window_rows(uid, role['role'], len(times), arrays, 'stride10', bag['status'])

    with (output / 'case_repair_results.jsonl').open('a' if resume else 'x', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, (record, c, r) in enumerate(pool.map(patch, pending), len(cases) + 1):
                cases.append(record); cap.extend(c); replay.extend(r)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
                if i % 100 == 0:
                    print('REAL_CSV_AND_REFERENCES_PERSISTED', i, '/465', flush=True)
    if len(cases) != 465 or len({r['case_uid'] for r in cases}) != 465:
        raise ValueError('Incomplete Real case membership')
    pd.DataFrame(cases).to_csv(output / 'case_manifest.csv', index=False)
    pd.DataFrame(cap).to_csv(output / 'cap64_reference_candidates.csv', index=False)
    pd.DataFrame(replay).to_csv(output / 'stride10_replay_window_links.csv', index=False)
    json_write(output / 'CSV_COMPLETE.json', {'cases': len(cases), 'verified_cases': sum(not r['quarantined'] for r in cases),
        'quarantined_cases': sum(r['quarantined'] for r in cases), 'changed_state_rows': sum(r['changed_state_rows'] for r in cases),
        'cap64_windows': len(cap), 'replay_windows': len(replay), 'cache_not_yet_built': True})


def build_cache(output):
    if (output / 'repair_manifest.json').exists() or (output / 'real_all_roles_evidence.pt').exists():
        raise FileExistsError('Refuse overwriting completed cache/version')
    running = json.loads((output / 'RUNNING.json').read_text(encoding='utf-8'))
    execution = json.loads((output / 'RESUME.json').read_text(encoding='utf-8')) if (output / 'RESUME.json').exists() else running
    if execution['script_sha256'] != digest(__file__):
        raise ValueError('Repair script changed between stages')
    for path, expected in running['input_identities'].items():
        checked_digest(Path(path), expected)
    roles, _, _ = load_inputs()
    cases = pd.read_csv(output / 'case_manifest.csv', keep_default_na=False).to_dict(orient='records')
    by_uid = {r['case_uid']: r for r in cases}
    candidates = pd.read_csv(output / 'cap64_reference_candidates.csv', keep_default_na=False)
    use = ['case_uid', 'domain', 'split', 'level2_label', 'window_start_row', 'window_end_row', 'source_csv']
    frames = []
    manifest_before = digest(MANIFEST)
    for chunk in pd.read_csv(MANIFEST, usecols=use, chunksize=200000):
        selected = chunk[(chunk.domain == 'REAL') & (chunk.split == 'external_test') & chunk.level2_label.isin(['normal', 'motor', 'sensor'])]
        if len(selected):
            frames.append(selected)
    full = pd.concat(frames, ignore_index=True)
    if set(full.case_uid) != set(by_uid):
        raise ValueError('Original manifest membership differs from frozen roles')
    parts = []
    # Matches existing cache builder group order and label-independent cap rule.
    for uid, group in full.groupby('case_uid', sort=False):
        group = group.sort_values('window_start_row')
        n = int(by_uid[uid]['rows'])
        if group.window_start_row.tolist() != list(range(0, n - 149, 5)) or not np.all(group.window_end_row - group.window_start_row == 150):
            raise ValueError('Original manifest grid differs from fixed crop')
        picks = np.unique(np.linspace(0, len(group) - 1, 64).round().astype(int)) if len(group) > 64 else np.arange(len(group))
        selected = group.iloc[picks].copy()
        selected['case_sample_index'] = np.arange(len(selected))
        parts.append(selected)
    original = pd.concat(parts, ignore_index=True)
    original.insert(0, 'source_cache_index', np.arange(len(original)))
    meta = original.merge(candidates, on=['case_uid', 'case_sample_index', 'window_start_row', 'window_end_row'],
        how='left', validate='one_to_one', indicator=True, sort=False)
    if not (meta['_merge'] == 'both').all() or len(meta) != len(candidates):
        raise ValueError('Original cap64 identity mismatch')
    meta = meta.drop(columns='_merge').sort_values('source_cache_index').reset_index(drop=True)
    if not np.array_equal(meta.historical_label.to_numpy(), (meta.level2_label != 'normal').astype(int).to_numpy()):
        raise ValueError('Historical labels disagree with original manifest')
    torch.set_num_threads(4)
    old = torch.load(OLD / 'real_holdout_test.pt', map_location='cpu', weights_only=True, mmap=True)
    n = len(meta)
    if set(old) != {'x', 'y'} or old['x'].shape != (n, 26, 150) or old['x'].dtype != torch.float32:
        raise ValueError('Unexpected original cache format')
    if not np.array_equal(old['y'].numpy(), meta.historical_label.to_numpy()):
        raise ValueError('Original cache y/order differs from original manifest')
    contract = json.loads(CONTRACT.read_text(encoding='utf-8-sig'))
    norm = json.loads((OLD / 'norm_stats.json').read_text(encoding='utf-8'))
    mapping = source_columns_for_domain(contract, 'REAL')
    for i, (uid, group) in enumerate(meta.groupby('case_uid', sort=False), 1):
        case = by_uid[uid]
        checked_digest(Path(case['final_output_path']), case['new_csv_sha256'])
        checked_digest(Path(case['historical_final_output_path']), case['output_sha256'])
        raw = pd.read_csv(case['historical_final_output_path'], usecols=sorted(set(mapping.values())))
        if not np.isfinite(raw.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)).all():
            raise ValueError('Nonfinite original required feature, no zero-fill certification')
        matrix, columns = read_case_features(case['historical_final_output_path'], 'REAL', contract)
        if columns != norm['columns']:
            raise ValueError('Channel order differs from frozen normalization')
        spans = list(zip(group.window_start_row.astype(int), group.window_end_row.astype(int)))
        expected = torch.from_numpy(window_batch(matrix, columns, spans))
        indices = torch.from_numpy(group.source_cache_index.to_numpy(dtype=np.int64))
        if not torch.isfinite(expected).all() or not torch.equal(expected, old['x'][indices]):
            raise ValueError('Original X/order/window extraction mismatch: ' + uid)
        if i % 100 == 0:
            print('REAL_CACHE_ORIGINAL_X_IDENTITIES_VERIFIED', i, '/465', flush=True)
    valid = meta.endpoint_label_valid.to_numpy(dtype=bool)
    ref = meta.ros_reference_label.to_numpy(dtype=np.int64)
    qualified = meta.qualified_label.to_numpy(dtype=np.int64)
    if not np.array_equal(qualified, np.where(valid, ref, -1)) or not np.isin(qualified, [-1, 0, 1]).all():
        raise ValueError('Qualified mask/target mismatch')
    if not np.isin(ref[valid], [0, 1]).all():
        raise ValueError('Invalid qualified binary targets')
    bundle = {'x': old['x'].clone(), 'historical_y': old['y'].clone(),
        'ros_reference_y': torch.from_numpy(ref.copy()), 'qualified_reference_y': torch.from_numpy(qualified.copy()),
        'endpoint_label_valid': torch.from_numpy(valid.copy()),
        'whole_window_label_valid': torch.from_numpy(meta.whole_window_label_valid.to_numpy(dtype=bool)),
        'role_id': torch.from_numpy(meta.role_id.to_numpy(dtype=np.int64)),
        'source_cache_index': torch.arange(n, dtype=torch.int64)}
    target = output / 'real_all_roles_evidence.pt'
    partial = output / 'real_all_roles_evidence.pt.partial'
    torch.save(bundle, partial); del bundle
    partial.replace(target)
    actual = torch.load(target, map_location='cpu', weights_only=True, mmap=True)
    for start in range(0, n, 512):
        if not torch.equal(actual['x'][start:start + 512], old['x'][start:start + 512]):
            raise ValueError('Persisted X changed')
    for key, expected in [('historical_y', old['y']), ('qualified_reference_y', torch.from_numpy(qualified)),
                          ('endpoint_label_valid', torch.from_numpy(valid))]:
        if not torch.equal(actual[key], expected):
            raise ValueError('Persisted evidence target changed: ' + key)
    meta = finalize_link_labels(meta, {r['case_uid']: r['level2_label'] for r in cases})
    meta['source_csv'] = meta.case_uid.map({r['case_uid']: r['final_output_path'] for r in cases})
    meta.to_csv(output / 'real_cap64_window_links.csv', index=False)
    meta[~meta.endpoint_label_valid].to_csv(output / 'evaluation_ineligible_window_ledger.csv', index=False)
    role_summary = {}
    for role, group in meta.groupby('role', sort=False):
        mask = group.endpoint_label_valid.to_numpy(dtype=bool)
        defined = group.ros_reference_label.to_numpy() >= 0
        changed = group.historical_label.to_numpy() != group.ros_reference_label.to_numpy()
        role_summary[role] = {'role_id': ROLE_IDS[role], 'cases': int(group.case_uid.nunique()), 'windows': len(group),
            'eligible_endpoints': int(mask.sum()), 'ineligible_endpoints': int((~mask).sum()),
            'defined_reference_changed_labels': int((defined & changed).sum()),
            'eligible_changed_labels': int((mask & changed).sum()),
            'review_reasons': dict(Counter(str(v) for v in group.loc[~group.endpoint_label_valid, 'review_reason']))}
    for path, expected in running['input_identities'].items():
        checked_digest(Path(path), expected)
    if digest(MANIFEST) != manifest_before:
        raise ValueError('Original manifest changed during build')
    if digest(output / 'norm_stats.json') != digest(OLD / 'norm_stats.json'):
        raise ValueError('Normalization changed')
    csv_status = json.loads((output / 'CSV_COMPLETE.json').read_text(encoding='utf-8'))
    csv_status['cache_not_yet_built'] = False
    json_write(output / 'repair_manifest.json', {'status': 'COMPLETE_REAL_LABELTIME_REPAIR_WITH_EXPLICIT_QUARANTINE',
        'version': VERSION, 'completed_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__),
        'role_ids': ROLE_IDS, 'roles': role_summary, 'csv': csv_status, 'windows': n,
        'input_identities': {**running['input_identities'], str(MANIFEST): manifest_before},
        'artifact_sha256': {name: digest(output / name) for name in ['real_all_roles_evidence.pt',
            'case_manifest.csv', 'real_cap64_window_links.csv', 'stride10_replay_window_links.csv',
            'evaluation_ineligible_window_ledger.csv', 'norm_stats.json']},
        'all_original_X_exactly_preserved': True, 'models_or_predictions_used': False, 'training_run': False,
        'input_quality_and_physical_onset_certified': False, 'event_scoring_ready': False,
        'warning': 'All-role evidence bundle, NOT train.pt/test.pt. Explicit role selection and endpoint validity required. No pooled all-role score. Row masks must be honored in any later event scoring. Old protocol and scores remain historical.'})
    print(json.dumps(role_summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('D:/UAV-Data/real_labeltime_v1'))
    parser.add_argument('--stage', choices=['csv', 'cache'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume and args.stage != 'csv':
        raise ValueError('Resume only supports unfinished CSV stage')
    if args.stage == 'csv':
        prepare_csv(output, resume=args.resume)
    else:
        build_cache(output)


if __name__ == '__main__':
    main()
