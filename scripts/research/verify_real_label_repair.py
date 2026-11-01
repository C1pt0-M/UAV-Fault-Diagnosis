"""Independent read-only verification of the materialized Real repair."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from pyulog import ULog

from audit_rflymad_raw_sources import digest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/UAV-Data/real_labeltime_v1')
ART = ROOT / 'artifacts/real_labeltime_repair_v1_20261031'
OLD = Path('D:/UAV-Data/online_cache_v2')
ID = 'rfly_ctrl_lxl_slash_send_data_id'
FLAGS = ['labeltime_state_reference', 'labeltime_supervision_valid', 'labeltime_input_clock_review']


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def active(ids):
    return (((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))).astype(np.int8)


def verify_csv(part=1):
    records = read_lines(DATA / 'case_repair_results.jsonl')
    assert len(records) == 465
    records = records[(part - 1) * 155:part * 155]
    bags = {r['case_uid']: r for r in read_lines(ART / 'bag_verification/case_results.jsonl')}
    cases = changed = quarantined = unknown_coverage_rows = disagreement_rows = 0
    for r in records:
        uid = r['case_uid']; b = bags[uid]
        assert digest(r['historical_final_output_path']) == r['output_sha256']
        assert digest(r['final_output_path']) == r['new_csv_sha256']
        assert digest(r['reference_file']) == r['reference_sha256']
        with np.load(r['reference_file'], allow_pickle=False) as d:
            saved = {key: d[key] for key in d.files}
        t = saved['timestamp_us']; n = len(t)
        assert n == r['rows'] and np.all(np.diff(t) == 20000)
        quarantine = b['status'] == 'QUARANTINE'
        if quarantine:
            for k in ['ros_state', 'ulog_state', 'qualified_state', 'reference_id', 'ros_query_ns']:
                assert np.all(saved[k] == -1)
            for k in ['ros_covered', 'ulog_covered', 'row_label_valid']:
                assert not saved[k].any()
            quarantined += 1
        else:
            source = ART / 'bag_verification' / b['reference_file']
            assert digest(source) == b['reference_sha256']
            with np.load(source, allow_pickle=False) as d:
                q = int(d['sync_ros_ns'][0]) + (t - int(d['sync_remote_ns'][0]) // 1000) * 1000
                times, ids = d['command_ros_ns'], d['command_id']
                positions = np.searchsorted(times, q, side='right') - 1
                covered = (positions >= 0) & (q <= times[-1])
                expected_id = np.full(n, -1, dtype=np.int64)
                expected_id[covered] = ids[positions[covered]]
            assert np.array_equal(q, saved['ros_query_ns'])
            assert np.array_equal(covered, saved['ros_covered'])
            assert np.array_equal(expected_id, saved['reference_id'])
            state = np.where(covered, active(expected_id), -1).astype(np.int8)
            assert np.array_equal(state, saved['ros_state'])
            unknown_coverage_rows += int((~covered).sum())
            raw = Path('D:/RflyMAD/Real').joinpath(*uid.split('/')[1:])
            log = ULog(str(next(raw.rglob('*.ulg'))), message_name_filter_list=['rfly_ctrl_lxl'])
            topic = next(d.data for d in log.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0)
            ut = topic['timestamp'].astype(np.int64)
            assert not (np.diff(ut) < 0).any()
            ix = np.searchsorted(ut, t, side='right') - 1
            uc = (ix >= 0) & (t <= ut[-1])
            ur = np.full(n, -1, dtype=np.int8)
            ur[uc] = active(topic['id'])[ix[uc]]
            assert np.array_equal(uc, saved['ulog_covered']) and np.array_equal(ur, saved['ulog_state'])
            valid = covered & uc & (state == ur)
            assert np.array_equal(valid, saved['row_label_valid'])
            assert np.array_equal(np.where(valid, state, -1), saved['qualified_state'])
            disagreement_rows += int((covered & uc & (state != ur)).sum())
        with Path(r['historical_final_output_path']).open(encoding='utf-8-sig', newline='') as old, Path(r['final_output_path']).open(encoding='utf-8', newline='') as new:
            ro, rn = csv.reader(old), csv.reader(new)
            header = next(ro); assert next(rn) == header + FLAGS
            ii, ti = header.index(ID), header.index('Timestamp')
            count = case_changes = 0
            for i, original in enumerate(ro):
                row = next(rn)
                assert len(row) == len(header) + len(FLAGS)
                assert float(original[ti]) == int(t[i])
                oldid = int(float(original[ii]))
                assert active(np.asarray([oldid]))[0] == saved['historical_state'][i]
                assert row[-3:] == [str(int(saved['ros_state'][i])), str(int(saved['row_label_valid'][i])), str(int(quarantine))]
                base = row[:len(header)]
                if quarantine:
                    assert base == original
                else:
                    assert base[:ii] + base[ii + 1:] == original[:ii] + original[ii + 1:]
                    assert int(base[ii]) == saved['reference_id'][i]
                    if saved['ros_covered'][i]:
                        case_changes += int(saved['historical_state'][i] != saved['ros_state'][i])
                count += 1
            assert count == n and next(rn, None) is None
            assert case_changes == r['changed_state_rows']
            changed += case_changes
        cases += 1
        if cases % 100 == 0:
            print('INDEPENDENT_CSV_VERIFICATION', part, cases, '/155', flush=True)
    assert cases == 155
    return {'part': part, 'cases': cases, 'quarantined_cases': quarantined, 'changed_state_rows': changed,
            'uncovered_ROS_rows_in_verified_cases': unknown_coverage_rows,
            'disagreeing_source_rows_in_verified_cases': disagreement_rows, 'all_original_field_strings_preserved_except_declared_ID_patch': True}


def verify_bags():
    sys.path.append('D:/tmp/real_labeltime_tools_20261031')
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    records = read_lines(ART / 'bag_verification/case_results.jsonl')
    verified = rejected = 0
    for r in records:
        assert digest(r['bag_path']) == r['bag_sha256']
        sent, sync = [], []
        store = get_typestore(Stores.ROS1_NOETIC)
        failed = None
        try:
            with Reader(r['bag_path']) as reader:
                conns = [c for c in reader.connections if c.topic in ['/mavros/rfly_ctrl_lxl/send_data', '/mavros/timesync_status']]
                for c in conns:
                    store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
                for c, timestamp, raw in reader.messages(connections=conns):
                    m = store.deserialize_ros1(raw, c.msgtype)
                    if c.topic == '/mavros/timesync_status':
                        sync.append((int(timestamp), int(m.remote_timestamp_ns)))
                    else:
                        sent.append((int(timestamp), int(m.id)))
        except Exception as exc:
            failed = type(exc).__name__ + ': ' + str(exc)
        if r['status'] == 'QUARANTINE':
            assert failed is not None, ('Quarantine not independently reproduced', r['case_uid'])
            rejected += 1
        else:
            assert failed is None, (r['case_uid'], failed)
            with np.load(ART / 'bag_verification' / r['reference_file'], allow_pickle=False) as d:
                assert np.array_equal(np.asarray(sent, dtype=np.int64), np.column_stack([d['command_ros_ns'], d['command_id']]))
                assert np.array_equal(np.asarray(sync, dtype=np.int64), np.column_stack([d['sync_ros_ns'], d['sync_remote_ns']]))
            verified += 1
        if (verified + rejected) % 100 == 0:
            print('INDEPENDENT_STANDARD_BAG_READER', verified + rejected, '/465', flush=True)
    assert verified == 456 and rejected == 9
    return {'standard_reader_verified_cases': verified, 'standard_reader_reproduced_failures': rejected,
            'all_bag_hashes_unchanged': True, 'verification_scope': 'Required command/sync records only; no global raw-payload or physical-truth certification'}


def verify_cache():
    torch.set_num_threads(4)
    manifest = json.loads((DATA / 'repair_manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['input_identities'].items():
        assert digest(path) == expected, path
    for name, expected in manifest['artifact_sha256'].items():
        assert digest(DATA / name) == expected, name
    old = torch.load(OLD / 'real_holdout_test.pt', map_location='cpu', weights_only=True, mmap=True)
    fixed = torch.load(DATA / 'real_all_roles_evidence.pt', map_location='cpu', weights_only=True, mmap=True)
    assert 'y' not in fixed
    n = len(old['y']); assert n == 29319
    assert torch.equal(fixed['historical_y'], old['y'])
    for i in range(0, n, 256):
        assert torch.equal(fixed['x'][i:i + 256], old['x'][i:i + 256])
        assert torch.isfinite(fixed['x'][i:i + 256]).all()
    assert torch.equal(fixed['source_cache_index'], torch.arange(n))
    meta = pd.read_csv(DATA / 'real_cap64_window_links.csv')
    replay = pd.read_csv(DATA / 'stride10_replay_window_links.csv')
    assert len(meta) == n and len(replay) == 196142
    assert np.array_equal(meta.source_cache_index, np.arange(n))
    roles = list(csv.DictReader((DATA / 'historical_real_roles.csv').open(encoding='utf-8-sig', newline='')))
    rolemap = {r['case_uid']: r['role'] for r in roles}
    expected_roles = {'adapt_normal': 10, 'calibrate_normal': 5, 'test_normal': 33, 'test_fault': 392, 'supplemental_normal_test': 25}
    assert Counter(rolemap.values()) == expected_roles
    counts = Counter()
    for uid, group in meta.groupby('case_uid', sort=False):
        with np.load(DATA / 'references' / (uid.replace('/', '__') + '.npz'), allow_pickle=False) as d:
            total = len(d['timestamp_us'])
            starts = np.arange(0, total - 149, 5)
            if len(starts) > 64:
                starts = starts[np.unique(np.linspace(0, len(starts) - 1, 64).round().astype(int))]
            assert group.window_start_row.tolist() == starts.tolist()
            ends = starts + 149; index = group.source_cache_index.to_numpy(dtype=np.int64)
            valid, ref = d['row_label_valid'][ends], d['ros_state'][ends]
            assert np.array_equal(group.qualified_label, np.where(valid, ref, -1))
            assert np.array_equal(group.historical_label, d['historical_state'][ends])
            assert np.array_equal(group.endpoint_label_valid, valid)
            for key, expected in [('ros_reference_y', ref), ('qualified_reference_y', np.where(valid, ref, -1)),
                                  ('endpoint_label_valid', valid)]:
                assert np.array_equal(fixed[key][index].numpy(), expected)
            assert (group.role == rolemap[uid]).all()
            assert np.array_equal(fixed['role_id'][index].numpy(), group.role_id)
            assert (group.loc[group.qualified_label < 0, 'level2_label'] == 'uncertain').all()
            assert (group.loc[group.qualified_label == 0, 'level2_label'] == 'normal').all()
            for row in group.itertuples():
                s, e = row.window_start_row, row.window_end_row
                assert row.whole_window_reference_covered == bool(d['ros_covered'][s:e].all())
                assert row.whole_window_label_valid == bool(d['row_label_valid'][s:e].all())
                if row.whole_window_reference_covered:
                    assert abs(row.fault_fraction - float((d['ros_state'][s:e] == 1).mean())) < 1e-12
                else:
                    assert pd.isna(row.fault_fraction)
            r = replay[replay.case_uid == uid]
            starts = np.arange(0, total - 149, 10); ends = starts + 149
            assert r.window_start_row.tolist() == starts.tolist()
            assert np.array_equal(r.qualified_label, d['qualified_state'][ends])
            assert np.array_equal(r.endpoint_label_valid, d['row_label_valid'][ends])
            counts[rolemap[uid]] += len(index)
    ledger = pd.read_csv(DATA / 'evaluation_ineligible_window_ledger.csv')
    assert ledger.source_cache_index.tolist() == meta.loc[~meta.endpoint_label_valid, 'source_cache_index'].tolist()
    assert digest(DATA / 'norm_stats.json') == digest(OLD / 'norm_stats.json')
    eligible = meta.endpoint_label_valid
    changed = (meta.historical_label != meta.qualified_label) & eligible
    assert int(eligible.sum()) == 28544 and int(changed.sum()) == 10
    assert len(ledger) == 775
    for role in ['adapt_normal', 'calibrate_normal']:
        part = meta[meta.role == role]
        assert part.endpoint_label_valid.all() and (part.qualified_label == 0).all()
    return {'windows': n, 'all_X_equal_original': True, 'role_counts_windows': dict(counts),
        'qualified_endpoints': int(eligible.sum()), 'unqualified_endpoints': len(ledger),
        'corrected_qualified_endpoint_labels': int(changed.sum()), 'replay_windows': len(replay),
        'quarantine_windows': int((meta.review_reason == 'original_bag_required_record_integrity_unresolved').sum()),
        'disagreement_windows': int((meta.review_reason == 'ROS_sent_ULog_received_state_disagree').sum()),
        'old_assets_and_normalization_unchanged': True, 'adaptation_calibration_identity_labels_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['csv', 'bags', 'cache'], required=True)
    parser.add_argument('--part', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    suffix = '_part' + str(args.part) if args.stage == 'csv' else ''
    target = ART / ('independent_' + args.stage + suffix + '_verification.json')
    if target.exists():
        raise FileExistsError('Preserve previous verification; use a new version')
    result = verify_csv(args.part) if args.stage == 'csv' else {'bags': verify_bags, 'cache': verify_cache}[args.stage]()
    result.update(status='PASS', verified_at=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=digest(__file__), models_or_training_used=False)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
