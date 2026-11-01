"""Independent verification for completed fixed-crop internal test repair."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from pyulog import ULog

from audit_source_event_boundaries import read_gtd
from audit_rflymad_raw_sources import digest
from validate_fixed_crop_labels import microseconds

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('D:/UAV-Data/internal_test_labeltime_v1')
ART = ROOT / 'artifacts/internal_test_labeltime_v1_20261031'
OLD = Path('D:/UAV-Data/online_cache_v2')


def active(ids):
    return (((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))).astype(np.int8)


def records(part):
    return [json.loads(s) for s in (OUT / 'parts' / ('part' + str(part) + '.jsonl')).read_text(encoding='utf-8').splitlines()]


def hold(t, states, q):
    assert len(t) and not (np.diff(t) < 0).any()
    ix = np.searchsorted(t, q, side='right') - 1
    cover = (ix >= 0) & (q <= t[-1])
    r = np.full(len(q), -1, dtype=np.int8)
    r[cover] = states[ix[cover]]
    return r, cover


def verify_part(part):
    members = records(part); changed = quarantined = 0
    candidate = pd.read_csv(OUT / 'parts' / ('part' + str(part) + '_windows.csv'))
    for j, r in enumerate(members, 1):
        old = Path(r['historical_final_output_path']); new = Path(r['final_output_path'])
        assert digest(old) == r['output_sha256']
        assert digest(new) == r['new_csv_sha256']
        assert digest(r['reference_file']) == r['sidecar_sha256']
        for path, expected in r['original_identities'].items(): assert digest(path) == expected
        with np.load(r['reference_file'], allow_pickle=False) as data:
            d = {key: data[key] for key in data.files}
        q = d['timestamp_us']; assert len(q) == int(r['row_count']) and np.all(np.diff(q) == 20000)
        raw = Path(r['source_path']); gt, gs = read_gtd(raw / 'TrueData/UAVState_data.xlsx')
        ref, gc = hold(microseconds(gt), gs.astype(np.int8), q)
        log = ULog(str(next((raw / 'Log').glob('*.ulg'))), message_name_filter_list=['rfly_ctrl_lxl', 'sensor_combined'])
        cmd = next(t.data for t in log.data_list if t.name == 'rfly_ctrl_lxl' and t.multi_id == 0)
        sensor = next(t.data for t in log.data_list if t.name == 'sensor_combined' and t.multi_id == 0)
        ur, uc = hold(cmd['timestamp'].astype(np.int64), active(cmd['id']), q)
        st = sensor['timestamp'].astype(np.int64); reversal = bool((np.diff(st) < 0).any())
        assert reversal == r['original_inertial_clock_review']
        assert np.array_equal(st, d['original_sensor_timestamp_us'])
        assert np.array_equal(ref, d['gtd_state_reference']) and np.array_equal(ur, d['command_state_reference'])
        assert np.array_equal(gc, d['gtd_covered']) and np.array_equal(uc, d['command_covered'])
        agree = gc & uc & (ref == ur)
        assert np.array_equal(agree, d['source_references_agree'])
        assert np.array_equal(np.where(agree, ref, -1), d['qualified_reference'])
        assert np.array_equal(agree & (not reversal) & d['required_input_row_finite'], d['row_label_valid'])
        count = case_changes = 0
        with old.open(encoding='utf-8-sig', newline='') as a, new.open(encoding='utf-8', newline='') as b:
            ar, br = csv.reader(a), csv.reader(b); h = next(ar); nh = next(br)
            assert nh == h + ['labeltime_supervision_valid', 'labeltime_input_clock_review']
            si, ti = h.index('UAVState_data_fault_state'), h.index('Timestamp')
            for i, original in enumerate(ar):
                row = next(br)
                assert len(row) == len(h) + 2
                assert original[:si] + original[si + 1:] == row[:si] + row[si + 1:len(h)]
                assert float(original[ti]) == q[i]
                assert int(float(original[si])) == d['existing_state'][i]
                assert int(row[si]) == ref[i]
                assert row[-2:] == [str(int(agree[i])), str(int(reversal))]
                case_changes += int(d['existing_state'][i] != ref[i]); count += 1
            assert count == len(q) and next(br, None) is None
        assert case_changes == r['changed_state_rows']; changed += case_changes
        selected = candidate[candidate.case_uid == r['case_uid']]
        starts = np.arange(0, len(q) - 149, 5); starts = starts[np.unique(np.linspace(0, len(starts) - 1, 64).round().astype(int))]
        assert selected.window_start_row.tolist() == starts.tolist()
        for w in selected.itertuples():
            s, e = w.window_start_row, w.window_end_row; ix = e - 1
            input_bad = bool(((st >= q[s]) & (st <= q[ix]) & d['original_sensor_required_field_nonfinite']).any()) or not d['required_input_row_finite'][s:e].all()
            valid = bool(agree[ix] and not reversal and not input_bad)
            assert w.endpoint_evaluation_valid == valid
            assert w.qualified_label == (int(ref[ix]) if valid else -1)
            assert w.existing_label == d['existing_state'][ix]
        quarantined += int(reversal)
        if j % 30 == 0: print('VERIFIED_INTERNAL_TEST_PART', part, j, '/', len(members), flush=True)
    return {'part': part, 'cases': len(members), 'windows': len(candidate), 'changed_state_rows': changed,
        'clock_review_cases': quarantined, 'original_hashes_unchanged': True, 'nonlabel_field_strings_unchanged': True}


def verify_cache():
    manifest = json.loads((OUT / 'repair_manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['input_identities'].items(): assert digest(path) == expected
    for name, expected in manifest['outputs_sha256'].items(): assert digest(OUT / name) == expected
    torch.set_num_threads(4)
    old = torch.load(OLD / 'test.pt', map_location='cpu', weights_only=True, mmap=True)
    new = torch.load(OUT / 'internal_test_evidence.pt', map_location='cpu', weights_only=True, mmap=True)
    assert 'y' not in new and len(new['historical_y']) == 35392
    assert torch.equal(old['y'], new['historical_y'])
    for s in range(0, 35392, 256):
        assert torch.equal(new['x'][s:s + 256], old['x'][s:s + 256])
    assert torch.equal(new['source_cache_index'], torch.arange(35392))
    meta = pd.read_csv(OUT / 'test_window_links.csv')
    ledger = pd.read_csv(OUT / 'evaluation_ineligible_window_ledger.csv')
    assert len(meta) == 35392 and meta.case_uid.nunique() == 553
    assert np.array_equal(meta.source_cache_index, np.arange(35392))
    assert np.array_equal(new['qualified_reference_y'].numpy(), meta.qualified_label)
    assert np.array_equal(new['endpoint_evaluation_valid'].numpy(), meta.endpoint_evaluation_valid)
    assert np.array_equal(new['gtd_reference_y'].numpy(), meta.gtd_reference_label)
    assert np.array_equal(new['command_reference_y'].numpy(), meta.command_reference_label)
    assert ledger.source_cache_index.tolist() == meta.loc[~meta.endpoint_evaluation_valid, 'source_cache_index'].tolist()
    assert np.array_equal(meta.qualified_label, np.where(meta.endpoint_evaluation_valid, meta.gtd_reference_label, -1))
    assert (meta.loc[meta.qualified_label < 0, 'level2_label'] == 'uncertain').all()
    assert digest(OUT / 'norm_stats.json') == digest(OLD / 'norm_stats.json')
    for uid, group in meta.groupby('case_uid', sort=False):
        with np.load(OUT / 'references' / (uid.replace('/', '__') + '.npz'), allow_pickle=False) as d:
            ends = group.window_end_row.to_numpy(dtype=int) - 1
            assert np.array_equal(group.gtd_reference_label, d['gtd_state_reference'][ends])
            assert np.array_equal(group.existing_label, d['existing_state'][ends])
    return {'cases': 553, 'windows': 35392, 'all_X_exactly_equal_original': True,
        'corrected_qualified_endpoint_labels': int(((meta.existing_label != meta.qualified_label) & meta.endpoint_evaluation_valid).sum()),
        'eligible_endpoints': int(meta.endpoint_evaluation_valid.sum()), 'ineligible_endpoints': len(ledger),
        'original_assets_and_norm_hashes_unchanged': True, 'no_generic_y': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['cases', 'cache'], required=True)
    parser.add_argument('--part', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    dest = ART / ('independent_part' + str(args.part) + '.json' if args.stage == 'cases' else 'independent_cache.json')
    if dest.exists(): raise FileExistsError('Preserve previous verified report')
    result = verify_part(args.part) if args.stage == 'cases' else verify_cache()
    result.update(status='PASS', verifier_sha256=digest(__file__), verified_at=datetime.now(timezone.utc).isoformat(), models_or_training_used=False)
    with dest.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
