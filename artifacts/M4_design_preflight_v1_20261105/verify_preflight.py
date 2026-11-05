"""Independent reconciliation of Train statistics and group folds, no models."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import csv
import gzip
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = Path('D:/UAV-Data/causal_input_v1')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    target = OUT / 'independent_verification.json'
    assert not target.exists()
    ident = json.loads((OUT / 'input_identity.json').read_text())
    assert sha(OUT / 'analyze_train.py') == ident['script_sha256']
    for name, expected in ident['inputs'].items():
        assert sha(DATA / name) == expected, name
    with gzip.open(OUT / 'window_state_ledger.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 162069
    ids = [int(r['original_cache_index']) for r in rows]
    assert len(set(ids)) == len(ids)
    assert [int(r['compact_index']) for r in rows] == list(range(len(rows)))
    state = defaultdict(lambda: {'windows': 0, 'normal': 0, 'fault': 0, 'cases': set(), 'groups': set()})
    for r in rows:
        x = state[r['state']];x['windows'] += 1;x['normal' if r['qualified_y'] == '0' else 'fault'] += 1
        x['cases'].add(r['case_uid']);x['groups'].add(r['association_group'])
    with (OUT / 'coverage_state.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            x = state[r['state']]
            assert (x['windows'], x['normal'], x['fault'], len(x['cases']), len(x['groups'])) == tuple(int(r[k]) for k in ['windows', 'normal_endpoint_windows', 'fault_endpoint_windows', 'cases', 'groups'])
    fold_checks = []
    for k in [5, 3]:
        with (OUT / f'fold_assignments_{k}.csv').open(encoding='utf-8', newline='') as f:
            a = list(csv.DictReader(f))
        assignment = {r['association_group']: int(r['outer_fold']) for r in a}
        assert len(assignment) == len(a) == 1296
        assert set(assignment) == {r['association_group'] for r in rows}
        seen = Counter()
        for fold in range(k):
            held = [r for r in rows if assignment[r['association_group']] == fold]
            train = [r for r in rows if assignment[r['association_group']] != fold]
            assert not {r['case_uid'] for r in held} & {r['case_uid'] for r in train}
            assert not {r['association_group'] for r in held} & {r['association_group'] for r in train}
            seen.update(int(r['original_cache_index']) for r in held)
        assert set(seen) == set(ids) and set(seen.values()) == {1}
        fold_checks.append({'k': k, 'all_original_ids_held_out_exactly_once': True, 'case_and_association_group_crossing': 0})
    payload = torch.load(DATA / 'cache/train.pt', map_location='cpu', mmap=True, weights_only=True)
    assert payload['split'] == 'train' and payload['data_version'] == 'rflymad_causal_publication_replay_v1'
    assert np.array_equal(payload['y'].numpy(), [int(r['qualified_y']) for r in rows])
    assert np.array_equal(payload['original_cache_index'].numpy(), ids)
    columns = json.loads((DATA / 'norm_stats.json').read_text())['columns']
    ci = {name: i for i, name in enumerate(columns)}
    # Three selected observations per case, checked with direct scalar formula.
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r['case_uid']].append(i)
    probes = sorted({indices[j] for indices in by_case.values() for j in [0, len(indices) // 2, len(indices) - 1]})
    for i in probes:
        raw = payload['x'][i].numpy().astype(float)
        speed = np.sqrt(sum(raw[ci['vel_' + axis]] ** 2 for axis in 'xyz'))
        rotation = np.sqrt(sum(raw[ci['gyro_' + axis]] ** 2 for axis in 'xyz'))
        vz = raw[ci['vel_z']]
        if np.quantile(rotation, .9) >= .3 or np.std(speed) >= .5:expected = 'maneuver_proxy'
        elif (vz <= -.5).mean() >= .8:expected = 'climb_proxy'
        elif (vz >= .5).mean() >= .8:expected = 'descent_proxy'
        elif (speed <= .3).mean() >= .8 and (rotation <= .15).mean() >= .8:expected = 'low_motion_proxy'
        elif (speed > .3).mean() >= .8:expected = 'translation_proxy'
        else:expected = 'other_motion'
        assert rows[i]['state'] == expected, i
        for name, value in [('speed_median_mps', np.median(speed)), ('speed_std_mps', np.std(speed)), ('gyro_p90_radps', np.quantile(rotation, .9))]:
            assert abs(float(rows[i][name]) - value) < 1e-12, (i, name)
    pairs = pd.read_csv(OUT / 'case_pre_post_state_pairs.csv', dtype={'association_group': str})
    by_id = {int(r['original_cache_index']): r for r in rows}
    for p in pairs.itertuples():
        a, b = by_id[p.pre_ID], by_id[p.post_ID]
        assert a['case_uid'] == b['case_uid'] == p.case_uid
        assert a['phase'] == 'pre_fault' and a['known_normal'] == 'True'
        assert b['phase'] == 'stable_fault' and b['pure_fault'] == 'True'
        assert bool(p.state_changed) == (a['state'] != b['state'])
    assert len(pairs) == 2198 and int(pairs.state_changed.sum()) == 692
    aucs = pd.read_csv(OUT / 'conditional_case_AUC_descriptive.csv')
    cf = pd.read_csv(OUT / 'case_state_feature_summaries.csv.gz', dtype={'association_group': str})
    checked = 0; max_error = 0.
    valid = aucs.dropna(subset=['fixed_signed_case_AUC'])
    for _, r in valid.iloc[np.linspace(0, len(valid) - 1, min(100, len(valid))).astype(int)].iterrows():
        q = cf[cf.state.eq(r.state) & cf.domain.eq(r.domain)]
        a = q.loc[q.known_label.eq('normal') & q.fault_directory.eq('no_fault'), r.feature].to_numpy()
        b = q.loc[q.known_label.eq('fault') & q.fault_directory.eq(r.fault_directory), r.feature].to_numpy()
        direct = float(((b[:, None] > a).sum() + .5 * (b[:, None] == a).sum()) / (len(a) * len(b)))
        err = abs(direct - r.fixed_signed_case_AUC); assert err < 1e-12
        max_error = max(max_error, err); checked += 1
    result = {'status': 'PASS_INDEPENDENT_TRAIN_PREFLIGHT_RECONCILIATION', 'source_Train_file_SHA_unchanged': True,
              'all_162069_original_ids_labels_and_state_totals_checked': True, 'case_first_middle_last_raw_probes': len(probes),
              'fold_checks': fold_checks, 'all_2198_pre_post_pairs_checked': True,
              'case_AUC_direct_pair_checks': checked, 'max_AUC_roundoff': max_error,
              'no_model_forward_training_GPU_server_Test_Real': True,
              'limits': 'This verifies the frozen descriptive calculation and grouping only. No independent encoder, predictive validation or physical causal attribution.'}
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
