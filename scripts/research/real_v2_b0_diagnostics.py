"""Frozen CPU B0 diagnostics: eligible Real dev only; no deployed models.

The split is authenticated before any Real shard access. Ledger chunks are
filtered immediately, then validated. The D0 gate precedes fitting or metrics.
Sensitivity reuses main session OOF scores (PI supplement 1).
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/UAV-Data/causal_input_v1')
SPLIT = ROOT / 'configs/data/real_v2_split.csv'
CFG = ROOT / 'configs/experiments/real_v2_b0_diagnostics.json'
SOURCE = ROOT / 'artifacts/ens_internal_test_v1_20261105/gbt_full_train.pkl'
REFERENCE = ROOT / 'artifacts/real_normal_adapt_v2_dev_20261106/real_dev_per_case.csv'
OUT = ROOT / 'artifacts/real_v2_b0_diagnostics'
SPLIT_SHA = '9232ee35e7a59f4fac4534fc8e98fc454380b51e6e5d8439aaa0f769cd1090ee'
SOURCE_SHA = 'df468524e2047205dd8422ab3e486ce7692f0ac312276e20394d6f88a1bb6774'
FULL_ROLES = {'test_normal', 'adapt_normal', 'calibrate_normal'}
RECIPE = dict(max_iter=400, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def stats(x):
    x = x.astype(np.float32); d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


def eligible_split(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Split SHA mismatch: STOP')
    split = pd.read_csv(path, dtype={'case_uid': str, 'association_group': str}, keep_default_na=False)
    if (not split.case_uid.is_unique or split.label_quarantine.dtype != bool
            or not split.partition.isin(['dev', 'sealed']).all()
            or not split.groupby('association_group').partition.nunique().eq(1).all()):
        raise ValueError('Invalid group-disjoint split metadata')
    dev = split[split.partition.eq('dev')].copy()
    if len(dev) != 320:
        raise ValueError('Expected 320 development cases')
    eligible = dev[~dev.label_quarantine].copy()
    eligible['dev_fold'] = pd.to_numeric(eligible.dev_fold, errors='raise').astype(np.int64)
    if (not eligible.dev_fold.isin(range(5)).all() or eligible.association_group.eq('').any()
            or not eligible.groupby('association_group').dev_fold.nunique().eq(1).all()
            or not eligible.family.isin(['normal', 'motor', 'sensor']).all()):
        raise ValueError('Invalid eligible metadata/folds')
    for uid in eligible.case_uid:
        parts = uid.split('/')
        if len(parts) != 4 or parts[0] != 'REAL' or any(p in ('', '.', '..') for p in parts):
            raise ValueError('Invalid Real case UID')
    return eligible.reset_index(drop=True), {'dev_before_quarantine': len(dev),
        'dev_quarantine': int(dev.label_quarantine.sum()), 'eligible_cases': len(eligible),
        'eligible_groups': int(eligible.association_group.nunique()), 'split_sha256': expected_sha}


def install_boundary(eligible, data=DATA, source=SOURCE):
    """Whitelist prevents sealed/quarantine/source/test NPZ reads and all writes."""
    data, source = Path(data).resolve(), Path(source).resolve()
    ledger = data / 'ledgers/real_included_windows.csv'
    allowed = {ledger}
    shard_kind = {}
    for uid in eligible.case_uid:
        for kind in ('windows', 'cases'):
            p = data / kind / (uid.replace('/', '__') + '.npz')
            allowed.add(p); shard_kind[p] = kind
    audit = {'windows': [], 'cases': [], 'blocked_accesses': []}
    old_test = source.parent
    final_test = (ROOT / 'artifacts/internal_test_final_v1').resolve()
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        writing = any(c in (args[1] or '') for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        bad = ((p.is_relative_to(data.parent) and (p not in allowed or writing))
               or (p.is_relative_to(old_test) and (p != source or writing))
               or p.is_relative_to(final_test))
        if bad:
            audit['blocked_accesses'].append(str(p))
            raise PermissionError('Only eligible Real dev inputs and frozen S allowed: ' + str(p))
        if p in shard_kind:
            audit[shard_kind[p]].append(str(p))
    sys.addaudithook(guard)
    return audit


def read_ledger(path, eligible):
    wanted = set(eligible.case_uid)
    parts = []
    for chunk in pd.read_csv(path, dtype={'case_uid': str, 'association_group': str},
                             keep_default_na=False, low_memory=False, chunksize=4096):
        # FIRST operation: discard every noneligible (including sealed) row.
        kept = chunk.loc[chunk.case_uid.isin(wanted)].copy()
        del chunk
        parts.append(kept)
    frame = pd.concat(parts, ignore_index=True)
    assert set(frame.case_uid) <= wanted
    if (not frame.original_cache_index.is_unique or not frame.qualified_y.isin([0, 1]).all()
            or not frame.domain.eq('REAL').all() or not frame.split.eq('real').all()):
        raise ValueError('Invalid eligible supervised Real ledger')
    for key in ('supervised_valid', 'input_valid', 'label_valid'):
        if frame[key].dtype != bool or not frame[key].all():
            raise ValueError('Non-supervised row in included population')
    meta = eligible.set_index('case_uid')
    for key in ('association_group',):
        if not np.array_equal(frame[key].to_numpy(), meta.loc[frame.case_uid, key].to_numpy()):
            raise ValueError('Ledger/split group mismatch')
    if not np.array_equal(frame.role.to_numpy(), meta.loc[frame.case_uid, 'role_v1'].to_numpy()):
        raise ValueError('Ledger role mismatch')
    for key in ('family', 'flight', 'stratum_family', 'role_v1', 'dev_fold'):
        frame[key] = meta.loc[frame.case_uid, key].to_numpy()
    return frame.sort_values('original_cache_index').reset_index(drop=True)


def window_features(path, ledger):
    with np.load(path, allow_pickle=False) as z:
        x, y, ids, starts = z['x'], z['y'], z['original_cache_index'], z['window_start_row']
    if (x.shape != (len(ids), 26, 150) or x.dtype != np.float32 or y.shape != (len(ids),)
            or not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or starts.shape != ids.shape or not np.isfinite(x).all()):
        raise ValueError('Invalid window tensor/index schema')
    if set(ids.tolist()) != set(ledger.original_cache_index.tolist()):
        raise ValueError('Window shard/ledger index coverage mismatch')
    pos = pd.Index(ids).get_indexer(ledger.original_cache_index)
    if not np.array_equal(y[pos], ledger.qualified_y.to_numpy()) or not np.array_equal(starts[pos], ledger.window_start_row.to_numpy()):
        raise ValueError('Window y/start differ from qualified ledger')
    if not len(ids):
        return np.empty((0, 130), np.float32)
    return stats(x[pos])


def real_weights(y, origin):
    y = np.asarray(y); origin = np.asarray(origin)
    weights = np.ones(len(y), float)
    n = int((y == 0).sum())
    if not n or not (y == 1).any():
        raise ValueError('Both Real training classes required')
    if not np.isin(origin[y == 0], ['normal_flight', 'normal_part_of_fault_case']).all():
        raise ValueError('Unknown normal origin; do not silently rebalance')
    for name in ('normal_flight', 'normal_part_of_fault_case'):
        mask = (y == 0) & (origin == name)
        if not mask.any():
            raise ValueError('Missing normal training origin')
        weights[mask] = .5 * n / int(mask.sum())
    return weights


def oof(x, y, frame, mask, task, note):
    p = np.full(len(frame), np.nan)
    logs = []
    for fold in range(5):
        train = mask & frame.dev_fold.ne(fold).to_numpy()
        held = mask & frame.dev_fold.eq(fold).to_numpy()
        if not held.any() or set(np.unique(y[train])) != {0, 1}:
            raise ValueError('Missing fold/class for registered OOF task')
        assert not set(frame.loc[train, 'association_group']) & set(frame.loc[held, 'association_group'])
        w = compute_sample_weight('balanced', y[train]) if task == 'session' else real_weights(y[train], frame.loc[train, 'normal_origin'])
        model = HistGradientBoostingClassifier(**RECIPE)
        model.fit(x[train], y[train], sample_weight=w)
        p[held] = model.predict_proba(x[held])[:, 1]
        logs.append({'fold': fold, 'train_windows': int(train.sum()), 'held_windows': int(held.sum()),
            'train_groups': int(frame.loc[train, 'association_group'].nunique()), 'held_groups': int(frame.loc[held, 'association_group'].nunique()),
            'class_weight_mass': {str(c): float(w[y[train] == c].sum()) for c in (0, 1)}, 'n_iter': int(model.n_iter_)})
        del model
        note('OOF_COMPLETE', task, fold)
    if not np.isfinite(p[mask]).all():
        raise ValueError('Missing/nonfinite OOF predictions')
    return p, logs


def auc_plan(prob, labels, mask):
    idx = np.flatnonzero(mask)
    if not np.isfinite(prob[idx]).all():
        raise ValueError('Nonfinite selected AUC scores')
    idx = idx[np.argsort(prob[idx], kind='stable')]
    starts = np.r_[0, np.flatnonzero(np.diff(prob[idx]) != 0) + 1] if len(idx) else np.array([], np.int64)
    return idx, np.asarray(labels)[idx].astype(bool), starts


def weighted_auc(plan, weights):
    idx, positive, starts = plan
    if not len(idx):
        return None
    w = weights[idx]
    pos = np.add.reduceat(w * positive, starts)
    neg = np.add.reduceat(w * ~positive, starts)
    np_, nn = float(pos.sum()), float(neg.sum())
    if not np_ or not nn:
        return None
    return float(np.dot(pos, np.cumsum(neg) - .5 * neg) / (np_ * nn))


def auc_intervals(plans, window_groups, case_groups, reps=2000, seed=20261108):
    groups = np.unique(np.asarray(case_groups))
    inverse = pd.Index(groups).get_indexer(window_groups)
    if (inverse < 0).any():
        raise ValueError('Window outside eligible groups')
    ones = np.ones(len(window_groups), np.int64)
    point = {name: weighted_auc(plan, ones) for name, plan in plans.items()}
    samples = {name: [] for name in plans}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        w = counts[inverse]
        for name, plan in plans.items():
            samples[name].append(weighted_auc(plan, w))
    out = {}
    for name, sample in samples.items():
        undefined = sum(v is None for v in sample)
        out[name] = {'auc': point[name], 'ci95': np.percentile(sample, [2.5, 97.5]).tolist() if not undefined else None,
                     'undefined_replicates': undefined}
    return out, {'replicates': reps, 'rng_seed': seed, 'groups': len(groups), 'shared_plan': True,
                 'population': 'all eligible dev association_groups; subset windows contribute according to each metric',
                 'no_pseudocount_drop_or_redraw': True}


def populations(frame):
    y = frame.qualified_y.to_numpy()
    fault_case = frame.family.isin(['motor', 'sensor']).to_numpy()
    full = frame.family.eq('normal').to_numpy() & frame.role_v1.isin(FULL_ROLES).to_numpy()
    normal_origin = frame.normal_origin.eq('normal_flight').to_numpy()
    pre = fault_case & frame.phase.eq('pre_fault').to_numpy()
    if (y[pre] != 0).any() or (y[full] != 0).any():
        raise ValueError('Session populations must have normal qualified labels')
    return {'normal_full': (y == 0) & full & normal_origin,
        'normal_segment': (y == 0) & frame.family.eq('normal').to_numpy() & ~full & normal_origin,
        'pre_fault': (y == 0) & pre,
        'post_fault_normal': (y == 0) & fault_case & frame.phase.eq('post_fault_normal').to_numpy(),
        'transition_normal': (y == 0) & frame.phase.eq('transition').to_numpy()}


def fpr_at_recall(prob, y, masks, recall):
    fault_scores = np.sort(prob[y == 1])[::-1]
    if not len(fault_scores):
        return {'threshold': None, 'actual_recall': None, 'normal_FPR': {k: None for k in masks}}
    k = int(np.ceil(recall * len(fault_scores)))
    threshold = float(fault_scores[k - 1])
    return {'requested_recall': recall, 'threshold': threshold, 'actual_recall': float((fault_scores >= threshold).mean()),
        'comparison': '>=; highest fault-score order statistic meeting requested recall, ties retained',
        'normal_FPR': {name: float((prob[mk] >= threshold).mean()) if mk.any() else None for name, mk in masks.items()},
        'normal_windows': {name: int(mk.sum()) for name, mk in masks.items()}}


def recommendation(b, strata):
    if b['auc'] is not None and b['auc'] >= .75:
        action = 'GO'
    elif b['ci95'] is not None and b['ci95'][1] < .65:
        action = 'STOP'
    else:
        action = '交用户决定'
    return {'recommendation': action, 'not_supported_in_B1': [name for name, v in strata.items() if v['auc'] is not None and v['auc'] < .65],
            'automatic_B1_launch': False}


def audits(frame, eligible, data):
    records = []
    for row in eligible.itertuples(index=False):
        f = frame[frame.case_uid.eq(row.case_uid)]
        valid_onsets = f.loc[f.fault_onset_row.ge(0), 'fault_onset_row'].unique()
        if len(valid_onsets) > 1 or (len(valid_onsets) and (f.fault_onset_row < 0).any()):
            raise ValueError('Inconsistent onset within a case')
        onset = int(valid_onsets[0]) if len(valid_onsets) else None
        record = {'record_type': 'case', 'case_uid': row.case_uid, 'family': row.family, 'flight': row.flight,
            'role_v1': row.role_v1, 'dev_fold': row.dev_fold, 'windows': len(f),
            'pre_fault_windows': int(f.phase.eq('pre_fault').sum()),
            'post_fault_normal_windows': int(f.phase.eq('post_fault_normal').sum()),
            'transition_normal_windows': int((f.phase.eq('transition') & f.qualified_y.eq(0)).sum()),
            'transition_fault_windows': int((f.phase.eq('transition') & f.qualified_y.eq(1)).sum()),
            'fault_onset_row': onset, 'pre_fault_length_s': .02 * onset if onset is not None and row.family != 'normal' else None,
            'normal_full_exposure_h': None, 'pre_fault_exposure_h': None}
        if row.family in ('motor', 'sensor') or row.role_v1 in FULL_ROLES:
            with np.load(data / 'cases' / (row.case_uid.replace('/', '__') + '.npz'), allow_pickle=False) as z:
                iv, lv = z['row_input_valid'], z['row_label_valid']
            if iv.ndim != 1 or iv.shape != lv.shape or iv.dtype != bool or lv.dtype != bool:
                raise ValueError('Invalid exposure masks')
            valid = iv & lv
            if row.family == 'normal':
                record['normal_full_exposure_h'] = float(valid.sum() * .02 / 3600)
            elif onset is not None:
                record['pre_fault_exposure_h'] = float((valid & (np.arange(len(valid)) < onset)).sum() * .02 / 3600)
            record['onset_outside_row_range'] = bool(onset is not None and onset >= len(valid))
        records.append(record)
    case = pd.DataFrame(records)
    counts = frame.groupby(['family', 'phase', 'qualified_y'], sort=True).size().reset_index(name='windows')
    distribution = {}
    fault = case[case.family.isin(['motor', 'sensor'])]
    for label, part in [('family', 'family'), ('flight', 'flight'), ('family_flight', ['family', 'flight'])]:
        distribution[label] = {}
        for key, f in fault.groupby(part, sort=True):
            key = '/'.join(key) if isinstance(key, tuple) else key
            x = f.pre_fault_length_s.dropna()
            distribution[label][key] = {'cases': len(f), 'defined': len(x), 'missing': int(f.pre_fault_length_s.isna().sum()),
                'min': float(x.min()) if len(x) else None, 'q25': float(x.quantile(.25)) if len(x) else None,
                'median': float(x.median()) if len(x) else None, 'q75': float(x.quantile(.75)) if len(x) else None,
                'max': float(x.max()) if len(x) else None}
    fold = []
    for (k, family), f in eligible.groupby(['dev_fold', 'family'], sort=True):
        fold.append({'dev_fold': int(k), 'family': family, 'cases': len(f),
                     'windows': int((frame.dev_fold.eq(k) & frame.family.eq(family)).sum())})
    special = {phase: [{'case_uid': uid, 'windows': len(f)} for uid, f in frame[frame.phase.eq(phase)].groupby('case_uid', sort=True)]
               for phase in ('post_fault_normal', 'transition')}
    for record in counts.to_dict('records'):
        records.append({'record_type': 'family_phase_label', **record})
    result = {'cases_by_family': case.groupby('family').size().to_dict(),
        'windows_by_family_phase_label': counts.to_dict('records'),
        'fault_cases_without_pre_fault_windows': {fam: int((f.pre_fault_windows == 0).sum()) for fam, f in fault.groupby('family')},
        'cases_without_pre_fault_windows_by_family': {fam: int((f.pre_fault_windows == 0).sum()) for fam, f in case.groupby('family')},
        'pre_fault_length_s': distribution, 'special_phase_cases': special,
        'normal_full_valid_exposure_h': float(case.normal_full_exposure_h.sum()),
        'fault_pre_valid_exposure_h': float(case.pre_fault_exposure_h.sum()),
        'missing_onset_fault_cases': fault.loc[fault.fault_onset_row.isna(), 'case_uid'].tolist(),
        'fold_family': fold}
    return records, result


def run():
    if OUT.exists():
        raise FileExistsError('Preserve previous attempt; fresh B0 output required')
    eligible, startup = eligible_split(SPLIT, SPLIT_SHA)
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    if cfg['status'] != 'FROZEN_BEFORE_REAL_VALUE_ACCESS' or sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen protocol/model identity mismatch')
    OUT.mkdir(parents=True, exist_ok=False)
    audit = install_boundary(eligible)
    started = time.perf_counter()
    stage = 'startup'
    with (OUT / 'run.log').open('x', encoding='utf-8', newline='\n') as journal:
        def note(*args):
            print(*args, flush=True); print(*args, file=journal, flush=True)
        def access():
            expected = {str((DATA / kind / (uid.replace('/', '__') + '.npz')).resolve())
                        for uid in eligible.case_uid for kind in ('windows', 'cases')}
            assert set(audit['windows']) | set(audit['cases']) <= expected
            return {'unique_window_npz': len(set(audit['windows'])), 'window_open_calls': len(audit['windows']),
                    'unique_case_npz': len(set(audit['cases'])), 'case_open_calls': len(audit['cases']),
                    'actual_window_files': sorted(set(audit['windows'])), 'actual_case_files': sorted(set(audit['cases'])),
                    'blocked_accesses': audit['blocked_accesses'], 'eligible_only': True}
        try:
            note('STARTUP_PASS', json.dumps(startup))
            frame = read_ledger(DATA / 'ledgers/real_included_windows.csv', eligible)
            with SOURCE.open('rb') as stream:
                source = pickle.load(stream)
            if not np.array_equal(source.classes_, [0, 1]) or source.n_features_in_ != 130:
                raise ValueError('Frozen S model has wrong classes/features')
            features, source_p = {}, {}
            def load_case(uid):
                if uid not in features:
                    part = frame[frame.case_uid.eq(uid)]
                    x = window_features(DATA / 'windows' / (uid.replace('/', '__') + '.npz'), part)
                    features[uid] = x
                    source_p[uid] = source.predict_proba(x)[:, 1] if len(x) else np.empty(0, float)
                    if not np.isfinite(source_p[uid]).all():
                        raise ValueError('Nonfinite S probability')
            gate_cases = eligible[eligible.role_v1.isin(['adapt_normal', 'calibrate_normal'])]
            if not len(gate_cases):
                raise ValueError('Reproduction gate must not be empty')
            ref = pd.read_csv(REFERENCE, usecols=['arm', 'case_uid', 'windows', 'GBT_win_fp05'])
            ref = ref[ref.case_uid.isin(gate_cases.case_uid) & ref.arm.eq('D0')].copy()
            if ref.case_uid.duplicated().any() or set(ref.case_uid) != set(gate_cases.case_uid):
                raise ValueError('Missing/ambiguous D0 gate reference')
            ref = ref.set_index('case_uid')
            gate = []
            stage = 'reproduction_gate'
            with threadpool_limits(limits=16):
                for uid in gate_cases.case_uid:
                    load_case(uid)
                    n, fp = len(source_p[uid]), int((source_p[uid] >= .5).sum())
                    gate.append({'case_uid': uid, 'windows': n, 'S_ge05': fp, 'expected_windows': int(ref.loc[uid, 'windows']),
                                 'expected_S_ge05': int(ref.loc[uid, 'GBT_win_fp05']),
                                 'pass': n == ref.loc[uid, 'windows'] and fp == ref.loc[uid, 'GBT_win_fp05']})
                gate_result = {'cases': len(gate), 'all_match': all(bool(x['pass']) for x in gate), 'rows': [{**x, 'pass': bool(x['pass'])} for x in gate]}
                save(OUT / 'reproduction_gate.json', gate_result)
                if not gate_result['all_match']:
                    save(OUT / 'results.json', {'status': 'STOPPED_GATE_MISMATCH', 'startup': startup, 'gate': gate_result, 'access': access(), 'B0_metrics_computed': False})
                    return 2
                note('REPRODUCTION_GATE_PASS', len(gate))
                stage = 'dev_features'
                for uid in eligible.case_uid:
                    load_case(uid)
                x = np.concatenate([features[uid] for uid in eligible.case_uid])
                ps = np.concatenate([source_p[uid] for uid in eligible.case_uid])
                ids = np.concatenate([frame.loc[frame.case_uid.eq(uid), 'original_cache_index'].to_numpy() for uid in eligible.case_uid])
                frame = frame.set_index('original_cache_index').loc[ids].reset_index()
                masks = populations(frame); y = frame.qualified_y.to_numpy()
                union = np.sum(np.stack(list(masks.values())), axis=0)
                if (union > 1).any() or ((union > 0) & (y != 0)).any():
                    raise ValueError('Registered normal populations overlap or include fault labels')
                unlisted_normal = (y == 0) & (union == 0)
                session_mask = masks['normal_full'] | masks['pre_fault']
                sy = masks['normal_full'].astype(np.int64)
                stage = 'session_oof'
                p_session, session_logs = oof(x, sy, frame, session_mask, 'session', note)
                stage = 'real_oof'
                pr, real_logs = oof(x, y, frame, np.ones(len(y), bool), 'real', note)
            export = frame[['case_uid', 'original_cache_index', 'dev_fold', 'family', 'phase', 'normal_origin', 'qualified_y']].copy()
            export['p_R'], export['p_session'], export['p_S'] = pr, p_session, ps
            export.to_csv(OUT / 'oof_predictions.csv', index=False)
            save(OUT / 'fit_receipt.json', {'session': session_logs, 'R': real_logs, 'models_saved': False,
                                           'recipe': RECIPE, 'CPU_native_thread_limit': 16})
            stage = 'auc_bootstrap'
            far = masks['pre_fault'] & (frame.window_end_row.to_numpy() <= frame.fault_onset_row.to_numpy() - 250)
            near = masks['pre_fault'] & ~far
            plans = {'session/main': auc_plan(p_session, sy, session_mask),
                     'session/far': auc_plan(p_session, sy, masks['normal_full'] | far),
                     'session/near': auc_plan(p_session, sy, masks['normal_full'] | near)}
            for flight in sorted(frame.flight.unique()):
                plans['session/flight/' + flight] = auc_plan(p_session, sy, session_mask & frame.flight.eq(flight).to_numpy())
            for arm, p in [('R', pr), ('S', ps)]:
                for stratum in ('pooled', 'motor', 'sensor'):
                    positive = (y == 1) & (True if stratum == 'pooled' else frame.family.eq(stratum).to_numpy())
                    for name, normal in [('a', y == 0), ('b', masks['pre_fault']), ('c', (y == 0) & frame.normal_origin.eq('normal_flight').to_numpy())]:
                        plans[f'{arm}/{stratum}/{name}'] = auc_plan(p, y, positive | normal)
            intervals, boot = auc_intervals(plans, frame.association_group.to_numpy(), eligible.association_group.to_numpy())
            source_desc = {name: {'windows': int(mask.sum()), 'cases': int(frame.loc[mask, 'case_uid'].nunique()),
                                 'median': float(np.median(ps[mask])) if mask.any() else None,
                                 'share_ge03744': float((ps[mask] >= .3744).mean()) if mask.any() else None,
                                 'share_ge05': float((ps[mask] >= .5).mean()) if mask.any() else None}
                           for name, mask in [('normal_full', masks['normal_full']), ('pre_fault', masks['pre_fault'])]}
            b01 = {'main': intervals['session/main'], 'sensitivity_far': intervals['session/far'],
                'near_only_descriptive': intervals['session/near'], 'near_windows_removed': int(near.sum()),
                'near_cases': int(frame.loc[near, 'case_uid'].nunique()), 'far_windows': int(far.sum()),
                'far_cases': int(frame.loc[far, 'case_uid'].nunique()), 'no_refit_supplement1': True,
                'by_flight': {f: intervals['session/flight/' + f] for f in sorted(frame.flight.unique())}, 'source_probability': source_desc}
            main_auc = b01['main']['auc']
            b01['interpretation'] = 'undefined' if main_auc is None else ('strong' if main_auc >= .80 else 'moderate' if main_auc >= .65 else 'weak')
            b02 = {}
            for arm, p in [('R', pr), ('S', ps)]:
                b02[arm] = {}
                for st in ('pooled', 'motor', 'sensor'):
                    tab = {name: intervals[f'{arm}/{st}/{name}'] for name in ('a', 'b', 'c')}
                    positive = (y == 1) & (True if st == 'pooled' else frame.family.eq(st).to_numpy())
                    mask = positive | (y == 0)
                    tab['AP_a'] = float(average_precision_score(y[mask], p[mask])) if positive.any() and (y[mask] == 0).any() else None
                    tab['c_minus_b'] = tab['c']['auc'] - tab['b']['auc'] if tab['c']['auc'] is not None and tab['b']['auc'] is not None else None
                    b02[arm][st] = tab
                b02[arm]['FPR_at_recall'] = {str(v): fpr_at_recall(p, y, masks, v) for v in (.5, .8)}
            b1 = recommendation(b02['R']['pooled']['b'], {k: b02['R'][k]['b'] for k in ('motor', 'sensor')})
            stage = 'label_exposure_audit'
            audit_rows, label = audits(frame, eligible, DATA)
            pd.DataFrame(audit_rows).to_csv(OUT / 'label_audit.csv', index=False)
            if sha(SOURCE) != SOURCE_SHA or sha(SPLIT) != SPLIT_SHA:
                raise ValueError('Frozen inputs changed')
            result = {'status': 'COMPLETE', 'protocol': str(CFG.relative_to(ROOT)), 'config_sha256': sha(CFG),
                'startup': startup, 'source_model_sha256': SOURCE_SHA, 'windows': len(frame), 'reproduction_gate': gate_result,
                'B0_1': b01, 'B0_2': b02, 'B1': b1, 'B0_3': label, 'bootstrap': boot,
                'normal_windows_outside_named_FPR_subsets': int(unlisted_normal.sum()),
                'access': access(), 'wall_time_s': time.perf_counter() - started, 'no_deployed_models': True,
                'no_working_point_or_event_replay': True, 'sealed_not_blind_history_disclosed': True}
            save(OUT / 'results.json', result)
            save(OUT / 'receipt.json', {'status': 'COMPLETE', 'startup': startup, 'access': access(),
                'source_model_sha256': SOURCE_SHA, 'config_sha256': sha(CFG), 'output_sha256': {name: sha(OUT / name) for name in ['results.json', 'oof_predictions.csv', 'label_audit.csv', 'reproduction_gate.json', 'fit_receipt.json']}})
            note('COMPLETE', json.dumps(b1, ensure_ascii=False))
            return 0
        except Exception as exc:
            note('FAILED_STOP', stage, repr(exc))
            save(OUT / 'failure.json', {'stage': stage, 'error': repr(exc), 'access': access(), 'no_retry': True})
            raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(run())
