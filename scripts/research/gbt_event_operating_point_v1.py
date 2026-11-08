"""Frozen GBT event operating point: CPU aggregation of existing scores only.

Selection is train-only. A failed reproduction gate or no feasible onset stops
before any val score is opened. No model/checkpoint/cache-feature loading.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
import numpy as np
import pandas as pd
import ens_event_threshold as reference

PROTOCOL = ROOT / 'configs/experiments/gbt_event_operating_point_v1.json'
SCORES = Path('D:/UAV-Data/ens_event_threshold_v1_scores')
DATA_ROOT = SCORES.parent
GROUP_LEDGER = Path('D:/UAV-Data/causal_input_v1/ledgers/train_all_windows.csv')
MODES = ROOT / 'artifacts/q2_gps_fault_modes.csv'
OUTPUT = ROOT / 'artifacts/gbt_event_operating_point_v1'
GRID = (.45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
CLASSES = ('GPS', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor')
BASE = dict(inference_every=1, clear_th=.4, k_on=3, k_off=3, min_event_s=.5, merge_gap_s=.5)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_metadata(scores, groups, modes):
    idx = pd.read_csv(Path(scores) / 'index.csv', dtype={'case_uid': str, 'split': str, 'domain': str})
    parts = idx.case_uid.str.split('/')
    if (len(idx) != 3135 or not idx.case_uid.is_unique or not parts.map(len).eq(4).all()
            or not idx.split.isin(['train', 'val']).all() or not idx.domain.isin(['SIL', 'HIL']).all()
            or not parts.str[0].eq(idx.domain).all()
            or parts.map(lambda p: any(x in ('', '.', '..') for x in p)).any()):
        raise ValueError('Invalid source-only case index')
    if idx.split.value_counts().to_dict() != {'train': 2580, 'val': 555}:
        raise ValueError('Wrong registered case populations')
    if not idx.loc[idx.split.eq('train'), 'fold'].isin([0, 1, 2]).all() or not idx.loc[idx.split.eq('val'), 'fold'].eq(-1).all():
        raise ValueError('Wrong held-out/val fold identifiers')
    idx['cls'], idx['flight'] = parts.str[2], parts.str[1]
    if not idx.cls.isin((*CLASSES, 'no_fault')).all():
        raise ValueError('Unexpected case class')
    mode = pd.read_csv(modes, dtype={'case_uid': str})
    if not mode.case_uid.is_unique:
        raise ValueError('Ambiguous GPS mode mapping')
    mode = mode.set_index('case_uid')['mode']
    gps = idx.cls.eq('GPS')
    idx['mode'] = idx.case_uid.map(mode)
    if idx.loc[gps, 'mode'].isna().any() or not idx.loc[gps, 'mode'].isin([1, 2, 3, 4]).all():
        raise ValueError('Every GPS case must have a valid mode; STOP')
    idx['excluded'] = gps & idx.flight.eq('hover') & idx['mode'].eq(4)
    counts = idx.loc[idx.excluded].groupby('split').size().to_dict()
    if counts != {'train': 36, 'val': 8}:
        raise ValueError('Wrong registered exclusion population: ' + str(counts))
    ledger = pd.read_csv(groups, usecols=['case_uid', 'association_group'], dtype=str, keep_default_na=False)
    wanted = set(idx.loc[idx.split.eq('train'), 'case_uid'])
    ledger = ledger[ledger.case_uid.isin(wanted)].drop_duplicates()
    if ledger.case_uid.duplicated().any() or ledger.association_group.eq('').any() or set(ledger.case_uid) != wanted:
        raise ValueError('Missing/ambiguous real association_group: STOP, never invent groups')
    group_map = ledger.set_index('case_uid').association_group
    idx['association_group'] = idx.case_uid.map(group_map)
    files = {p.name for p in Path(scores).glob('*.npz')}
    expected = {uid.replace('/', '__') + '.npz' for uid in idx.case_uid}
    if files != expected:
        raise ValueError('Score filenames do not match the complete case index')
    return idx, {'excluded_cases': counts, 'GPS_modes_missing': 0, 'train_groups': int(ledger.association_group.nunique()),
                 'group_source': str(Path(groups).resolve()), 'train_cases_with_groups': len(ledger)}


def install_source_boundary(scores, groups, output, train_uids, val_uids):
    """Applied in parent AND spawned workers; val requires train-only selection."""
    scores, output = Path(scores).resolve(), Path(output).resolve()
    allowed = {scores / 'index.csv', Path(groups).resolve()}
    allowed.update(scores / (u.replace('/', '__') + '.npz') for u in (*train_uids, *val_uids))
    val_paths = {scores / (u.replace('/', '__') + '.npz') for u in val_uids}
    data_root = DATA_ROOT.resolve()
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(data_root):
            if path not in allowed or any(c in (args[1] or '') for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError('Only registered read-only score/train-ledger files allowed: ' + str(path))
            if path in val_paths:
                selected = output / 'selection_train_only.json'
                if not selected.exists():
                    raise PermissionError('Val score access before train-only onset selection')
                choices = json.loads(selected.read_text(encoding='utf-8'))['chosen_onsets']
                if any(choices.get(a) not in GRID for a in ('GBT', 'M0')):
                    raise PermissionError('Val forbidden when there is no feasible train onset')
    sys.addaudithook(guard)


def case_records(job):
    uid, specifications, directory = job
    from replay_m1_continuous import aggregate_scores
    from uav_fault.online.detection_policy import OnlineDetectionPolicy
    with np.load(Path(directory) / (uid.replace('/', '__') + '.npz'), allow_pickle=False) as z:
        times_us, m0, gbt = z['timestamp_us'], z['m0'].astype(float), z['gbt'].astype(float)
        ref, valid = z['reference_state'], z['row_label_valid']
    if (times_us.ndim != 1 or not len(times_us) or times_us.dtype != np.int64
            or any(a.shape != times_us.shape for a in (m0, gbt, ref, valid))
            or valid.dtype != np.bool_ or (np.diff(times_us) <= 0).any()
            or not np.isin(ref[valid], [0, 1]).all() or np.isinf(m0).any() or np.isinf(gbt).any()):
        raise ValueError('Invalid saved score arrays: ' + uid)
    scored = np.isfinite(m0)
    if not np.array_equal(scored, np.isfinite(gbt)):
        raise ValueError('M0/GBT score availability differs: ' + uid)
    if any(((p[scored] < 0) | (p[scored] > 1)).any() for p in (m0, gbt)):
        raise ValueError('Saved probability outside [0,1]')
    times = times_us / 1e6
    normal = valid & (ref == 0)
    on = np.flatnonzero(valid & (ref == 1))
    onset = float(times[on[0]]) if len(on) else None
    rows = []
    for arm, thresholds in specifications:
        prob = {'M0': m0, 'GBT': gbt, 'ENS': (m0 + gbt) / 2}[arm]
        for threshold in thresholds:
            events = aggregate_scores(times, prob, policy=OnlineDetectionPolicy(onset_th=threshold, **BASE))
            finalized = events['finalized_events']
            occupancy = np.zeros(len(times), dtype=bool)
            false_count = 0
            for event in finalized:
                mask = (times >= event['start_t']) & (times <= event['end_t'])
                if mask.any() and normal[mask].all():
                    false_count += 1
                    occupancy |= mask
            confirmations = [c for c in events['confirmations'] if onset is not None and c['candidate_start_t'] >= onset]
            first = float(confirmations[0]['confirmed_t']) if confirmations else np.nan
            rows.append({'case_uid': uid, 'arm': arm, 'onset_th': threshold, 'false_events': false_count,
                'normal_exposure_s': float((normal & scored).sum() * .02),
                'false_occupancy_s': float((occupancy & normal & scored).sum() * .02),
                'onset_s': onset if onset is not None else np.nan,
                'post_onset_scored': bool(onset is not None and (scored & (times >= onset)).any()),
                'detected': bool(confirmations), 'first_confirmed_t': first,
                'confirm_latency_s': first - onset if confirmations else np.nan,
                'spans_onset': bool(onset is not None and any(e['start_t'] < onset <= e['end_t'] for e in finalized)),
                'scored_frames': int(scored.sum())})
    return rows


def collect(pool, idx, specifications, scores):
    jobs = ((uid, specifications, str(scores)) for uid in idx.case_uid)
    rows = []
    for number, batch in enumerate(pool.map(case_records, jobs, chunksize=8), 1):
        rows.extend(batch)
        if number % 250 == 0 or number == len(idx):
            print('AGGREGATED', number, '/', len(idx), specifications, flush=True)
    return pd.DataFrame(rows).merge(idx, on='case_uid', validate='many_to_one')


def paired_latency(data, arm, threshold, *, exclude=True):
    data = data[~data.excluded] if exclude else data
    base = data[data.arm.eq('M0') & data.onset_th.eq(.60)].set_index('case_uid')
    candidate = data[data.arm.eq(arm) & data.onset_th.eq(threshold)].set_index('case_uid')
    if not base.index.equals(candidate.index):
        candidate = candidate.reindex(base.index)
    both = base.detected.eq(True) & candidate.detected.eq(True)
    a, b = base.loc[both, 'confirm_latency_s'], candidate.loc[both, 'confirm_latency_s']
    delta = b - a
    return {'cases': int(both.sum()), 'median': float(delta.median()) if len(delta) else None,
            'p90': float(delta.quantile(.9)) if len(delta) else None,
            'baseline_median_s': float(a.median()) if len(a) else None,
            'candidate_median_s': float(b.median()) if len(b) else None}


def selection_table(train):
    allowed = train[~train.excluded]
    target = reference.summarize(allowed, 'M0', .60)['detected_by_class']
    if set(target) != set(CLASSES):
        raise ValueError('Baseline must contain all six scored fault classes')
    rows = []
    chosen = {}
    for arm in ('GBT', 'M0'):
        candidates = []
        for threshold in GRID:
            summary = reference.summarize(allowed, arm, threshold)
            full = reference.summarize(train, arm, threshold)
            counts = {c: summary['detected_by_class'].get(c, 0) for c in CLASSES}
            latency = paired_latency(train, arm, threshold)
            a = all(counts[c] >= target[c] for c in CLASSES)
            b = latency['median'] is not None and latency['median'] <= .5
            row = {'arm': arm, 'onset_th': threshold, 'constraint_a': a, 'constraint_b': b, 'feasible': a and b,
                'detected_by_class': counts, 'baseline_targets': target, 'paired_latency': latency,
                'indep_false_events': full['indep_false_events'], 'indep_events_per_h': full['indep_events_per_h'],
                'detected_total': full['fault_detected'], 'detected_excluding': summary['fault_detected']}
            rows.append(row)
            if row['feasible']:
                candidates.append(row)
        best = min(candidates, key=lambda r: (r['indep_events_per_h'], -r['detected_total'], r['onset_th'])) if candidates else None
        chosen[arm] = best['onset_th'] if best else None
    return rows, chosen


def acceptance(train, val, chosen):
    threshold = chosen['GBT']
    tr0, tr1 = reference.summarize(train, 'M0', .6), reference.summarize(train, 'GBT', threshold)
    va0, va1 = reference.summarize(val, 'M0', .6), reference.summarize(val, 'GBT', threshold)
    clean = val[~val.excluded]
    c0, c1 = reference.summarize(clean, 'M0', .6), reference.summarize(clean, 'GBT', threshold)
    allowance = .01 * va0['fault_cases']  # Registered scored-case denominator; detected totals remove exclusions.
    deficits = {c: c0['detected_by_class'].get(c, 0) - c1['detected_by_class'].get(c, 0) for c in CLASSES}
    values = {'EV1': {'pass': bool(tr1['indep_events_per_h'] < tr0['indep_events_per_h']),
        'GBT_per_h': tr1['indep_events_per_h'], 'M0_per_h': tr0['indep_events_per_h']},
        'EV2': {'pass': va1['indep_false_events'] <= va0['indep_false_events'],
        'GBT_events': va1['indep_false_events'], 'M0_events': va0['indep_false_events']},
        'EV3': {'pass': c1['fault_detected'] >= c0['fault_detected'] - allowance,
        'GBT_detected_excluding': c1['fault_detected'], 'M0_detected_excluding': c0['fault_detected'],
        'scored_fault_cases': va0['fault_cases'], 'scored_fault_cases_excluding': c0['fault_cases'],
        'allowance_cases': allowance, 'minimum_detected': c0['fault_detected'] - allowance,
        'per_class_deficits': deficits, 'flagged_classes_deficit_gt2': [c for c, n in deficits.items() if n > 2]}}
    return {'criteria': values, 'verdict': 'PASS' if all(v['pass'] for v in values.values()) else 'FAIL'}


def per_class(data, arm, threshold):
    x = data[data.arm.eq(arm) & data.onset_th.eq(threshold)]
    fault = x[x.cls.ne('no_fault') & x.post_onset_scored.eq(True)]
    result = {}
    for cls in CLASSES:
        f = fault[fault.cls.eq(cls)]
        latency = f.loc[f.detected, 'confirm_latency_s']
        result[cls] = {'scored_cases': len(f), 'detected': int(f.detected.sum()), 'spans_onset': int(f.spans_onset.sum()),
            'confirm_median_s': float(latency.median()) if len(latency) else None,
            'confirm_p90_s': float(latency.quantile(.9)) if len(latency) else None}
    return result


def summarize_selected(data, arm, threshold):
    selected = data[data.arm.eq(arm) & data.onset_th.eq(threshold)]
    return {'all_cases': reference.summarize(data, arm, threshold),
        'excluding': reference.summarize(data[~data.excluded], arm, threshold),
        'per_class_all': per_class(data, arm, threshold),
        'per_class_excluding': per_class(data[~data.excluded], arm, threshold),
        'by_flight': {flight: reference.summarize(selected[selected.flight.eq(flight)], arm, threshold)
                      for flight in sorted(selected.flight.unique())},
        'excluded': {'cases': int(selected.excluded.sum()), 'scored_cases': int((selected.excluded & selected.post_onset_scored).sum()),
            'detected': int((selected.excluded & selected.detected).sum()),
            'first_confirmed_before30': int((selected.excluded & selected.detected & selected.first_confirmed_t.lt(30)).sum()),
            'first_confirmed_at30': int((selected.excluded & selected.detected & selected.first_confirmed_t.eq(30)).sum()),
            'first_confirmed_after30': int((selected.excluded & selected.detected & selected.first_confirmed_t.gt(30)).sum())}}


def bootstrap_rates(train, chosen, reps=2000, seed=20261106):
    groups = train[['case_uid', 'association_group']].drop_duplicates().set_index('case_uid')
    if groups.association_group.isna().any() or groups.association_group.eq('').any():
        raise ValueError('Missing actual association group')
    cases = sorted(groups.index)
    unique, inverse = np.unique(groups.loc[cases, 'association_group'].to_numpy(), return_inverse=True)
    definitions = [('GBT', chosen['GBT']), ('M0', .6), ('M0', chosen['M0'])]
    values = []
    for arm, threshold in definitions:
        d = train[train.arm.eq(arm) & train.onset_th.eq(threshold)].set_index('case_uid').loc[cases]
        normal = d.cls.eq('no_fault').to_numpy()
        values.append((d.false_events.to_numpy(dtype=float) * normal, d.normal_exposure_s.to_numpy() * normal))
    rng = np.random.default_rng(seed)
    samples = [[], []]
    for _ in range(reps):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = count[inverse]
        rates = [float(np.dot(weight, f) * 3600 / np.dot(weight, h)) if np.dot(weight, h) > 0 else None for f, h in values]
        for i, base in enumerate(rates[1:]):
            samples[i].append(rates[0] / base if base is not None and base > 0 and rates[0] is not None else None)
    out = {}
    for label, sample in zip(('GBT_over_M0_0.60', 'GBT_over_M0_tuned'), samples):
        undefined = sum(v is None for v in sample)
        out[label] = {'ci95': np.percentile(sample, [2.5, 97.5]).tolist() if not undefined else None,
                      'undefined_replicates': undefined}
    return {'replicates': reps, 'rng_seed': seed, 'groups': len(unique), 'population': 'all train case association_groups; only independent-normal cases contribute false events/exposure',
            'onset_reselected': False, 'paired_shared_plan': True, 'comparisons': out}


def cap64_check():
    feature = ROOT / 'artifacts/feature_groupcv_v1_20261105'
    troot = ROOT / 'artifacts/t_selection_mstca_v1/cv'
    needed = [troot / f'seed42_fold{f}/heldout_predictions.npz' for f in range(3)]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if missing:
        return {'status': '无法核对', 'reason': '本地缺少真实T0 seed42 CV held-out概率；不使用smoke/合成测试产物冒充，不另行推理。', 'missing': missing,
                'feature_fold_files_present': all((feature / f'fold{f}/heldout_predictions.npz').is_file() for f in range(3))}
    # The current registered input layout is handled only when real T trajectories exist.
    values = []
    for fold, path in enumerate(needed):
        with np.load(feature / f'fold{fold}/heldout_predictions.npz', allow_pickle=False) as old, np.load(path, allow_pickle=False) as new:
            selected = json.loads((path.parent / 'selections.json').read_text(encoding='utf-8'))['T0']['epoch']
            p = pd.Series(new['probabilities'][list(new['epochs']).index(selected)], index=new['original_cache_index'])
            ids = old['original_indices']
            if not pd.Index(ids).isin(p.index).all():
                return {'status': '无法核对', 'reason': '原折cap64窗口索引无法在T0原始窗口索引中完整对齐；未猜测映射。'}
            a, b = old['probabilities'][:, 1], p.loc[ids].to_numpy()
            values.append({'fold': fold, 'windows': len(ids), 'equal': bool(np.array_equal(a, b)), 'max_abs_difference': float(np.max(np.abs(a.astype(float) - b.astype(float))))})
    return {'status': 'CHECKED', 'folds': values, 'equal': all(r['equal'] for r in values)}


def run(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Fresh output required; preserve every earlier attempt')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata, audit = read_metadata(args.scores, args.groups, args.modes)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if tuple(protocol['event_policy']['onset_grid']) != GRID or protocol['status'] != 'FROZEN_BEFORE_RUN':
        raise ValueError('Frozen protocol/grid differs')
    train_idx = metadata[metadata.split.eq('train')]
    val_idx = metadata[metadata.split.eq('val')]
    result = {'protocol': str(PROTOCOL.relative_to(ROOT)), 'protocol_commit': '3a9ac66dfe8898d590ab89ffb38acf85a27d3e43',
        'status': 'RUNNING', 'metadata': audit, 'grid': GRID, 'policy': BASE, 'workers': args.workers,
        'created_at': datetime.now(timezone.utc).isoformat(), 'no_training_no_new_inference': True}
    boundary = (args.scores, args.groups, out, tuple(train_idx.case_uid), tuple(val_idx.case_uid))
    install_source_boundary(*boundary)
    with ProcessPoolExecutor(args.workers, initializer=install_source_boundary, initargs=boundary) as pool:
        gate_data = collect(pool, train_idx, [('M0', (.6,)), ('ENS', (.75,))], args.scores)
        base = reference.summarize(gate_data, 'M0', .6)
        ens = reference.summarize(gate_data, 'ENS', .75)
        observed = [base['indep_false_events'], base['fault_detected'], ens['indep_false_events'], ens['fault_detected']]
        result['reproduction_gate'] = {'actual': observed, 'expected': [113, 1513, 81, 1705], 'pass': observed == [113, 1513, 81, 1705]}
        write_json(out / 'reproduction_gate.json', result['reproduction_gate'])
        if not result['reproduction_gate']['pass']:
            result.update(status='STOPPED_REPRODUCTION_GATE', wall_time_s=time.perf_counter() - started)
            gate_data.to_csv(out / 'events_per_case.csv', index=False)
            write_json(out / 'results.json', result)
            return result
        print('REPRODUCTION_PASS', observed, flush=True)
        rest = collect(pool, train_idx, [('GBT', GRID), ('M0', tuple(th for th in GRID if th != .6))], args.scores)
        train = pd.concat([gate_data[gate_data.arm.eq('M0')], rest], ignore_index=True)
        table, chosen = selection_table(train)
        result.update(selection_table=table, chosen_onsets=chosen)
        write_json(out / 'selection_train_only.json', {'table': table, 'chosen_onsets': chosen, 'val_scores_read': False})
        if chosen['GBT'] is None or chosen['M0'] is None:
            result.update(status='STOPPED_NO_FEASIBLE_ONSET', val_scores_read=False, wall_time_s=time.perf_counter() - started)
            pd.concat([train, gate_data[gate_data.arm.eq('ENS')]], ignore_index=True).to_csv(out / 'events_per_case.csv', index=False)
            pd.DataFrame([{**reference.summarize(train, a, th), 'split': 'train', 'arm': a, 'onset_th': th} for a in ('GBT', 'M0') for th in GRID]).to_csv(out / 'sweep.csv', index=False)
            write_json(out / 'results.json', result)
            return result
        print('ONSET_SELECTED_TRAIN_ONLY', chosen, flush=True)
        val = collect(pool, val_idx, [('GBT', GRID), ('M0', GRID)], args.scores)
    result.update(status='COMPLETE', acceptance=acceptance(train, val, chosen),
        bootstrap=bootstrap_rates(train, chosen), cap64_check=cap64_check(),
        summaries={split: {label: summarize_selected(data, arm, th) for label, arm, th in
            [('M0_0.60', 'M0', .6), ('GBT_selected', 'GBT', chosen['GBT']), ('M0_tuned', 'M0', chosen['M0'])]}
            for split, data in [('train', train), ('val', val)]},
        paired_latency={split: {label: paired_latency(data, arm, th) for label, arm, th in
            [('GBT_vs_baseline', 'GBT', chosen['GBT']), ('M0_tuned_vs_baseline', 'M0', chosen['M0'])]}
            for split, data in [('train', train), ('val', val)]}, wall_time_s=time.perf_counter() - started)
    pd.concat([train, val, gate_data[gate_data.arm.eq('ENS')]], ignore_index=True).to_csv(out / 'events_per_case.csv', index=False)
    pd.DataFrame([{**reference.summarize(data, arm, th), 'split': split, 'arm': arm, 'onset_th': th}
        for split, data in [('train', train), ('val', val)] for arm in ('GBT', 'M0') for th in GRID]).to_csv(out / 'sweep.csv', index=False)
    write_json(out / 'results.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, default=SCORES)
    p.add_argument('--groups', type=Path, default=GROUP_LEDGER)
    p.add_argument('--modes', type=Path, default=MODES)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--workers', type=int, default=8)
    return p


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    result = run(parser().parse_args())
    print(json.dumps({'status': result['status'], 'chosen': result.get('chosen_onsets'), 'acceptance': result.get('acceptance')}, ensure_ascii=False), flush=True)
    if result['status'].startswith('STOPPED_'):
        raise SystemExit(2)
