"""One final internal-test evaluation, four fixed operating points only.

Real execution cannot resume: existing output OR nonempty score directory is refused.
An existing empty score directory is allowed by PI supplement 1.
No metrics are called until all frame/window parity checks pass and the new
TEST_EVALUATED.lock has been exclusively created. Synthetic tests call only
pure helpers, never this entry point or real test data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
import numpy as np
import pandas as pd
import gbt_event_operating_point_v1 as events

DATA = Path('D:/UAV-Data/causal_input_v1')
DATA_ROOT = DATA.parent
SCORES = Path('D:/UAV-Data/internal_test_final_v1_scores')
OUT = ROOT / 'artifacts/internal_test_final_v1'
CFG = ROOT / 'configs/experiments/internal_test_final_v1.json'
SAVED = ROOT / 'artifacts/ens_internal_test_v1_20261105/test_predictions.npz'
OLD_LOCK = SAVED.parent / 'TEST_EVALUATED.lock'
SAVED_SHA = '45360dbf5e136dc30e881bf9989cb0307683fe1cd6ff19ce8cdfde886411e8f8'
GBT_WINDOW = .3744
GBT_EVENT = .95
M0_EVENT = .60
CLASSES = events.CLASSES
POLICY = events.BASE


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_targets(output, scores):
    scores = Path(scores)
    if Path(output).exists() or (scores.exists() and (not scores.is_dir() or any(scores.iterdir()))):
        raise FileExistsError('Final test is one-shot: existing output/lock or nonempty scores forbid restart')


def align_windows(ledger, original_indices):
    ids = np.asarray(original_indices)
    if ids.dtype != np.int64 or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique int64 saved original indices required')
    if not ledger.original_cache_index.is_unique or set(ids.tolist()) != set(ledger.original_cache_index.tolist()):
        raise ValueError('Saved predictions and supervised window ledger do not cover the same indices')
    result = ledger.set_index('original_cache_index').loc[ids].reset_index()
    if not ((result.window_end_row - result.window_start_row) == 150).all():
        raise ValueError('Expected end-exclusive 150-row windows; never guess alignment offsets')
    result['physical_end_row'] = result.window_start_row + 149
    return result


def startup():
    fresh_targets(OUT, SCORES)
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    if cfg['status'] != 'FROZEN_BEFORE_TEST_ACCESS' or not cfg.get('user_approval'):
        raise ValueError('Explicit frozen final-test approval required')
    models = {}
    for key, description in cfg['frozen_models'].items():
        path = ROOT / description.split(', sha256 ')[0]
        match = re.search(r'sha256 ([0-9a-f]{64})', description)
        actual = sha(path)
        if match is None or actual != match.group(1):
            raise ValueError('Model identity mismatch: ' + key)
        models[key] = {'path': str(path), 'sha256': actual}
    if sha(SAVED) != SAVED_SHA or not OLD_LOCK.is_file():
        raise ValueError('Saved test predictions identity or historical lock missing/mismatched')
    old_lock = {'sha256': sha(OLD_LOCK), 'mtime_ns': OLD_LOCK.stat().st_mtime_ns}
    tasks = [t for t in json.loads((DATA / 'tasks.json').read_text(encoding='utf-8')) if t['split'] == 'test']
    if len(tasks) != 553 or len({t['case_uid'] for t in tasks}) != 553:
        raise ValueError('Exactly 553 unique test tasks required')
    records = []
    mode_table = pd.read_csv(ROOT / 'artifacts/q2_gps_fault_modes.csv', dtype={'case_uid': str})
    if not mode_table.case_uid.is_unique:
        raise ValueError('Ambiguous fault mode table')
    mode_map = mode_table.set_index('case_uid')['mode']
    for task in tasks:
        parts = task['case_uid'].split('/')
        group = task.get('group')
        if (len(parts) != 4 or any(p in ('', '.', '..') for p in parts) or parts[0] != task['domain']
                or task['domain'] not in ('SIL', 'HIL') or parts[2] not in (*CLASSES, 'no_fault')
                or task.get('role') != 'test' or not isinstance(group, str) or not group):
            raise ValueError('Invalid source-test task/group; no fabricated grouping')
        mode = None
        if parts[2] == 'GPS':
            if task['case_uid'] not in mode_map.index or mode_map[task['case_uid']] not in (1, 2, 3, 4):
                raise ValueError('Every test GPS case must have a valid mode')
            mode = int(mode_map[task['case_uid']])
        records.append({'case_uid': task['case_uid'], 'split': 'test', 'domain': task['domain'], 'flight': parts[1],
            'cls': parts[2], 'association_group': group, 'mode': mode,
            'excluded': parts[1] == 'hover' and parts[2] == 'GPS' and mode == 4})
    metadata = pd.DataFrame(records)
    if int(metadata.cls.eq('GPS').sum()) != 48:
        raise ValueError('Expected 48 test GPS cases')
    ledger = pd.read_csv(DATA / 'ledgers/test_included_windows.csv',
        dtype={'case_uid': str, 'association_group': str, 'domain': str, 'split': str, 'role': str}, keep_default_na=False, low_memory=False)
    if (len(ledger) != 34802 or not ledger.compact_index.eq(np.arange(len(ledger))).all()
            or not ledger.split.eq('test').all() or not ledger.role.eq('test').all()
            or not ledger.qualified_y.isin([0, 1]).all()):
        raise ValueError('Wrong frozen supervised test population')
    for column in ('input_valid', 'label_valid', 'supervised_valid'):
        if ledger[column].dtype != bool or not ledger[column].all():
            raise ValueError('Unqualified test ledger row')
    by_case = metadata.set_index('case_uid')
    if not ledger.case_uid.isin(by_case.index).all():
        raise ValueError('Window case outside authorized test tasks')
    if not np.array_equal(ledger.association_group.to_numpy(), by_case.association_group.loc[ledger.case_uid].to_numpy()):
        raise ValueError('Window/case association_group mismatch')
    with np.load(SAVED, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in ('m0_probabilities', 'gbt_prob', 'labels', 'original_indices')}
    n = len(ledger)
    if (arrays['m0_probabilities'].shape != (n, 2) or arrays['gbt_prob'].shape != (n,)
            or arrays['labels'].dtype != np.int64 or arrays['labels'].shape != (n,)
            or not np.isin(arrays['labels'], [0, 1]).all()):
        raise ValueError('Invalid saved prediction schema')
    for key in ('m0_probabilities', 'gbt_prob'):
        a = arrays[key]
        if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
            raise ValueError('Invalid saved probabilities')
    aligned = align_windows(ledger, arrays['original_indices'])
    normalization = json.loads((DATA / 'norm_stats.json').read_text(encoding='utf-8'))
    if normalization['fit_split'] != 'source_train_only' or len(normalization['columns']) != 26:
        raise ValueError('Frozen source-train normalization required')
    mu = np.asarray(normalization['mean'], np.float32)[None, :, None]
    sd = np.asarray(normalization['std'], np.float32)[None, :, None]
    if mu.shape != (1, 26, 1) or sd.shape != (1, 26, 1) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError('Invalid frozen normalization')
    info = {'status': 'STARTUP_PASS', 'config_sha256': sha(CFG), 'models': models, 'saved_predictions_sha256': sha(SAVED),
        'historical_lock': old_lock, 'test_cases': 553, 'supervised_windows': n, 'GPS_cases': 48,
        'excluded_case_count': int(metadata.excluded.sum()), 'excluded_case_uids': metadata.loc[metadata.excluded, 'case_uid'].tolist(),
        'association_groups': int(metadata.association_group.nunique()), 'normalization_sha256': sha(DATA / 'norm_stats.json'),
        'ledger_sha256': sha(DATA / 'ledgers/test_included_windows.csv'), 'window_end_rule': 'end-exclusive; physical final index=start+149=end-1',
        'freeze_commit': subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(CFG.relative_to(ROOT))], cwd=ROOT, text=True).strip()}
    return info, metadata, aligned, arrays, mu, sd


def input_boundary(metadata):
    """Only selected test-case payloads; new score output may be written."""
    readonly = {DATA / 'tasks.json', DATA / 'norm_stats.json', DATA / 'input_contract.json', DATA / 'ledgers/test_included_windows.csv'}
    readonly.update(DATA / 'cases' / (uid.replace('/', '__') + '.npz') for uid in metadata.case_uid)
    readonly = {p.resolve() for p in readonly}
    protected = {OLD_LOCK.resolve(), SAVED.resolve()}
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    protected.update((ROOT / x.split(', sha256 ')[0]).resolve() for x in cfg['frozen_models'].values())
    data_root, score_root = DATA_ROOT.resolve(), SCORES.resolve()
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            writing = any(c in (args[1] or '') for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if p in protected and writing:
                raise PermissionError('Frozen asset/historical lock is read-only')
            if p.is_relative_to(data_root) and not p.is_relative_to(score_root):
                if p not in readonly or writing:
                    raise PermissionError('Final test forbids other source/Real/hold-out access or source writes: ' + str(p))
        if event in ('os.remove', 'os.rename'):
            for item in args[:2]:
                if isinstance(item, (str, bytes, os.PathLike)):
                    p = Path(os.fsdecode(item)).resolve()
                    if p in protected or p in readonly:
                        raise PermissionError('Cannot remove/move frozen inputs')
    sys.addaudithook(guard)


def score_arrays(rows, valid, m0, gbt, mu, sd):
    """Exact val scoring recipe; raw stats for GBT, frozen normalization for M0."""
    import torch
    from replay_ens_events_val import case_windows, stats
    n = len(rows)
    if rows.shape != (n, 26) or valid.shape != (n,) or valid.dtype != bool:
        raise ValueError('Invalid canonical case input layout')
    pm, pg = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    if n < 150:
        return pm, pg
    ends, xr = case_windows(rows, valid)
    if len(ends):
        if not np.isfinite(xr).all():
            raise ValueError('Nonfinite valid canonical windows; no imputation or retry')
        with torch.no_grad():
            pm[ends] = np.concatenate([m0(torch.from_numpy((xr[i:i + 2048] - mu) / sd)).softmax(1)[:, 1].numpy()
                                      for i in range(0, len(xr), 2048)])
        pg[ends] = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
        if not np.isfinite(pm[ends]).all() or not np.isfinite(pg[ends]).all():
            raise ValueError('Nonfinite model output at a valid input window; do not silently drop it')
    if not np.array_equal(np.isfinite(pm), np.isfinite(pg)) or np.isinf(pm).any() or np.isinf(pg).any():
        raise ValueError('Invalid or unequal frozen-model score availability')
    for p in (pm, pg):
        if ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
            raise ValueError('Frozen model probability outside [0,1]')
    return pm, pg


def score_cases(metadata, info, mu, sd, note):
    import torch
    from score_ens_frames import load_m0
    torch.set_num_threads(16)
    m0 = load_m0(Path(info['models']['M0_baseline']['path']))
    with Path(info['models']['GBT']['path']).open('rb') as stream:
        gbt = pickle.load(stream)
    hashes = {}
    for number, row in enumerate(metadata.itertuples(index=False), 1):
        key = row.case_uid.replace('/', '__') + '.npz'
        with np.load(DATA / 'cases' / key, allow_pickle=False) as z:
            timestamps, rows, valid = z['timestamp_us'], z['canonical_rows'], z['row_input_valid']
            ref, label_valid = z['reference_state'], z['row_label_valid']
        if (timestamps.dtype != np.int64 or timestamps.shape != (len(rows),) or not len(rows)
                or (np.diff(timestamps) <= 0).any() or ref.shape != timestamps.shape
                or label_valid.shape != timestamps.shape or label_valid.dtype != bool
                or not np.isin(ref[label_valid], [0, 1]).all()):
            raise ValueError('Invalid test case timestamps/reference: ' + row.case_uid)
        pm, pg = score_arrays(rows, valid, m0, gbt, mu, sd)
        path = SCORES / key
        with path.open('xb') as stream:
            np.savez_compressed(stream, timestamp_us=timestamps, m0=pm, gbt=pg, reference_state=ref, row_label_valid=label_valid)
        hashes[row.case_uid] = {'file': key, 'sha256': sha(path), 'frames': len(timestamps), 'scored_frames': int(np.isfinite(pm).sum())}
        if number % 25 == 0 or number == len(metadata):
            note('SCORED_CASES', number, '/', len(metadata), row.case_uid)
    metadata.to_csv(SCORES / 'index.csv', index=False)
    return {'directory': str(SCORES), 'cases': len(hashes), 'index_sha256': sha(SCORES / 'index.csv'), 'files': hashes,
            'torch_threads': 16, 'M0_batch': 2048, 'GBT_batch': 4096, 'device': 'cpu', 'no_fit_calls': True}


def parity_gate(ledger, saved, scores):
    aligned = align_windows(ledger, saved['original_indices'])
    matched, missing = 0, []
    max_m0, max_gbt = 0., 0.
    for uid, positions in aligned.groupby('case_uid', sort=False).indices.items():
        positions = np.asarray(positions)
        end = aligned.iloc[positions].physical_end_row.to_numpy(dtype=np.int64)
        with np.load(Path(scores) / (uid.replace('/', '__') + '.npz'), allow_pickle=False) as z:
            pm, pg = z['m0'], z['gbt']
        ok = (end >= 0) & (end < len(pm))
        finite = np.zeros(len(end), dtype=bool)
        finite[ok] = np.isfinite(pm[end[ok]]) & np.isfinite(pg[end[ok]])
        if not finite.all():
            missing.extend(aligned.iloc[positions[~finite]].original_cache_index.tolist())
        rows = positions[finite]
        physical = end[finite]
        matched += len(rows)
        if len(rows):
            max_m0 = max(max_m0, float(np.max(np.abs(pm[physical].astype(float) - saved['m0_probabilities'][rows, 1].astype(float)))))
            max_gbt = max(max_gbt, float(np.max(np.abs(pg[physical].astype(float) - saved['gbt_prob'][rows].astype(float)))))
    labels_equal = bool(np.array_equal(aligned.qualified_y.to_numpy(), saved['labels']))
    passed = matched == len(aligned) and not missing and labels_equal and max_gbt <= 1e-6 and max_m0 <= .002
    return {'status': 'PASS' if passed else 'FAIL', 'expected_windows': len(aligned), 'aligned_windows': matched,
            'missing_window_original_indices': missing, 'labels_equal': labels_equal,
            'GBT_max_abs_difference': max_gbt, 'GBT_limit': 1e-6,
            'M0_max_abs_difference': max_m0, 'M0_limit': .002, 'pass': bool(passed)}


def write_lock(path, info, parity):
    if not parity['pass']:
        raise RuntimeError('No metric lock may be written after a failed parity gate')
    record = {'time': datetime.now(timezone.utc).isoformat(), 'config_sha256': info['config_sha256'],
              'models': info['models'], 'saved_predictions_sha256': info['saved_predictions_sha256'],
              'parity_gate': parity, 'operating_points': {'GBT_window': .3744, 'M0_window': 'argmax', 'GBT_event': .95, 'M0_event': .60},
              'policy': POLICY, 'final_use_no_rerun': True}
    save(path, record)
    return record


def window_metrics(meta, saved, metadata):
    by_case = metadata.set_index('case_uid')
    frame = meta.copy()
    for field in ('domain', 'flight', 'cls', 'mode', 'excluded'):
        frame[field] = by_case.loc[frame.case_uid, field].to_numpy()
    y = saved['labels']
    # Exactly the two registered WINDOW points, once. No alternative thresholds.
    predictions = {'GBT': saved['gbt_prob'] >= GBT_WINDOW, 'M0': saved['m0_probabilities'].argmax(axis=1) == 1}
    late_mode4 = ((y == 1) & frame.cls.eq('GPS').to_numpy() & frame.flight.eq('hover').to_numpy()
                  & frame['mode'].eq(4).to_numpy() & (frame.window_end_row.to_numpy() * .02 > 30))
    def count(mask):
        return {arm: {'windows': int(mask.sum()), 'normal_windows': int((mask & (y == 0)).sum()),
            'fault_windows': int((mask & (y == 1)).sum()),
            'normal_FP': int((mask & (y == 0) & p).sum()), 'fault_TP': int((mask & (y == 1) & p).sum())}
            for arm, p in predictions.items()}
    classes = {}
    for cls in CLASSES:
        mask = (y == 1) & frame.cls.eq(cls).to_numpy() & (~late_mode4 if cls == 'GPS' else True)
        n = int(mask.sum())
        tp = {arm: int(p[mask].sum()) for arm, p in predictions.items()}
        classes[cls] = {'windows': n, 'M0_TP': tp['M0'], 'GBT_TP': tp['GBT'],
            'M0_recall': tp['M0'] / n if n else None, 'GBT_recall': tp['GBT'] / n if n else None,
            'delta_pp': 100 * (tp['GBT'] - tp['M0']) / n if n else None,
            'pass': bool(n and 100 * tp['GBT'] >= 100 * tp['M0'] - n)}
    excluded = {'windows': int(late_mode4.sum()), 'cases': int(frame.loc[late_mode4, 'case_uid'].nunique()),
        'recall': {a: float(p[late_mode4].mean()) if late_mode4.any() else None for a, p in predictions.items()}}
    result = {'operating_points': {'GBT': .3744, 'M0': 'argmax'}, 'overall': count(np.ones(len(y), bool)),
        'F3': classes, 'excluded_late_hover_mode4': excluded,
        'by_domain': {v: count(frame.domain.eq(v).to_numpy()) for v in sorted(frame.domain.unique())},
        'by_flight': {v: count(frame.flight.eq(v).to_numpy()) for v in sorted(frame.flight.unique())},
        'ENS_known_normal_FP_reference_only': 342}
    return result, predictions, frame


def event_primary(summaries):
    b, p = summaries['M0'], summaries['GBT']
    b0, p0 = b['all_cases'], p['all_cases']
    bc, pc = b['excluding'], p['excluding']
    n = b0['fault_cases']
    allowance = .01 * n
    deficits = {c: bc['detected_by_class'].get(c, 0) - pc['detected_by_class'].get(c, 0) for c in CLASSES}
    f1 = bool(p0['indep_events_per_h'] < b0['indep_events_per_h'])
    f2 = bool(pc['fault_detected'] >= bc['fault_detected'] - allowance)
    return {'F1': {'pass': f1, 'GBT_events': p0['indep_false_events'], 'M0_events': b0['indep_false_events'],
                   'GBT_per_h': p0['indep_events_per_h'], 'M0_per_h': b0['indep_events_per_h']},
        'F2': {'pass': f2, 'GBT_detected_excluding': pc['fault_detected'], 'M0_detected_excluding': bc['fault_detected'],
               'scored_fault_cases': n, 'scored_fault_cases_excluding': bc['fault_cases'], 'allowance_cases': allowance,
               'minimum_detected': bc['fault_detected'] - allowance, 'per_class_deficits': deficits,
               'flagged_classes_deficit_gt2': [c for c, x in deficits.items() if x > 2]},
        'verdict': 'CONFIRMED' if f1 and f2 else 'NOT_CONFIRMED'}


def latency_metrics(summaries, data):
    table = {}
    for cls in CLASSES:
        b = summaries['M0']['per_class_all'][cls]
        p = summaries['GBT']['per_class_all'][cls]
        delta = p['confirm_median_s'] - b['confirm_median_s'] if p['confirm_median_s'] is not None and b['confirm_median_s'] is not None else None
        table[cls] = {'M0': b, 'GBT': p, 'median_delta_s': delta, 'flag_slower_gt2s': delta > 2 if delta is not None else None}
    return {'F4_population': 'all scored fault cases, including excluded cases; per-arm detected cases for marginal latency',
            'F4': table, 'paired_all_common_detected': events.paired_latency(data, 'GBT', GBT_EVENT, exclude=False)}


def window_fp_bootstrap(frame, y, predictions, metadata, replicates=2000, seed=20261106):
    cases = metadata.sort_values('case_uid').set_index('case_uid')
    unique, inverse = np.unique(cases.association_group.to_numpy(), return_inverse=True)
    positions = pd.Index(cases.index).get_indexer(frame.case_uid)
    if (positions < 0).any():
        raise ValueError('Window outside registered cases')
    values = {}
    for arm, pred in predictions.items():
        counts = np.zeros(len(cases), dtype=np.int64)
        np.add.at(counts, positions[(y == 0) & pred], 1)
        values[arm] = counts
    rng = np.random.default_rng(seed)
    sample = []
    for _ in range(replicates):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = count[inverse]
        b = int(np.dot(weight, values['M0']))
        p = int(np.dot(weight, values['GBT']))
        sample.append(p / b if b else None)
    undefined = sum(x is None for x in sample)
    return {'replicates': replicates, 'rng_seed': seed, 'groups': len(unique),
            'population': 'all 553 registered test case association_groups, only normal windows contribute FP',
            'ratio': int(values['GBT'].sum()) / int(values['M0'].sum()) if values['M0'].sum() else None,
            'ci95': np.percentile(sample, [2.5, 97.5]).tolist() if not undefined else None,
            'undefined_replicates': undefined, 'no_pseudocount_drop_redraw': True}


def run_once():
    start = time.perf_counter()
    info, metadata, ledger, saved, mu, sd = startup()
    OUT.mkdir(parents=True, exist_ok=False)
    SCORES.mkdir(parents=True, exist_ok=True)
    input_boundary(metadata)
    stage = 'STARTUP_PASS'
    with (OUT / 'run.log').open('x', encoding='utf-8', newline='\n') as journal:
        def note(*args):
            print(*args, flush=True)
            print(*args, file=journal, flush=True)
        note('STARTUP_PASS', json.dumps(info, ensure_ascii=False))
        save(OUT / 'startup_checks.json', info)
        try:
            stage = 'FRAME_SCORING'
            score_receipt = score_cases(metadata, info, mu, sd, note)
            save(OUT / 'scoring_receipt.json', score_receipt)
            stage = 'PARITY_GATE'
            parity = parity_gate(ledger, saved, SCORES)
            save(OUT / 'parity_gate.json', parity)
            note('PARITY_GATE', json.dumps(parity))
            if not parity['pass']:
                receipt = {'status': 'STOPPED_PARITY_GATE', 'startup': info, 'scoring': score_receipt,
                           'parity_gate': parity, 'metrics_computed': False, 'lock_written': False}
                save(OUT / 'receipt.json', receipt)
                save(OUT / 'results.json', {'status': 'STOPPED_PARITY_GATE', 'parity_gate': parity, 'metrics_computed': False})
                return receipt
            if sha(CFG) != info['config_sha256'] or sha(SAVED) != info['saved_predictions_sha256']:
                raise RuntimeError('Frozen inputs changed during scoring: STOP')
            for model in info['models'].values():
                if sha(model['path']) != model['sha256']:
                    raise RuntimeError('Model changed during scoring: STOP')
            if sha(OLD_LOCK) != info['historical_lock']['sha256'] or OLD_LOCK.stat().st_mtime_ns != info['historical_lock']['mtime_ns']:
                raise RuntimeError('Historical lock changed')
            stage = 'WRITE_NEW_LOCK'
            lock = write_lock(OUT / 'TEST_EVALUATED.lock', info, parity)
            note('NEW_LOCK_WRITTEN_BEFORE_METRICS', lock['time'])
            stage = 'EVENT_METRICS'
            # Only these TWO event points. No ENS or threshold/onset sweep.
            rows = []
            for i, uid in enumerate(metadata.case_uid, 1):
                rows.extend(events.case_records((uid, [('GBT', (GBT_EVENT,)), ('M0', (M0_EVENT,))], SCORES)))
                if i % 100 == 0 or i == len(metadata):
                    note('EVENT_CASES', i, '/', len(metadata))
            data = pd.DataFrame(rows).merge(metadata, on='case_uid', validate='many_to_one')
            data.to_csv(OUT / 'events_per_case.csv', index=False)
            summaries = {a: events.summarize_selected(data, a, th) for a, th in [('GBT', GBT_EVENT), ('M0', M0_EVENT)]}
            primary = event_primary(summaries)
            latency = latency_metrics(summaries, data)
            by_domain = {domain: {a: events.summarize_selected(data[data.domain.eq(domain)], a, th)
                         for a, th in [('GBT', GBT_EVENT), ('M0', M0_EVENT)]} for domain in ('SIL', 'HIL')}
            event_result = {'summaries': summaries, 'primary': primary, 'latency': latency, 'by_domain': by_domain}
            save(OUT / 'event_metrics.json', event_result)
            stage = 'WINDOW_METRICS'
            windows, predictions, frame = window_metrics(ledger, saved, metadata)
            save(OUT / 'window_metrics.json', windows)
            stage = 'EVENT_BOOTSTRAP'
            # Direct reuse. M0-tuned alias equals the same registered baseline,
            # so the helper's second identical comparison is not a new point.
            eb = events.bootstrap_rates(data, {'GBT': GBT_EVENT, 'M0': M0_EVENT}, reps=2000, seed=20261106)
            event_ci = {'replicates': eb['replicates'], 'rng_seed': eb['rng_seed'], 'groups': eb['groups'],
                        'population': eb['population'].replace('train', 'test'),
                        **eb['comparisons']['GBT_over_M0_0.60'], 'onset_reselected': False}
            b0, p0 = summaries['M0']['all_cases'], summaries['GBT']['all_cases']
            event_ci['ratio'] = p0['indep_events_per_h'] / b0['indep_events_per_h'] if b0['indep_events_per_h'] else None
            save(OUT / 'event_bootstrap.json', event_ci)
            stage = 'WINDOW_BOOTSTRAP'
            window_ci = window_fp_bootstrap(frame, saved['labels'], predictions, metadata)
            save(OUT / 'window_bootstrap.json', window_ci)
            if event_ci['groups'] != window_ci['groups']:
                raise RuntimeError('Case/window group plans are not aligned')
            results = {'status': 'COMPLETE', 'protocol': str(CFG.relative_to(ROOT)), 'config_sha256': info['config_sha256'],
                'test_use': 'SECOND_AND_FINAL', 'parity_gate': parity, 'operating_points': lock['operating_points'], 'policy': POLICY,
                'event': event_result, 'window': windows, 'bootstrap': {'event_false_rate_ratio': event_ci, 'window_FP_ratio': window_ci,
                    'shared_plan': 'same sorted case/group ordering, same RNG seed/count; fixed operating points, no reselection'},
                'no_training_no_refit_no_sweep': True, 'historical_test_exposure_disclosed': True,
                'wall_time_s': time.perf_counter() - start, 'test_permanently_closed': True}
            save(OUT / 'results.json', results)
            receipt = {'status': 'COMPLETE', 'startup': info, 'scoring': score_receipt, 'parity_gate': parity,
                'new_lock_sha256': sha(OUT / 'TEST_EVALUATED.lock'), 'metrics_computed_once': True,
                'metric_stages_saved': ['event_metrics.json', 'window_metrics.json', 'event_bootstrap.json', 'window_bootstrap.json'],
                'results_sha256': sha(OUT / 'results.json'), 'time': datetime.now(timezone.utc).isoformat(),
                'test_permanently_closed': True, 'wall_time_s': time.perf_counter() - start}
            save(OUT / 'receipt.json', receipt)
            note('COMPLETE', primary['verdict'])
            return receipt
        except Exception as exc:
            note('FAILED_STOP_NO_RERUN', stage, repr(exc))
            if not (OUT / 'receipt.json').exists():
                save(OUT / 'receipt.json', {'status': 'FAILED_STOP_NO_RERUN', 'failed_stage': stage, 'error': repr(exc),
                    'startup': info, 'lock_exists': (OUT / 'TEST_EVALUATED.lock').exists(), 'no_restart_permitted': True})
            raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    value = run_once()
    if value['status'] != 'COMPLETE':
        raise SystemExit(2)
