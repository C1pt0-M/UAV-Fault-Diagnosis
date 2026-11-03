"""Read-only paired M3 error analysis from saved source-Val probabilities.

No torch, model forward, training, feature fitting, replay or threshold search.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ['C0', 'CO', 'CN', 'CON', 'CO9']
PAIRS = [('C0', 'CO'), ('C0', 'CN'), ('C0', 'CON'), ('C0', 'CO9'),
         ('CN', 'CON'), ('CO', 'CO9'), ('CO', 'CON')]
EVENTS = ['FN_to_TP', 'TP_to_FN', 'FP_to_TN', 'TN_to_FP',
          'both_FN', 'both_TP', 'both_FP', 'both_TN']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def load_predictions(path, ids, labels):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'probabilities', 'labels', 'original_indices'}:
            raise ValueError('Unexpected prediction schema: ' + str(path))
        q, y, actual_ids = z['probabilities'], z['labels'], z['original_indices']
    if not np.array_equal(actual_ids, ids) or not np.array_equal(y, labels):
        raise ValueError('ID/label/order mismatch: ' + str(path))
    if q.shape != (len(ids), 2) or not np.isfinite(q).all():
        raise ValueError('Invalid probability shape or finite check: ' + str(path))
    if np.any((q < 0) | (q > 1)) or not np.allclose(q.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError('Invalid probability range/sum: ' + str(path))
    # Frozen two-class argmax: an exact tie is normal. No threshold search.
    return q[:, 1].astype(np.float64), q.argmax(1)


def transitions(y, ref, candidate):
    y, ref, candidate = [np.asarray(v) for v in (y, ref, candidate)]
    if any(v.shape != y.shape or not np.isin(v, [0, 1]).all() for v in (y, ref, candidate)):
        raise ValueError('Expected matching binary labels/predictions')
    event = np.empty(len(y), dtype=object)
    event[(y == 1) & (ref == 0) & (candidate == 1)] = 'FN_to_TP'
    event[(y == 1) & (ref == 1) & (candidate == 0)] = 'TP_to_FN'
    event[(y == 0) & (ref == 1) & (candidate == 0)] = 'FP_to_TN'
    event[(y == 0) & (ref == 0) & (candidate == 1)] = 'TN_to_FP'
    event[(y == 1) & (ref == 0) & (candidate == 0)] = 'both_FN'
    event[(y == 1) & (ref == 1) & (candidate == 1)] = 'both_TP'
    event[(y == 0) & (ref == 1) & (candidate == 1)] = 'both_FP'
    event[(y == 0) & (ref == 0) & (candidate == 0)] = 'both_TN'
    return event


def counts(frame):
    c = frame.event.value_counts().reindex(EVENTS, fill_value=0)
    result = {name: int(c[name]) for name in EVENTS}
    result.update(windows=len(frame), cases=int(frame.case_uid.nunique()),
                  association_groups=int(frame.association_group.nunique()))
    result['net_fault_TP_delta'] = int(c.FN_to_TP - c.TP_to_FN)
    result['net_normal_FP_delta'] = int(c.TN_to_FP - c.FP_to_TN)
    for event, denominator in [('FN_to_TP', c.FN_to_TP + c.both_FN),
                               ('TP_to_FN', c.TP_to_FN + c.both_TP),
                               ('FP_to_TN', c.FP_to_TN + c.both_FP),
                               ('TN_to_FP', c.TN_to_FP + c.both_TN)]:
        result[event + '_reference_risk_windows'] = int(denominator)
        result[event + '_fraction_of_reference_risk'] = float(c[event] / denominator) if denominator else None
    return result


def probability_summary(frame):
    row = dict(windows=len(frame), cases=int(frame.case_uid.nunique()),
               association_groups=int(frame.association_group.nunique()))
    for name in ['reference_probability', 'candidate_probability', 'delta_probability']:
        values = frame[name].to_numpy()
        for q, suffix in [(.1, 'p10'), (.5, 'median'), (.9, 'p90')]:
            row[name + '_' + suffix] = float(np.quantile(values, q))
    row['score_down_windows'] = int(frame.delta_probability.lt(0).sum())
    row['score_up_windows'] = int(frame.delta_probability.gt(0).sum())
    row['reference_0p4_to_0p6_windows'] = int(frame.reference_probability.between(.4, .6).sum())
    row['reference_ge_0p6_windows'] = int(frame.reference_probability.ge(.6).sum())
    row['reference_ge_0p8_windows'] = int(frame.reference_probability.ge(.8).sum())
    return row


def linked_patterns(windows):
    c0, cn, con = [windows['pred_' + v].to_numpy() for v in ['C0', 'CN', 'CON']]
    fault = windows.label.to_numpy() == 1
    return {
        'CN_lost_C0_TP': fault & (c0 == 1) & (cn == 0),
        'CN_lost_C0_TP_recovered_CON': fault & (c0 == 1) & (cn == 0) & (con == 1),
        'CN_lost_C0_TP_still_FN_CON': fault & (c0 == 1) & (cn == 0) & (con == 0),
        'CN_recovered_C0_FN_lost_CON': fault & (c0 == 0) & (cn == 1) & (con == 0),
        'both_C0_CN_FN_recovered_CON': fault & (c0 == 0) & (cn == 0) & (con == 1),
        'both_C0_CN_TP_lost_CON': fault & (c0 == 1) & (cn == 1) & (con == 0),
        'CN_corrected_C0_FP': ~fault & (c0 == 1) & (cn == 0),
        'CN_corrected_C0_FP_undone_CON': ~fault & (c0 == 1) & (cn == 0) & (con == 1),
        'both_C0_CN_TN_new_FP_CON': ~fault & (c0 == 0) & (cn == 0) & (con == 1),
        'CN_new_FP_corrected_CON': ~fault & (c0 == 0) & (cn == 1) & (con == 0),
        'both_C0_CN_FP_corrected_CON': ~fault & (c0 == 1) & (cn == 1) & (con == 0)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected', type=Path, default=ROOT / 'artifacts/M3_results_v1_20261103/collected')
    parser.add_argument('--metadata', type=Path, default=Path('D:/UAV-Data/causal_m1_continuous_v1/ledgers/val_included_windows.csv'))
    parser.add_argument('--base-ledger', type=Path, default=Path('D:/UAV-Data/causal_input_v1/ledgers/val_included_windows.csv'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError('Do not overwrite an existing analysis: ' + str(a.output))
    preflight_path = ROOT / 'configs/experiments/M3_preflight_v1.json'
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if sha(preflight_path) != preflight_path.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Frozen protocol hash mismatch')
    expected = preflight['data']['source_split_identities']['val']
    formal = json.loads((ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json').read_text())
    if sha(a.base_ledger) != formal['data_identities']['ledgers/val_included_windows.csv']:
        raise ValueError('Frozen base ledger hash mismatch')
    source = json.loads((a.collected / 'source_manifest.json').read_text())
    identities = {str(p): sha(p) for p in [a.metadata, a.base_ledger, preflight_path]}
    w = pd.read_csv(a.metadata, keep_default_na=False, dtype={'association_group': str})
    base = pd.read_csv(a.base_ledger, keep_default_na=False, dtype={'association_group': str})
    ids, labels = w.original_cache_index.to_numpy(dtype=np.int64), w.qualified_y.to_numpy(dtype=np.int64)
    if len(w) != 32381 or len(np.unique(ids)) != 32381 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid common validation population')
    if digest(ids) != expected['original_ids_sha256'] or digest(labels) != expected['labels_sha256']:
        raise ValueError('Frozen Val ID/label/order digest mismatch')
    if not w.split.eq('val').all() or not w.domain.isin(['SIL', 'HIL']).all():
        raise ValueError('Source Val only')
    selected = base.iloc[w.compact_index.to_numpy()].reset_index(drop=True)
    if not selected.equals(w[selected.columns].reset_index(drop=True)):
        raise ValueError('M1 metadata differs from frozen Base26 metadata')
    time_hash = hashlib.sha256(w[['original_cache_index', 'window_start_row', 'window_end_row']].to_csv(index=False).encode()).hexdigest()
    if time_hash != expected['window_id_time_rows_sha256']:
        raise ValueError('Frozen window time/ID rows differ')
    if not w.residual_valid_positions.eq(150).all() or not (w.window_end_row - w.window_start_row).eq(150).all():
        raise ValueError('Not the frozen full-150 common domain')
    if w.case_uid.nunique() != expected['metadata_case_count']:
        raise ValueError('Frozen case population mismatch')
    w['label'] = labels
    w['flight_status'] = w.case_uid.str.split('/').str[1]
    w['fault_directory'] = w.case_uid.str.split('/').str[2]
    w['fault_type'] = np.where(labels == 1, w.fault_directory, 'known_normal')
    w['crop_end_seconds'] = (w.window_end_row - 1) * .02
    w['hover_segment'] = np.where(w.flight_status.ne('hover'), 'not_hover',
                                  np.where(w.crop_end_seconds.le(30), 'first30s', 'after30s'))
    for v in VARIANTS:
        path = a.collected / 'runs' / v / 'best_validation_predictions.npz'
        expected_file = source['files']['runs/' + v + '/best_validation_predictions.npz']
        if sha(path) != expected_file['sha256'] or path.stat().st_size != expected_file['bytes']:
            raise ValueError('Original saved prediction hash differs: ' + v)
        identities[str(path)] = sha(path)
        w['p_' + v], w['pred_' + v] = load_predictions(path, ids, labels)
        report = json.loads((a.collected / 'runs' / v / 'report.json').read_text())
        cm = np.bincount(labels * 2 + w['pred_' + v].to_numpy(), minlength=4).reshape(2, 2)
        if cm.tolist() != report['source_validation']['overall']['confusion_matrix']:
            raise ValueError('Saved prediction confusion differs: ' + v)
    groups = [['fault_type'], ['domain'], ['flight_status'], ['phase'], ['normal_origin'],
              ['hover_segment'], ['fault_type', 'domain'], ['fault_type', 'flight_status'],
              ['domain', 'flight_status']]
    totals, grouped, probability, cases, changed = [], [], [], [], []
    for ref, cand in PAIRS:
        frame = w.copy()
        frame['reference'], frame['candidate'] = ref, cand
        frame['reference_probability'], frame['candidate_probability'] = frame['p_' + ref], frame['p_' + cand]
        frame['delta_probability'] = frame.candidate_probability - frame.reference_probability
        frame['event'] = transitions(labels, w['pred_' + ref], w['pred_' + cand])
        totals.append(dict(reference=ref, candidate=cand, **counts(frame)))
        changed.append(frame[frame.event.isin(EVENTS[:4])])
        for keys in groups:
            for _, subset in frame.groupby(keys, sort=True):
                grouped.append(dict(reference=ref, candidate=cand, grouping='+'.join(keys),
                                    group_value='|'.join(str(subset[k].iloc[0]) for k in keys), **counts(subset)))
        for name, subset in [('all_fault', frame[frame.label.eq(1)]), ('all_normal', frame[frame.label.eq(0)])]:
            probability.append(dict(reference=ref, candidate=cand, event=name, **probability_summary(subset)))
        for event, subset in frame.groupby('event', sort=True):
            probability.append(dict(reference=ref, candidate=cand, event=event, **probability_summary(subset)))
        for uid, subset in frame.groupby('case_uid', sort=True):
            cases.append(dict(reference=ref, candidate=cand, case_uid=uid,
                              domain=subset.domain.iloc[0], flight_status=subset.flight_status.iloc[0],
                              fault_directory=subset.fault_directory.iloc[0], **counts(subset)))
    changes = pd.concat(changed, ignore_index=True)
    concentration = []
    examples = []
    # Transparent descriptive example rule: top contributing cases per event,
    # closest probability delta to that case/event median; original-ID tie break.
    for (ref, cand, event), subset in changes.groupby(['reference', 'candidate', 'event'], sort=True):
        per_case = subset.groupby('case_uid').size().sort_values(ascending=False, kind='stable')
        concentration.append(dict(reference=ref, candidate=cand, event=event, windows=len(subset),
                                  cases=len(per_case), association_groups=int(subset.association_group.nunique()),
                                  top1_case_windows=int(per_case.iloc[0]), top5_case_windows=int(per_case.head(5).sum()),
                                  top1_case_fraction=float(per_case.iloc[0] / len(subset)),
                                  top5_case_fraction=float(per_case.head(5).sum() / len(subset))))
        for rank, uid in enumerate(per_case.head(2).index, 1):
            f = subset[subset.case_uid.eq(uid)].copy()
            f['median_delta_distance'] = (f.delta_probability - f.delta_probability.median()).abs()
            row = f.sort_values(['median_delta_distance', 'original_cache_index']).iloc[0].to_dict()
            row.update(example_rule='top case count, median delta, original-ID tie break',
                       case_rank_for_event=rank, case_event_windows=len(f))
            examples.append(row)
    linked = []
    for pattern, mask in linked_patterns(w).items():
        sub = w[mask]
        linked.append(dict(pattern=pattern, grouping='overall', group_value='all', windows=len(sub),
                           cases=int(sub.case_uid.nunique())))
        for keys in [['fault_type'], ['domain'], ['flight_status']]:
            for _, f in sub.groupby(keys, sort=True):
                linked.append(dict(pattern=pattern, grouping='+'.join(keys),
                                   group_value='|'.join(str(f[k].iloc[0]) for k in keys),
                                   windows=len(f), cases=int(f.case_uid.nunique())))
    a.output.mkdir(parents=True, exist_ok=False)
    tables = {'window_comparison.csv': w, 'changed_windows.csv': changes,
              'transition_totals.csv': pd.DataFrame(totals), 'transition_groups.csv': pd.DataFrame(grouped),
              'probability_changes.csv': pd.DataFrame(probability), 'case_concentration.csv': pd.DataFrame(cases),
              'event_concentration.csv': pd.DataFrame(concentration), 'typical_examples.csv': pd.DataFrame(examples),
              'CN_CON_linked_patterns.csv': pd.DataFrame(linked)}
    for name, table in tables.items():
        table.to_csv(a.output / name, index=False)
    if any(sha(Path(p)) != h for p, h in identities.items()):
        raise ValueError('Input bytes changed during read-only analysis')
    receipt = {'status': 'PASS_M3_TARGETED_SAVED_PREDICTION_ERROR_ANALYSIS',
               'created_at': datetime.now(timezone.utc).isoformat(), 'validation_windows': len(w),
               'normal_windows': int((labels == 0).sum()), 'fault_windows': int((labels == 1).sum()),
               'cases': int(w.case_uid.nunique()), 'models': VARIANTS,
               'original_ID_order_sha256': digest(ids), 'label_order_sha256': digest(labels),
               'window_time_rows_sha256': time_hash, 'input_sha256': identities,
               'outputs': {name: {'sha256': sha(a.output / name), 'rows': len(t)} for name, t in tables.items()},
               'missing_metadata': ['Per-window observed physical flight intention/state; only existing directory scenario, phase and crop-relative time used.',
                                    'Adapter margins/activations and per-feature effects are not in saved predictions; no mechanism or single-feature causal attribution.'],
               'no_training_model_forward_replay_threshold_search_Test_Real': True,
               'original_inputs_unchanged': True, 'probability_summary_is_not_causal_or_calibration_evidence': True,
               'example_rule': 'Two highest-count cases per pair/event; closest to within-case median probability delta, original-ID tie break. Descriptive examples only; all changes retained.'}
    (a.output / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'totals': totals}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
