"""Frozen V criteria on logits. Previously used validation split, NOT a test.

All class comparisons use paired integer TP counts before the final pp division
so the registered 1/2 pp boundaries are not changed by subtracting rounded recalls.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_v_confirm as train
from train_v_confirm import s, t, m
import evaluate_t_selection as te
import numpy as np
import torch

ARMS = ('V0', 'V1')
CRITERIA = ('C1_false_alarms', 'C2_matched_fp_class_recall', 'C3_r97_class_cap', 'C4_ap')


def valid_interval(value):
    return value is not None and len(value) == 2 and all(v is not None and math.isfinite(v) for v in value) and value[0] <= value[1]


def criteria(ratios, ratio_ci, matched_deltas_pp, r97_drops_pp, ap_ci):
    point = m.geometric_mean(ratios) if len(ratios) == 5 else None
    c1 = (point is not None and point < 1 and all(v is not None for v in ratios)
          and sum(v < 1 for v in ratios) >= 4 and valid_interval(ratio_ci) and ratio_ci[1] < 1)
    c2 = bool(matched_deltas_pp) and all(v is not None and math.isfinite(v) and v >= -1 for v in matched_deltas_pp.values())
    c3 = bool(r97_drops_pp) and all(v is not None and math.isfinite(v) and v <= 2 for v in r97_drops_pp.values())
    c4 = valid_interval(ap_ci) and ap_ci[1] >= 0
    checks = dict(zip(CRITERIA, map(bool, (c1, c2, c3, c4))))
    failed = [name for name, passed in checks.items() if not passed]
    return {'verdict': 'NOT_CONFIRMED_KEEP_T0' if failed else 'CONFIRMED_ADOPT_FIXED_100',
            'criteria': checks, 'failed_criteria': failed}


def matched_fp_threshold(y, score, k, weights=None, *, normal_order=None):
    y, score = np.asarray(y), np.asarray(score)
    w = np.ones(len(y), dtype=np.int64) if weights is None else np.asarray(weights)
    if (y.ndim != 1 or y.shape != score.shape or w.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all()
            or ((score < 0) | (score > 1)).any() or (w < 0).any()
            or not np.issubdtype(w.dtype, np.integer)):
        raise ValueError('Matched-FP vectors/weights are invalid')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        return {'threshold': None, 'target_fp': k, 'realised_fp': None, 'reason': 'k_zero_or_undefined'}
    order = np.argsort(-score, kind='stable') if normal_order is None else normal_order
    order = order[y[order] == 0]
    cumulative = np.cumsum(w[order], dtype=np.int64)
    if not len(cumulative) or k > cumulative[-1]:
        return {'threshold': None, 'target_fp': k, 'realised_fp': None, 'reason': 'normal_mass_unavailable'}
    index = int(np.searchsorted(cumulative, k, side='left'))
    threshold = float(score[order[index]])
    return {'threshold': threshold, 'target_fp': int(k),
            'realised_fp': int(w[(y == 0) & (score >= threshold)].sum())}


def prepare(frame, scores):
    y, _, classes, _, late = m.strata(frame)
    if set(scores) != set(ARMS) or any(set(scores[a]) != set(t.SEEDS) for a in ARMS):
        raise ValueError('Exactly V0/V1 and five registered seeds required')
    cache = {(a, seed): m.MetricCache(y, scores[a][seed]) for a in ARMS for seed in t.SEEDS}
    class_masks = {c: (y == 1) & ~late & (classes == c) for c in sorted(set(classes[y == 1]))}
    ensemble = te.ensemble([scores['V1'][seed] for seed in t.SEEDS])
    return {'y': y, 'scores': scores, 'cache': cache, 'class_masks': class_masks,
            'ensemble': m.MetricCache(y, ensemble)}


def measure(prepared, weights=None):
    y = prepared['y']
    w = np.ones(len(y), dtype=np.int64) if weights is None else weights
    basic = {key: cache.basic(w) for key, cache in prepared['cache'].items()}
    ratios, aps, matched = [], [], {}
    for seed in t.SEEDS:
        a, b = basic[('V0', seed)], basic[('V1', seed)]
        ratios.append(m.ratio(b['fp'], a['fp']))
        aps.append(b['ap'] - a['ap'] if a['ap'] is not None and b['ap'] is not None else None)
        cache = prepared['cache'][('V1', seed)]
        matched[seed] = matched_fp_threshold(y, cache.score, a['fp'], w, normal_order=cache.order)
    deltas, drops = {}, {}
    for cls, mask in prepared['class_masks'].items():
        mass = int(w[mask].sum())
        delta_counts, drop_counts = [], []
        for seed in t.SEEDS:
            zero, one = basic[('V0', seed)]['threshold'], basic[('V1', seed)]['threshold']
            tm = matched[seed]['threshold']
            p0, p1 = prepared['scores']['V0'][seed], prepared['scores']['V1'][seed]
            tp0 = int(w[mask & (p0 >= zero)].sum()) if zero is not None else None
            tp1 = int(w[mask & (p1 >= one)].sum()) if one is not None else None
            tpm = int(w[mask & (p1 >= tm)].sum()) if tm is not None else None
            delta_counts.append(tpm - tp0 if tpm is not None and tp0 is not None else None)
            drop_counts.append(tp0 - tp1 if tp0 is not None and tp1 is not None else None)
        deltas[cls] = 100 * sum(delta_counts) / (5 * mass) if mass and all(v is not None for v in delta_counts) else None
        drops[cls] = 100 * sum(drop_counts) / (5 * mass) if mass and all(v is not None for v in drop_counts) else None
    en = prepared['ensemble'].basic(w)
    return {'per_seed_ratios': dict(zip(map(str, t.SEEDS), ratios)), 'aggregate_ratio': m.geometric_mean(ratios),
        'per_seed_delta_ap': dict(zip(map(str, t.SEEDS), aps)),
        'aggregate_delta_ap': float(np.mean(aps)) if all(ap is not None for ap in aps) else None,
        'C2_seed_mean_recall_delta_pp': deltas, 'C3_seed_mean_recall_drop_pp': drops,
        'matched_fp': {str(seed): value for seed, value in matched.items()},
        'V3_fp_r97': en['fp'], 'V3_ratio_to_geometric_mean_V1': te.ensemble_ratio(en['fp'], [basic[('V1', sd)]['fp'] for sd in t.SEEDS])}


def paired_bootstrap(prepared, groups, replicates=2000, seed=20261106):
    groups = np.asarray(groups)
    if len(groups) != len(prepared['y']) or not len(groups) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('Bootstrap groups must be aligned nonempty strings')
    if type(replicates) is not int or replicates < 1:
        raise ValueError('Positive replicate count required')
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    ratios, aps, en_ratios = [], [], []
    per_seed_ratios = {str(sd): [] for sd in t.SEEDS}
    per_seed_aps = {str(sd): [] for sd in t.SEEDS}
    class_delta = {name: [] for name in prepared['class_masks']}
    plan_hash = hashlib.sha256()
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        plan_hash.update(counts.astype('<i8').tobytes())
        value = measure(prepared, counts[inverse])
        ratios.append(value['aggregate_ratio'])
        aps.append(value['aggregate_delta_ap'])
        en_ratios.append(value['V3_ratio_to_geometric_mean_V1'])
        for sd in map(str, t.SEEDS):
            per_seed_ratios[sd].append(value['per_seed_ratios'][sd])
            per_seed_aps[sd].append(value['per_seed_delta_ap'][sd])
        for name in class_delta:
            class_delta[name].append(value['C2_seed_mean_recall_delta_pp'][name])
    return {'replicates': replicates, 'rng_seed': seed, 'groups': len(unique),
        'shared_resample_plan_sha256': plan_hash.hexdigest(), 'paired_across_all_arms_and_seeds': True,
        'aggregate_ratio_ci': m.percentile(ratios), 'aggregate_delta_ap_ci': m.percentile(aps),
        'per_seed_ratio_ci': {sd: m.percentile(values) for sd, values in per_seed_ratios.items()},
        'per_seed_delta_ap_ci': {sd: m.percentile(values) for sd, values in per_seed_aps.items()},
        'C2_class_delta_ci_pp': {name: m.percentile(values) for name, values in class_delta.items()},
        'undefined_ratio_replicates': sum(v is None for v in ratios),
        'undefined_ap_replicates': sum(v is None for v in aps),
        'C2_undefined_class_replicates': {name: sum(v is None for v in values) for name, values in class_delta.items()},
        'V3_ratio_ci': m.percentile(en_ratios), 'V3_undefined_replicates': sum(v is None for v in en_ratios),
        'undefined_policy': 'No pseudocounts, redraws, or dropped replicates; C2 CI is descriptive, its point estimate decides.'}


def metrics_from_logits(frame, logits, device='cpu'):
    score = train.logits_to_probability(logits, device)
    result = te.score_metrics(frame, score)
    labels = torch.as_tensor(frame.qualified_y.to_numpy(dtype=np.int64), device=device)
    result['class_balanced_ce'] = float(s.class_balanced_ce(torch.as_tensor(logits, device=device), labels).cpu())
    result['ce_source'] = 'float64 cross_entropy from raw float32 logits; no probability clipping'
    return result


def load_logits(path, frame, epoch):
    with np.load(path, allow_pickle=False) as data:
        logits = data['logits']
        if logits.dtype != np.float32 or logits.shape != (len(frame), 2) or not np.isfinite(logits).all():
            raise ValueError('Invalid finite float32 val logits [n,2]')
        if data['original_cache_index'].dtype != np.int64 or not np.array_equal(data['original_cache_index'], frame.original_cache_index.to_numpy()):
            raise ValueError('Val window IDs/order mismatch')
        if data['labels'].dtype != np.int64 or not np.array_equal(data['labels'], frame.qualified_y.to_numpy()):
            raise ValueError('Val labels/order mismatch')
        if not np.array_equal(data['epochs'], [epoch]) or str(data['population_scope']) != 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION':
            raise ValueError('Wrong epoch or STAND_IN cannot enter formal V evaluation')
    return logits


def load_runs(root, frame, origin_sha, device='cuda:0'):
    scores, logits = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    reports, selected = {}, {}
    manifests = set()
    manifest = train.implementation_manifest()
    for seed in t.SEEDS:
        folder = Path(root) / f'seed{seed}'
        report = s.read_json(folder / 'report.json')
        if (report.get('status') != 'COMPLETE' or report.get('cpu_smoke') or report.get('phase') != 'val'
                or report.get('prediction_scope') != 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION'
                or report.get('seed') != seed or report.get('epochs') != 100 or len(report.get('history', [])) != 100
                or report.get('protocol_sha256') != train.PROTOCOL_SHA):
            raise ValueError('Only complete formal 100-epoch V runs, never STAND_IN')
        t.require_formal_environment(report['environment'])
        if report['implementation_files'] != manifest['files'] or report['initial_digest'] != t.golden_digests()[('mstca', seed)]:
            raise ValueError('V code/golden mismatch')
        expected = s.read_json(s.BASELINE)['data_identities']
        if report['data_identities']['matched_sha256'] != expected or report['data_identities']['skipped_sha256']:
            raise ValueError('Formal run must verify nine data identities')
        if report['origin']['sha256'] != origin_sha:
            raise ValueError('V origin SHA mismatch')
        manifests.add(report['implementation_manifest_sha256'])
        inner_path = folder / 'inner_logits.npz'
        if s.sha(inner_path) != report['inner_logits_sha256']:
            raise ValueError('V inner logits SHA mismatch')
        with np.load(inner_path, allow_pickle=False) as inner:
            n_inner = report['used_window_counts']['inner']
            population = s.read_json(folder / 'population.json')['inner']
            if (inner['logits'].dtype != np.float32 or inner['logits'].shape != (100, n_inner, 2)
                    or not np.isfinite(inner['logits']).all() or not np.array_equal(inner['epochs'], np.arange(1, 101))
                    or inner['original_cache_index'].dtype != np.int64
                    or not np.array_equal(inner['original_cache_index'], population)
                    or inner['labels'].dtype != np.int64 or inner['labels'].shape != (n_inner,)
                    or not np.isin(inner['labels'], [0, 1]).all()
                    or str(inner['population_scope']) != 'SOURCE_TRAIN_INNER'):
                raise ValueError('V inner logits trajectory/population mismatch')
        choice = train.selections([row['inner_class_balanced_ce'] for row in report['history']])
        if choice != s.read_json(folder / 'selections.json'):
            raise ValueError('V selections differ from inner CE replay')
        epochs = {choice['V0']['epoch'], 100}
        if set(report['prediction_artifacts']) != set(map(str, epochs)) or report['val_prediction_passes'] != len(epochs):
            raise ValueError('Exactly one val pass per distinct selected checkpoint required')
        for epoch in range(1, 101):
            if not (folder / f'checkpoints/epoch_{epoch}.pt').is_file() or report['history'][epoch - 1]['epoch'] != epoch:
                raise ValueError('Missing/unordered V checkpoint trajectory')
            if report['history'][epoch - 1]['state_digest'] != report['checkpoint_state_digests'][str(epoch)]:
                raise ValueError('V checkpoint digest metadata mismatch')
        for arm, epoch in [('V0', choice['V0']['epoch']), ('V1', 100)]:
            entry = report['prediction_artifacts'][str(epoch)]
            expected_name = f'val_logits_epoch_{epoch}.npz'
            if entry['file'] != expected_name or s.sha(folder / expected_name) != entry['sha256']:
                raise ValueError('V logits artifact SHA/name mismatch')
            logits[arm][seed] = load_logits(folder / expected_name, frame, epoch)
            scores[arm][seed] = train.logits_to_probability(logits[arm][seed], device)
        if seed <= 44:
            train.reproduction_gate(seed, report['history'], choice,
                m.fp_at_recall(frame.qualified_y.to_numpy(), scores['V0'][seed])['fp'])
        reports[str(folder)] = report
        selected[str(seed)] = choice
    if len(manifests) != 1:
        raise ValueError('V manifests differ across seeds')
    return scores, logits, reports, selected


def per_seed_fp_ratios(metrics):
    return {str(seed): {'FP_R97_V1_over_V0': m.ratio(metrics['V1'][str(seed)]['primary']['fp'],
                                                metrics['V0'][str(seed)]['primary']['fp']),
                       'FP_R95_V1_over_V0': m.ratio(metrics['V1'][str(seed)]['fp_at_r95']['fp'],
                                                metrics['V0'][str(seed)]['fp_at_r95']['fp'])}
            for seed in t.SEEDS}


def historical_s_comparison(metrics, path=None):
    path = ROOT / 'artifacts/s_ablation_mstca_v1/evaluation_val/evaluation.json' if path is None else Path(path)
    result = s.read_json(path)
    if result.get('status') != 'COMPLETE' or result.get('phase') != 'val' or result.get('protocol_sha256') != s.PROTOCOL_SHA:
        raise ValueError('Invalid historical S phase-2 reference')
    return {'source_sha256': s.sha(path), 'scope': 'DESCRIPTIVE_ONLY_S0_EARLY_STOPPING_BEST_LAST',
        'seeds': {str(seed): {'S_best_fp_r97': result['metrics_best']['S0'][str(seed)]['primary']['fp'],
            'S_last_fp_r97': result['last_epoch_sensitivity']['S0'][str(seed)]['primary']['fp'],
            'V0_fp_r97': metrics['V0'][str(seed)]['primary']['fp'], 'V1_fp_r97': metrics['V1'][str(seed)]['primary']['fp']}
            for seed in (42, 43, 44)}}


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh V evaluation output required')
    protocol, pins = train.load_protocol(args.protocol)
    origin = train.origin_gate(args.t_phase1_evaluation or train.ORIGIN, protocol['origin']['t_phase1_evaluation_sha256'])
    t.require_formal_environment()
    manifest = train.implementation_manifest()
    expected = s.read_json(s.BASELINE)['data_identities']['ledgers/val_included_windows.csv']
    if s.sha(Path(args.data) / 'ledgers/val_included_windows.csv') != expected:
        raise ValueError('V val ledger identity mismatch')
    frame = s.read_ledger(args.data, 'val')
    if len(frame) != 34710 or int(frame.qualified_y.eq(0).sum()) != 17740:
        raise ValueError('Wrong full val population')
    output.mkdir(parents=True, exist_ok=False)
    try:
        scores, logits, reports, selected = load_runs(args.runs, frame, origin['sha256'])
        prepared = prepare(frame, scores)
        point = measure(prepared)
        uncertainty = paired_bootstrap(prepared, frame.association_group.to_numpy())
        verdict = criteria(list(point['per_seed_ratios'].values()), uncertainty['aggregate_ratio_ci'],
            point['C2_seed_mean_recall_delta_pp'], point['C3_seed_mean_recall_drop_pp'], uncertainty['aggregate_delta_ap_ci'])
        metrics = {arm: {str(seed): metrics_from_logits(frame, logits[arm][seed], 'cuda:0') for seed in t.SEEDS} for arm in ARMS}
        stability = {}
        for arm in ARMS:
            fps = [metrics[arm][str(seed)]['primary']['fp'] for seed in t.SEEDS]
            stability[arm] = {f'ddof{ddof}': float(np.std(np.log(fps), ddof=ddof))
                if all(fp is not None and fp > 0 for fp in fps) else None for ddof in (0, 1)}
        result = {'status': 'COMPLETE', 'protocol_sha256': train.PROTOCOL_SHA, 'performance_claim': True,
            'population': len(frame), 'disclosure': protocol['origin']['val_exposure_disclosure'],
            'origin': origin, 'T_verdicts_and_adoption_unchanged': True, 'metrics': metrics,
            'comparison': point, 'uncertainty': uncertainty, **verdict, 'selections': selected,
            'per_seed_fp_ratios': per_seed_fp_ratios(metrics),
            'seed_stability_log_fp_sd': stability,
            'V3_descriptive': {'fp_r97': point['V3_fp_r97'], 'ratio_to_geometric_mean_V1': point['V3_ratio_to_geometric_mean_V1'],
                               'ratio_ci': uncertainty['V3_ratio_ci'], 'mac_cost_multiplier': 5},
            'S_phase2_descriptive': historical_s_comparison(metrics),
            'run_report_sha256': {name: s.sha(Path(name) / 'report.json') for name in reports},
            'resources': [{key: r[key] for key in ('seed', 'epochs', 'wall_time_s', 'param_count', 'macs')} for r in reports.values()]}
        if train.load_protocol(args.protocol)[1] != pins or train.implementation_manifest() != manifest:
            raise ValueError('V code/protocol drift during evaluation')
        train.origin_gate(origin['path'], origin['sha256'])
        s.write_json(output / 'evaluation.json', result)
        s.write_json(output / 'evaluation_sha256.json', {'sha256': s.sha(output / 'evaluation.json')})
        with (output / 'summary.md').open('x', encoding='utf-8') as stream:
            stream.write('# V confirmation on previously used validation, NOT an untouched test\n' + result['verdict'] + '\nFailed: ' + str(result['failed_criteria']) + '\n')
        return result
    except Exception as exc:
        s.write_json(output / 'FAILED.json', {'status': 'FAILED', 'error': repr(exc)})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, default=train.DEFAULT_PROTOCOL)
    result.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    result.add_argument('--runs', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--t-phase1-evaluation', type=Path)
    return result


if __name__ == '__main__':
    print(evaluate(parser().parse_args())['verdict'])
