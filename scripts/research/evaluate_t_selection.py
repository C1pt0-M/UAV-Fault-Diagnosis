"""Paired T-rule evaluation. Oracle curves are descriptive NOT SELECTABLE.

All rules are read out from frozen inner-only selections, never held-out/val.
No local/smoke input can yield a COMPLETE formal verdict or phase-2 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import numpy as np
import train_t_selection as train
from train_t_selection import s
import evaluate_s_ablation as m

ARMS = ('T0', 'T1', 'T2')
COMPARISONS = {'T1_vs_T0': ('T1', 'T0'), 'T2_vs_T0': ('T2', 'T0'), 'T2_vs_T1': ('T2', 'T1')}


def ensemble(probabilities):
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] != 5 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('T3 requires exactly five aligned finite probability vectors')
    return values.astype(np.float64).mean(axis=0)


def ensemble_ratio(ensemble_fp, single_model_fps):
    # Algebraically FP(T3)/GM(FP(T1)). Taking ratios first preserves exact 1
    # for identical models instead of exp(log(integer)) roundoff creating gain.
    return m.geometric_mean([m.ratio(ensemble_fp, fp) for fp in single_model_fps])


def ensemble_delta_ap(ensemble_ap, single_model_aps):
    if ensemble_ap is None or any(ap is None for ap in single_model_aps):
        return None
    # Mean of paired differences equals AP(T3)-mean(AP(T1)), and is exactly
    # zero for identical APs; no tolerance, clipping or pseudocount is applied.
    return float(np.mean([ensemble_ap - ap for ap in single_model_aps]))


def decide(ratios, ratio_ci, ap_ci, class_drops):
    result = {'verdict': 'INCONCLUSIVE', 'reason': None}
    if len(ratios) != 5 or any(v is None or not math.isfinite(v) for v in ratios):
        result['reason'] = 'ratio_undefined'
    elif ratio_ci is None:
        result['reason'] = 'ratio_ci_unestimable'
    elif ap_ci is None:
        result['reason'] = 'ap_ci_unestimable'
    elif not class_drops or any(v is None or not math.isfinite(v) for v in class_drops.values()):
        result['reason'] = 'class_constraint_unestimable'
    else:
        equivalent = 1 / 1.10 <= ratio_ci[0] <= ratio_ci[1] <= 1.10
        consistent = max(sum(r < 1 for r in ratios), sum(r > 1 for r in ratios)) >= 4
        if not consistent:
            if max(class_drops.values()) > 1.:
                result.update(verdict='TRADE_OFF', reason='class_constraint_seed_mixed')
            elif equivalent:
                result.update(verdict='EQUIVALENT', reason='equivalent_seed_mixed')
            else:
                result['reason'] = 'seed_consistency'
        elif max(class_drops.values()) > 1.:
            result.update(verdict='TRADE_OFF', reason='class_constraint')
        elif ratio_ci[1] < 1.:
            result.update(verdict='BETTER', reason='upper_bound_lt_one')
        elif ratio_ci[0] > 1.:
            result.update(verdict='WORSE', reason='lower_bound_gt_one')
        elif equivalent:
            result.update(verdict='EQUIVALENT', reason='inside_equivalence_margin')
        else:
            result['reason'] = 'ratio_ci_crosses_margin'
        if (result['verdict'] == 'BETTER' and ap_ci[1] < 0) or (result['verdict'] == 'WORSE' and ap_ci[0] > 0):
            result.update(pre_ap_verdict=result['verdict'], verdict='INCONCLUSIVE', reason='AP_consistency')
    return result


def adopt(decisions, t2_vs_t1_ci):
    better = [arm for arm in ('T1', 'T2') if decisions[arm]['verdict'] == 'BETTER']
    if len(better) == 2:
        return 'T2' if t2_vs_t1_ci is not None and t2_vs_t1_ci[1] < 1 else 'T1'
    if better:
        return better[0]
    return 'T1' if decisions['T1']['verdict'] == 'EQUIVALENT' else 'T0'


def t3_verdict(ratio_ci, ap_ci):
    # The inherited undefined-policy gate applies before the T3 directional rule.
    if ratio_ci is None or ap_ci is None:
        return {'verdict': 'INCONCLUSIVE', 'reason': 'undefined_interval', 'mac_cost_multiplier': 5}
    return {'verdict': 'ENSEMBLE_HELPS' if ratio_ci[1] < 1 and ap_ci[1] >= 0 else 'NO_CLEAR_GAIN',
            'mac_cost_multiplier': 5}


def summarize_pair(current, reference):
    ratios = [m.ratio(current[str(seed)]['primary']['fp'], reference[str(seed)]['primary']['fp']) for seed in train.SEEDS]
    differences, deltas = {}, {}
    for seed in map(str, train.SEEDS):
        a, b = current[seed], reference[seed]
        differences[seed] = a['primary']['fp'] - b['primary']['fp'] if a['primary']['fp'] is not None and b['primary']['fp'] is not None else None
        deltas[seed] = a['ap'] - b['ap'] if a['ap'] is not None and b['ap'] is not None else None
    return {'per_seed_ratios': dict(zip(map(str, train.SEEDS), ratios)), 'aggregate_ratio': m.geometric_mean(ratios),
            'per_seed_fp_difference': differences, 'per_seed_delta_ap': deltas,
            'aggregate_delta_ap': float(np.mean(list(deltas.values()))) if all(v is not None for v in deltas.values()) else None}


def paired_bootstrap(y, scores, groups, replicates=2000, seed=20261106):
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError('Positive bootstrap replicate count required')
    groups = np.asarray(groups)
    if len(groups) != len(y) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('Bootstrap needs aligned nonempty string group IDs')
    unique, inverse = np.unique(groups, return_inverse=True)
    cache = {(a, sd): m.MetricCache(y, scores[a][sd]) for a in ARMS for sd in train.SEEDS}
    ensemble_cache = m.MetricCache(y, ensemble([scores['T1'][sd] for sd in train.SEEDS]))
    names = tuple(COMPARISONS) + ('T3_vs_T1',)
    samples = {name: {'ratios': [], 'deltas': [],
        'per_seed_ratios': {str(sd): [] for sd in train.SEEDS},
        'per_seed_deltas': {str(sd): [] for sd in train.SEEDS}} for name in names}
    rng = np.random.default_rng(seed)
    plan_hash = hashlib.sha256()
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        plan_hash.update(counts.astype('<i8').tobytes())
        weights = counts[inverse]
        metrics = {key: item.basic(weights) for key, item in cache.items()}
        en = ensemble_cache.basic(weights)
        for name, (a, b) in COMPARISONS.items():
            ratios, deltas = [], []
            for sd in train.SEEDS:
                x, z = metrics[(a, sd)], metrics[(b, sd)]
                r = m.ratio(x['fp'], z['fp'])
                d = x['ap'] - z['ap'] if x['ap'] is not None and z['ap'] is not None else None
                samples[name]['per_seed_ratios'][str(sd)].append(r)
                samples[name]['per_seed_deltas'][str(sd)].append(d)
                ratios.append(r)
                deltas.append(d)
            samples[name]['ratios'].append(m.geometric_mean(ratios))
            samples[name]['deltas'].append(float(np.mean(deltas)) if all(d is not None for d in deltas) else None)
        base_fp = [metrics[('T1', sd)]['fp'] for sd in train.SEEDS]
        base_ap = [metrics[('T1', sd)]['ap'] for sd in train.SEEDS]
        samples['T3_vs_T1']['ratios'].append(ensemble_ratio(en['fp'], base_fp))
        samples['T3_vs_T1']['deltas'].append(ensemble_delta_ap(en['ap'], base_ap))
    result = {}
    for name, values in samples.items():
        result[name] = {'aggregate_ratio_ci': m.percentile(values['ratios']),
            'aggregate_delta_ap_ci': m.percentile(values['deltas']),
            'undefined_ratio_replicates': sum(v is None for v in values['ratios']),
            'undefined_ap_replicates': sum(v is None for v in values['deltas'])}
        if name != 'T3_vs_T1':
            result[name].update(per_seed_ratio_ci={sd: m.percentile(v) for sd, v in values['per_seed_ratios'].items()},
                per_seed_delta_ap_ci={sd: m.percentile(v) for sd, v in values['per_seed_deltas'].items()})
    return {'replicates': replicates, 'rng_seed': seed, 'groups': len(unique),
        'shared_resample_plan_sha256': plan_hash.hexdigest(), 'paired_across_all_arms_and_seeds': True,
        'comparisons': result}


def score_metrics(frame, score):
    result = m.score_metrics(frame, score)
    y, _, classes, independent, late = m.strata(frame)
    threshold = result['primary']['threshold']
    matched = np.asarray(score) >= threshold if threshold is not None else np.zeros(len(y), dtype=bool)
    result['matched_late_hover_normal_fp'] = {
        domain: int(matched[independent & late & frame.domain.eq(domain).to_numpy()].sum()) for domain in ('SIL', 'HIL')}
    result['matched_late_hover_fault_recall'] = {}
    for fault in ('GPS', 'barometer'):
        mask = (y == 1) & late & (classes == fault)
        result['matched_late_hover_fault_recall'][fault] = float(matched[mask].mean()) if mask.any() else None
    return result


def class_drops(current, reference):
    result = {}
    classes = reference['42']['matched_class_recall_outside_late_hover_fault']
    for name in classes:
        a = [current[str(sd)]['matched_class_recall_outside_late_hover_fault'][name] for sd in train.SEEDS]
        b = [reference[str(sd)]['matched_class_recall_outside_late_hover_fault'][name] for sd in train.SEEDS]
        result[name] = 100 * float(np.mean(np.subtract(b, a))) if all(v is not None for v in a + b) else None
    return result


def seed_stability(metrics):
    # The protocol does not specify the SD divisor. Report both, explicitly,
    # rather than silently choosing a convention; neither enters a verdict.
    result = {}
    for arm in ARMS:
        fps = [metrics[arm][str(sd)]['primary']['fp'] for sd in train.SEEDS]
        if any(fp is None or fp <= 0 for fp in fps):
            result[arm] = {'sample_ddof1': None, 'population_ddof0': None}
        else:
            values = np.log(fps)
            result[arm] = {'sample_ddof1': float(np.std(values, ddof=1)),
                           'population_ddof0': float(np.std(values, ddof=0))}
    return result


def evaluation_reproduction(metrics):
    expected = {42: 3595, 43: 3588, 44: 4610}
    actual = {seed: metrics['T0'][str(seed)]['primary']['fp'] for seed in expected}
    if actual != expected:
        raise RuntimeError(f'T0 pooled S0 FP reproduction mismatch: {actual} != {expected}; STOP')
    return {'status': 'PASS', 'fp_r97': actual}


def load_prediction(path, expected_ids, expected_labels, rows, *, epochs=None):
    with np.load(path, allow_pickle=False) as data:
        p = data['probabilities']
        if p.dtype != np.float32 or p.shape != (rows, len(expected_ids)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid float32 trajectory probability matrix')
        if data['original_cache_index'].dtype != np.int64 or not np.array_equal(data['original_cache_index'], expected_ids):
            raise ValueError('Prediction window IDs/order differ from frozen population')
        if data['labels'].dtype != np.int64 or not np.array_equal(data['labels'], expected_labels):
            raise ValueError('Prediction labels differ from frozen population')
        expected_epochs = np.arange(1, rows + 1) if epochs is None else np.asarray(epochs)
        if not np.array_equal(data['epochs'], expected_epochs):
            raise ValueError('Prediction epoch IDs/order mismatch')
    return p


def load_runs(root, frame, phase, phase1_sha=None, adopted_rule=None, *, train_frame=None):
    _, outer = s.split_ledger(frame, 0)
    scores = {a: {} for a in ARMS}
    trajectories, reports, epoch_choices = {}, {}, {}
    manifests = set()
    goldens = train.golden_digests()
    expected_data = s.read_json(s.BASELINE)['data_identities']
    for seed in train.SEEDS:
        pools = {a: np.full(len(frame), np.nan) for a in ARMS}
        trajectories[seed] = {}
        for fold in (range(3) if phase == 'cv' else [0]):
            folder = Path(root) / f'seed{seed}_fold{fold}'
            r = s.read_json(folder / 'report.json')
            if r.get('status') != 'COMPLETE' or r.get('cpu_smoke') or r.get('determinism_probe'):
                raise ValueError('Only COMPLETE formal T runs are eligible')
            if (r['seed'], r['fold'], r['phase'], r['protocol_sha256']) != (seed, fold, phase, train.PROTOCOL_SHA):
                raise ValueError('T run identity mismatch')
            if r['epochs'] != 100 or len(r['history']) != 100 or [x['epoch'] for x in r['history']] != list(range(1, 101)):
                raise ValueError('T requires all 100 epochs')
            train.require_formal_environment(r['environment'])
            if r['initial_digest'] != goldens[('mstca', seed)] or r['implementation_files'] != train.implementation_manifest()['files']:
                raise ValueError('Run golden/implementation mismatch')
            if r['data_identities']['matched_sha256'] != expected_data or r['data_identities']['skipped_sha256']:
                raise ValueError('All nine identities must match')
            manifests.add(r['implementation_manifest_sha256'])
            selected = s.read_json(folder / 'selections.json')
            for epoch in range(1, 101):
                if not (folder / f'checkpoints/epoch_{epoch}.pt').is_file():
                    raise ValueError('Missing per-epoch checkpoint')
                if r['history'][epoch - 1]['state_digest'] != r['checkpoint_state_digests'][str(epoch)]:
                    raise ValueError('Checkpoint digest metadata mismatch')
            source_frame = frame if phase == 'cv' else train_frame
            if source_frame is None:
                raise ValueError('Explicit authenticated train ledger required for val evaluation')
            positions, _ = s.split_ledger(source_frame, fold, phase)
            inner_frame = source_frame.iloc[positions['inner']]
            inner_path = folder / 'inner_predictions.npz'
            if s.sha(inner_path) != r['prediction_sha256']['inner']:
                raise ValueError('Inner trajectory SHA mismatch')
            inner = load_prediction(inner_path, inner_frame.original_cache_index.to_numpy(dtype=np.int64),
                inner_frame.qualified_y.to_numpy(dtype=np.int64), 100)
            replay = train.select_epochs([x['inner_class_balanced_ce'] for x in r['history']], inner, inner_frame.qualified_y.to_numpy())
            if replay != selected:
                raise ValueError('Stored selections differ from inner-only replay')
            idx = np.flatnonzero(outer == fold) if phase == 'cv' else np.arange(len(frame))
            ids = frame.iloc[idx].original_cache_index.to_numpy(dtype=np.int64)
            labels = frame.iloc[idx].qualified_y.to_numpy(dtype=np.int64)
            if phase == 'cv':
                if seed <= 44:
                    train.reproduction_gate(seed, fold, r['history'], selected)
                path = folder / 'heldout_predictions.npz'
                if s.sha(path) != r['prediction_sha256']['heldout']:
                    raise ValueError('Held-out trajectory SHA mismatch')
                trajectory = load_prediction(path, ids, labels, 100)
                trajectories[seed][fold] = (idx, trajectory)
                for arm in ARMS:
                    pools[arm][idx] = trajectory[selected[arm]['epoch'] - 1]
            else:
                if r['phase1_verdicts_sha256'] != phase1_sha or r['adopted_rule'] != adopted_rule:
                    raise ValueError('Val run phase-1 gate/adoption mismatch')
                chosen = {selected['T0']['epoch'], selected[adopted_rule]['epoch']}
                if set(r['val_prediction_sha256']) != set(map(str, chosen)) or r['val_prediction_passes'] != len(chosen):
                    raise ValueError('Val must have exactly one prediction pass per distinct selected checkpoint')
                saved = {}
                for epoch in chosen:
                    path = folder / f'val_epoch_{epoch}.npz'
                    if s.sha(path) != r['val_prediction_sha256'][str(epoch)]:
                        raise ValueError('Val prediction SHA mismatch')
                    saved[epoch] = load_prediction(path, ids, labels, 1, epochs=[epoch])[0]
                pools['T0'][idx] = saved[selected['T0']['epoch']]
                pools[adopted_rule][idx] = saved[selected[adopted_rule]['epoch']]
            epoch_choices[f'seed{seed}_fold{fold}'] = selected
            reports[str(folder)] = r
        for arm in ARMS if phase == 'cv' else set(('T0', adopted_rule)):
            if not np.isfinite(pools[arm]).all():
                raise ValueError('Incomplete/overlapping pooled predictions')
            scores[arm][seed] = pools[arm]
    if len(manifests) != 1:
        raise ValueError('Implementation manifest differs across runs')
    return scores, trajectories, reports, epoch_choices


def oracle_curves(frame, trajectories):
    y = frame.qualified_y.to_numpy()
    result = {}
    for seed in train.SEEDS:
        curve = []
        for epoch in range(100):
            pooled = np.full(len(frame), np.nan)
            for idx, values in trajectories[seed].values():
                pooled[idx] = values[epoch]
            curve.append(m.fp_at_recall(y, pooled)['fp'])
        oracle = min(range(100), key=lambda i: curve[i]) + 1 if all(v is not None for v in curve) else None
        result[str(seed)] = {'label': 'NOT SELECTABLE', 'fp_r97_by_epoch': curve,
                             'oracle_epoch': oracle, 'oracle_fp': curve[oracle - 1] if oracle is not None else None}
    return result


def confirmation(phase1, metrics):
    adopted = phase1['adopted_rule']
    point = summarize_pair(metrics[adopted], metrics['T0'])
    value = point['aggregate_ratio']
    verdict = phase1['decisions'].get(adopted, {}).get('verdict')
    if adopted == 'T0':
        label = 'NOT_APPLICABLE_KEEP_T0'
    elif value is None:
        label = 'INCONCLUSIVE'
    else:
        ok = value < 1 if verdict == 'BETTER' else 1 / 1.10 <= value <= 1.10
        label = 'CONFIRMED' if ok else 'NOT_CONFIRMED'
    return {**point, 'confirmation': label, 'adopted_rule_unchanged': adopted}


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh evaluation output required')
    phase1 = train.phase1_gate(args.phase1_verdicts, args.phase1_verdicts_sha256, args.adopted_rule) if args.phase == 'val' else None
    train.require_formal_environment()
    train.load_protocol(args.protocol)
    manifest = train.implementation_manifest()
    split = 'train' if args.phase == 'cv' else 'val'
    expected = s.read_json(s.BASELINE)['data_identities'][f'ledgers/{split}_included_windows.csv']
    if s.sha(Path(args.data) / f'ledgers/{split}_included_windows.csv') != expected:
        raise ValueError('Evaluation ledger identity mismatch')
    frame = s.read_ledger(args.data, split)
    if args.phase == 'cv':
        s.fold_sizes(frame, s.read_json(s.DEFAULT_PROTOCOL))
    output.mkdir(parents=True, exist_ok=False)
    try:
        train_frame = None
        if args.phase == 'val':
            train_path = Path(args.data) / 'ledgers/train_included_windows.csv'
            expected_train = s.read_json(s.BASELINE)['data_identities']['ledgers/train_included_windows.csv']
            if s.sha(train_path) != expected_train:
                raise ValueError('Phase-2 train ledger identity mismatch')
            train_frame = s.read_ledger(args.data)
        scores, trajectories, reports, selections = load_runs(args.runs, frame, args.phase,
            getattr(args, 'phase1_verdicts_sha256', None), getattr(args, 'adopted_rule', None), train_frame=train_frame)
        arms = ARMS if args.phase == 'cv' else set(('T0', args.adopted_rule))
        metrics = {a: {str(sd): score_metrics(frame, scores[a][sd]) for sd in train.SEEDS} for a in arms}
        result = {'status': 'COMPLETE', 'phase': args.phase, 'protocol_sha256': train.PROTOCOL_SHA,
            'synthetic_demo': False, 'performance_claim': True,
            'population': len(frame), 'metrics': metrics, 'selections': selections,
            'run_report_sha256': {path: s.sha(Path(path) / 'report.json') for path in reports},
            'resources': [{k: r[k] for k in ('seed', 'fold', 'epochs', 'wall_time_s', 'param_count', 'macs')} for r in reports.values()]}
        if args.phase == 'cv':
            result['evaluation_reproduction'] = evaluation_reproduction(metrics)
            uncertainty = paired_bootstrap(frame.qualified_y.to_numpy(), scores, frame.association_group.to_numpy())
            comparisons, decisions = {}, {}
            for name, (a, b) in COMPARISONS.items():
                point = summarize_pair(metrics[a], metrics[b])
                drops = class_drops(metrics[a], metrics[b])
                intervals = uncertainty['comparisons'][name]
                comparisons[name] = {**point, **intervals, 'class_recall_drop_seed_mean_pp': drops}
                if b == 'T0':
                    decisions[a] = decide(list(point['per_seed_ratios'].values()), intervals['aggregate_ratio_ci'],
                                          intervals['aggregate_delta_ap_ci'], drops)
            en = score_metrics(frame, ensemble([scores['T1'][sd] for sd in train.SEEDS]))
            base_fp = [metrics['T1'][str(sd)]['primary']['fp'] for sd in train.SEEDS]
            base_ap = [metrics['T1'][str(sd)]['ap'] for sd in train.SEEDS]
            en_uncertainty = uncertainty['comparisons']['T3_vs_T1']
            result.update(uncertainty=uncertainty, comparisons=comparisons, decisions=decisions,
                adopted_rule=adopt(decisions, comparisons['T2_vs_T1']['aggregate_ratio_ci']),
                T3={'metrics': en, 'ratio_vs_geometric_mean_T1_fp': ensemble_ratio(en['primary']['fp'], base_fp),
                    'delta_ap_vs_arithmetic_mean_T1': ensemble_delta_ap(en['ap'], base_ap),
                    **en_uncertainty, **t3_verdict(en_uncertainty['aggregate_ratio_ci'], en_uncertainty['aggregate_delta_ap_ci'])},
                seed_stability_log_fp_sd=seed_stability(metrics),
                oracle=oracle_curves(frame, trajectories))
        else:
            result.update(decisions=phase1['decisions'], adopted_rule=phase1['adopted_rule'],
                phase1_verdicts_sha256=args.phase1_verdicts_sha256, confirmation=confirmation(phase1, metrics))
            if args.adopted_rule == 'T1':
                en = score_metrics(frame, ensemble([scores['T1'][sd] for sd in train.SEEDS]))
                result['T3_descriptive_only'] = {'mac_cost_multiplier': 5, 'metrics': en,
                    'ratio_vs_geometric_mean_T1_fp': ensemble_ratio(en['primary']['fp'],
                        [metrics['T1'][str(sd)]['primary']['fp'] for sd in train.SEEDS]),
                    'delta_ap_vs_arithmetic_mean_T1': ensemble_delta_ap(en['ap'],
                        [metrics['T1'][str(sd)]['ap'] for sd in train.SEEDS])}
        train.load_protocol(args.protocol)
        if train.implementation_manifest() != manifest:
            raise ValueError('T implementation changed during evaluation')
        s.write_json(output / 'evaluation.json', result)
        s.write_json(output / 'evaluation_sha256.json', {'sha256': s.sha(output / 'evaluation.json')})
        with (output / 'summary.md').open('x', encoding='utf-8') as stream:
            stream.write('# T selection evaluation\n' + str(result.get('decisions', {})) + '\nAdopted: ' + result['adopted_rule'] + '\nOracle: NOT SELECTABLE\n')
        return result
    except Exception as exc:
        s.write_json(output / 'FAILED.json', {'status': 'FAILED', 'error': repr(exc)})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, default=train.DEFAULT_PROTOCOL)
    result.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    result.add_argument('--runs', type=Path, required=True)
    result.add_argument('--phase', choices=('cv', 'val'), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--phase1-verdicts', type=Path)
    result.add_argument('--phase1-verdicts-sha256')
    result.add_argument('--adopted-rule', choices=('T0', 'T1', 'T2'))
    return result


if __name__ == '__main__':
    print(evaluate(parser().parse_args())['status'])
