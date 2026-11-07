"""E-group: ENS (MSTCA + window-statistics GBT, plain mean) on existing T CV predictions.

Protocol: configs/experiments/e_ens_t_cv_v1.json (frozen before any result).
Zero training; reads only train-CV held-out predictions. No val/test/Real.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_s_ablation as s  # noqa: E402
import evaluate_s_ablation as m  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
T_ROOT = ROOT / 'artifacts/t_selection_mstca_v1/cv'
GBT = ROOT / 'artifacts/feature_groupcv_v1_20261105/gbt_heldout.npz'
W0_EXPECTED = {42: 3595, 43: 3588, 44: 4610}
PRIMARY = ('E0', 'E1')            # vs T0, decide
DESCRIPTIVE = {'GBT_vs_T0': ('GBT', 'T0'), 'E1_vs_T1': ('E1', 'T1'), 'T1_vs_T0': ('T1', 'T0')}


def load_scores(frame, outer):
    pos = {int(v): i for i, v in enumerate(frame.original_cache_index.to_numpy())}
    y = frame.qualified_y.to_numpy(dtype=np.int64)
    with np.load(GBT, allow_pickle=False) as g:
        gi = np.array([pos[int(v)] for v in g['original_indices']])
        if len(set(gi.tolist())) != len(frame):
            raise RuntimeError('GBT does not cover the population exactly')
        gbt = np.full(len(frame), np.nan); gbt[gi] = g['prob']
        gfold = np.full(len(frame), -1); gfold[gi] = g['fold']
    if not np.array_equal(gfold, outer):
        raise RuntimeError('GBT folds differ from T outer folds; STOP')
    t0, t1, epochs = {}, {}, {}
    for seed in SEEDS:
        a, b = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        for fold in range(3):
            d = T_ROOT / f'seed{seed}_fold{fold}'
            sel = json.loads((d / 'selections.json').read_text())
            with np.load(d / 'heldout_predictions.npz', allow_pickle=False) as p:
                rows = np.array([pos[int(v)] for v in p['original_cache_index']])
                if not np.array_equal(p['labels'], y[rows]) or not (outer[rows] == fold).all():
                    raise RuntimeError(f'Held-out identity mismatch seed{seed} fold{fold}')
                ep = list(p['epochs'])
                a[rows] = p['probabilities'][ep.index(sel['T0']['epoch'])]
                b[rows] = p['probabilities'][ep.index(100)]
            epochs[f'{seed}_{fold}'] = sel['T0']['epoch']
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise RuntimeError('Incomplete pooled predictions')
        t0[seed], t1[seed] = a, b
    scores = {'T0': t0, 'T1': t1, 'GBT': {sd: gbt for sd in SEEDS},
              'E0': {sd: (t0[sd] + gbt) / 2 for sd in SEEDS},
              'E1': {sd: (t1[sd] + gbt) / 2 for sd in SEEDS}}
    return scores, epochs


def matched_threshold(y, cache, k):
    normals = cache.order[y[cache.order] == 0]
    return float(cache.score[normals[k - 1]]) if 0 < k <= len(normals) else None


def class_tables(y, classes, late, scores, caches, basic, arm, ref):
    """C2 (matched FP) and C3 (own R97) seed-mean pp per class, all windows and V-style."""
    out = {}
    for label, use_late in (('all_fault_windows', True), ('excl_late_hover', False)):
        c2, c3 = {}, {}
        for cls in sorted(set(classes[y == 1])):
            mask = (y == 1) & (classes == cls) & (True if use_late else ~late)
            n = int(mask.sum())
            d2, d3 = [], []
            for sd in SEEDS:
                r0 = scores[ref][sd] >= basic[(ref, sd)]['threshold']
                tp0 = int(r0[mask].sum())
                tp1 = int((scores[arm][sd] >= basic[(arm, sd)]['threshold'])[mask].sum())
                tm = matched_threshold(y, caches[(arm, sd)], basic[(ref, sd)]['fp'])
                tpm = int((scores[arm][sd] >= tm)[mask].sum()) if tm is not None else None
                d2.append(None if tpm is None else tpm - tp0)
                d3.append(tp0 - tp1)
            c2[cls] = 100 * sum(d2) / (5 * n) if None not in d2 else None
            c3[cls] = 100 * sum(d3) / (5 * n)
        out[label] = {'C2_matched_fp_recall_delta_pp': c2, 'C3_r97_recall_drop_pp': c3}
    return out


def phase_fp(frame, y, flight, independent, scores, basic, arm):
    onset = frame[frame.fault_onset_row >= 0].drop_duplicates('case_uid')
    oflight = onset.case_uid.str.split('/').str[1]
    ref = (onset.fault_onset_row * 0.02).groupby(oflight).median().to_dict()
    t_end = frame.window_end_row.to_numpy() * 0.02
    post = t_end > np.array([ref[f] for f in flight])
    res = {}
    for f in sorted(ref):
        for ph, pm in (('pre', ~post), ('post', post)):
            mask = independent & (flight == f) & pm
            rates = [float((scores[arm][sd] >= basic[(arm, sd)]['threshold'])[mask].mean()) for sd in SEEDS]
            res[f'{f}_{ph}'] = {'windows': int(mask.sum()), 'fp_rate_seed_mean': float(np.mean(rates))}
    return {'onset_reference_s': ref, 'independent_normal_fp': res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--replicates', type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    frame = s.read_ledger(args.data)
    _, outer = s.split_ledger(frame, 0)
    y, flight, classes, independent, late = m.strata(frame)
    scores, epochs = load_scores(frame, outer)
    arms = list(scores)
    caches = {(a, sd): m.MetricCache(y, scores[a][sd]) for a in arms for sd in SEEDS}
    basic = {k: c.basic() for k, c in caches.items()}
    gate = {sd: basic[('T0', sd)]['fp'] for sd in W0_EXPECTED}
    if gate != W0_EXPECTED:
        raise RuntimeError(f'W0 gate failed {gate}; STOP')
    pairs = {f'{a}_vs_T0': (a, 'T0') for a in PRIMARY} | DESCRIPTIVE

    # bootstrap (one shared plan)
    groups = frame.association_group.to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(20261106)
    h = hashlib.sha256()
    boot = {p: {'ratio': [], 'dap': []} for p in pairs}
    for _ in range(args.replicates):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        h.update(cnt.astype('<i8').tobytes())
        w = cnt[inv]
        bb = {k: c.basic(w) for k, c in caches.items() if k[0] != 'GBT' or k[1] == 42}
        for sd in SEEDS:
            bb[('GBT', sd)] = bb[('GBT', 42)]
        for p, (a, r) in pairs.items():
            ratios = [m.ratio(bb[(a, sd)]['fp'], bb[(r, sd)]['fp']) for sd in SEEDS]
            boot[p]['ratio'].append(m.geometric_mean(ratios))
            boot[p]['dap'].append(float(np.mean([bb[(a, sd)]['ap'] - bb[(r, sd)]['ap'] for sd in SEEDS])))

    result = {'protocol': 'configs/experiments/e_ens_t_cv_v1.json', 'population': len(frame),
              'w0_gate': {str(k): v for k, v in gate.items()}, 't0_epochs': epochs,
              'bootstrap': {'replicates': args.replicates, 'rng_seed': 20261106, 'groups': len(uniq),
                            'plan_sha256': h.hexdigest()},
              'fp_r97': {a: {str(sd): basic[(a, sd)]['fp'] for sd in SEEDS} for a in arms},
              'ap': {a: {str(sd): basic[(a, sd)]['ap'] for sd in SEEDS} for a in arms},
              'comparisons': {}, 'verdicts': {}}
    for p, (a, r) in pairs.items():
        ratios = [m.ratio(basic[(a, sd)]['fp'], basic[(r, sd)]['fp']) for sd in SEEDS]
        dap = float(np.mean([basic[(a, sd)]['ap'] - basic[(r, sd)]['ap'] for sd in SEEDS]))
        rci, aci = m.percentile(boot[p]['ratio']), m.percentile(boot[p]['dap'])
        cls = class_tables(y, classes, late, scores, caches, basic, a, r)
        gm = m.geometric_mean(ratios)
        c = {'C1': bool(gm < 1 and sum(v < 1 for v in ratios) >= 4 and rci[1] < 1),
             'C2': all(v is not None and v >= -1 for v in cls['all_fault_windows']['C2_matched_fp_recall_delta_pp'].values()),
             'C3': all(v <= 2 for v in cls['all_fault_windows']['C3_r97_recall_drop_pp'].values()),
             'C4': bool(aci[1] >= 0)}
        late_hover = {}
        for sd in SEEDS:
            pa = scores[a][sd] >= basic[(a, sd)]['threshold']
            late_hover[str(sd)] = {'late_hover_indep_normal_fp': int(pa[independent & late].sum()),
                                   **{f'late_hover_{k}_recall': float(pa[(y == 1) & late & (classes == k)].mean())
                                      for k in ('GPS', 'barometer')}}
        result['comparisons'][p] = {'per_seed_ratio': dict(zip(map(str, SEEDS), ratios)), 'aggregate_ratio': gm,
            'ratio_ci': rci, 'delta_ap': dap, 'delta_ap_ci': aci, **cls, 'criteria': c,
            'late_hover': late_hover}
        if p in ('E0_vs_T0', 'E1_vs_T0'):
            failed = [k for k, v in c.items() if not v]
            result['verdicts'][a] = 'PASS' if not failed else 'FAIL ' + ','.join(failed)
    result['phase_fp'] = {a: phase_fp(frame, y, flight, independent, scores, basic, a) for a in ('T0', 'T1', 'GBT', 'E0', 'E1')}
    result['flights_with_alarm'] = {a: float(np.mean([frame[independent & (scores[a][sd] >= basic[(a, sd)]['threshold'])].case_uid.nunique() for sd in SEEDS])) for a in arms}
    (out / 'evaluation.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'verdicts': result['verdicts'], 'fp_r97': result['fp_r97'],
                      **{p: {k: v for k, v in result['comparisons'][p].items() if k in ('aggregate_ratio', 'ratio_ci', 'delta_ap_ci', 'criteria')} for p in pairs}}, indent=1))


if __name__ == '__main__':
    main()
