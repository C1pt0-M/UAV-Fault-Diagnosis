"""Q1: unified evaluation of existing source-domain candidates under a per-class operating point.

Protocol: configs/experiments/q1_source_model_selection_v1.json (frozen before W/S results).
Zero training; train-CV held-out predictions only. No val/test/Real.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import e_ens_t_cv as e  # noqa: E402
from e_ens_t_cv import s, m  # noqa: E402

S5, S3 = (42, 43, 44, 45, 46), (42, 43, 44)
W_ROOT = ROOT / 'artifacts/w_phase_balance_mstca_v1/cv'
S_ROOT = ROOT / 'artifacts/s_ablation_mstca_v1/cv'
SINGLE = ('T1', 'W1', 'W2', 'S1', 'S2', 'S3', 'GBT')
ENS = ('E0', 'E1', 'W1+GBT', 'W2+GBT', 'S1+GBT', 'S2+GBT', 'S3+GBT')
SENS = ('S1last', 'S2last', 'S3last', 'S1last+GBT', 'S2last+GBT', 'S3last+GBT')
G = {}  # globals shared with forked bootstrap workers


def load_w(frame, outer, pos, arm):
    out = {}
    for sd in S5:
        a = np.full(len(frame), np.nan)
        for f in range(3):
            d = W_ROOT / f'{arm}_seed{sd}_fold{f}'
            ep = json.loads((d / 'selections.json').read_text())['T0']['epoch']
            with np.load(d / 'heldout_predictions.npz') as p:
                rows = np.array([pos[int(v)] for v in p['original_cache_index']])
                if not (outer[rows] == f).all():
                    raise RuntimeError(f'{arm} seed{sd} fold{f} not held-out')
                a[rows] = p['probabilities'][list(p['epochs']).index(ep)]
        out[sd] = a
    return out


def load_s(frame, outer, pos, arm, which):
    out = {}
    for sd in S3:
        a = np.full(len(frame), np.nan)
        for f in range(3):
            with np.load(S_ROOT / f'{arm}_seed{sd}_fold{f}' / f'heldout_predictions_{which}.npz') as p:
                rows = np.array([pos[int(v)] for v in p['original_cache_index']])
                if not (outer[rows] == f).all():
                    raise RuntimeError(f'{arm} seed{sd} fold{f} not held-out')
                a[rows] = p['probabilities'][:, 1]
        out[sd] = a
    return out


class Ranked:
    """Score orderings for weighted threshold / count queries."""

    def __init__(self, score, y, strata):
        self.n_idx = np.flatnonzero(y == 0)
        self.n_idx = self.n_idx[np.argsort(-score[self.n_idx], kind='stable')]
        self.n_neg = -score[self.n_idx]
        self.f_idx = np.flatnonzero(y == 1)
        self.f_idx = self.f_idx[np.argsort(-score[self.f_idx], kind='stable')]
        self.f_sc = score[self.f_idx]
        self.st = []
        for mask in strata:
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(-score[idx], kind='stable')]
            self.st.append((idx, score[idx]))

    def r97(self, w):
        cum = np.cumsum(w[self.f_idx])
        return float(self.f_sc[np.searchsorted(cum, 0.97 * cum[-1], side='left')])

    def tp(self, w, t):
        out = []
        for idx, sc in self.st:
            k = np.searchsorted(-sc, -t, side='right')
            out.append(int(w[idx[:k]].sum()))
        return out

    def threshold(self, w, targets, n_strata):
        t = np.inf
        for (idx, sc), tgt in list(zip(self.st, targets))[:n_strata]:
            if tgt <= 0:
                continue
            cum = np.cumsum(w[idx])
            j = np.searchsorted(cum, tgt, side='left')
            t = min(t, float(sc[j]) if j < len(sc) else 0.0)
        return t

    def fp(self, w, t):
        k = np.searchsorted(self.n_neg, -t, side='right')
        return int(w[self.n_idx[:k]].sum())


def evaluate(w, plan):
    """Return {(arm, seed): (fp_main, fp_strict)} for weights w (T0 included)."""
    R, n6 = G['ranked'], G['n_classes']
    res = {}
    for sd in S5:
        r0 = R[('T0', sd)]
        t0 = r0.r97(w)
        tgt = r0.tp(w, t0)
        for arm, seeds in plan:
            if sd not in seeds:
                continue
            ra = R[(arm, sd)]
            res[(arm, sd)] = (ra.fp(w, ra.threshold(w, tgt, n6)), ra.fp(w, ra.threshold(w, tgt, n6 + 1)))
    return res


def boot_worker(counts):
    inv = G['inv']
    return [evaluate(c[inv], G['plan']) for c in counts]


def gm_ratio(fp, a, b, seeds, k):
    return m.geometric_mean([m.ratio(fp[(a, sd)][k], fp[(b, sd)][k]) for sd in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--replicates', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    frame = s.read_ledger(args.data)
    _, outer = s.split_ledger(frame, 0)
    y, flight, classes, independent, late = m.strata(frame)
    pos = {int(v): i for i, v in enumerate(frame.original_cache_index.to_numpy())}
    base, _ = e.load_scores(frame, outer)
    scores = {'T0': base['T0'], 'T1': base['T1'], 'GBT': base['GBT'], 'E0': base['E0'], 'E1': base['E1']}
    gbt = base['GBT'][42]
    for arm in ('W1', 'W2'):
        scores[arm] = load_w(frame, outer, pos, arm)
    for arm in ('S0', 'S1', 'S2', 'S3'):
        scores[arm] = load_s(frame, outer, pos, arm, 'best')
        if arm != 'S0':
            scores[arm + 'last'] = load_s(frame, outer, pos, arm, 'last')
    for arm in ('W1', 'W2', 'S1', 'S2', 'S3', 'S1last', 'S2last', 'S3last'):
        scores[arm + '+GBT'] = {sd: (v + gbt) / 2 for sd, v in scores[arm].items()}
    for arm, d in scores.items():
        for sd, v in d.items():
            if not np.isfinite(v).all():
                raise RuntimeError(f'incomplete {arm} {sd}')

    gate = {sd: m.MetricCache(y, scores['T0'][sd]).basic()['fp'] for sd in S5}
    gate_s0 = {sd: m.MetricCache(y, scores['S0'][sd]).basic()['fp'] for sd in S3}
    if any(gate[sd] != e.W0_EXPECTED[sd] or gate_s0[sd] != e.W0_EXPECTED[sd] for sd in S3):
        raise RuntimeError(f'gate failed T0 {gate} S0 {gate_s0}')

    cls_names = sorted(set(classes[y == 1]))
    strata = [(y == 1) & (classes == c) for c in cls_names] + [(y == 1) & late & (classes == 'GPS')]
    arms = ('T0',) + SINGLE + ENS + SENS
    seeds_of = {a: (S3 if a.startswith('S') else S5) for a in arms}
    plan = [(a, seeds_of[a]) for a in arms]
    G.update(ranked={(a, sd): Ranked(scores[a][sd], y, strata) for a, ss in plan for sd in ss},
             n_classes=len(cls_names), plan=plan)

    ones = np.ones(len(y), dtype=np.int64)
    point = evaluate(ones, plan)

    # bootstrap
    groups = frame.association_group.to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(20261106)
    counts = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(args.replicates)]
    G['inv'] = inv
    chunks = [counts[i::args.workers] for i in range(args.workers)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        parts = pool.map(boot_worker, chunks)
    boot = [r for part in parts for r in part]

    # descriptive per arm at main operating point
    def at_main(a, sd):
        r0 = G['ranked'][('T0', sd)]
        tgt = r0.tp(ones, r0.r97(ones))
        t = G['ranked'][(a, sd)].threshold(ones, tgt, len(cls_names))
        pred = scores[a][sd] >= t
        return {'threshold': t,
                'late_hover_gps_recall': float(pred[strata[-1]].mean()),
                'late_hover_indep_normal_fp': int(pred[independent & late].sum()),
                'binding_classes': [c for c, mk in zip(cls_names, strata) if int(pred[mk].sum()) == tgt[cls_names.index(c)]]}

    def oof(a, sd):
        fp_a = fp_0 = 0
        tp_a = np.zeros(len(strata)); tp_0 = np.zeros(len(strata))
        r0, ra = G['ranked'][('T0', sd)], G['ranked'][(a, sd)]
        for k in range(3):
            wo, wk = (outer != k).astype(np.int64), (outer == k).astype(np.int64)
            t0 = r0.r97(wo)
            ta = ra.threshold(wo, r0.tp(wo, t0), len(cls_names))
            fp_0 += r0.fp(wk, t0); fp_a += ra.fp(wk, ta)
            tp_0 += r0.tp(wk, t0); tp_a += ra.tp(wk, ta)
        n = np.array([mk.sum() for mk in strata])
        return {'fp_arm': fp_a, 'fp_t0': fp_0, 'recall_diff_pp': (100 * (tp_a - tp_0) / n).tolist()}

    names = cls_names + ['late_hover_GPS']
    result = {'protocol': 'configs/experiments/q1_source_model_selection_v1.json', 'population': len(y),
              'gate_t0': gate, 'gate_s0': gate_s0, 'classes': names,
              'bootstrap': {'replicates': args.replicates, 'rng_seed': 20261106, 'groups': len(uniq)}, 'arms': {}}
    for a in arms[1:]:
        ss = seeds_of[a]
        main_r = [m.ratio(point[(a, sd)][0], point[('T0', sd)][0]) for sd in ss]
        strict_r = [m.ratio(point[(a, sd)][1], point[('T0', sd)][1]) for sd in ss]
        bm = [gm_ratio(b, a, 'T0', ss, 0) for b in boot]
        bs = [gm_ratio(b, a, 'T0', ss, 1) for b in boot]
        fps = [point[(a, sd)][0] for sd in ss]
        o = {sd: oof(a, sd) for sd in ss}
        result['arms'][a] = {
            'seeds': list(ss), 'fp_pc': dict(zip(map(str, ss), fps)),
            't0_fp': {str(sd): point[('T0', sd)][0] for sd in ss},
            'fp_pc_sd': float(np.std(fps, ddof=1)), 'fp_pc_cv': float(np.std(fps, ddof=1) / np.mean(fps)),
            'ratio_per_seed': dict(zip(map(str, ss), main_r)), 'ratio': m.geometric_mean(main_r), 'ratio_ci': m.percentile(bm),
            'strict_ratio_per_seed': dict(zip(map(str, ss), strict_r)), 'strict_ratio': m.geometric_mean(strict_r),
            'strict_ratio_ci': m.percentile(bs),
            'ap_delta': float(np.mean([m.MetricCache(y, scores[a][sd]).basic()['ap'] - m.MetricCache(y, scores['T0'][sd]).basic()['ap'] for sd in ss])),
            'at_main': {str(sd): at_main(a, sd) for sd in ss},
            't0_late_hover': {str(sd): at_main('T0', sd) for sd in ss},
            'oof': {'fp_ratio_gm': m.geometric_mean([o[sd]['fp_arm'] / o[sd]['fp_t0'] for sd in ss]),
                    'recall_diff_pp_seed_mean': dict(zip(names, np.mean([o[sd]['recall_diff_pp'] for sd in ss], axis=0).tolist())),
                    'per_seed': {str(sd): o[sd] for sd in ss}}}
        r = result['arms'][a]
        r['eligible'] = bool(a not in SENS and r['ratio'] < 1 and all(v < 1 for v in main_r) and r['ratio_ci'][1] < 1)

    # selection
    cand = {a: v for a, v in result['arms'].items() if v['eligible']}
    best = min(cand, key=lambda a: cand[a]['ratio']) if cand else None
    pars = {}
    if best:
        for a in cand:
            if a in SINGLE and a != best:
                common = [sd for sd in seeds_of[best] if sd in seeds_of[a]]
                ci = m.percentile([gm_ratio(b, best, a, common, 0) for b in boot])
                pars[a] = {'common_seeds': common, 'point': gm_ratio(point, best, a, common, 0), 'ci': ci}
    keep = best
    if best and best not in SINGLE:
        ties = [a for a, v in pars.items() if v['ci'][1] >= 1]
        if ties:
            keep = min(ties, key=lambda a: cand[a]['ratio'])
    flags = {}
    if keep:
        r = result['arms'][keep]
        flags = {'strict_ratio_ge_1': r['strict_ratio'] >= 1,
                 'oof_class_deficit_gt_1pp': [c for c, v in r['oof']['recall_diff_pp_seed_mean'].items() if c != 'late_hover_GPS' and v < -1],
                 'three_seeds_only': keep.startswith('S')}
    result['selection'] = {'eligible': sorted(cand, key=lambda a: cand[a]['ratio']), 'best': best,
                           'parsimony_checks': pars, 'keep': keep, 'flags': flags}
    (out / 'evaluation.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    for a, v in result['arms'].items():
        print(f"{a:12s} ratio {v['ratio']:.3f} CI {v['ratio_ci'][0]:.3f}-{v['ratio_ci'][1]:.3f} strict {v['strict_ratio']:.3f} "
              f"cv {v['fp_pc_cv']:.3f} oof {v['oof']['fp_ratio_gm']:.3f} elig {v['eligible']}")
    print(json.dumps(result['selection'], indent=1))


if __name__ == '__main__':
    main()
