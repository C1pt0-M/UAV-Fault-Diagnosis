"""Source operating point v1: GBT deployment threshold under strict per-class recall >= T0,
excluding late-hover mode-4 GPS windows from the hard constraints.

Protocol: configs/experiments/source_operating_point_v1.json (frozen before results).
Zero training; train-CV held-out predictions only. No val/test/Real.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import e_ens_t_cv as e  # noqa: E402
from e_ens_t_cv import s, m  # noqa: E402
from q1_source_model_selection import Ranked  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
G = {}


def deploy(w):
    """Per-seed T0 targets -> GBT per-seed thresholds -> deployment threshold (min). Returns dict."""
    R, gbt = G['ranked'], G['gbt']
    t0, tgt, fp0, tg = {}, {}, {}, {}
    for sd in SEEDS:
        r0 = R[sd]
        t0[sd] = r0.r97(w)
        tgt[sd] = r0.tp(w, t0[sd])
        fp0[sd] = r0.fp(w, t0[sd])
        tg[sd] = gbt.threshold(w, tgt[sd], 7)
    td = min(tg.values())
    return {'t0': t0, 'targets': tgt, 'fp_t0': fp0, 'thr_seed': tg, 'thr': td, 'fp': gbt.fp(w, td),
            'fp_seed': {sd: gbt.fp(w, tg[sd]) for sd in SEEDS}}


def gm(d):
    return m.geometric_mean([d['fp'] / d['fp_t0'][sd] for sd in SEEDS])


def boot_worker(counts):
    inv = G['inv']
    return [gm(deploy(c[inv])) for c in counts]


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
    base, _ = e.load_scores(frame, outer)
    gate = {sd: m.MetricCache(y, base['T0'][sd]).basic()['fp'] for sd in SEEDS}
    if any(gate[sd] != e.W0_EXPECTED[sd] for sd in (42, 43, 44)):
        raise RuntimeError(f'T0 gate failed {gate}')
    gbt = base['GBT'][42]

    modes = pd.read_csv(ROOT / 'artifacts/q2_gps_fault_modes.csv').set_index('case_uid')
    uid = frame.case_uid.to_numpy()
    lgps = (y == 1) & late & (classes == 'GPS')
    mode = np.array([int(modes.loc[u, 'mode']) if l else 0 for u, l in zip(uid, lgps)])
    params = np.array([modes.loc[u, 'params'] if l else '' for u, l in zip(uid, lgps)], dtype=object)
    if (lgps & (mode == 0)).any():
        raise RuntimeError('late-hover GPS window without fault mode')
    excl = lgps & (mode == 4)
    zeq1 = excl & np.array([p.split()[2] == '1.' if p else False for p in params])
    ident = excl & (params == '1. 1. 1.')
    names = ['GPS_excl_late_mode4', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor', 'late_hover_GPS_mode3']
    strata = [(y == 1) & (classes == 'GPS') & ~excl] + \
             [(y == 1) & (classes == c) for c in names[1:6]] + [lgps & (mode == 3)]
    disclose = {'late_mode4_all': excl, 'late_mode4_z_ne_1': excl & ~zeq1, 'late_mode4_z_eq_1_non_identity': zeq1 & ~ident,
                'late_mode4_identity': ident, 'late_GPS_all': lgps}

    G.update(ranked={sd: Ranked(base['T0'][sd], y, strata) for sd in SEEDS}, gbt=Ranked(gbt, y, strata))
    ones = np.ones(len(y), dtype=np.int64)
    pt = deploy(ones)

    groups = frame.association_group.to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(20261106)
    counts = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(args.replicates)]
    G['inv'] = inv
    with mp.get_context('fork').Pool(args.workers) as pool:
        boot = [r for part in pool.map(boot_worker, [counts[i::args.workers] for i in range(args.workers)]) for r in part]

    pred = gbt >= pt['thr']
    rec = lambda p, mk: float(p[mk].mean())
    t0pred = {sd: base['T0'][sd] >= pt['t0'][sd] for sd in SEEDS}
    strata_tab = {n: {'n': int(mk.sum()), 'gbt': rec(pred, mk), 't0_seed': {str(sd): rec(t0pred[sd], mk) for sd in SEEDS},
                      't0_max': max(rec(t0pred[sd], mk) for sd in SEEDS)} for n, mk in zip(names, strata)}
    disc_tab = {n: {'n': int(mk.sum()), 'cases': int(len(set(uid[mk]))), 'gbt': rec(pred, mk),
                    't0_mean': float(np.mean([rec(t0pred[sd], mk) for sd in SEEDS]))} for n, mk in disclose.items()}
    late_norm = independent & late

    # out-of-fold transfer
    fp_g = 0; fp_0 = {sd: 0 for sd in SEEDS}
    tp_g = np.zeros(7); tp_0 = {sd: np.zeros(7) for sd in SEEDS}
    for k in range(3):
        wo, wk = (outer != k).astype(np.int64), (outer == k).astype(np.int64)
        d = deploy(wo)
        fp_g += G['gbt'].fp(wk, d['thr']); tp_g += G['gbt'].tp(wk, d['thr'])
        for sd in SEEDS:
            fp_0[sd] += G['ranked'][sd].fp(wk, d['t0'][sd]); tp_0[sd] += G['ranked'][sd].tp(wk, d['t0'][sd])
    nst = np.array([mk.sum() for mk in strata])
    oof = {'fp_gbt': int(fp_g), 'fp_t0': {str(sd): int(v) for sd, v in fp_0.items()},
           'fp_ratio_gm': m.geometric_mean([fp_g / fp_0[sd] for sd in SEEDS]),
           'recall_diff_pp_seed_mean': dict(zip(names, np.mean([100 * (tp_g - tp_0[sd]) / nst for sd in SEEDS], axis=0).tolist())),
           'recall_diff_pp_worst_seed': dict(zip(names, np.min([100 * (tp_g - tp_0[sd]) / nst for sd in SEEDS], axis=0).tolist()))}

    # descriptive references (per-seed thresholds, no min rule)
    refs = {}
    for arm in ('T1', 'E1'):
        r = []
        for sd in SEEDS:
            ra = Ranked(base[arm][sd], y, strata)
            r.append(ra.fp(ones, ra.threshold(ones, pt['targets'][sd], 7)) / pt['fp_t0'][sd])
        refs[arm] = {'ratio_per_seed': dict(zip(map(str, SEEDS), r)), 'ratio_gm': m.geometric_mean(r)}
    refs['GBT_per_seed_threshold'] = {'ratio_gm': m.geometric_mean([pt['fp_seed'][sd] / pt['fp_t0'][sd] for sd in SEEDS])}

    ratio = gm(pt)
    ci = m.percentile(boot)
    a2 = all(v >= -1 for v in oof['recall_diff_pp_seed_mean'].values())
    res = {'protocol': 'configs/experiments/source_operating_point_v1.json', 'population': int(len(y)), 'gate_t0': gate,
           'excluded_windows': int(excl.sum()),
           'deployment_threshold': pt['thr'], 'threshold_per_seed': {str(k): v for k, v in pt['thr_seed'].items()},
           'binding_seed': str(min(pt['thr_seed'], key=pt['thr_seed'].get)),
           'fp_gbt': pt['fp'], 'fp_t0': {str(k): v for k, v in pt['fp_t0'].items()},
           'fp_ratio_per_seed': {str(sd): pt['fp'] / pt['fp_t0'][sd] for sd in SEEDS},
           'fp_ratio_gm': ratio, 'fp_ratio_ci': ci,
           'n_normal': int((y == 0).sum()), 'late_hover_indep_normal': {'n': int(late_norm.sum()), 'fp_gbt': int(pred[late_norm].sum()),
                                                                      'fp_t0_mean': float(np.mean([t0pred[sd][late_norm].sum() for sd in SEEDS]))},
           'constrained_recall': strata_tab, 'disclosure_excluded_recall': disc_tab, 'oof': oof, 'references': refs,
           'acceptance': {'A1': bool(ratio < 1 and ci[1] < 1), 'A2': bool(a2)},
           'bootstrap': {'replicates': args.replicates, 'rng_seed': 20261106, 'groups': int(len(uniq))}}
    (out / 'evaluation.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
