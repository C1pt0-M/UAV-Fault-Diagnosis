"""Q2 Part A: is late-hover GPS fault information present in the 26 input channels?

Protocol: configs/experiments/q2_late_hover_gps_observability_v1.json (frozen before results).
CPU probes on train only. No val/test/Real.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import e_ens_t_cv as e  # noqa: E402
from e_ens_t_cv import s, m  # noqa: E402

DT = 0.02
GROUPS = {'actuator_ctrl': range(0, 4), 'actuator_out': range(4, 8), 'gyro': range(8, 11),
          'accel': [11, 12, 13, 25], 'baro': [14], 'attitude': range(15, 19), 'velocity': range(19, 22),
          'mag': range(22, 25)}
BANDS = [(1, 3), (3, 9), (9, 24), (24, 76)]  # rfft bins at 1/3 Hz: 0.33-1, 1-3, 3-8, 8-25 Hz


def f0(x):
    d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


def f1_extra(x):
    t = np.arange(x.shape[2]) - (x.shape[2] - 1) / 2
    slope = (x * t).sum(2) / (t ** 2).sum()
    q = np.quantile(x, [0.1, 0.5, 0.9], axis=2)
    spec = np.abs(np.fft.rfft(x - x.mean(2, keepdims=True), axis=2)) ** 2
    bands = [np.log1p(spec[:, :, a:b].sum(2)) for a, b in BANDS]
    return np.concatenate([q[0], q[1], q[2], slope, x[:, :, -1] - x[:, :, 0]] + bands, axis=1)


def physics(x):
    w, qx, qy, qz = x[:, 15], x[:, 16], x[:, 17], x[:, 18]
    f = x[:, 11:14]
    R = np.stack([np.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - w*qz), 2 * (qx*qz + w*qy)], 1),
                  np.stack([2 * (qx*qy + w*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - w*qx)], 1),
                  np.stack([2 * (qx*qz - w*qy), 2 * (qy*qz + w*qx), 1 - 2 * (qx**2 + qy**2)], 1)], 1)  # (n,3,3,T)
    a = np.einsum('nijt,njt->nit', R, f)
    a[:, 2] += 9.80665
    v = x[:, 19:22]
    k = 5
    dv = (v[:, :, k:] - v[:, :, :-k]) / (k * DT)
    am = np.stack([np.convolve(np.ones(k) / k, row, mode='valid')[:dv.shape[2]] for row in a.reshape(-1, a.shape[2])]).reshape(a.shape[0], 3, -1)
    r = dv - am
    dvw = v[:, :, -1] - v[:, :, 0] - a[:, :, :-1].sum(2) * DT
    b = x[:, 14]
    rb = (b[:, k:] - b[:, :-k]) / (k * DT) + (v[:, 2, k:] + v[:, 2, :-k]) / 2
    db = b[:, -1] - b[:, 0] + v[:, 2, :-1].sum(1) * DT
    hs = np.hypot(v[:, 0], v[:, 1])
    tilt = np.arccos(np.clip(1 - 2 * (qx**2 + qy**2), -1, 1))
    return np.concatenate([r.mean(2), r.std(2), np.abs(r).max(2), dvw, rb.mean(1)[:, None], rb.std(1)[:, None],
                           db[:, None], hs.mean(1)[:, None], hs.max(1)[:, None], tilt.mean(1)[:, None],
                           tilt.std(1)[:, None]], axis=1)


def features(x):
    x = x.astype(np.float64)
    a, b, c = f0(x), f1_extra(x), physics(x)
    return a, b, c


def group_cols(ch):
    """F1 column indices (F0 130 + extra 234) belonging to channels ch."""
    ch = list(ch)
    cols = [k * 26 + i for k in range(5) for i in ch] + [130 + k * 26 + i for k in range(9) for i in ch]
    return np.array(cols)


def fold_of(g):
    return int(hashlib.sha256(('h1cv|' + g).encode()).hexdigest(), 16) % 3


def probe(X, y, folds):
    p = np.full(len(y), np.nan)
    for f in range(3):
        tr = folds != f
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=0).fit(X[tr], y[tr], sample_weight=w)
        p[~tr] = mdl.predict_proba(X[~tr])[:, 1]
    return p


def recall_at_fpr(score, y, fpr):
    t = np.quantile(score[y == 0], 1 - fpr)
    return float((score[y == 1] > t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--modes', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--replicates', type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    frame = s.read_ledger(args.data)
    _, outer = s.split_ledger(frame, 0)
    y_all, flight, classes, independent, late = m.strata(frame)
    end_s = frame.window_end_row.to_numpy() * DT
    modes = pd.read_csv(args.modes).set_index('case_uid')
    pos_mask = late & (classes == 'GPS') & (y_all == 1)
    neg_mask = late & independent
    idx = np.flatnonzero(pos_mask | neg_mask)
    uid = frame.case_uid.to_numpy()
    par = modes.reindex(uid[idx])
    stratum = np.full(len(idx), 'normal', dtype=object)
    ispos = pos_mask[idx]
    mode = par['mode'].to_numpy(); prm = par['params'].fillna('').to_numpy()
    zf = np.array([p.split()[2] if p else '' for p in prm])
    stratum[ispos & (mode == 3)] = 'mode3'
    stratum[ispos & (mode == 4) & (zf != '1.')] = 'mode4_z_ne_1'
    stratum[ispos & (mode == 4) & (zf == '1.')] = 'mode4_z_eq_1'
    if (stratum[ispos] == 'normal').any():
        raise RuntimeError('unmapped GPS case')
    identity = ispos & (prm == '1. 1. 1.')

    # confound control windows
    cpos = (flight == 'hover') & (classes == 'GPS') & (y_all == 0) & (end_s >= 10) & (end_s <= 21.7)
    cneg = (flight == 'hover') & independent & (end_s >= 10) & (end_s <= 21.7)
    cidx = np.flatnonzero(cpos | cneg)

    data = torch.load(Path(args.data) / 'cache/train.pt', weights_only=True, mmap=True)
    if not (data['original_cache_index'].numpy() == frame.original_cache_index.to_numpy()).all():
        raise RuntimeError('cache/ledger order mismatch')
    X0, X1, XP = features(data['x'][torch.from_numpy(idx)].numpy())
    C0, C1, CP = features(data['x'][torch.from_numpy(cidx)].numpy())
    F = {'F0': X0, 'F1': np.hstack([X0, X1]), 'F2': np.hstack([X0, X1, XP])}
    y = ispos.astype(int)
    groups = frame.association_group.to_numpy()[idx]
    folds = np.array([fold_of(g) for g in groups])
    if not (folds == outer[idx]).all():
        raise RuntimeError('fold mismatch')

    probes = {k: probe(v, y, folds) for k, v in F.items()}
    for g, ch in GROUPS.items():
        probes[f'group_{g}'] = probe(F['F1'][:, group_cols(ch)], y, folds)
    probes['group_physics'] = probe(XP, y, folds)
    # stratum-specific F2 probes (clarification added before results: 'best probe' includes them)
    for st in ('mode3', 'mode4_z_ne_1', 'mode4_z_eq_1'):
        keep = (stratum == st) | (y == 0)
        p = np.full(len(y), np.nan)
        p[keep] = probe(F['F2'][keep], y[keep], folds[keep])
        probes[f'F2_only_{st}'] = p

    base, _ = e.load_scores(frame, outer)
    existing = {'T0': {sd: base['T0'][sd][idx] for sd in e.SEEDS}, 'T1': {sd: base['T1'][sd][idx] for sd in e.SEEDS},
                'GBT': {42: base['GBT'][42][idx]}, 'E1': {sd: base['E1'][sd][idx] for sd in e.SEEDS}}

    cy = cpos[cidx].astype(int)
    cfold = outer[cidx]
    cprobe = probe(np.hstack([C0, C1, CP]), cy, cfold)

    strata = {'all': ispos, 'mode3': stratum == 'mode3', 'mode4_z_ne_1': stratum == 'mode4_z_ne_1',
              'mode4_z_eq_1': stratum == 'mode4_z_eq_1', 'mode4_identity': identity}
    uniq, inv = np.unique(groups, return_inverse=True)
    cuniq, cinv = np.unique(frame.association_group.to_numpy()[cidx], return_inverse=True)
    rng = np.random.default_rng(20261106)
    plans = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv] for _ in range(args.replicates)]
    rng2 = np.random.default_rng(20261106)
    cplans = [np.bincount(rng2.integers(0, len(cuniq), len(cuniq)), minlength=len(cuniq))[cinv] for _ in range(args.replicates)]

    def auc(score, sel, w=None):
        mk = (sel | (y == 0)) & np.isfinite(score)
        ww = None if w is None else w[mk]
        if ww is not None and (ww[y[mk] == 1].sum() == 0 or ww[y[mk] == 0].sum() == 0):
            return np.nan
        return float(roc_auc_score(y[mk], score[mk], sample_weight=ww))

    def model_auc(name, sel, w=None):
        return float(np.mean([auc(v, sel, w) for v in existing[name].values()]))

    res = {'protocol': 'configs/experiments/q2_late_hover_gps_observability_v1.json',
           'counts': {k: {'windows': int(v.sum()), 'cases': int(len(set(uid[idx][v])))} for k, v in strata.items()},
           'normals': {'windows': int((y == 0).sum()), 'cases': int(len(set(uid[idx][y == 0])))},
           'strata': {}}
    for st, sel in strata.items():
        r = {'probes': {}, 'existing': {}}
        for name, sc in probes.items():
            if name.startswith('F2_only_') and name != f'F2_only_{st}' and not (st == 'mode4_identity' and name == 'F2_only_mode4_z_eq_1'):
                continue
            boot = [auc(sc, sel, w) for w in plans]
            mk = (sel | (y == 0)) & np.isfinite(sc)
            r['probes'][name] = {'auc': auc(sc, sel), 'ci': m.percentile([b for b in boot if np.isfinite(b)]),
                                 'recall_fpr5': recall_at_fpr(sc[mk], y[mk], 0.05), 'recall_fpr10': recall_at_fpr(sc[mk], y[mk], 0.10)}
        for name in existing:
            r['existing'][name] = {'auc': model_auc(name, sel)}
        best_p = max(r['probes'], key=lambda k: r['probes'][k]['auc'])
        best_m = max(r['existing'], key=lambda k: r['existing'][k]['auc'])
        diff = [auc(probes[best_p], sel, w) - model_auc(best_m, sel, w) for w in plans]
        r['best_probe'] = best_p
        r['best_existing'] = best_m
        r['probe_minus_existing'] = {'point': r['probes'][best_p]['auc'] - r['existing'][best_m]['auc'],
                                     'ci': m.percentile([d for d in diff if np.isfinite(d)])}
        res['strata'][st] = r
    cboot = [float(roc_auc_score(cy, cprobe, sample_weight=w)) for w in cplans]
    res['confound_control'] = {'auc': float(roc_auc_score(cy, cprobe)), 'ci': m.percentile(cboot),
                               'pos_windows': int(cy.sum()), 'neg_windows': int((cy == 0).sum())}
    # verdicts
    conf_ok = res['confound_control']['auc'] <= 0.70
    for st, r in res['strata'].items():
        bp = r['probes'][r['best_probe']]
        if bp['auc'] >= 0.90 and bp['ci'][0] >= 0.85 and conf_ok:
            v = 'info_present'
        elif bp['ci'][1] < 0.75:
            v = 'not_observable_26ch'
        else:
            v = 'partial'
        r['verdict_26ch'] = v
        r['model_capability_gap'] = bool(v == 'info_present' and r['probe_minus_existing']['ci'][0] > 0.05)
    np.savez_compressed(out / 'probe_scores.npz', idx=idx, y=y, stratum=stratum.astype(str),
                        **{k: v for k, v in probes.items()})
    (out / 'evaluation.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    for st, r in res['strata'].items():
        print(st, r['verdict_26ch'], 'best', r['best_probe'], round(r['probes'][r['best_probe']]['auc'], 3),
              [round(c, 3) for c in r['probes'][r['best_probe']]['ci']], 'existing', r['best_existing'],
              round(r['existing'][r['best_existing']]['auc'], 3), 'gap', round(r['probe_minus_existing']['point'], 3))
    print('confound', res['confound_control'])


if __name__ == '__main__':
    main()
