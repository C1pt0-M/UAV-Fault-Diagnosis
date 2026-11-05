"""Separability of late-hover normal vs fault windows (read-only, no training).

Population: source val, hover directory, window end > 30 s, supervised-valid.
Normal = independent normal_flight cases; fault = endpoint-labelled fault windows.
1) Model-score AUC per fault class: M0 B best, H1 best, H1 epoch 49.
2) Per-channel AUC of simple window statistics (mean, std) per fault class,
   using unnormalized windows from causal_input_v1/windows (train-only norm not
   needed for rank statistics). Direction-free AUC = max(a, 1-a).
Also the same channel AUCs on train (more cases) as a stability check.
"""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = 'D:/UAV-Fault-Diagnosis/artifacts/'
DATA = 'D:/UAV-Data/causal_input_v1/'
OUT = ROOT + 'hover_separability_v1_20261105/'
cols = json.load(open(DATA + 'input_contract.json'))['columns']


def population(split):
    L = pd.read_csv(DATA + f'ledgers/{split}_included_windows.csv', low_memory=False)
    p = L.case_uid.str.split('/')
    L['flight'], L['cls'] = p.str[1], p.str[2]
    L = L[(L.flight == 'hover') & (L.window_end_row * 0.02 > 30)]
    n = L[(L.qualified_y == 0) & (L.normal_origin == 'normal_flight')].assign(cls='normal')
    f = L[L.qualified_y == 1]
    return pd.concat([n, f])


def features(P):
    feats = []
    cache = {}
    for _, r in P.iterrows():
        k = r.case_uid.replace('/', '__')
        if k not in cache:
            z = np.load(DATA + 'windows/' + k + '.npz')
            cache = {k: dict(zip(z['original_cache_index'].tolist(), z['x']))}
        x = cache[k][r.original_cache_index]
        feats.append(np.concatenate([x.mean(1), x.std(1)]))
    return np.array(feats)


def auc(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


res = {}
for split in ['val', 'train']:
    P = population(split).reset_index(drop=True)
    F = features(P)
    names = [c + '_mean' for c in cols] + [c + '_std' for c in cols]
    rows = {}
    for cls in sorted(set(P.cls) - {'normal'}):
        m = P.cls.isin(['normal', cls]).to_numpy()
        y = (P.cls[m] == cls).astype(int).to_numpy()
        a = np.array([auc(y, F[m, j]) for j in range(F.shape[1])])
        top = np.argsort(-a)[:5]
        rows[cls] = {'n_fault': int(y.sum()), 'n_normal': int((1 - y).sum()),
                     'fault_cases': int(P[m & (P.cls == cls).to_numpy()].case_uid.nunique()),
                     'best_channel_auc': float(a.max()),
                     'top5': [(names[j], round(float(a[j]), 3)) for j in top]}
    res[split] = {'normal_windows': int((P.cls == 'normal').sum()), 'normal_cases': int(P[P.cls == 'normal'].case_uid.nunique()), 'classes': rows}

# model-score AUC on val
L = pd.read_csv(DATA + 'ledgers/val_all_windows.csv', low_memory=False).set_index('original_cache_index')
Pv = population('val').set_index('original_cache_index')
scores = {'M0_B': ROOT + 'causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/source_validation_predictions.npz',
          'H1_best': ROOT + 'h1_hover_coverage_v1_20261105/run/source_validation_predictions.npz',
          'H1_ep49': ROOT + 'h1_hover_coverage_v1_20261105/diag_last_epoch49.npz'}
model = {}
for name, path in scores.items():
    z = np.load(path)
    s = pd.Series(z['probabilities'][:, 1], index=z['original_indices'])
    sub = Pv.assign(score=s.reindex(Pv.index).values)
    model[name] = {}
    for cls in sorted(set(sub.cls) - {'normal'}):
        m = sub.cls.isin(['normal', cls])
        y = (sub.cls[m] == cls).astype(int)
        model[name][cls] = round(float(roc_auc_score(y, sub.score[m])), 3)
res['model_score_auc_val'] = model

import os
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(OUT + 'separability.json', 'w'), indent=2)
print('model-score AUC (val, late hover, fault class vs independent normal):')
print(pd.DataFrame(model))
for split in ['val', 'train']:
    r = res[split]
    print(f'\n{split}: normal windows {r["normal_windows"]} / cases {r["normal_cases"]}')
    for cls, v in r['classes'].items():
        print(f'  {cls:14s} fault {v["n_fault"]:5d} ({v["fault_cases"]:3d} cases)  best single-stat AUC {v["best_channel_auc"]:.3f}  {v["top5"][:3]}')
