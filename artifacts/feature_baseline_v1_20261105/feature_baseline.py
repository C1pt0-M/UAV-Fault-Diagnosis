"""Global feature baseline on full source train/val (CPU, no neural training).

Question: is M0's late-hover failure due to the representation (raw-sequence
CNN) or the training population (late hover ~= fault)? 2x2:
  features: window statistics (130-d, fixed)
  population: original train  vs  train + H1 refill windows
Model: HistGradientBoosting, Balanced-equivalent sample weights (same pool
masses as M0 B / H1). Evaluated on full source val 34710 with fixed 0.5 and with
the same stratified counts as the H1 report.
"""
import json, time
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = 'D:/UAV-Data/causal_input_v1/'
REFILL = 'D:/UAV-Data/h1_hover_refill_v1/'
OUT = 'D:/UAV-Fault-Diagnosis/artifacts/feature_baseline_v1_20261105/'
ART = 'D:/UAV-Fault-Diagnosis/artifacts/'


def stats(x):
    x = x.astype(np.float32)
    d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


def cache_features(split):
    p = torch.load(DATA + f'cache/{split}.pt', weights_only=True, mmap=True)
    x = p['x']; out = []
    for i in range(0, len(x), 4096):
        out.append(stats(x[i:i + 4096].numpy()))
    return np.concatenate(out), p['y'].numpy(), p['original_cache_index'].numpy()


def ledger(split):
    L = pd.read_csv(DATA + f'ledgers/{split}_included_windows.csv', low_memory=False)
    s = L.case_uid.str.split('/')
    L['flight'], L['cls'] = s.str[1], s.str[2]
    L['t'] = L.window_end_row * 0.02
    return L


def balanced_weights(y, origin, normal_mass):
    w = np.ones(len(y))
    for o in ['normal_flight', 'normal_part_of_fault_case']:
        m = origin == o
        w[m] = 0.5 * normal_mass / m.sum()
    return w


def evaluate(V, pred, prob):
    n = V[V.qualified_y == 0]; f = V[V.qualified_y == 1]
    ind = n[n.normal_origin == 'normal_flight']
    r = {
        'late_hover_normal_FP': int(pred[ind[(ind.flight == 'hover') & (ind.t > 30)].index].sum()),
        'hover_le30_normal_FP': int(pred[ind[(ind.flight == 'hover') & (ind.t <= 30)].index].sum()),
        'indep_nonhover_FP': int(pred[ind[ind.flight != 'hover'].index].sum()),
        'all_normal_FP': int(pred[n.index].sum()),
        'fault_TP': int(pred[f.index].sum()),
        'hover_late_fault_TP': int(pred[f[(f.flight == 'hover') & (f.t > 30)].index].sum()),
        'class_TP': {c: int(pred[g.index].sum()) for c, g in f.groupby('cls')},
        'auc': float(roc_auc_score(V.qualified_y, prob)),
    }
    lh = V[((V.normal_origin == 'normal_flight') | (V.qualified_y == 1)) & (V.flight == 'hover') & (V.t > 30)]
    r['late_hover_class_auc'] = {c: round(float(roc_auc_score((g.qualified_y == 1).astype(int), prob[g.index])), 3)
                                 for c, g in [(c, lh[lh.cls.isin([c, 'no_fault'])]) for c in sorted(set(lh.cls) - {'no_fault'})]}
    return r


t0 = time.time()
XT, yT, iT = cache_features('train'); XV, yV, iV = cache_features('val')
LT, LV = ledger('train'), ledger('val')
assert (LT.original_cache_index.values == iT).all() and (LV.original_cache_index.values == iV).all()
assert (LT.qualified_y.values == yT).all() and (LV.qualified_y.values == yV).all()
xr = np.load(REFILL + 'x.npy', mmap_mode='r')
XR = np.concatenate([stats(np.asarray(xr[i:i + 4096])) for i in range(0, len(xr), 4096)])
print('features built', round(time.time() - t0), 's', XT.shape, XV.shape, XR.shape, flush=True)

normal_mass = float((yT == 0).sum())
oT = LT.normal_origin.values
runs = {
    'GBT_orig': (XT, yT, balanced_weights(yT, oT, normal_mass)),
    'GBT_refill': (np.concatenate([XT, XR]), np.concatenate([yT, np.zeros(len(XR), int)]),
                   balanced_weights(np.concatenate([yT, np.zeros(len(XR), int)]),
                                    np.concatenate([oT, np.array(['normal_flight'] * len(XR))]), normal_mass)),
}
results = {}
for name, (X, y, w) in runs.items():
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
    prob = m.predict_proba(XV)[:, 1]
    results[name] = evaluate(LV, (prob >= 0.5).astype(int), prob)
    np.save(OUT + f'{name}_val_prob.npy', prob)
    print(name, 'done', round(time.time() - t0), 's', flush=True)

for name, path in {'M0_B': ART + 'causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/source_validation_predictions.npz',
                   'H1_best': ART + 'h1_hover_coverage_v1_20261105/run/source_validation_predictions.npz',
                   'H1_ep49': ART + 'h1_hover_coverage_v1_20261105/diag_last_epoch49.npz'}.items():
    z = np.load(path); assert (z['original_indices'] == iV).all()
    prob = z['probabilities'][:, 1]
    results[name] = evaluate(LV, (z['probabilities'][:, 1] > z['probabilities'][:, 0]).astype(int), prob)

json.dump(results, open(OUT + 'results.json', 'w'), indent=2)
keys = ['late_hover_normal_FP', 'hover_le30_normal_FP', 'indep_nonhover_FP', 'all_normal_FP', 'fault_TP', 'hover_late_fault_TP', 'auc']
print(pd.DataFrame({k: {c: results[k][c] for c in keys} for k in results}).to_string())
print(pd.DataFrame({k: results[k]['class_TP'] for k in results}).to_string())
print(pd.DataFrame({k: results[k]['late_hover_class_auc'] for k in results}).to_string())
