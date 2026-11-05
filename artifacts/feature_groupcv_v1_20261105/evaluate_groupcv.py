"""Apply feature_groupcv_v1 pass rule to held-out M0, GBT and ENS predictions."""
import json
import numpy as np, pandas as pd

DATA = 'D:/UAV-Data/causal_input_v1/'
OUT = 'D:/UAV-Fault-Diagnosis/artifacts/feature_groupcv_v1_20261105/'
L = pd.read_csv(DATA + 'ledgers/train_included_windows.csv', low_memory=False)
g = np.load(OUT + 'gbt_heldout.npz')
assert (L.original_cache_index.values == g['original_indices']).all()
L['fold'] = g['fold']; L['GBT'] = g['prob']
m0 = np.full(len(L), np.nan); pos = {v: i for i, v in enumerate(L.original_cache_index)}
for f in range(3):
    z = np.load(OUT + f'fold{f}/heldout_predictions.npz')
    ii = [pos[v] for v in z['original_indices']]
    assert (L.fold.values[ii] == f).all()
    m0[ii] = z['probabilities'][:, 1]
assert np.isfinite(m0).all()
L['M0'] = m0; L['ENS'] = (L.M0 + L.GBT) / 2
s = L.case_uid.str.split('/'); L['flight'], L['cls'] = s.str[1], s.str[2]; L['t'] = L.window_end_row * 0.02


def stats(d, arm):
    p = (d[arm] >= 0.5) if arm != 'M0' else (d.M0 > 0.5)
    n = d.qualified_y == 0; f = ~n
    ind = n & (d.normal_origin == 'normal_flight')
    late = ind & (d.flight == 'hover') & (d.t > 30)
    return {'all_normal_FP': int(p[n].sum()), 'fault_TP': int(p[f].sum()), 'fault_n': int(f.sum()),
            'late_hover_FP': int(p[late].sum()), 'late_hover_n': int(late.sum()),
            'indep_nonhover_FP': int(p[ind & (d.flight != 'hover')].sum()),
            'class_TP': {c: int(p[f & (d.cls == c)].sum()) for c in sorted(d.cls[f].unique())},
            'class_n': {c: int((f & (d.cls == c)).sum()) for c in sorted(d.cls[f].unique())},
            'domain_late_hover_FP': {k: int(p[late & (d.domain == k)].sum()) for k in ['SIL', 'HIL']}}


res = {'per_fold': {}, 'pooled': {}}
for arm in ['M0', 'GBT', 'ENS']:
    res['pooled'][arm] = stats(L, arm)
    res['per_fold'][arm] = {f: stats(L[L.fold == f], arm) for f in range(3)}
verdict = {}
for arm in ['GBT', 'ENS']:
    a = all(res['per_fold'][arm][f]['all_normal_FP'] < res['per_fold']['M0'][f]['all_normal_FP'] for f in range(3))
    b = all(res['per_fold'][arm][f]['fault_TP'] >= res['per_fold']['M0'][f]['fault_TP'] - 0.005 * res['per_fold']['M0'][f]['fault_n'] for f in range(3))
    c = res['pooled'][arm]['late_hover_FP'] < res['pooled']['M0']['late_hover_FP']
    P, B = res['pooled'][arm], res['pooled']['M0']
    drops = {k: 100 * (B['class_TP'][k] - P['class_TP'][k]) / B['class_n'][k] for k in B['class_TP']}
    d = max(drops.values()) <= 1.0
    verdict[arm] = {'a_all_normal_FP_lower_each_fold': a, 'b_fault_TP_within_0.5pct_each_fold': b,
                    'c_pooled_late_hover_FP_lower': c, 'd_class_drop_le_1pp': d,
                    'class_drop_pp': {k: round(v, 3) for k, v in drops.items()}, 'PASS': bool(a and b and c and d)}
res['verdict'] = verdict
json.dump(res, open(OUT + 'evaluation.json', 'w'), indent=2)
rows = ['all_normal_FP', 'fault_TP', 'late_hover_FP', 'indep_nonhover_FP']
print('pooled held-out (train, 3 folds):')
print(pd.DataFrame({a: {r: res['pooled'][a][r] for r in rows} for a in ['M0', 'GBT', 'ENS']}).to_string())
print('late hover FP by domain', {a: res['pooled'][a]['domain_late_hover_FP'] for a in ['M0', 'GBT', 'ENS']}, 'n', res['pooled']['M0']['late_hover_n'])
print(pd.DataFrame({a: res['pooled'][a]['class_TP'] for a in ['M0', 'GBT', 'ENS']}).assign(n=res['pooled']['M0']['class_n']).to_string())
print('per fold all_normal_FP / fault_TP:')
for f in range(3):
    print(f, {a: (res['per_fold'][a][f]['all_normal_FP'], res['per_fold'][a][f]['fault_TP']) for a in ['M0', 'GBT', 'ENS']}, 'fault_n', res['per_fold']['M0'][f]['fault_n'])
for a, v in verdict.items():
    print(a, v)
