"""Apply m0s_stats_branch_v1 pass rule: M0S held-out vs M0 held-out (feature_groupcv_v1)."""
import json
import numpy as np, pandas as pd

DATA = 'D:/UAV-Data/causal_input_v1/'
ART = 'D:/UAV-Fault-Diagnosis/artifacts/'
CV = ART + 'feature_groupcv_v1_20261105/'
OUT = ART + 'm0s_stats_branch_v1_20261105/'

L = pd.read_csv(DATA + 'ledgers/train_included_windows.csv', low_memory=False)
pos = {v: i for i, v in enumerate(L.original_cache_index)}
g = np.load(CV + 'gbt_heldout.npz'); L['fold'] = g['fold']
for arm, root in [('M0', CV), ('M0S', OUT)]:
    col = np.full(len(L), np.nan)
    for f in range(3):
        z = np.load(root + f'fold{f}/heldout_predictions.npz')
        ii = [pos[v] for v in z['original_indices']]
        assert (L.fold.values[ii] == f).all()
        col[ii] = z['probabilities'][:, 1] - z['probabilities'][:, 0]  # >0 means argmax fault
    assert np.isfinite(col).all(); L[arm] = col > 0
s = L.case_uid.str.split('/'); L['flight'], L['cls'] = s.str[1], s.str[2]; L['t'] = L.window_end_row * 0.02
L['late'] = (L.flight == 'hover') & (L.t > 30)


def st(d, a):
    n = d.qualified_y == 0; f = ~n; ind = n & (d.normal_origin == 'normal_flight')
    return {'all_normal_FP': int(d[a][n].sum()), 'fault_TP': int(d[a][f].sum()), 'fault_n': int(f.sum()),
            'late_hover_FP': int(d[a][ind & d.late].sum()), 'late_hover_n': int((ind & d.late).sum()),
            'late_hover_FP_by_domain': {k: int(d[a][ind & d.late & (d.domain == k)].sum()) for k in ['SIL', 'HIL']},
            'indep_nonhover_FP': int(d[a][ind & (d.flight != 'hover')].sum()),
            'class_TP_outside_late': {c: int(d[a][f & ~d.late & (d.cls == c)].sum()) for c in sorted(d.cls[f].unique())},
            'class_n_outside_late': {c: int((f & ~d.late & (d.cls == c)).sum()) for c in sorted(d.cls[f].unique())},
            'class_TP_late': {c: int(d[a][f & d.late & (d.cls == c)].sum()) for c in sorted(d.cls[f].unique())},
            'class_n_late': {c: int((f & d.late & (d.cls == c)).sum()) for c in sorted(d.cls[f].unique())}}


res = {a: {'pooled': st(L, a), 'folds': {f: st(L[L.fold == f], a) for f in range(3)}} for a in ['M0', 'M0S']}
B, P = res['M0'], res['M0S']
a = all(P['folds'][f]['all_normal_FP'] < B['folds'][f]['all_normal_FP'] for f in range(3))
b = all(P['folds'][f]['fault_TP'] >= B['folds'][f]['fault_TP'] - 0.005 * B['folds'][f]['fault_n'] for f in range(3))
c = P['pooled']['late_hover_FP'] < B['pooled']['late_hover_FP']
drop = {k: 100 * (B['pooled']['class_TP_outside_late'][k] - P['pooled']['class_TP_outside_late'][k]) / B['pooled']['class_n_outside_late'][k]
        for k in B['pooled']['class_TP_outside_late']}
d = max(drop.values()) <= 1.0
late = {k: (B['pooled']['class_TP_late'][k], P['pooled']['class_TP_late'][k], B['pooled']['class_n_late'][k]) for k in ['GPS', 'barometer']}
res['verdict'] = {'a': a, 'b': b, 'c': c, 'd': d, 'class_drop_outside_late_pp': {k: round(v, 3) for k, v in drop.items()},
                  'e_late_hover_GPS_baro_TP(M0,M0S,n)': late, 'PASS': bool(a and b and c and d)}
json.dump(res, open(OUT + 'evaluation.json', 'w'), indent=2)
for k in ['all_normal_FP', 'fault_TP', 'late_hover_FP', 'indep_nonhover_FP']:
    print(f'{k:20s} M0 {B["pooled"][k]:>6}  M0S {P["pooled"][k]:>6}')
print('late hover FP by domain', B['pooled']['late_hover_FP_by_domain'], P['pooled']['late_hover_FP_by_domain'])
for f in range(3):
    print('fold', f, 'normal FP', B['folds'][f]['all_normal_FP'], '->', P['folds'][f]['all_normal_FP'], ' fault TP', B['folds'][f]['fault_TP'], '->', P['folds'][f]['fault_TP'], '/', B['folds'][f]['fault_n'])
print(json.dumps(res['verdict'], indent=1))
