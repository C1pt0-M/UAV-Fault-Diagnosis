"""Score ens_internal_test_v1 from saved test predictions (pure arithmetic)."""
import json
import numpy as np, pandas as pd

D = 'D:/UAV-Data/causal_input_v1/'
OUT = 'D:/UAV-Fault-Diagnosis/artifacts/ens_internal_test_v1_20261105/'
z = np.load(OUT + 'test_predictions.npz')
L = pd.read_csv(D + 'ledgers/test_included_windows.csv', low_memory=False)
assert (L.original_cache_index.values == z['original_indices']).all() and (L.qualified_y.values == z['labels']).all()
m0 = z['m0_probabilities']
L['M0'] = m0[:, 1] > m0[:, 0]
L['ENS'] = (m0[:, 1] + z['gbt_prob']) / 2 >= 0.5
L['GBT'] = z['gbt_prob'] >= 0.5
s = L.case_uid.str.split('/'); L['flight'], L['cls'] = s.str[1], s.str[2]
L['late'] = (L.flight == 'hover') & (L.window_end_row * 0.02 > 30)
n = L.qualified_y == 0; f = ~n; ind = n & (L.normal_origin == 'normal_flight')


def st(a):
    p = L[a]
    return {'all_normal_FP': int(p[n].sum()), 'all_normal_n': int(n.sum()),
            'fault_TP': int(p[f].sum()), 'fault_n': int(f.sum()),
            'late_hover_FP': int(p[ind & L.late].sum()), 'late_hover_n': int((ind & L.late).sum()),
            'late_hover_FP_by_domain': {k: int(p[ind & L.late & (L.domain == k)].sum()) for k in ['SIL', 'HIL']},
            'indep_FP_by_flight': {k: int(p[ind & (L.flight == k)].sum()) for k in sorted(L.flight.unique())},
            'indep_n_by_flight': {k: int((ind & (L.flight == k)).sum()) for k in sorted(L.flight.unique())},
            'prefault_FP': int(p[n & ~ind].sum()),
            'normal_FP_by_domain': {k: int(p[n & (L.domain == k)].sum()) for k in ['SIL', 'HIL']},
            'fault_TP_by_domain': {k: int(p[f & (L.domain == k)].sum()) for k in ['SIL', 'HIL']},
            'class_TP': {c: int(p[f & (L.cls == c)].sum()) for c in sorted(L.cls[f].unique())},
            'class_n': {c: int((f & (L.cls == c)).sum()) for c in sorted(L.cls[f].unique())},
            'class_TP_outside_late': {c: int(p[f & ~L.late & (L.cls == c)].sum()) for c in sorted(L.cls[f].unique())},
            'class_n_outside_late': {c: int((f & ~L.late & (L.cls == c)).sum()) for c in sorted(L.cls[f].unique())},
            'late_GPS_baro_TP': {c: (int(p[f & L.late & (L.cls == c)].sum()), int((f & L.late & (L.cls == c)).sum())) for c in ['GPS', 'barometer']},
            'case_indep_normal_any_FP': (int(L[ind].assign(p=p[ind]).groupby('case_uid').p.any().sum()), int(L[ind].case_uid.nunique())),
            'case_fault_any_TP': (int(L[f].assign(p=p[f]).groupby('case_uid').p.any().sum()), int(L[f].case_uid.nunique()))}


R = {a: st(a) for a in ['M0', 'GBT', 'ENS']}
B, P = R['M0'], R['ENS']
drop = {k: round(100 * (B['class_TP_outside_late'][k] - P['class_TP_outside_late'][k]) / B['class_n_outside_late'][k], 3) for k in B['class_TP']}
v = {'a': P['all_normal_FP'] < B['all_normal_FP'],
     'b': P['fault_TP'] >= B['fault_TP'] - 0.005 * B['fault_n'],
     'c': P['late_hover_FP'] < B['late_hover_FP'],
     'd': max(drop.values()) <= 1.0, 'class_drop_outside_late_pp': drop}
v['PASS'] = bool(v['a'] and v['b'] and v['c'] and v['d'])
R['verdict'] = v
json.dump(R, open(OUT + 'evaluation.json', 'w'), indent=2)
rows = ['all_normal_FP', 'fault_TP', 'late_hover_FP', 'prefault_FP']
print(pd.DataFrame({a: {r: R[a][r] for r in rows} for a in ['M0', 'GBT', 'ENS']}).to_string())
print('n: normal', B['all_normal_n'], 'fault', B['fault_n'], 'late hover normal', B['late_hover_n'])
for k in ['late_hover_FP_by_domain', 'normal_FP_by_domain', 'fault_TP_by_domain', 'indep_FP_by_flight', 'class_TP', 'late_GPS_baro_TP', 'case_indep_normal_any_FP', 'case_fault_any_TP']:
    print(k, {a: R[a][k] for a in ['M0', 'ENS']})
print('indep n by flight', B['indep_n_by_flight'], 'class n', B['class_n'])
print(json.dumps(v, indent=1))
