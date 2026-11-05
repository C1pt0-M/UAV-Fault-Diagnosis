"""Evaluate H1 against M0 B with the frozen success criteria (read-only)."""
import json
import numpy as np
import pandas as pd

ROOT = 'D:/UAV-Fault-Diagnosis/artifacts/'
L = pd.read_csv('D:/UAV-Data/causal_input_v1/ledgers/val_all_windows.csv', low_memory=False).set_index('original_cache_index')
RUNS = {'M0_B': ROOT + 'causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal',
        'H1': ROOT + 'h1_hover_coverage_v1_20261105/run'}


def load(path):
    z = np.load(path + '/source_validation_predictions.npz')
    d = L.loc[z['original_indices']].copy()
    assert (d.qualified_y.values == z['labels']).all() and d.supervised_valid.all() and len(d) == 34710
    p = z['probabilities']
    d['p'] = p[:, 1]
    d['pred'] = (p[:, 1] > p[:, 0]).astype(int)
    d['t'] = d.window_end_row * 0.02
    parts = d.case_uid.str.split('/')
    d['flight'] = parts.str[1]; d['cls'] = parts.str[2]
    return d


def summary(d):
    n = d[d.qualified_y == 0]; f = d[d.qualified_y == 1]
    ind = n[n.normal_origin == 'normal_flight']
    late = ind[(ind.flight == 'hover') & (ind.t > 30)]
    hf = f[(f.flight == 'hover') & (f.t > 30)]
    tp = lambda x: int(x.pred.sum())
    return {
        'late_hover_FP': tp(late), 'late_hover_n': len(late),
        'late_hover_case_fpr': late.groupby('case_uid').pred.mean().to_dict(),
        'hover_le30_FP': tp(ind[(ind.flight == 'hover') & (ind.t <= 30)]),
        'fault_TP': tp(f), 'fault_n': len(f),
        'class_TP': f.groupby('cls').pred.sum().to_dict(), 'class_n': f.groupby('cls').size().to_dict(),
        'domain_TP': f.groupby('domain').pred.sum().to_dict(), 'domain_n': f.groupby('domain').size().to_dict(),
        'hover_late_fault_TP': tp(hf), 'hover_late_fault_n': len(hf),
        'indep_FP_by_flight': ind.groupby('flight').pred.sum().to_dict(),
        'indep_nonhover_FP': tp(ind[ind.flight != 'hover']),
        'all_normal_FP': tp(n), 'all_normal_n': len(n),
        'prefault_FP': tp(n[n.normal_origin != 'normal_flight']),
        'normal_FP_by_domain': n.groupby('domain').pred.sum().to_dict(),
    }


S = {k: summary(load(v)) for k, v in RUNS.items()}
b, h = S['M0_B'], S['H1']
pp = lambda a, n: 100 * a / n
checks = {}
checks['1_late_hover_FP<=140'] = (h['late_hover_FP'], h['late_hover_FP'] <= 140)
lower = [c for c in b['late_hover_case_fpr'] if h['late_hover_case_fpr'][c] < b['late_hover_case_fpr'][c]]
pairs = {}
for c in b['late_hover_case_fpr']:
    key = c.split('/')[-1].split('_')[1]
    pairs.setdefault(key, []).append(h['late_hover_case_fpr'][c] < b['late_hover_case_fpr'][c])
pair_both = sum(all(v) and len(v) == 2 for v in pairs.values())
checks['2_cases_lower>=8_pairs_both>=4'] = ((len(lower), pair_both), len(lower) >= 8 and pair_both >= 4)
checks['3_fault_TP>=16343'] = (h['fault_TP'], h['fault_TP'] >= 16343)
cls_drop = {c: pp(b['class_TP'][c] - h['class_TP'][c], b['class_n'][c]) for c in b['class_TP']}
dom_drop = {c: pp(b['domain_TP'][c] - h['domain_TP'][c], b['domain_n'][c]) for c in b['domain_TP']}
checks['4_class_drop<=1pp_domain<=0.5pp'] = ({**{k: round(v, 3) for k, v in cls_drop.items()}, **{k: round(v, 3) for k, v in dom_drop.items()}},
                                             max(cls_drop.values()) <= 1.0 and max(dom_drop.values()) <= 0.5)
hl_drop = pp(b['hover_late_fault_TP'] - h['hover_late_fault_TP'], b['hover_late_fault_n'])
checks['5_hover_late_fault_drop<=1pp'] = (round(hl_drop, 3), hl_drop <= 1.0)
checks['6_indep_nonhover_FP<=168'] = (h['indep_nonhover_FP'], h['indep_nonhover_FP'] <= 168)
checks['7_all_normal_FP<=417'] = (h['all_normal_FP'], h['all_normal_FP'] <= 417)

out = {'summaries': S, 'checks': {k: {'value': v[0], 'pass': bool(v[1])} for k, v in checks.items()},
       'all_pass': all(v[1] for v in checks.values())}
json.dump(out, open(ROOT + 'h1_hover_coverage_v1_20261105/evaluation.json', 'w'), indent=2, default=str)
for k in ['late_hover_FP', 'hover_le30_FP', 'fault_TP', 'hover_late_fault_TP', 'all_normal_FP', 'prefault_FP', 'indep_nonhover_FP']:
    print(f'{k:22s} M0_B {b[k]:>6}  H1 {h[k]:>6}')
print('class TP', {c: (b['class_TP'][c], h['class_TP'][c], b['class_n'][c]) for c in b['class_TP']})
print('domain TP', {c: (b['domain_TP'][c], h['domain_TP'][c]) for c in b['domain_TP']})
print('indep FP by flight', {c: (b['indep_FP_by_flight'][c], h['indep_FP_by_flight'][c]) for c in b['indep_FP_by_flight']})
print('normal FP by domain', b['normal_FP_by_domain'], h['normal_FP_by_domain'])
print('late case FPR', {c.split('/')[0] + c.split('/')[-1][8:11]: (round(b['late_hover_case_fpr'][c], 2), round(h['late_hover_case_fpr'][c], 2)) for c in b['late_hover_case_fpr']})
for k, v in out['checks'].items():
    print(k, v)
print('ALL_PASS', out['all_pass'])
