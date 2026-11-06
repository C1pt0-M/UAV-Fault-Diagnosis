"""Summarize real_event_eval_v1 per_case.csv, plus secondary window-level metrics."""
import json
import numpy as np, pandas as pd

OUT = 'D:/UAV-Fault-Diagnosis/artifacts/real_event_eval_v1_20261106/'
DATA = 'D:/UAV-Data/causal_input_v1/'
d = pd.read_csv(OUT + 'per_case.csv')
ARMS = ['M0@0.60', 'M0@0.70', 'ENS@0.75']


def normal(x):
    h = x.normal_exposure_s.sum() / 3600
    return {'cases': int(len(x)), 'cases_with_false_event': int((x.false_events > 0).sum()),
            'case_false_alarm_rate': round(float((x.false_events > 0).mean()), 4), 'false_events': int(x.false_events.sum()),
            'exposure_h': round(h, 3), 'events_per_h': round(x.false_events.sum() / h, 2) if h else None,
            'occupancy_pct': round(100 * x.false_occupancy_s.sum() / x.normal_exposure_s.sum(), 2) if h else None}


def fault(x):
    x = x[(x.quarantine == False) & (x.post_onset_scored == True)]
    det = x.detected.astype(bool); sp = x.spans_onset.astype(bool); lat = x.confirm_latency_s[det]
    return {'cases': int(len(x)), 'recall': round(float(det.mean()), 4), 'detected': int(det.sum()), 'spans_onset': int(sp.sum()),
            'neither': int((~det & ~sp).sum()),
            'latency_median_s': round(float(lat.median()), 3) if len(lat) else None, 'latency_p90_s': round(float(lat.quantile(.9)), 3) if len(lat) else None}


res = {}
for a in ARMS:
    x = d[d.arm == a]; f = x[x.role == 'test_fault']
    res[a] = {'test_normal': normal(x[x.role == 'test_normal']),
              'test_normal_by_flight': {k: normal(g) for k, g in x[x.role == 'test_normal'].groupby('flight')},
              'supplemental_normal': normal(x[x.role == 'supplemental_normal_test']),
              'prefault_normal_in_fault_cases': normal(f),
              'fault': fault(f), 'fault_motor': fault(f[f.family == 'motor']), 'fault_sensor': fault(f[f.family == 'sensor']),
              'fault_by_flight': {k: fault(g) for k, g in f.groupby('flight')}}
f = d[(d.role == 'test_fault') & (d.quarantine == False) & (d.post_onset_scored == True)]
pair = {}
for a in ['M0@0.60', 'ENS@0.75']:
    A = f[f.arm == 'M0@0.70'].set_index('case_uid'); B = f[f.arm == a].set_index('case_uid')
    both = A.index[A.detected.astype(bool) & B.detected.reindex(A.index).astype(bool)]
    dl = B.confirm_latency_s[both] - A.confirm_latency_s[both]
    pair[a + '_minus_M0@0.70'] = {'cases': int(len(both)), 'median_s': round(float(dl.median()), 3), 'p90_s': round(float(dl.quantile(.9)), 3)}
    only_b = int((B.detected.reindex(A.index).astype(bool) & ~A.detected.astype(bool)).sum()); only_a = int((A.detected.astype(bool) & ~B.detected.reindex(A.index).astype(bool)).sum())
    pair[a + '_vs_M0@0.70_discordant'] = {'only_' + a: only_b, 'only_M0@0.70': only_a}
res['paired'] = pair
res['quarantined_fault_cases_excluded'] = int(d[(d.arm == ARMS[0]) & (d.role == 'test_fault') & d.quarantine].shape[0])
res['fault_cases_without_post_onset_score'] = int(d[(d.arm == ARMS[0]) & (d.role == 'test_fault') & (d.quarantine == False) & (d.post_onset_scored != True)].shape[0])

# Secondary: window-level at 0.5 on cap64 supervised Real windows of the main roles, scored from saved frames.
L = pd.read_csv(DATA + 'ledgers/real_included_windows.csv', low_memory=False)
L = L[L.role.isin(['test_normal', 'test_fault'])]
win = {'M0': [], 'ENS': []}
for uid, g in L.groupby('case_uid'):
    z = np.load(OUT + 'frames/' + uid.replace('/', '__') + '.npz')
    e = g.window_end_row.values.astype(int) - 1
    win['M0'].append(z['m0'][e]); win['ENS'].append((z['m0'][e] + z['gbt'][e]) / 2)
order = np.concatenate([g.qualified_y.values for _, g in L.groupby('case_uid')])
sec = {}
for k in win:
    p = np.concatenate(win[k]); assert np.isfinite(p).all()
    sec[k] = {'normal_fpr': round(float((p[order == 0] >= .5).mean()), 4), 'fault_recall': round(float((p[order == 1] >= .5).mean()), 4),
              'normal_windows': int((order == 0).sum()), 'fault_windows': int((order == 1).sum())}
res['secondary_window_0.5'] = sec
json.dump(res, open(OUT + 'summary.json', 'w'), indent=2)
for a in ARMS:
    r = res[a]
    print('==', a)
    print(' test_normal', r['test_normal'])
    print(' supplemental', r['supplemental_normal'])
    print(' prefault', {k: r['prefault_normal_in_fault_cases'][k] for k in ['false_events', 'events_per_h', 'exposure_h']})
    print(' fault', r['fault']); print(' motor', r['fault_motor']); print(' sensor', r['fault_sensor'])
    print(' normal by flight', {k: (v['false_events'], v['cases_with_false_event'], v['cases']) for k, v in r['test_normal_by_flight'].items()})
print(json.dumps({k: res[k] for k in ['paired', 'quarantined_fault_cases_excluded', 'fault_cases_without_post_onset_score', 'secondary_window_0.5']}, indent=1))
