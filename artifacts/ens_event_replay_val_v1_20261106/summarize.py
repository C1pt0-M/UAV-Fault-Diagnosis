"""Summarize ens_event_replay_val_v1 per_case.csv (descriptive)."""
import json
import numpy as np, pandas as pd

OUT = 'D:/UAV-Fault-Diagnosis/artifacts/ens_event_replay_val_v1_20261106/'
d = pd.read_csv(OUT + 'per_case.csv')
res = {}


def normal_block(x):
    h = x.normal_exposure_s.sum() / 3600
    return {'cases': int(x.case_uid.nunique()), 'cases_with_false_event': int((x.false_events > 0).sum()),
            'false_events': int(x.false_events.sum()), 'exposure_h': round(h, 3),
            'events_per_h': round(x.false_events.sum() / h, 2) if h else None,
            'alarm_occupancy_pct': round(100 * x.false_occupancy_s.sum() / x.normal_exposure_s.sum(), 2) if h else None}


for arm in ['M0', 'ENS']:
    a = d[d.arm == arm]
    ind = a[a.cls == 'no_fault']; fc = a[a.cls != 'no_fault']
    r = {'independent_normal': normal_block(ind),
         'independent_normal_by_domain': {k: normal_block(g) for k, g in ind.groupby('domain')},
         'independent_normal_by_flight': {k: normal_block(g) for k, g in ind.groupby('flight')},
         'prefault_normal_in_fault_cases': normal_block(fc)}
    m = fc[fc.post_onset_scored == True]
    clean = m[~m.event_spans_onset.astype(bool)]
    lat = clean.confirm_latency_s.dropna()
    r['fault'] = {'cases_with_onset': int(len(fc)), 'cases_scored_after_onset': int(len(m)),
                  'cases_event_spanning_onset': int(m.event_spans_onset.astype(bool).sum()),
                  'detected_new_confirmation': int(m.detected.astype(bool).sum()),
                  'clean_cases': int(len(clean)), 'clean_detected': int(clean.detected.astype(bool).sum()),
                  'confirm_latency_s_clean': {'median': round(float(lat.median()), 3), 'p90': round(float(lat.quantile(.9)), 3), 'max': round(float(lat.max()), 3)} if len(lat) else None,
                  'final_emit_reason_counts': m.final_emit_reason.value_counts().to_dict()}
    r['fault_by_class'] = {c: {'cases': int(len(g)), 'detected': int(g.detected.astype(bool).sum()),
                               'spans_onset': int(g.event_spans_onset.astype(bool).sum()),
                               'confirm_latency_median_s': round(float(g[~g.event_spans_onset.astype(bool)].confirm_latency_s.dropna().median()), 3) if g.confirm_latency_s.notna().any() else None}
                           for c, g in m.groupby('cls')}
    res[arm] = r
json.dump(res, open(OUT + 'summary.json', 'w'), indent=2)
for arm in ['M0', 'ENS']:
    r = res[arm]
    print('==', arm)
    print(' independent normal', r['independent_normal'])
    print(' by domain', r['independent_normal_by_domain'])
    print(' by flight', {k: (v['false_events'], v['cases_with_false_event'], v['cases'], v['alarm_occupancy_pct']) for k, v in r['independent_normal_by_flight'].items()})
    print(' prefault normal', r['prefault_normal_in_fault_cases'])
    print(' fault', r['fault'])
    print(' by class', r['fault_by_class'])
