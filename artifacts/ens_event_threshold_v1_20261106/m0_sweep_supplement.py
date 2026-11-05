"""Supplement (not part of the frozen rule): M0 on the same onset grid, to compare at matched false-event rate."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('D:/UAV-Fault-Diagnosis'); sys.path.insert(0, str(ROOT / 'scripts/research')); sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
import ens_event_threshold as E

def m0_events(uid):
    from replay_m1_continuous import aggregate_scores
    from uav_fault.online.detection_policy import OnlineDetectionPolicy
    with np.load(E.SCORES / (uid.replace('/', '__') + '.npz')) as z:
        t = z['timestamp_us'] / 1e6; m0 = z['m0'].astype(float); ref = z['reference_state']; lv = z['row_label_valid']
    normal = lv & (ref == 0); scored = np.isfinite(m0); on = np.flatnonzero(lv & (ref == 1)); onset = float(t[on[0]]) if len(on) else None
    out = []
    for th in E.GRID[1:]:
        ev = aggregate_scores(t, m0, policy=OnlineDetectionPolicy(onset_th=th, **E.BASE)); fin = ev['finalized_events']; occ = np.zeros(len(t), bool); nf = 0
        for e in fin:
            m = (t >= e['start_t']) & (t <= e['end_t'])
            if m.any() and normal[m].all(): nf += 1; occ |= m
        r = {'case_uid': uid, 'arm': 'M0', 'onset_th': th, 'false_events': nf, 'normal_exposure_s': float((normal & scored).sum() * .02), 'false_occupancy_s': float((occ & normal & scored).sum() * .02)}
        if onset is not None:
            conf = [c for c in ev['confirmations'] if c['candidate_start_t'] >= onset]
            r.update(onset_s=onset, post_onset_scored=bool((scored & (t >= onset)).any()), detected=bool(conf), confirm_latency_s=conf[0]['confirmed_t'] - onset if conf else np.nan, spans_onset=any(e['start_t'] < onset <= e['end_t'] for e in fin))
        out.append(r)
    return out

if __name__ == '__main__':
    idx = pd.read_csv(E.SCORES / 'index.csv')
    with ProcessPoolExecutor(16) as ex: rows = [r for rs in ex.map(m0_events, idx.case_uid, chunksize=8) for r in rs]
    d = pd.DataFrame(rows).merge(idx, on='case_uid'); d['cls'] = d.case_uid.str.split('/').str[2]
    base = pd.read_csv(E.OUT / 'events_per_case.csv'); d = pd.concat([base, d], ignore_index=True)
    res = {}
    for split in ['train', 'val']:
        part = d[d.split == split]
        res[split] = {f'{arm}@{th}': {k: v for k, v in E.summarize(part, arm, th).items() if k in ['indep_false_events', 'indep_events_per_h', 'fault_detected', 'fault_spans_onset', 'fault_cases', 'detected_by_class']}
                      for arm in ['M0', 'ENS'] for th in E.GRID}
    json.dump(res, open(E.OUT / 'm0_sweep_supplement.json', 'w'), indent=2, default=float)
    for split in res:
        print('==', split)
        for th in E.GRID:
            a, b = res[split][f'M0@{th}'], res[split][f'ENS@{th}']
            print(f'  {th:.2f}  M0 false {a["indep_false_events"]:4d} det {a["fault_detected"]:5d} span {a["fault_spans_onset"]:4d} | ENS false {b["indep_false_events"]:4d} det {b["fault_detected"]:5d} span {b["fault_spans_onset"]:4d}')
