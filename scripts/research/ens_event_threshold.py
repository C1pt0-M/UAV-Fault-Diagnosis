"""ens_event_threshold_v1 step 2: event aggregation, threshold selection on train held-out, val confirmation."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
SCORES = Path('D:/UAV-Data/ens_event_threshold_v1_scores')
OUT = ROOT / 'artifacts/ens_event_threshold_v1_20261106'
GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BASE = dict(inference_every=1, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5)


def case_events(uid):
    from replay_m1_continuous import aggregate_scores
    from uav_fault.online.detection_policy import OnlineDetectionPolicy
    with np.load(SCORES / (uid.replace('/', '__') + '.npz')) as z:
        t = z['timestamp_us'] / 1e6; m0 = z['m0'].astype(float); g = z['gbt'].astype(float)
        ref = z['reference_state']; lv = z['row_label_valid']
    ens = (m0 + g) / 2
    normal = lv & (ref == 0); scored = np.isfinite(m0)
    on = np.flatnonzero(lv & (ref == 1)); onset = float(t[on[0]]) if len(on) else None
    out = []
    for arm, prob, ths in [('M0', m0, [0.60]), ('ENS', ens, GRID)]:
        for th in ths:
            ev = aggregate_scores(t, prob, policy=OnlineDetectionPolicy(onset_th=th, **BASE))
            fin = ev['finalized_events']; occ = np.zeros(len(t), bool); nfalse = 0
            for e in fin:
                m = (t >= e['start_t']) & (t <= e['end_t'])
                if m.any() and normal[m].all():
                    nfalse += 1; occ |= m
            r = {'case_uid': uid, 'arm': arm, 'onset_th': th, 'false_events': nfalse,
                 'normal_exposure_s': float((normal & scored).sum() * .02), 'false_occupancy_s': float((occ & normal & scored).sum() * .02)}
            if onset is not None:
                conf = [c for c in ev['confirmations'] if c['candidate_start_t'] >= onset]
                r.update(onset_s=onset, post_onset_scored=bool((scored & (t >= onset)).any()), detected=bool(conf),
                         confirm_latency_s=conf[0]['confirmed_t'] - onset if conf else np.nan,
                         spans_onset=any(e['start_t'] < onset <= e['end_t'] for e in fin))
            out.append(r)
    return out


def summarize(d, arm, th):
    x = d[(d.arm == arm) & (d.onset_th == th)]
    ind = x[x.cls == 'no_fault']; f = x[(x.cls != 'no_fault') & (x.post_onset_scored == True)]
    h = ind.normal_exposure_s.sum() / 3600
    return {'indep_false_events': int(ind.false_events.sum()), 'indep_cases_with_false': int((ind.false_events > 0).sum()),
            'indep_cases': int(len(ind)), 'indep_exposure_h': round(h, 3), 'indep_events_per_h': ind.false_events.sum() / h,
            'indep_occupancy_pct': 100 * ind.false_occupancy_s.sum() / ind.normal_exposure_s.sum(),
            'prefault_false_events': int(x[x.cls != 'no_fault'].false_events.sum()),
            'fault_cases': int(len(f)), 'fault_detected': int(f.detected.astype(bool).sum()),
            'fault_spans_onset': int(f.spans_onset.astype(bool).sum()),
            'detected_by_class': f.groupby('cls').detected.apply(lambda s: int(s.astype(bool).sum())).to_dict()}


def main():
    idx = pd.read_csv(SCORES / 'index.csv')
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(case_events, idx.case_uid, chunksize=8) for r in rs]
    d = pd.DataFrame(rows).merge(idx, on='case_uid')
    d['cls'] = d.case_uid.str.split('/').str[2]; d['flight'] = d.case_uid.str.split('/').str[1]
    d.to_csv(OUT / 'events_per_case.csv', index=False)
    tr, va = d[d.split == 'train'], d[d.split == 'val']
    sweep = {th: summarize(tr, 'ENS', th) for th in GRID}
    base_tr = summarize(tr, 'M0', 0.60)
    chosen = next((th for th in GRID if sweep[th]['indep_events_per_h'] <= base_tr['indep_events_per_h']), None)
    res = {'train_M0_0.60': base_tr, 'train_ENS_sweep': sweep, 'chosen_onset': chosen}
    if chosen is not None:
        res['val_M0_0.60'] = summarize(va, 'M0', 0.60); res['val_ENS_chosen'] = summarize(va, 'ENS', chosen)
        res['val_ENS_0.60'] = summarize(va, 'ENS', 0.60)
        verdict = {}
        for name, B, P in [('train', base_tr, sweep[chosen]), ('val', res['val_M0_0.60'], res['val_ENS_chosen'])]:
            verdict[name] = {'a_false_events': P['indep_false_events'] <= B['indep_false_events'],
                             'b_detected': P['fault_detected'] >= B['fault_detected'] - 0.01 * B['fault_cases']}
        # paired latency on cases detected by both
        for name, part in [('train', tr), ('val', va)]:
            a = part[(part.arm == 'M0') & (part.onset_th == 0.60)].set_index('case_uid')
            b = part[(part.arm == 'ENS') & (part.onset_th == chosen)].set_index('case_uid')
            both = a.index[(a.detected == True) & (b.detected.reindex(a.index) == True)]
            dl = (b.confirm_latency_s[both] - a.confirm_latency_s[both])
            res[f'{name}_paired_latency_ENS_minus_M0_s'] = {'cases': int(len(both)), 'median': float(dl.median()), 'p90': float(dl.quantile(.9)),
                                                           'M0_median': float(a.confirm_latency_s[both].median()), 'ENS_median': float(b.confirm_latency_s[both].median())}
        verdict['PASS'] = all(v['a_false_events'] and v['b_detected'] for k, v in verdict.items() if k != 'PASS')
        res['verdict'] = verdict
    json.dump(res, open(OUT / 'results.json', 'w'), indent=2, default=float)
    print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))


if __name__ == '__main__':
    main()
