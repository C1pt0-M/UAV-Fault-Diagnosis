"""real_normal_adapt_v2_test: one-shot exploratory Real test of target-normal GBT arms."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_ens_events_val import case_windows, stats, sha, DATA  # noqa: E402
from real_normal_adapt_dev import load_case, align  # noqa: E402
from replay_m1_continuous import aggregate_scores  # noqa: E402
from uav_fault.online.detection_policy import OnlineDetectionPolicy  # noqa: E402

DEV = ROOT / 'artifacts/real_normal_adapt_v2_dev_20261106'
OUT = ROOT / 'artifacts/real_normal_adapt_v2_test_20261106'
CFG = ROOT / 'configs/experiments/real_normal_adapt_v2_test.json'
POLICY = dict(onset_th=0.75, inference_every=1, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5)
ROLES = ['test_normal', 'test_fault', 'supplemental_normal_test']


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rec = {'config_sha256': sha(CFG), 'gbt_D2_sha256': sha(DEV / 'gbt_D2.pkl'), 'gbt_D3_sha256': sha(DEV / 'gbt_D3.pkl'),
           'alignment_sha256': sha(DEV / 'affine_alignment.json'), 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    (OUT / 'REAL_TEST_EVALUATED.lock').write_text(json.dumps(rec, indent=2))
    A = {k: np.asarray(v) for k, v in json.load(open(DEV / 'affine_alignment.json')).items() if k in ('mu_S', 'sd_S', 'mu_R', 'sd_R')}
    arms = {'GBT_D2': (pickle.load(open(DEV / 'gbt_D2.pkl', 'rb')), None), 'GBT_D3': (pickle.load(open(DEV / 'gbt_D3.pkl', 'rb')), A)}
    tasks = [t for t in json.load(open(DATA / 'tasks.json')) if t['split'] == 'real' and t['role'] in ROLES]
    RL = pd.read_csv(DATA / 'ledgers/real_included_windows.csv', low_memory=False)
    RL = RL[RL.role.isin(['test_normal', 'test_fault'])]
    rows, win = [], {a: ([], []) for a in arms}
    for k, t in enumerate(tasks):
        uid = t['case_uid']; g, r, v, ref, lv = load_case(uid)
        normal = lv & (ref == 0); on = np.flatnonzero(lv & (ref == 1)); onset = float(g[on[0]]) if len(on) else None
        we = RL[RL.case_uid == uid]
        for arm, (gbt, Aa) in arms.items():
            ends, xr = case_windows(align(r, Aa), v)
            p = np.full(len(g), np.nan)
            if len(ends):
                p[ends] = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
            if len(we):
                pw = p[we.window_end_row.values.astype(int) - 1]; assert np.isfinite(pw).all()
                win[arm][0].append(pw); win[arm][1].append(we.qualified_y.values)
            scored = np.isfinite(p)
            ev = aggregate_scores(g, p, policy=OnlineDetectionPolicy(**POLICY))
            fin = ev['finalized_events']; occ = np.zeros(len(g), bool); nf = 0
            for e in fin:
                m = (g >= e['start_t']) & (g <= e['end_t'])
                if m.any() and normal[m].all():
                    nf += 1; occ |= m
            row = {'case_uid': uid, 'role': t['role'], 'family': t['family'], 'flight': uid.split('/')[1],
                   'quarantine': bool(t.get('label_quarantine', False)), 'arm': arm,
                   'normal_exposure_s': float((normal & scored).sum() * .02), 'false_events': nf,
                   'false_occupancy_s': float((occ & normal & scored).sum() * .02), 'onset_s': onset,
                   'post_onset_frac_over_onset': float((p[scored & (g >= onset)] >= .75).mean()) if onset is not None and (scored & (g >= onset)).any() else np.nan}
            if onset is not None:
                conf = [c for c in ev['confirmations'] if c['candidate_start_t'] >= onset]
                row.update(post_onset_scored=bool((scored & (g >= onset)).any()), detected=bool(conf),
                           confirm_latency_s=conf[0]['confirmed_t'] - onset if conf else np.nan,
                           spans_onset=any(e['start_t'] < onset <= e['end_t'] for e in fin))
            rows.append(row)
        if k % 50 == 0:
            print(k, uid, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'per_case.csv', index=False)
    sec = {}
    for arm, (ps, ys) in win.items():
        p = np.concatenate(ps); y = np.concatenate(ys)
        sec[arm] = {'normal_fpr05': round(float((p[y == 0] >= .5).mean()), 4), 'fault_recall05': round(float((p[y == 1] >= .5).mean()), 4),
                    'normal_windows': int((y == 0).sum()), 'fault_windows': int((y == 1).sum())}
    json.dump(sec, open(OUT / 'window_secondary.json', 'w'), indent=2)
    rec.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), per_case_sha256=sha(OUT / 'per_case.csv'))
    (OUT / 'REAL_TEST_EVALUATED.lock').write_text(json.dumps(rec, indent=2))
    print(json.dumps(sec, indent=1))


if __name__ == '__main__':
    main()
