"""real_event_eval_v1: one-shot Real event-level evaluation of M0@0.60, M0@0.70, ENS@0.75."""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_ens_events_val import case_windows, stats, sha, DATA, M0_PT, GBT_PKL  # noqa: E402
from replay_m1_continuous import aggregate_scores  # noqa: E402
from uav_fault.online.detection_policy import OnlineDetectionPolicy  # noqa: E402
from uav_fault.models.cnn.lstm import OnlineMSTCA  # noqa: E402

OUT = ROOT / 'artifacts/real_event_eval_v1_20261106'
CFG = ROOT / 'configs/experiments/real_event_eval_v1.json'
ARMS = [('M0@0.60', 'M0', .60), ('M0@0.70', 'M0', .70), ('ENS@0.75', 'ENS', .75)]
BASE = dict(inference_every=1, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5)
ROLES = ['test_normal', 'test_fault', 'supplemental_normal_test']


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'frames').mkdir()
    lock = OUT / 'REAL_EVALUATED.lock'
    record = {'config_sha256': sha(CFG), 'm0_sha256': sha(M0_PT), 'gbt_sha256': sha(GBT_PKL),
              'selection_sha256': sha(ROOT / 'artifacts/ens_event_threshold_v1_20261106/common_fa_target_selection.json'),
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    lock.write_text(json.dumps(record, indent=2))
    st = json.load(open(DATA / 'norm_stats.json'))
    mu = np.asarray(st['mean'], np.float32)[None, :, None]; sd = np.asarray(st['std'], np.float32)[None, :, None]
    gbt = pickle.load(open(GBT_PKL, 'rb'))
    m0 = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=0.2).float()
    m0.load_state_dict(torch.load(M0_PT, map_location='cpu', weights_only=True)); m0.eval(); torch.set_num_threads(16)
    tasks = [t for t in json.load(open(DATA / 'tasks.json')) if t['split'] == 'real' and t['role'] in ROLES]
    rows = []; t0 = time.time()
    for k, t in enumerate(tasks):
        uid = t['case_uid']; key = uid.replace('/', '__')
        with np.load(DATA / 'cases' / (key + '.npz')) as z:
            grid = z['timestamp_us'] / 1e6; raw = z['canonical_rows']; valid = z['row_input_valid']
            ref = z['reference_state']; lv = z['row_label_valid']
        ends, xr = case_windows(raw, valid)
        pm = np.full(len(grid), np.nan); pg = np.full(len(grid), np.nan)
        if len(ends):
            with torch.no_grad():
                pm[ends] = np.concatenate([m0(torch.from_numpy((xr[i:i + 2048] - mu) / sd)).softmax(1)[:, 1].numpy() for i in range(0, len(xr), 2048)])
            pg[ends] = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
        np.savez_compressed(OUT / 'frames' / (key + '.npz'),
                            timestamp_us=(grid * 1e6).astype(np.int64), m0=pm.astype(np.float32), gbt=pg.astype(np.float32))
        normal = lv & (ref == 0); scored = np.isfinite(pm)
        on = np.flatnonzero(lv & (ref == 1)); onset = float(grid[on[0]]) if len(on) else None
        for name, kind, th in ARMS:
            prob = pm if kind == 'M0' else (pm + pg) / 2
            ev = aggregate_scores(grid, prob, policy=OnlineDetectionPolicy(onset_th=th, **BASE))
            fin = ev['finalized_events']; occ = np.zeros(len(grid), bool); nf = 0
            for e in fin:
                m = (grid >= e['start_t']) & (grid <= e['end_t'])
                if m.any() and normal[m].all():
                    nf += 1; occ |= m
            r = {'case_uid': uid, 'role': t['role'], 'family': t['family'], 'domain': t['domain'],
                 'flight': uid.split('/')[1], 'quarantine': bool(t.get('label_quarantine', False)), 'arm': name,
                 'scored_frames': int(scored.sum()), 'frames': int(len(grid)),
                 'normal_exposure_s': float((normal & scored).sum() * .02), 'false_events': nf,
                 'false_occupancy_s': float((occ & normal & scored).sum() * .02), 'onset_s': onset}
            if onset is not None:
                conf = [c for c in ev['confirmations'] if c['candidate_start_t'] >= onset]
                r.update(post_onset_scored=bool((scored & (grid >= onset)).any()), detected=bool(conf),
                         confirm_latency_s=conf[0]['confirmed_t'] - onset if conf else np.nan,
                         spans_onset=any(e['start_t'] < onset <= e['end_t'] for e in fin))
            rows.append(r)
        if k % 50 == 0:
            print(k, uid, round(time.time() - t0), 's', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'per_case.csv', index=False)
    record.update(cases=len(tasks), finished=time.strftime('%Y-%m-%d %H:%M:%S'), per_case_sha256=sha(OUT / 'per_case.csv'))
    lock.write_text(json.dumps(record, indent=2))
    print('done', len(tasks))


if __name__ == '__main__':
    main()
