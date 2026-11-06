"""real_normal_adapt_v2_dev: target-normal adaptation development on Real adapt/calibrate normals only."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_ens_events_val import case_windows, stats, DATA, M0_PT, GBT_PKL  # noqa: E402
from replay_m1_continuous import aggregate_scores  # noqa: E402
from uav_fault.online.detection_policy import OnlineDetectionPolicy  # noqa: E402
from uav_fault.models.cnn.lstm import OnlineMSTCA  # noqa: E402

OUT = ROOT / 'artifacts/real_normal_adapt_v2_dev_20261106'
POLICY = dict(inference_every=1, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5)
ALLOWED = {'adapt_normal', 'calibrate_normal'}
tasks = json.load(open(DATA / 'tasks.json'))
st = json.load(open(DATA / 'norm_stats.json'))
MU = np.asarray(st['mean'], np.float32)[None, :, None]; SD = np.asarray(st['std'], np.float32)[None, :, None]


def load_case(uid):
    with np.load(DATA / 'cases' / (uid.replace('/', '__') + '.npz')) as z:
        return z['timestamp_us'] / 1e6, z['canonical_rows'].astype(np.float64), z['row_input_valid'], z['reference_state'], z['row_label_valid']


def frame_moments(uids):
    s = np.zeros(26); s2 = np.zeros(26); n = 0
    for u in uids:
        _, r, v, _, _ = load_case(u)
        r = r[v].copy(); r[:, 25] = np.sqrt((r[:, 11:14] ** 2).sum(1))
        s += r.sum(0); s2 += (r ** 2).sum(0); n += len(r)
    m = s / n
    return m, np.sqrt(np.maximum(s2 / n - m ** 2, 1e-12)), n


def align(rows, A):
    if A is None:
        return rows
    out = (rows - A['mu_R']) / A['sd_R'] * A['sd_S'] + A['mu_S']
    return out  # acc_norm recomputed from aligned accel inside case_windows


def score_case(uid, m0, gbt, A):
    t, r, v, ref, lv = load_case(uid)
    ends, xr = case_windows(align(r, A), v)
    pm = np.full(len(t), np.nan); pg = np.full(len(t), np.nan)
    with torch.no_grad():
        pm[ends] = np.concatenate([m0(torch.from_numpy((xr[i:i + 2048] - MU) / SD)).softmax(1)[:, 1].numpy() for i in range(0, len(xr), 2048)])
    pg[ends] = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
    return t, pm, pg, ends, xr


def events(t, p, th):
    ev = aggregate_scores(t, p, policy=OnlineDetectionPolicy(onset_th=th, **POLICY))
    fin = ev['finalized_events']; occ = np.zeros(len(t), bool)
    for e in fin:
        occ |= (t >= e['start_t']) & (t <= e['end_t'])
    return len(fin), float(occ[np.isfinite(p)].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real = {t['case_uid']: t['role'] for t in tasks if t['split'] == 'real'}
    adapt = sorted(u for u, r in real.items() if r == 'adapt_normal'); calib = sorted(u for u, r in real.items() if r == 'calibrate_normal')
    assert len(adapt) == 10 and len(calib) == 5 and set(real[u] for u in adapt + calib) <= ALLOWED
    L = pd.read_csv(DATA / 'ledgers/train_included_windows.csv', low_memory=False)
    src_norm = sorted(L[L.normal_origin == 'normal_flight'].case_uid.unique())
    mu_S, sd_S, nS = frame_moments(src_norm); mu_R, sd_R, nR = frame_moments(adapt)
    A = {'mu_S': mu_S, 'sd_S': sd_S, 'mu_R': mu_R, 'sd_R': sd_R}
    json.dump({k: v.tolist() for k, v in A.items()} | {'n_source_frames': nS, 'n_adapt_frames': nR, 'columns': st['columns']},
              open(OUT / 'affine_alignment.json', 'w'), indent=1)
    m0 = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=0.2).float()
    m0.load_state_dict(torch.load(M0_PT, map_location='cpu', weights_only=True)); m0.eval(); torch.set_num_threads(16)
    gbt0 = pickle.load(open(GBT_PKL, 'rb'))

    # D2: refit GBT with adapt-normal windows (stride 5 valid ends), raw and aligned variants.
    p = torch.load(DATA / 'cache/train.pt', weights_only=True, mmap=True)
    XT = np.concatenate([stats(p['x'][i:i + 4096].numpy().astype(np.float32)) for i in range(0, len(p['x']), 4096)])
    yT = p['y'].numpy(); del p
    w = np.ones(len(yT)); nm = float((yT == 0).sum())
    for o in ['normal_flight', 'normal_part_of_fault_case']:
        m = L.normal_origin.values == o; w[m] = 0.5 * nm / m.sum()
    src_normal_weight = w[yT == 0].sum()

    def refit(Aa):
        Xa = []
        for u in adapt:
            _, r, v, _, _ = load_case(u); _, xr = case_windows(align(r, Aa), v); Xa.append(stats(xr[::5]))
        Xa = np.concatenate(Xa); wa = np.full(len(Xa), 0.1 * src_normal_weight / len(Xa))
        g = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        return g.fit(np.concatenate([XT, Xa]), np.r_[yT, np.zeros(len(Xa), int)], sample_weight=np.r_[w, wa]), len(Xa)
    gbt_D2, nD2 = refit(None); gbt_D3, nD3 = refit(A)
    pickle.dump(gbt_D2, open(OUT / 'gbt_D2.pkl', 'wb')); pickle.dump(gbt_D3, open(OUT / 'gbt_D3.pkl', 'wb'))
    arms = {'D0': (gbt0, None), 'D1_affine': (gbt0, A), 'D2_gbt_target_normal': (gbt_D2, None), 'D3': (gbt_D3, A)}

    # Real adapt/calibrate: supervised window FPR at 0.5, frame fraction over onset, events on full timeline.
    RL = pd.read_csv(DATA / 'ledgers/real_included_windows.csv', low_memory=False)
    rows = []
    for arm, (g, Aa) in arms.items():
        for u in adapt + calib:
            t, pm, pg, ends, _ = score_case(u, m0, g, Aa); pe = (pm + pg) / 2
            we = RL[RL.case_uid == u].window_end_row.values.astype(int) - 1
            r = {'arm': arm, 'case_uid': u, 'role': real[u], 'windows': len(we)}
            for name, pr, th in [('M0', pm, .70), ('GBT', pg, .75), ('ENS', pe, .75)]:
                fv = pr[np.isfinite(pr)]
                r[f'{name}_win_fp05'] = int((pr[we] >= .5).sum()); r[f'{name}_frames_over_onset'] = float((fv >= th).mean())
                if name != 'GBT':
                    r[f'{name}_events'], r[f'{name}_occupancy'] = events(t, pr, th)
            rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(OUT / 'real_dev_per_case.csv', index=False)
    summ = {}
    for (arm, role), x in d.groupby(['arm', 'role']):
        n = x.windows.sum()
        summ[f'{arm}|{role}'] = {f'{m}_win_fpr05': round(x[f'{m}_win_fp05'].sum() / n, 4) for m in ['M0', 'GBT', 'ENS']} | {
            'M0_events': int(x.M0_events.sum()), 'ENS_events': int(x.ENS_events.sum()),
            'M0_occupancy': round(float(x.M0_occupancy.mean()), 4), 'ENS_occupancy': round(float(x.ENS_occupancy.mean()), 4), 'cases': len(x)}

    # Source val check for refit GBTs (D2/D3 change the GBT; source inputs are never aligned).
    pv = torch.load(DATA / 'cache/val.pt', weights_only=True, mmap=True)
    XV = np.concatenate([stats(pv['x'][i:i + 4096].numpy().astype(np.float32)) for i in range(0, len(pv['x']), 4096)]); yV = pv['y'].numpy()
    val = {}
    for arm, g in [('D0', gbt0), ('D2', gbt_D2), ('D3', gbt_D3)]:
        pr = g.predict_proba(XV)[:, 1]
        val[arm] = {'gbt_normal_fpr05': round(float((pr[yV == 0] >= .5).mean()), 4), 'gbt_fault_recall05': round(float((pr[yV == 1] >= .5).mean()), 4)}
    res = {'real_dev': summ, 'source_val_gbt': val, 'adapt_windows_added': {'D2': nD2, 'D3': nD3},
           'note': 'adapt rows are in-sample for all arms; calibrate rows are held out'}
    json.dump(res, open(OUT / 'results.json', 'w'), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
