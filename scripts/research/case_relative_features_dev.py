"""case_relative_features_v1_dev: GBT on window stats normalized by each case's first 10 s baseline."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_ens_events_val import case_windows, stats, DATA, GBT_PKL  # noqa: E402

OUT = ROOT / 'artifacts/case_relative_features_v1_dev_20261106'
BASE_S = 10.0
SRC_SD = np.asarray(json.load(open(DATA / 'norm_stats.json'))['std'], np.float64)


def case_arrays(uid):
    with np.load(DATA / 'cases' / (uid.replace('/', '__') + '.npz')) as z:
        t = z['timestamp_us'] / 1e6; r = z['canonical_rows'].astype(np.float64); v = z['row_input_valid']
    r[:, 25] = np.sqrt((r[:, 11:14] ** 2).sum(1))
    return t, r, v


def baseline(t, r, v):
    iv = np.flatnonzero(v)
    t0 = t[iv[0]]; b = v & (t < t0 + BASE_S); x = r[b]
    bm = x.mean(0); bsd = np.maximum(x.std(0), 1e-3 * SRC_SD)
    pair = b[1:] & b[:-1]
    bad = np.maximum(np.abs(np.diff(r, axis=0))[pair].mean(0), 1e-3 * SRC_SD)
    return t0 + BASE_S, bm, bsd, bad


def relative(S, B):
    _, bm, bsd, bad = B; c = 26
    return np.concatenate([(S[:, :c] - bm) / bsd, S[:, c:2 * c] / bsd, (S[:, 2 * c:3 * c] - bm) / bsd,
                           (S[:, 3 * c:4 * c] - bm) / bsd, S[:, 4 * c:] / bad], axis=1)


def split_features(split):
    """Raw and relative stats for eligible ledger windows of a split, with ledger rows."""
    p = torch.load(DATA / f'cache/{split}.pt', weights_only=True, mmap=True)
    L = pd.read_csv(DATA / f'ledgers/{split}_included_windows.csv', dtype={'association_group': str}, keep_default_na=False, low_memory=False)
    S = np.concatenate([stats(p['x'][i:i + 4096].numpy().astype(np.float32)) for i in range(0, len(p['x']), 4096)])
    y = p['y'].numpy(); R = np.full_like(S, np.nan); ok = np.zeros(len(L), bool)
    for uid, idx in L.groupby('case_uid').indices.items():
        t, r, v = case_arrays(uid); B = baseline(t, r, v)
        e = L.window_end_row.values[idx].astype(int) - 1
        el = t[e] >= B[0]; ok[idx[el]] = True
        R[idx[el]] = relative(S[idx[el]], B)
    return L, S, R, y, ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LT, ST, RT, yT, okT = split_features('train')
    w = np.ones(len(yT)); nm = float((yT[okT] == 0).sum())
    for o in ['normal_flight', 'normal_part_of_fault_case']:
        m = okT & (LT.normal_origin.values == o); w[m] = 0.5 * nm / m.sum()
    gbt = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=0).fit(RT[okT], yT[okT], sample_weight=w[okT])
    pickle.dump(gbt, open(OUT / 'gbt_relative.pkl', 'wb'))
    raw = pickle.load(open(GBT_PKL, 'rb'))

    LV, SV, RV, yV, okV = split_features('val')
    pr = gbt.predict_proba(RV[okV])[:, 1]; pw = raw.predict_proba(SV[okV])[:, 1]
    lv = LV[okV].reset_index(drop=True); y = yV[okV]
    cls = lv.case_uid.str.split('/').str[2].values; flight = lv.case_uid.str.split('/').str[1].values
    tend = []  # time since t0 of window end, for hover >30 s
    for uid, idx in lv.groupby('case_uid').indices.items():
        t, r, v = case_arrays(uid); t0 = t[np.flatnonzero(v)[0]]
        tend.append(pd.Series(t[lv.window_end_row.values[idx].astype(int) - 1] - t0, index=idx))
    tend = pd.concat(tend).sort_index().values
    late = (flight == 'hover') & (tend > 30) & (lv.normal_origin.values == 'normal_flight')

    def m(p):
        out = {'normal_fpr': round(float((p[y == 0] >= .5).mean()), 4), 'fault_recall': round(float((p[y == 1] >= .5).mean()), 4),
               'late_hover_indep_normal_fp': f'{int((p[late] >= .5).sum())}/{int(late.sum())}'}
        out['recall_by_class'] = {c: round(float((p[(y == 1) & (cls == c)] >= .5).mean()), 4) for c in sorted(set(cls[y == 1]))}
        return out
    val = {'relative_gbt': m(pr), 'raw_gbt_same_windows': m(pw), 'eligible_windows': int(okV.sum()), 'all_windows': int(len(okV))}

    tasks = json.load(open(DATA / 'tasks.json'))
    rn = sorted(t['case_uid'] for t in tasks if t['split'] == 'real' and t['role'] in ('adapt_normal', 'calibrate_normal'))
    assert len(rn) == 15
    RL = pd.read_csv(DATA / 'ledgers/real_included_windows.csv', low_memory=False)
    RL = RL[RL.case_uid.isin(rn)]
    assert set(RL.role) <= {'adapt_normal', 'calibrate_normal'}
    real_rows, XR, GR = [], [], []
    for uid in rn:
        t, r, v = case_arrays(uid); B = baseline(t, r, v)
        ends, xr = case_windows(r, v)
        el = t[ends] >= B[0]
        Rf = relative(stats(xr[el]), B); pf = gbt.predict_proba(Rf)[:, 1]; pf_raw = raw.predict_proba(stats(xr[el]))[:, 1]
        sup = set(RL[RL.case_uid == uid].window_end_row.astype(int) - 1)
        sm = np.array([e in sup for e in ends[el]])
        real_rows.append({'case_uid': uid, 'role': RL[RL.case_uid == uid].role.iloc[0], 'sup_windows': int(sm.sum()),
                          'sup_fp05': int((pf[sm] >= .5).sum()), 'sup_fp05_raw': int((pf_raw[sm] >= .5).sum()),
                          'frame_frac05': float((pf >= .5).mean()), 'frame_frac05_raw': float((pf_raw >= .5).mean()),
                          'frame_frac075': float((pf >= .75).mean())})
        XR.append(Rf[::5]); GR += [uid] * len(Rf[::5])
    rd = pd.DataFrame(real_rows); rd.to_csv(OUT / 'real_normal_per_case.csv', index=False)
    real = {'sup_window_fpr05': round(rd.sup_fp05.sum() / rd.sup_windows.sum(), 4),
            'sup_window_fpr05_raw_gbt': round(rd.sup_fp05_raw.sum() / rd.sup_windows.sum(), 4),
            'frame_frac05_mean': round(float(rd.frame_frac05.mean()), 4), 'frame_frac075_mean': round(float(rd.frame_frac075.mean()), 4),
            'by_role': rd.groupby('role').apply(lambda x: round(x.sup_fp05.sum() / x.sup_windows.sum(), 4)).to_dict()}

    ii = np.sort(np.random.default_rng(0).choice(np.flatnonzero(okT), 20000, replace=False))
    Xa = np.concatenate([RT[ii], np.concatenate(XR)]); ya = np.r_[np.zeros(len(ii)), np.ones(len(GR))]
    Ga = np.r_[LT.case_uid.values[ii], np.array(GR)]; pd_ = np.zeros(len(ya))
    for tr, te in GroupKFold(5).split(Xa, ya, Ga):
        pd_[te] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
    gate = {'real_normal_fpr_le_20pct': real['sup_window_fpr05'] <= .20, 'val_recall_ge_95pct': val['relative_gbt']['fault_recall'] >= .95,
            'val_normal_fpr_le_3pct': val['relative_gbt']['normal_fpr'] <= .03}
    gate['PASS'] = all(gate.values())
    res = {'source_val': val, 'real_normal_dev': real, 'domain_probe_auc_relative': round(float(roc_auc_score(ya, pd_)), 4),
           'train_eligible_windows': int(okT.sum()), 'train_all_windows': int(len(okT)), 'gate': gate}
    json.dump(res, open(OUT / 'results.json', 'w'), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
