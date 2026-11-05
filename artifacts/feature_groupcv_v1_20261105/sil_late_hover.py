"""Why late-hover independent normals stay alarmed in SIL (read-only).

Uses group-CV GBT held-out probabilities on train (more cases than val) and
window statistics. Compares SIL vs HIL late-hover normal windows and the late-
hover fault windows of the same domain.
"""
import json
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

DATA = 'D:/UAV-Data/causal_input_v1/'
OUT = 'D:/UAV-Fault-Diagnosis/artifacts/feature_groupcv_v1_20261105/'
cols = json.load(open(DATA + 'input_contract.json'))['columns']
z = np.load(OUT + 'gbt_heldout.npz')
L = pd.read_csv(DATA + 'ledgers/train_included_windows.csv', dtype={'association_group': str}, keep_default_na=False, low_memory=False)
assert (L.original_cache_index.values == z['original_indices']).all()
L['prob'] = z['prob']; s = L.case_uid.str.split('/'); L['flight'], L['cls'] = s.str[1], s.str[2]
L['t'] = L.window_end_row * 0.02
late = (L.flight == 'hover') & (L.t > 30)
N = L[late & (L.normal_origin == 'normal_flight')]
F = L[late & (L.qualified_y == 1)]

print('held-out GBT, late-hover independent normal FPR by domain:')
print(N.groupby('domain').apply(lambda g: pd.Series({'windows': len(g), 'cases': g.case_uid.nunique(), 'FPR': round((g.prob >= .5).mean(), 3)})))
print('\nper-case FPR:')
print(N.groupby(['domain', 'case_uid']).prob.apply(lambda p: round((p >= .5).mean(), 2)).unstack(0).describe().round(2))
print('\nlate-hover fault vs normal AUC by domain:')
for d in ['SIL', 'HIL']:
    n = N[N.domain == d]; f = F[F.domain == d]
    y = np.r_[np.zeros(len(n)), np.ones(len(f))]; pr = np.r_[n.prob, f.prob]
    print(d, 'all-fault AUC', round(roc_auc_score(y, pr), 3), {c: round(roc_auc_score(np.r_[np.zeros(len(n)), np.ones((f.cls == c).sum())], np.r_[n.prob, f[f.cls == c].prob]), 3) for c in sorted(f.cls.unique())})

# channel statistics: compare SIL normal vs HIL normal and vs same-domain fault
p = torch.load(DATA + 'cache/train.pt', weights_only=True, mmap=True)
pos = {v: i for i, v in enumerate(p['original_cache_index'].numpy())}


def win(frame):
    x = p['x'][[pos[i] for i in frame.original_cache_index]].numpy()
    return x.mean(2), x.std(2)


rows = []
for d in ['SIL', 'HIL']:
    nm, ns = win(N[N.domain == d]); fm, fs = win(F[F.domain == d])
    for j, c in enumerate(cols):
        rows.append({'domain': d, 'channel': c, 'normal_mean': nm[:, j].mean(), 'fault_mean': fm[:, j].mean(),
                     'normal_std': ns[:, j].mean(), 'fault_std': fs[:, j].mean(),
                     'auc_mean': max(a := roc_auc_score(np.r_[np.zeros(len(nm)), np.ones(len(fm))], np.r_[nm[:, j], fm[:, j]]), 1 - a),
                     'auc_std': max(a := roc_auc_score(np.r_[np.zeros(len(ns)), np.ones(len(fs))], np.r_[ns[:, j], fs[:, j]]), 1 - a)})
R = pd.DataFrame(rows)
R.to_csv(OUT + 'sil_hil_channel_stats.csv', index=False)
for d in ['SIL', 'HIL']:
    r = R[R.domain == d].assign(best=lambda x: x[['auc_mean', 'auc_std']].max(1)).sort_values('best', ascending=False)
    print(f'\n{d}: top channels separating late-hover normal from fault')
    print(r.head(8)[['channel', 'auc_mean', 'auc_std', 'normal_std', 'fault_std']].round(4).to_string(index=False))
# SIL vs HIL normal std ratio for actuator / imu channels
piv = R.pivot(index='channel', columns='domain', values='normal_std')
piv['SIL/HIL'] = piv.SIL / piv.HIL
print('\nnormal window std, SIL vs HIL:')
print(piv.round(5).sort_values('SIL/HIL').to_string())
