"""Group-CV GBT on train (frozen config feature_groupcv_v1). CPU only.

Writes held-out GBT probabilities per fold. Comparison with M0 needs the M0
fold predictions from scripts/research/train_m0_groupcv.py (GPU).
"""
import hashlib, json, time
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = 'D:/UAV-Data/causal_input_v1/'
OUT = 'D:/UAV-Fault-Diagnosis/artifacts/feature_groupcv_v1_20261105/'


def stats(x):
    x = x.astype(np.float32); d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


def fold_of(g):
    return int(hashlib.sha256(('h1cv|' + g).encode()).hexdigest(), 16) % 3


t0 = time.time()
p = torch.load(DATA + 'cache/train.pt', weights_only=True, mmap=True)
X = np.concatenate([stats(p['x'][i:i + 4096].numpy()) for i in range(0, len(p['x']), 4096)])
y = p['y'].numpy(); idx = p['original_cache_index'].numpy()
L = pd.read_csv(DATA + 'ledgers/train_included_windows.csv', dtype={'association_group': str}, keep_default_na=False, low_memory=False)
assert (L.original_cache_index.values == idx).all() and (L.qualified_y.values == y).all()
fold = L.association_group.map(fold_of).to_numpy()
origin = L.normal_origin.to_numpy()
prob = np.full(len(y), np.nan)
for f in range(3):
    tr = fold != f
    w = np.ones(tr.sum()); yt = y[tr]; ot = origin[tr]
    nm = float((yt == 0).sum())
    for o in ['normal_flight', 'normal_part_of_fault_case']:
        m = ot == o; w[m] = 0.5 * nm / m.sum()
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0).fit(X[tr], yt, sample_weight=w)
    prob[~tr] = model.predict_proba(X[~tr])[:, 1]
    print('fold', f, 'done', round(time.time() - t0), 's', flush=True)
assert np.isfinite(prob).all()
np.savez_compressed(OUT + 'gbt_heldout.npz', prob=prob, fold=fold, original_indices=idx)
print('saved')
