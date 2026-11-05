"""ens_event_threshold_v1 step 1: per-frame M0 and GBT scores for train (held-out) and val.

Train cases are scored by fold models that did not see them. GBT per fold is
refit with the frozen recipe and checked against feature_groupcv_v1 held-out
probabilities at cap64 windows. Output: one npz per case with timestamps, labels
and both scores; no event aggregation here.
"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_ens_events_val import case_windows, stats, sha, DATA, W  # noqa: E402
from uav_fault.models.cnn.lstm import OnlineMSTCA  # noqa: E402

A = ROOT / 'artifacts/ens_event_threshold_v1_20261106'
CV = ROOT / 'artifacts/feature_groupcv_v1_20261105'
OUT = Path('D:/UAV-Data/ens_event_threshold_v1_scores')


def fold_of(g):
    return int(hashlib.sha256(('h1cv|' + g).encode()).hexdigest(), 16) % 3


def load_m0(path):
    m = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=0.2).float()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)); m.eval(); return m


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(16)
    st = json.load(open(DATA / 'norm_stats.json'))
    mu = np.asarray(st['mean'], np.float32)[None, :, None]; sd = np.asarray(st['std'], np.float32)[None, :, None]
    # Refit fold GBTs (frozen recipe) and check against saved held-out probabilities.
    p = torch.load(DATA / 'cache/train.pt', weights_only=True, mmap=True)
    X = np.concatenate([stats(p['x'][i:i + 4096].numpy().astype(np.float32)) for i in range(0, len(p['x']), 4096)])
    y = p['y'].numpy()
    L = pd.read_csv(DATA / 'ledgers/train_included_windows.csv', dtype={'association_group': str}, keep_default_na=False, low_memory=False)
    fold = L.association_group.map(fold_of).to_numpy(); origin = L.normal_origin.to_numpy()
    saved = np.load(CV / 'gbt_heldout.npz')
    gbts = {}
    for f in range(3):
        tr = fold != f; yt = y[tr]; w = np.ones(tr.sum()); nm = float((yt == 0).sum())
        for o in ['normal_flight', 'normal_part_of_fault_case']:
            m = origin[tr] == o; w[m] = 0.5 * nm / m.sum()
        g = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           random_state=0).fit(X[tr], yt, sample_weight=w)
        diff = float(np.abs(g.predict_proba(X[~tr])[:, 1] - saved['prob'][~tr]).max())
        if diff > 1e-9:
            raise ValueError(f'Fold {f} GBT refit does not reproduce saved held-out ({diff})')
        gbts[f] = g; print('fold', f, 'GBT reproduced, max diff', diff, flush=True)
    del X, p
    models = {('train', f): (load_m0(A / f'models/m0_fold{f}.pt'), gbts[f]) for f in range(3)}
    models[('val', -1)] = (load_m0(ROOT / 'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/best.pt'),
                           pickle.load(open(ROOT / 'artifacts/ens_internal_test_v1_20261105/gbt_full_train.pkl', 'rb')))
    tasks = [t for t in json.load(open(DATA / 'tasks.json')) if t['split'] in ('train', 'val')]
    group = dict(zip(L.case_uid, L.association_group))
    t0 = time.time(); index = []
    for k, t in enumerate(tasks):
        uid = t['case_uid']; key = uid.replace('/', '__')
        if t['split'] == 'train':
            g = group.get(uid, t.get('group', ''))
            f = fold_of(g); mk = ('train', f)
        else:
            f = -1; mk = ('val', -1)
        m0, gbt = models[mk]
        with np.load(DATA / 'cases' / (key + '.npz')) as z:
            grid = z['timestamp_us']; rows = z['canonical_rows']; valid = z['row_input_valid']
            ref = z['reference_state']; lv = z['row_label_valid']
        ends, xr = case_windows(rows, valid)
        pm = np.full(len(grid), np.nan, np.float32); pg = np.full(len(grid), np.nan, np.float32)
        if len(ends):
            with torch.no_grad():
                pm[ends] = np.concatenate([m0(torch.from_numpy((xr[i:i + 2048] - mu) / sd)).softmax(1)[:, 1].numpy() for i in range(0, len(xr), 2048)])
            pg[ends] = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
        np.savez_compressed(OUT / (key + '.npz'), timestamp_us=grid, m0=pm, gbt=pg, reference_state=ref, row_label_valid=lv)
        index.append({'case_uid': uid, 'split': t['split'], 'fold': f, 'domain': t['domain']})
        if k % 200 == 0:
            print(k, uid, round(time.time() - t0), 's', flush=True)
    pd.DataFrame(index).to_csv(OUT / 'index.csv', index=False)
    json.dump({'cases': len(index), 'elapsed_s': time.time() - t0,
               'm0_fold_sha256': {f: sha(A / f'models/m0_fold{f}.pt') for f in range(3)}}, open(OUT / 'receipt.json', 'w'), indent=2)
    print('done', len(index))


if __name__ == '__main__':
    main()
