"""ens_internal_test_v1: one-shot internal test evaluation of frozen ENS vs M0.

Order enforced in code: fit GBT on train and record model identity, then open
test. Writes a lock file so a second evaluation refuses to run.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from uav_fault.data.causal_cache import CausalCacheDataset, VERSION  # noqa: E402
from uav_fault.models.cnn.lstm import OnlineMSTCA  # noqa: E402

DATA = Path('D:/UAV-Data/causal_input_v1')
OUT = ROOT / 'artifacts/ens_internal_test_v1_20261105'
CFG = ROOT / 'configs/experiments/ens_internal_test_v1.json'
M0_DIR = ROOT / 'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = x.astype(np.float32); d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


OUT.mkdir(parents=True, exist_ok=True)
lock = OUT / 'TEST_EVALUATED.lock'
if lock.exists():
    raise SystemExit('Internal test already evaluated; refusing a second run')
t0 = time.time()
record = {'config_sha256': sha(CFG), 'm0_best_sha256': sha(M0_DIR / 'best.pt'), 'started': time.strftime('%Y-%m-%d %H:%M:%S')}

# 1) Fit GBT on full train (no test access yet).
p = torch.load(DATA / 'cache/train.pt', weights_only=True, mmap=True)
XT = np.concatenate([stats(p['x'][i:i + 4096].numpy()) for i in range(0, len(p['x']), 4096)])
yT = p['y'].numpy()
LT = pd.read_csv(DATA / 'ledgers/train_included_windows.csv', low_memory=False)
assert (LT.original_cache_index.values == p['original_cache_index'].numpy()).all() and (LT.qualified_y.values == yT).all()
w = np.ones(len(yT)); nm = float((yT == 0).sum())
for o in ['normal_flight', 'normal_part_of_fault_case']:
    m = LT.normal_origin.values == o; w[m] = 0.5 * nm / m.sum()
gbt = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                     random_state=0).fit(XT, yT, sample_weight=w)
import pickle
pickle.dump(gbt, open(OUT / 'gbt_full_train.pkl', 'wb'))
record['gbt_pickle_sha256'] = sha(OUT / 'gbt_full_train.pkl'); record['gbt_n_iter'] = int(gbt.n_iter_)
del XT, p
print('GBT fitted', round(time.time() - t0), 's', flush=True)

# 2) Open test once.
lock.write_text(json.dumps(record))
ds = CausalCacheDataset(DATA, expected_version=VERSION, split='test', purpose='evaluation')
idx = ds.original_cache_indices.numpy()
raw = torch.load(DATA / 'cache/test_evidence.pt', weights_only=True, mmap=True)
sel = ds._selected
XR = raw['x'][sel]  # unnormalized, for GBT
gbt_p = np.concatenate([gbt.predict_proba(stats(XR[i:i + 4096].numpy()))[:, 1] for i in range(0, len(XR), 4096)])
st = json.load(open(DATA / 'norm_stats.json'))
mu = torch.tensor(st['mean']).view(1, -1, 1); sd = torch.tensor(st['std']).view(1, -1, 1)
model = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=0.2).float()
model.load_state_dict(torch.load(M0_DIR / 'best.pt', map_location='cpu', weights_only=True)); model.eval()
torch.set_num_threads(16)
probs = []
with torch.no_grad():
    for i in range(0, len(XR), 1024):
        probs.append(model((XR[i:i + 1024] - mu) / sd).softmax(1).numpy())
m0p = np.concatenate(probs)
labels = np.asarray(ds.labels)
np.savez_compressed(OUT / 'test_predictions.npz', m0_probabilities=m0p, gbt_prob=gbt_p, labels=labels, original_indices=idx)
record.update(test_windows=int(len(idx)), test_original_population=int(ds.original_population),
              test_excluded=int(ds.excluded_population), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
              predictions_sha256=sha(OUT / 'test_predictions.npz'))
lock.write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
