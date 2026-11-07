"""Architect-side independent recomputation of T phase-1 results (read-only)."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path('/root/autodl-tmp/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_s_ablation as s
DATA = Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')
RUNS = ROOT / 'artifacts/t_selection_mstca_v1/cv'
frame = s.read_ledger(DATA)
y = frame.qualified_y.to_numpy(np.int64)
parts = frame.case_uid.str.split('/')
flight = parts.str[1].to_numpy(); cls = parts.str[2].to_numpy()
late = (flight == 'hover') & (frame.window_end_row.to_numpy() * .02 > 30)
ids_all = frame.original_cache_index.to_numpy(np.int64)
pos = {v: i for i, v in enumerate(ids_all)}

def fp97(score, r=.97):
    f = np.sort(score[y == 1])[::-1]
    k = int(np.ceil(r * len(f)))  # need >= k faults detected
    t = f[k - 1]
    return int((score[y == 0] >= t).sum()), t

SEEDS = [42, 43, 44, 45, 46]
pools = {a: {} for a in ('T0', 'T1', 'T2')}
traj = {sd: [] for sd in SEEDS}
sel_table = {}
for sd in SEEDS:
    P = {a: np.full(len(y), np.nan) for a in pools}
    for f in range(3):
        d = RUNS / f'seed{sd}_fold{f}'
        sel = json.load(open(d / 'selections.json'))
        sel_table[f'{sd}_{f}'] = {a: sel[a]['epoch'] for a in pools}
        z = np.load(d / 'heldout_predictions.npz')
        idx = np.array([pos[i] for i in z['original_cache_index']])
        assert (z['labels'] == y[idx]).all()
        pr = z['probabilities']
        traj[sd].append((idx, pr))
        for a in pools:
            P[a][idx] = pr[sel[a]['epoch'] - 1]
    for a in pools:
        assert np.isfinite(P[a]).all()
        pools[a][sd] = P[a].astype(np.float64)

out = {'fp': {}, 'class_recall_outside_late': {}, 'selected': sel_table}
classes = sorted(set(cls[y == 1]))
for a in pools:
    out['fp'][a] = {}
    out['class_recall_outside_late'][a] = {}
    for sd in SEEDS:
        fp, t = fp97(pools[a][sd])
        out['fp'][a][sd] = fp
        m = pools[a][sd] >= t
        out['class_recall_outside_late'][a][sd] = {c: float(m[(y == 1) & ~late & (cls == c)].mean()) for c in classes}
ens = np.mean([pools['T1'][sd] for sd in SEEDS], axis=0)
out['fp']['T3'] = fp97(ens)[0]
for a in ('T1', 'T2'):
    r = [out['fp'][a][sd] / out['fp']['T0'][sd] for sd in SEEDS]
    out[f'{a}_ratio_gm'] = float(np.exp(np.mean(np.log(r))))
    out[f'{a}_class_drop_pp'] = {c: 100 * float(np.mean([out['class_recall_outside_late']['T0'][sd][c] - out['class_recall_outside_late'][a][sd][c] for sd in SEEDS])) for c in classes}
    out[f'{a}_gps_drop_per_seed_pp'] = {sd: 100 * (out['class_recall_outside_late']['T0'][sd]['GPS'] - out['class_recall_outside_late'][a][sd]['GPS']) for sd in SEEDS}
# descriptive: class recall of T1 / T0 when thresholded to the SAME FP as T0 (not frozen)
desc = {}
for sd in SEEDS:
    fp0 = out['fp']['T0'][sd]
    n = np.sort(pools['T1'][sd][y == 0])[::-1]
    t = n[fp0 - 1]
    m = pools['T1'][sd] >= t
    desc[sd] = {c: float(m[(y == 1) & ~late & (cls == c)].mean()) for c in classes}
out['descriptive_T1_at_T0_fp_class_recall'] = desc
# descriptive oracle curve (pooled per epoch)
curve = {}
for sd in SEEDS:
    c = []
    for e in range(100):
        P = np.full(len(y), np.nan)
        for idx, pr in traj[sd]:
            P[idx] = pr[e]
        c.append(fp97(P.astype(np.float64))[0])
    curve[sd] = c
out['curve_fp_by_epoch'] = curve
json.dump(out, open('/tmp/t_verify.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in out if k not in ('curve_fp_by_epoch', 'selected', 'class_recall_outside_late', 'descriptive_T1_at_T0_fp_class_recall')}, indent=1))
for sd in SEEDS:
    c = curve[sd]; print(sd, 'oracle', int(np.argmin(c)) + 1, min(c), 'ep50', c[49], 'ep70', c[69], 'ep90', c[89], 'ep100', c[99])
