"""Architect independent recomputation of V verdict from raw val logits (server-side).

Reimplements FP@R97, strata, matched-FP C2, R97 C3, AP and the C1 cluster bootstrap
without importing the V evaluator. Read-only; writes architect_verify.json beside the
evaluation output.
"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/UAV-Fault-Diagnosis')
DATA = Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')
V = ROOT / 'artifacts/v_confirm_fixed100_mstca_v1'
SEEDS = [42, 43, 44, 45, 46]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

frame = pd.read_csv(DATA / 'ledgers/val_included_windows.csv')
y = frame.qualified_y.to_numpy(np.int64)
parts = frame.case_uid.str.split('/')
flight, cls = parts.str[1].to_numpy(), parts.str[2].to_numpy()
late = (flight == 'hover') & (frame.window_end_row.to_numpy() * .02 > 30)
assert len(y) == 34710 and (y == 0).sum() == 17740

def prob(path, epoch):
    d = np.load(path)
    assert d['logits'].dtype == np.float32 and d['logits'].shape == (len(y), 2)
    assert np.array_equal(d['original_cache_index'], frame.original_cache_index.to_numpy())
    assert np.array_equal(d['labels'], y) and list(d['epochs']) == [epoch]
    t = torch.as_tensor(d['logits'], device='cuda:0').double().softmax(1)[:, 1]
    return t.cpu().numpy()

def r97(score, w=None, recall=.97):
    w = np.ones(len(y), np.int64) if w is None else w
    order = np.argsort(-score, kind='stable')
    fo = order[y[order] == 1]
    cum = np.cumsum(w[fo], dtype=np.float64)
    thr = score[fo[np.searchsorted(cum, recall * cum[-1], 'left')]]
    return thr, int(w[(y == 0) & (score >= thr)].sum())

def ap(score):
    order = np.argsort(-score, kind='stable'); s = score[order]; yy = y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(yy)[ends].astype(float); tot = (ends + 1).astype(float)
    return float(np.sum(np.diff(np.r_[0, tp / yy.sum()]) * tp / tot))

P = {}
out = {'seeds': {}}
for sd in SEEDS:
    f = V / f'val/seed{sd}'
    rep = json.load(open(f / 'report.json'))
    sel = json.load(open(f / 'selections.json'))
    e0 = sel['V0']['epoch']
    for e, a in [(e0, 'V0'), (100, 'V1')]:
        art = rep['prediction_artifacts'][str(e)]
        assert sha(f / art['file']) == art['sha256']
        P[a, sd] = prob(f / art['file'], e)
    t0, fp0 = r97(P['V0', sd]); t1, fp1 = r97(P['V1', sd])
    n1 = np.sort(P['V1', sd][y == 0])[::-1]
    tm = n1[fp0 - 1]
    out['seeds'][sd] = {'V0_epoch': e0, 'status': rep['status'], 'epochs': rep['epochs'],
        'reproduction': rep.get('reproduction', {}).get('status'),
        'fp': [fp0, fp1], 'ratio': fp1 / fp0, 'ap': [ap(P['V0', sd]), ap(P['V1', sd])],
        'matched_realised_fp': int((P['V1', sd][y == 0] >= tm).sum()), 'thr': [t0, t1, tm]}
classes = sorted(set(cls[y == 1]))
c2, c3 = {}, {}
for c in classes:
    m = (y == 1) & ~late & (cls == c)
    d2 = d3 = 0
    for sd in SEEDS:
        t0, t1, tm = out['seeds'][sd]['thr']
        tp0 = int((m & (P['V0', sd] >= t0)).sum())
        d2 += int((m & (P['V1', sd] >= tm)).sum()) - tp0
        d3 += tp0 - int((m & (P['V1', sd] >= t1)).sum())
    c2[c] = 100 * d2 / (5 * m.sum()); c3[c] = 100 * d3 / (5 * m.sum())
    c3[c + '_n_windows'] = int(m.sum())
ratios = [out['seeds'][s]['ratio'] for s in SEEDS]
gm = float(np.exp(np.mean(np.log(ratios))))
# C1 cluster bootstrap with the registered plan
groups = frame.association_group.astype(str).to_numpy()
uniq, inv = np.unique(groups, return_inverse=True)
rng = np.random.default_rng(20261106)
boot = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(uniq), size=len(uniq)), minlength=len(uniq))[inv]
    boot.append(float(np.exp(np.mean([np.log(r97(P['V1', s], w)[1] / r97(P['V0', s], w)[1]) for s in SEEDS]))))
ci = np.percentile(boot, [2.5, 97.5]).tolist()
dap = float(np.mean([out['seeds'][s]['ap'][1] - out['seeds'][s]['ap'][0] for s in SEEDS]))
ev = V / 'evaluation_val/evaluation.json'
E = json.load(open(ev))
out.update(C1={'gm': gm, 'ci': ci, 'n_below_1': sum(r < 1 for r in ratios)}, C2=c2, C3=c3,
    C4_point_dAP=dap, evaluation_sha256=sha(ev),
    evaluator={'verdict': E['verdict'], 'failed': E['failed_criteria'], 'ratio_ci': E['uncertainty']['aggregate_ratio_ci'],
               'C2': E['comparison']['C2_seed_mean_recall_delta_pp'], 'C3': E['comparison']['C3_seed_mean_recall_drop_pp']})
# per-seed GPS drop at R97 (descriptive)
m = (y == 1) & ~late & (cls == 'GPS')
out['GPS_drop_per_seed_pp'] = {sd: 100 * (int((m & (P['V0', sd] >= out['seeds'][sd]['thr'][0])).sum())
    - int((m & (P['V1', sd] >= out['seeds'][sd]['thr'][1])).sum())) / m.sum() for sd in SEEDS}
json.dump(out, open(V / 'architect_verify.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
