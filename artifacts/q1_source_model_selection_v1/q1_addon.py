"""Descriptive add-on to Q1 (not part of the frozen rule): paired E1/GBT under strict variant; 3-seed comparisons."""
import sys, json, numpy as np, multiprocessing as mp
sys.path.insert(0, 'scripts/research')
import q1_source_model_selection as q
from q1_source_model_selection import e, s, m, G
data = '/root/autodl-tmp/causal_source_v1_20261031/unpacked/data'
frame = s.read_ledger(data); _, outer = s.split_ledger(frame, 0)
y, flight, classes, independent, late = m.strata(frame)
pos = {int(v): i for i, v in enumerate(frame.original_cache_index.to_numpy())}
base, _ = e.load_scores(frame, outer)
gbt = base['GBT'][42]
sc = {'T0': base['T0'], 'T1': base['T1'], 'GBT': base['GBT'], 'E1': base['E1'], 'E0': base['E0']}
sc['W1'] = q.load_w(frame, outer, pos, 'W1'); sc['W1+GBT'] = {k: (v + gbt) / 2 for k, v in sc['W1'].items()}
sc['S2'] = q.load_s(frame, outer, pos, 'S2', 'best'); sc['S2+GBT'] = {k: (v + gbt) / 2 for k, v in sc['S2'].items()}
cls = sorted(set(classes[y == 1]))
strata = [(y == 1) & (classes == c) for c in cls] + [(y == 1) & late & (classes == 'GPS')]
plan = [(a, q.S3 if a.startswith('S') else q.S5) for a in sc]
G.update(ranked={(a, sd): q.Ranked(sc[a][sd], y, strata) for a, ss in plan for sd in ss}, n_classes=len(cls), plan=plan)
point = q.evaluate(np.ones(len(y), dtype=np.int64), plan)
uniq, inv = np.unique(frame.association_group.to_numpy(), return_inverse=True)
rng = np.random.default_rng(20261106)
counts = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(2000)]
G['inv'] = inv
with mp.get_context('fork').Pool(48) as p:
    boot = [r for part in p.map(q.boot_worker, [counts[i::48] for i in range(48)]) for r in part]
out = {}
for a, b, ss, k in [('E1', 'GBT', q.S5, 1), ('E1', 'GBT', q.S5, 0), ('E1', 'E0', q.S5, 0), ('E1', 'W1+GBT', q.S5, 0),
                    ('S2+GBT', 'E1', q.S3, 0), ('S2+GBT', 'E0', q.S3, 0), ('S2+GBT', 'GBT', q.S3, 0), ('E1', 'GBT', q.S3, 0)]:
    key = f'{a}/{b} {"strict" if k else "main"} seeds{ss[0]}-{ss[-1]}'
    out[key] = {'point': q.gm_ratio(point, a, b, ss, k), 'ci': m.percentile([q.gm_ratio(bb, a, b, ss, k) for bb in boot])}
    print(key, round(out[key]['point'], 3), [round(x, 3) for x in out[key]['ci']])
json.dump(out, open('artifacts/q1_source_model_selection_v1/descriptive_pairs.json', 'w'), indent=1)
