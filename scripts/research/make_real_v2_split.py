"""real_v2 split: re-split Real into a development pool (~70%) and a sealed final test (~30%).

Metadata only (tasks.json): no sensor values, labels-in-time or predictions are read.
Rule (user approval 2026-11-08):
  unit = association group; stratum = (flight directory, family key), where family key is
  motor / sensor if the group contains a fault case of that family, otherwise normal_full
  (group contains a full normal flight role) or normal_segment (supplemental normal only).
  Within a stratum, groups are ordered by sha256('real_v2|' + group); the first
  floor(0.3 * n + 0.5) are sealed, the rest are dev and get folds 0..4 round-robin in the same order.
"""
import collections, csv, hashlib, json, math, sys
from pathlib import Path

DATA = Path('D:/UAV-Data/causal_input_v1')
OUT = Path('D:/UAV-Fault-Diagnosis/configs/data/real_v2_split.csv')
FULL_NORMAL_ROLES = {'test_normal', 'adapt_normal', 'calibrate_normal'}


def h(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def main():
    tasks = [t for t in json.load(open(DATA / 'tasks.json', encoding='utf-8')) if t['split'] == 'real']
    assert len(tasks) == 465
    groups = collections.defaultdict(list)
    for t in tasks:
        groups[t['group']].append(t)
    stratum = {}
    for g, mem in groups.items():
        flights = {m['case_uid'].split('/')[1] for m in mem}
        assert len(flights) == 1, g
        fams = {m['family'] for m in mem}
        faults = fams - {'normal'}
        assert len(faults) <= 1, g
        if faults:
            key = faults.pop()
        else:
            key = 'normal_full' if any(m['role'] in FULL_NORMAL_ROLES for m in mem) else 'normal_segment'
        stratum[g] = (flights.pop(), key)
    part, fold = {}, {}
    by = collections.defaultdict(list)
    for g, s in stratum.items():
        by[s].append(g)
    for s, gs in by.items():
        gs.sort(key=lambda g: h('real_v2|' + g))
        k = math.floor(0.3 * len(gs) + 0.5)
        for i, g in enumerate(gs):
            if i < k:
                part[g], fold[g] = 'sealed', ''
            else:
                part[g], fold[g] = 'dev', (i - k) % 5
    rows = sorted(({'case_uid': t['case_uid'], 'association_group': t['group'], 'flight': stratum[t['group']][0],
                    'family': t['family'], 'stratum_family': stratum[t['group']][1], 'role_v1': t['role'],
                    'label_quarantine': bool(t['label_quarantine']), 'partition': part[t['group']],
                    'dev_fold': fold[t['group']]} for t in tasks), key=lambda r: r['case_uid'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='\n', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    c = collections.Counter((r['partition'], r['stratum_family']) for r in rows)
    cg = collections.Counter((part[g], stratum[g][1]) for g in stratum)
    for k in sorted(c):
        print(k, 'cases', c[k], 'groups', cg[k])
    print('dev folds (cases):', sorted(collections.Counter(r['dev_fold'] for r in rows if r['partition'] == 'dev').items()))
    print('quarantine:', collections.Counter(r['partition'] for r in rows if r['label_quarantine']))
    print('sha256', hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    sys.exit(main())
