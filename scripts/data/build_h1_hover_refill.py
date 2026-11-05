"""H1: extra stride-5 windows for train independent-normal hover cases.

Reads causal_input_v1 cases/ (read-only). Uses the same eligibility rule and the
same canonical_window transform (identity normalization, like the original
build) as scripts/data/build_full_causal_input.py. Windows already in cap64 are
skipped so the training population gains only new start rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from uav_fault.causal_input import canonical_window, window_input_valid  # noqa: E402

WINDOW, STRIDE = 150, 5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse overwrite: ' + str(args.output))
    contract = json.loads((args.data / 'input_contract.json').read_text(encoding='utf-8'))
    columns = contract['columns']
    ledger = pd.read_csv(args.data / 'ledgers/train_included_windows.csv',
                         dtype={'association_group': str}, keep_default_na=False, low_memory=False)
    target = ledger[ledger.case_uid.str.contains('/hover/') & (ledger.case_level2_label == 'normal')]
    cases = sorted(target.case_uid.unique())
    if len(cases) != 56 or not (target.normal_origin == 'normal_flight').all():
        raise ValueError('Unexpected target population')

    xs, rows = [], []
    for uid in cases:
        meta = target[target.case_uid == uid]
        with np.load(args.data / 'cases' / (uid.replace('/', '__') + '.npz')) as z:
            matrix = z['canonical_rows']; valid = z['row_input_valid']
            label_valid = z['row_label_valid']; reference = z['reference_state']
        starts = np.arange(0, len(valid) - WINDOW + 1, STRIDE, dtype=np.int64)
        ends = starts + WINDOW
        ok = window_input_valid(valid, starts, window_frames=WINDOW) & label_valid[ends - 1] & np.isin(reference[ends - 1], [0, 1])
        if (reference[ends[ok] - 1] != 0).any():
            raise ValueError('Normal case has fault endpoint: ' + uid)
        existing = set(meta.window_start_row.astype(np.int64))
        if not existing.issubset(set(starts[ok])):
            raise ValueError('cap64 window not on eligible stride grid: ' + uid)
        new = [s for s in starts[ok] if s not in existing]
        for s in new:
            xs.append(canonical_window(matrix[s:s + WINDOW], columns, np.ones(WINDOW, dtype=bool),
                                       mean=np.zeros(26), std=np.ones(26)))
            rows.append({'case_uid': uid, 'domain': meta.domain.iloc[0], 'association_group': meta.association_group.iloc[0],
                         'window_start_row': int(s), 'window_end_row': int(s + WINDOW), 'qualified_y': 0,
                         'case_level2_label': 'normal', 'normal_origin': 'normal_flight'})
    x = np.stack(xs).astype(np.float32)
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True)
    np.save(args.output / 'x.npy', x)
    frame.to_csv(args.output / 'windows.csv', index=False)
    manifest = {
        'name': 'h1_hover_refill_v1', 'source_data_version': contract['version'],
        'rule': 'train independent-normal hover cases; all stride-5 150-frame windows with all rows input-valid, endpoint label valid and reference 0; cap64 starts excluded',
        'transform': 'canonical_window with identity normalization (unnormalized, same as cache/train.pt)',
        'cases': len(cases), 'windows': len(frame),
        'late_gt30s_windows': int((frame.window_end_row * 0.02 > 30).sum()),
        'by_domain': frame.domain.value_counts().to_dict(),
        'x_shape': list(x.shape), 'x_sha256': sha(args.output / 'x.npy'),
        'windows_csv_sha256': sha(args.output / 'windows.csv'),
        'source_identities': {name: sha(args.data / name) for name in ['input_contract.json', 'norm_stats.json', 'ledgers/train_included_windows.csv']},
        'builder_sha256': sha(__file__),
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
