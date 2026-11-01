"""Exclude already-invalid pilot windows; never delete source files or invent labels.

This is input-only selection for the existing 16-case pilot, NOT a full-dataset
training release. Preserve original train/val and window identities and all masks.
"""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from audit_rflymad_raw_sources import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/causal_input_parity_v1_20261031'
OUTPUT = ROOT / 'artifacts/causal_input_selection_v1_20261031'


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior selection; use an independent version')
    cases = [json.loads(s) for s in (SOURCE / 'case_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases) == 16
    included, excluded, file_summary = [], [], []
    identities = {str(SOURCE / name): digest(SOURCE / name) for name in
                  ['case_results.jsonl', 'summary.json', 'input_policy.json', 'pilot_norm_stats.json']}
    for case in cases:
        path = SOURCE / case['output_file']
        assert digest(path) == case['output_sha256']
        identities[str(path)] = case['output_sha256']
        # Saved diagnostics established these invalid rows were stale, not
        # missing/nonfinite. Do not silently relabel other failure types.
        assert all(c['missing'] == 0 and c['nonfinite'] == 0
                   and c['stale'] == c['invalid_frames'] for c in case['group_invalid_counts'].values())
        with np.load(path, allow_pickle=False) as data:
            starts = data['window_start_row']; grid = data['grid_us']
            rows_valid = data['row_input_valid']; saved = data['window_input_valid']
            assert starts.dtype == np.int64 and rows_valid.dtype == np.bool_
            actual = np.asarray([rows_valid[s:s+150].all() for s in starts], dtype=bool)
            assert np.array_equal(actual, saved)
            assert int(saved.sum()) == case['valid_windows']
            for index, start in enumerate(starts):
                end = int(start) + 150
                row = {'case_uid': case['case_uid'], 'split': case['split'],
                    'pilot_phase': case['phase'], 'original_case_window_index': index,
                    'window_start_row': int(start), 'window_end_row': end,
                    'window_start_us': int(grid[start]), 'window_endpoint_us': int(grid[end-1]),
                    'input_eligible': bool(saved[index]), 'invalid_input_frames': int((~rows_valid[start:end]).sum()),
                    'reason': '' if saved[index] else 'required_input_stale_under_frozen_pilot_policy',
                    'source_reference': str(path), 'training_authorized': False}
                (included if saved[index] else excluded).append(row)
        file_summary.append({'case_uid': case['case_uid'], 'split': case['split'],
            'original_windows': case['original_stride5_windows'], 'included_windows': case['valid_windows'],
            'excluded_windows': case['ineligible_windows_retained'],
            'whole_file_excluded': case['valid_windows'] == 0})
    key = lambda r: (r['case_uid'], r['window_start_row'], r['window_end_row'])
    good = {key(r) for r in included}; bad = {key(r) for r in excluded}
    assert len(good) == 7729 and len(bad) == 214 and not good.intersection(bad)
    assert len(good.union(bad)) == 7943
    assert sum(r['excluded_windows'] > 0 for r in file_summary) == 5
    assert not any(r['whole_file_excluded'] for r in file_summary)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT / 'included_input_windows.csv', included)
    write_csv(OUTPUT / 'excluded_input_windows.csv', excluded)
    write_csv(OUTPUT / 'case_selection_summary.csv', file_summary)
    for name, expected in identities.items():
        assert digest(name) == expected
    for name, expected_count in [('included_input_windows.csv', 7729), ('excluded_input_windows.csv', 214)]:
        with (OUTPUT / name).open(encoding='utf-8', newline='') as stream:
            persisted = list(csv.DictReader(stream))
        assert len(persisted) == expected_count
        expected = included if name.startswith('included') else excluded
        assert [key(r) for r in persisted] == [(r['case_uid'], str(r['window_start_row']), str(r['window_end_row'])) for r in expected]
    summary = {'status': 'APPLIED_INPUT_ONLY_PILOT_WINDOW_EXCLUSION', 'files_examined': 16,
        'partially_affected_files': 5, 'whole_files_excluded': 0, 'original_windows': 7943,
        'input_eligible_windows': 7729, 'input_excluded_windows': 214,
        'excluded_by_original_split': dict(Counter(r['split'] for r in excluded)),
        'included_by_original_split': dict(Counter(r['split'] for r in included)),
        'exclusion_reason': 'Input observation age exceeds frozen pilot per-group budget in at least one required frame.',
        'original_inputs_and_masks_unchanged': True, 'original_case_window_ids_and_split_preserved': True,
        'physical_files_deleted': False, 'model_or_scores_used': False, 'training_authorized': False,
        'scope': 'Only existing fixed 16-case source train/val pilot; not a full-dataset count or deletion. Input eligibility alone does not establish label eligibility or live receive-time provenance. Full continuous replay and original exclusion denominators remain intact.',
        'source_identities': identities, 'script_sha256': digest(__file__),
        'outputs_sha256': {name: digest(OUTPUT / name) for name in
            ['included_input_windows.csv', 'excluded_input_windows.csv', 'case_selection_summary.csv']}}
    with (OUTPUT / 'selection_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_identities','outputs_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
