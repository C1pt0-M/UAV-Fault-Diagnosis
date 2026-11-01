"""Read-only independent checks of the saved source-only causal input pilot.

No raw/test/Real expansion, models or fitting choices. Independent statistics
recomputation allows tiny float64 summation-order roundoff, not window mismatch.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/causal_input_parity_v1_20261031'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = ART / 'independent_verification.json'
    if target.exists():
        raise FileExistsError('Preserve previous verification')
    records = [json.loads(s) for s in (ART / 'case_results.jsonl').read_text(encoding='utf-8').splitlines()]
    norm = json.loads((ART / 'pilot_norm_stats.json').read_text(encoding='utf-8'))
    policy = json.loads((ART / 'input_policy.json').read_text(encoding='utf-8'))
    summary = json.loads((ART / 'summary.json').read_text(encoding='utf-8'))
    assert sha(ART / 'input_policy.json') == summary['policy_sha256']
    assert sha(ART / 'pilot_norm_stats.json') == summary['normalization_sha256']
    assert sha(ART / 'pipeline_at_pilot_run.py') == summary['pipeline_sha256']
    columns = policy['columns']; mu = np.asarray(norm['mean'], dtype=np.float32); sd = np.asarray(norm['std'], dtype=np.float32)
    sums = np.zeros(26); squares = np.zeros(26); fit_windows = frames = checked = 0
    for record in records:
        path = ART / record['output_file']
        assert sha(path) == record['output_sha256']
        with np.load(path, allow_pickle=False) as d:
            matrix = d['canonical_rows']; valid = d['row_input_valid']; starts = d['window_start_row']; wm = d['window_input_valid']; grid = d['grid_us']
            assert matrix.shape == (record['rows'], 26) and np.all(np.diff(grid) == 20000)
            assert np.array_equal(np.isfinite(matrix).all(axis=1), valid)
            assert np.array_equal([valid[s:s+150].all() for s in starts], wm)
            for start in starts[wm]:
                x = matrix[start:start+150].copy()
                acceleration = x[:, [columns.index('accel_' + axis) for axis in 'xyz']]
                assert np.array_equal(x[:, columns.index('acc_norm')], np.sqrt((acceleration * acceleration).sum(axis=1)))
                x[:, columns.index('baro_alt_relative')] -= x[0, columns.index('baro_alt_relative')]
                normalized = (x.astype(np.float32) - mu) / sd
                assert normalized.shape == (150, 26) and np.isfinite(normalized).all()
                checked += 1
            if record['phase'] == 'fit':
                picks = np.unique(np.linspace(0, len(starts)-1, 64).round().astype(int)) if len(starts)>64 else np.arange(len(starts))
                selected, good = starts[picks], wm[picks]
                assert norm['fit_cases'][record['case_uid']]['valid_cap64_windows'] == int(good.sum())
                for start in selected[good]:
                    v = matrix[start:start+150].copy()
                    v[:, columns.index('baro_alt_relative')] -= v[0, columns.index('baro_alt_relative')]
                    v = v.astype(np.float32).astype(np.float64)
                    sums += v.sum(axis=0); squares += (v*v).sum(axis=0); frames += len(v); fit_windows += 1
            if record['gap_endpoint']:
                hits = np.flatnonzero(grid == record['gap_endpoint']['endpoint_us'])
                assert len(hits) == 1 and not valid[hits[0]]
        for path, expected in record['identities'].items():
            assert sha(path) == expected
    mean = sums / frames; sigma = np.sqrt(np.maximum(squares / frames - mean * mean, 0)); sigma[sigma == 0] = 1
    mean_error = float(np.max(np.abs(mean - np.asarray(norm['mean']))))
    std_error = float(np.max(np.abs(sigma - np.asarray(norm['std']))))
    # Independent contiguous/strided reductions differ only in float64 order.
    # This tolerance applies to stats recomputation, NEVER batch/stream tensors.
    assert mean_error < 1e-10 and std_error < 1e-10
    assert fit_windows == 760 and frames == 114000 and checked == 7729
    assert len(records) == 16 and sum(r['ineligible_windows_retained'] for r in records) == 214
    result = {'status': 'PASS_INDEPENDENT_SAVED_PILOT_INPUT_AND_NORM_CHECK',
        'cases': 16, 'rows': sum(r['rows'] for r in records), 'original_windows': sum(r['original_stride5_windows'] for r in records),
        'eligible_normalized_windows_checked': checked, 'ineligible_windows_retained': 214,
        'source_train_normalization_windows': fit_windows, 'source_train_normalization_frame_contributions': frames,
        'normalization_recompute_mean_max_abs_error': mean_error, 'normalization_recompute_std_max_abs_error': std_error,
        'normalization_recompute_only_tolerance': 1e-10, 'batch_stream_tensor_tolerance': 0,
        'all_original_used_input_hashes_unchanged': True, 'live_chain_or_model_tested': False, 'verifier_sha256': sha(__file__)}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
