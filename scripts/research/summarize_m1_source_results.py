"""Recompute fixed-threshold source-val M1 comparisons; no model or tuning."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
from smoke_normal_source_sampling import stratified_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(labels, probabilities):
    predicted = probabilities.argmax(1)
    cm = np.bincount(labels*2+predicted, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = map(int, cm.ravel())
    return {'windows': len(labels), 'confusion_matrix': cm.tolist(),
            'precision': tp/(tp+fp) if tp+fp else None,
            'fault_recall': tp/(tp+fn) if tp+fn else None,
            'normal_fpr': fp/(tn+fp) if tn+fp else None,
            'fault_misses': fn, 'normal_false_positives': fp}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'features', 'results', 'verified', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve source comparison')
    summary = read(a.verified/'summary.json')
    assert summary['status'] == 'PASS_REGISTERED_M1_AND_MATCHED_SOURCE_RESULTS'
    export = torch.load(a.features/'cache/val.pt', map_location='cpu', weights_only=True, mmap=True)
    source = pd.read_csv(a.data/'ledgers/val_included_windows.csv', keep_default_na=False, low_memory=False)
    f = source.iloc[export['base_compact_index'].numpy()].copy().reset_index(drop=True)
    f['label'] = f.qualified_y.astype(int)
    f['normal_origin'] = np.where(f.label == 1, 'fault_endpoint', np.where(f.case_level2_label == 'normal', 'original_normal_case', 'normal_part_of_fault_case'))
    f['flight_status'] = f.case_uid.str.split('/').str[1]
    f['fault_directory'] = f.case_uid.str.split('/').str[2]
    f['end_relative_to_crop_s'] = (f.window_end_row-1)*.02
    assert len(f) == 32381 and np.array_equal(f.original_cache_index, export['original_cache_index'])
    a.output.mkdir(parents=True)
    rows = []
    for item in summary['rows']:
        path = a.results/'runs'/item['variant']/item['arm']/'source_validation_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            y, q = z['labels'], z['probabilities']
            assert np.array_equal(y, f.label) and np.array_equal(z['original_indices'], f.original_cache_index)
        original = stratified_metrics(y, q.argmax(1), f)
        assert original['overall']['confusion_matrix'] == item['confusion_matrix']
        row = dict(item, **{'fixed_threshold': .5, 'overall_fixed_threshold': metrics(y, q)})
        row['six_fault_directories'] = {}
        for subtype, part in f[f.label == 1].groupby('fault_directory', sort=True):
            row['six_fault_directories'][subtype] = metrics(y[part.index], q[part.index])
        assert set(row['six_fault_directories']) == {'motor','GPS','accelerometer','gyroscope','barometer','magnetometer'}
        assert sum(x['windows'] for x in row['six_fault_directories'].values()) == int((y == 1).sum())
        row['all_normal_by_flight_status'] = {str(k): metrics(y[g.index], q[g.index]) for k, g in f[f.label == 0].groupby('flight_status', sort=True)}
        row['independent_normal_by_flight_status'] = {str(k): metrics(y[g.index], q[g.index]) for k, g in f[f.normal_origin == 'original_normal_case'].groupby('flight_status', sort=True)}
        hover = f[(f.normal_origin == 'original_normal_case') & (f.flight_status == 'hover')]
        row['independent_normal_hover_crop_time'] = {}
        for key, mask in [('first30s', hover.end_relative_to_crop_s <= 30), ('after30s', hover.end_relative_to_crop_s > 30)]:
            g = hover[mask]
            row['independent_normal_hover_crop_time'][key] = metrics(y[g.index], q[g.index])
        case_rows = []
        for uid, g in hover.groupby('case_uid', sort=True):
            case_rows.append({'case_uid': uid, **metrics(y[g.index], q[g.index]), 'after30s': metrics(y[g[g.end_relative_to_crop_s > 30].index], q[g[g.end_relative_to_crop_s > 30].index])})
        row['independent_normal_hover_cases'] = case_rows
        rows.append(row)
    differences = []
    for arm in ['uniform_replacement', 'balanced_source_normal']:
        base = next(r for r in rows if r['arm'] == arm and r['variant'] == 'M0_matched')
        m1 = next(r for r in rows if r['arm'] == arm and r['variant'] == 'M1')
        differences.append({'arm': arm, 'M1_minus_M0': {k: m1[k]-base[k] for k in ['accuracy','macro_f1','fault_recall','normal_fpr','original_normal_fpr','hover_fpr','average_precision_step_integral']},
                            'fault_directory_recall_differences': {k: m1['six_fault_directories'][k]['fault_recall']-v['fault_recall'] for k,v in base['six_fault_directories'].items()},
                            'normal_false_positive_count_difference': m1['overall_fixed_threshold']['normal_false_positives']-base['overall_fixed_threshold']['normal_false_positives'],
                            'fault_miss_count_difference': m1['overall_fixed_threshold']['fault_misses']-base['overall_fixed_threshold']['fault_misses']})
    result = {'status': 'PASS_FIXED_SOURCE_VAL_DIRECTORY_AND_HOVER_ANALYSIS', 'checked_at': datetime.now(timezone.utc).isoformat(),
              'verified_summary_sha256': hashlib.sha256((a.verified/'summary.json').read_bytes()).hexdigest(), 'rows': rows, 'paired_differences': differences,
              'scope': 'Frozen0.5 only, source-val common150 IDs, same arm pretraining. AP already computed by independent verifier. No threshold selection, new training, independent-seed significance or final test/Real.'}
    (a.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(differences, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
