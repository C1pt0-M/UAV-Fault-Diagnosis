"""Verify recovered corrected-budget source results, no model inference."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
sys.path.insert(0, str(ROOT / 'scripts/research'))
from smoke_normal_source_sampling import stratified_metrics
from train_source_normal_sampling import selection_step, state_digest
from uav_fault.models.train_online_detector import get_warmup_cosine_lr


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    output = HERE / 'local_results_verification.json'
    if output.exists():
        raise FileExistsError('Preserve completed verification')
    data = Path('D:/UAV-Data/causal_input_v1')
    saved = HERE / 'results'
    short = ROOT / 'artifacts/causal_source_remote_v1_20261031/results/runs'
    config = read(HERE / 'corrected_longbudget_protocol.json')
    assert sha(HERE / 'corrected_longbudget_protocol.json') == (HERE / 'corrected_longbudget_protocol.json.sha256').read_text().strip()
    for relative, expected in config['data_identities'].items():
        assert sha(data / relative) == expected, relative
    for name, identity in read(saved / 'verified_results_manifest.json')['files'].items():
        assert sha(saved / name) == identity['sha256'], name
        assert (saved / name).stat().st_size == identity['bytes']
    frame = pd.read_csv(data / 'ledgers/val_included_windows.csv', keep_default_na=False, low_memory=False)
    frame['label'] = frame.qualified_y.astype(int)
    frame['normal_origin'] = np.where(frame.label == 1, 'fault_endpoint', np.where(frame.case_level2_label == 'normal', 'original_normal_case', 'normal_part_of_fault_case'))
    frame['flight_status'] = frame.case_uid.str.split('/').str[1]
    frame['cache_index'] = np.arange(len(frame))
    assert len(frame) == 34710
    lengths = read(ROOT / 'artifacts/causal_train_normal_diagnosis_v1_20261101/source_normal_crop_lengths.json')['rows_by_case_uid']
    pure = frame.case_level2_label.eq('normal').to_numpy()
    hover = pure & frame.flight_status.eq('hover').to_numpy()
    assert int(hover.sum()) == 754
    progress = (frame.loc[hover, 'window_end_row'].to_numpy() - 1) / (frame.loc[hover, 'case_uid'].map(lengths).to_numpy() - 1)
    quarters = np.minimum(3, np.floor(progress * 4).astype(int))
    records = {}; initials = []
    for arm in config['arms']:
        root = saved / 'runs' / arm
        report = read(root / 'report.json'); history = read(root / 'history.json')
        assert report['status'] == 'COMPLETE_FINITE_SOURCE_RETRAIN' and report['seed'] == 42 and not report['cpu_check']
        assert report['protocol_sha256'] == sha(HERE / 'corrected_longbudget_protocol.json')
        assert report['checkpoint_reload_matches'] and report['no_final_test_or_Real_reads']
        assert report['validation_windows'] == 34710 and report['source_train_windows'] == 162069
        assert report['epochs'] == len(history) <= 100
        state = dict(best_score=-float('inf'), best_epoch=None, significant_reference=-float('inf'), bad_epochs=0)
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['gradient_steps'] == 317 and np.isfinite(row['train_loss'])
            assert sum(row['realized_training_origins'].values()) == 162069
            expected_lr = get_warmup_cosine_lr(epoch - 1, 5, 100, .0003, .000001)
            # Windows/Linux libm cosine differs by at most5.43e-20 here.
            # Numerical verification tolerance only; never changes training.
            assert np.isclose(row['lr'], expected_lr, rtol=0, atol=1e-18)
            state, _, stop = selection_step(state, row['source_validation']['overall']['macro_f1'], epoch, config['validation']['early_stop'])
            assert state == row['selection']
            if stop: assert epoch == len(history)
        assert len(history) == 100 or stop
        assert state['best_epoch'] == report['best_epoch']
        assert report['gradient_steps'] == 317 * len(history)
        with np.load(root / 'source_validation_predictions.npz', allow_pickle=False) as payload:
            assert np.array_equal(payload['compact_indices'], frame.cache_index)
            assert np.array_equal(payload['original_indices'], frame.original_cache_index)
            assert np.array_equal(payload['labels'], frame.label)
            probabilities = payload['probabilities']
            assert probabilities.shape == (34710, 2) and np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)
            pred = probabilities.argmax(1)
        metrics = stratified_metrics(frame.label.to_numpy(), pred, frame)
        assert metrics == report['source_validation'] == history[report['best_epoch'] - 1]['source_validation']
        initial = torch.load(root / 'initial.pt', weights_only=True, map_location='cpu')
        previous_initial = torch.load(short / arm / 'initial.pt', weights_only=True, map_location='cpu')
        assert all(torch.equal(v, previous_initial[k]) for k, v in initial.items())
        assert state_digest(initial) == report['initial_state_digest']
        initials.append(initial)
        assert sha(root / 'best.pt') == report['best_checkpoint_sha256']
        best = torch.load(root / 'best.pt', weights_only=True, map_location='cpu')
        assert all(torch.isfinite(v).all() for v in best.values())
        previous = read(short / arm / 'report.json')['source_validation']
        rows = []
        for q in range(4):
            selected = quarters == q
            rows.append({'crop_quarter': q + 1, 'windows': int(selected.sum()), 'false_positives': int(pred[hover][selected].sum())})
        records[arm] = {'epochs': len(history), 'best_epoch': report['best_epoch'], 'gradient_steps': report['gradient_steps'],
            'metrics': metrics, 'hover_by_original_crop_quarter': rows,
            'hover_by_domain': {domain: {'windows': int((hover & frame.domain.eq(domain).to_numpy()).sum()),
                'false_positives': int(pred[hover & frame.domain.eq(domain).to_numpy()].sum())} for domain in ['SIL', 'HIL']},
            'short_budget_metrics': previous,
            'budget_change_effect': {key: metrics['overall'][key] - previous['overall'][key] for key in ['macro_f1', 'fault_recall', 'normal_window_fpr']},
            'hover_fpr_change_percentage_points': 100 * (metrics['original_normal_by_status']['hover']['normal_window_fpr'] - previous['original_normal_by_status']['hover']['normal_window_fpr']),
            'all_initial_tensors_equal_to_short_budget': True, 'elapsed_s': report['elapsed_s']}
    assert all(torch.equal(v, initials[1][k]) for k, v in initials[0].items())
    result = {'status': 'PASS_LOCAL_FULL_SOURCE_RESULTS_RECOMPUTED', 'seed': 42, 'rows': records,
        'all_source_data_hashes_unchanged': True, 'same_initial_tensors_both_arms_and_short_runs': True,
        'full_population_metrics_selection_early_stop_and_schedule_recomputed': True, 'no_new_inference_or_training': True,
        'test_or_Real_read': False, 'verification_script_sha256': sha(__file__)}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({arm: {k: r[k] for k in ['epochs', 'best_epoch', 'hover_fpr_change_percentage_points', 'hover_by_original_crop_quarter']} for arm, r in records.items()}, indent=2))
    print('PASS_LOCAL_FULL_SOURCE_RESULTS_RECOMPUTED')


if __name__ == '__main__':
    main()
