"""Verify completed M0 source-only runs, no model forward/training or retries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import tarfile

import numpy as np
import pandas as pd
import torch

BASE = Path('/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102')
CODE = BASE / 'code'
DATA = Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')
RUNS = BASE / 'runs'
PROTOCOL = CODE / 'configs/experiments/causal_pure_mstca_m0_v1.json'
sys.path.insert(0, str(CODE / 'scripts/research'))
sys.path.insert(0, str(CODE / 'packages/uav_fault'))
from smoke_normal_source_sampling import stratified_metrics
from train_source_normal_sampling import state_digest, selection_step
from uav_fault.models.train_online_detector import get_warmup_cosine_lr


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    if (BASE / 'verified_m0_results_manifest.json').exists():
        raise FileExistsError('Preserve verified M0 results')
    assert read(BASE / 'remote_status.json')['phase'] == 'COMPLETED_TWO_ARM_FORMAL_SEQUENCE'
    p = read(PROTOCOL)
    digest = sha(PROTOCOL)
    assert digest == PROTOCOL.with_suffix('.json.sha256').read_text().strip()
    for relative, expected in p['data_identities'].items():
        assert sha(DATA / relative) == expected, relative
    for relative, expected in p['code_identities'].items():
        assert sha(CODE / relative) == expected, relative
    baseline_path = CODE / p['baseline']['protocol_path']
    assert sha(baseline_path) == p['baseline']['protocol_sha256']
    metadata = pd.read_csv(DATA / 'ledgers/val_included_windows.csv', keep_default_na=False, low_memory=False)
    metadata['label'] = metadata.qualified_y.astype(int)
    metadata['normal_origin'] = np.where(metadata.label == 1, 'fault_endpoint',
        np.where(metadata.case_level2_label == 'normal', 'original_normal_case', 'normal_part_of_fault_case'))
    metadata['flight_status'] = metadata.case_uid.str.split('/').str[1]
    metadata['cache_index'] = np.arange(len(metadata))
    assert len(metadata) == 34710 and set(metadata.domain) <= {'SIL', 'HIL'}
    rows, reports, initials, files = {}, {}, [], []
    for arm in p['arms']:
        root = RUNS / arm
        report, history = read(root / 'report.json'), read(root / 'history.json')
        assert report['status'] == 'COMPLETE_FINITE_SOURCE_RETRAIN' and report['model'] == 'OnlineMSTCA'
        assert report['parameters'] == 105938 and report['input_shape'] == [26, 150]
        assert report['protocol_sha256'] == digest and report['seed'] == 42 and not report['cpu_check']
        assert report['validation_windows'] == 34710 and report['source_train_windows'] == 162069
        assert report['checkpoint_reload_matches'] and report['parameters_changed'] and report['no_final_test_or_Real_reads']
        for key, value in p['formal_environment'].items():
            assert report['environment'][key] == value
        assert report['device'] == 'cuda:0' and 1 <= report['epochs'] <= 100 and len(history) == report['epochs']
        state = {'best_score': -float('inf'), 'best_epoch': None, 'significant_reference': -float('inf'), 'bad_epochs': 0}
        for epoch, h in enumerate(history, 1):
            assert h['epoch'] == epoch and np.isfinite(h['train_loss']) and h['gradient_steps'] == 317
            assert sum(h['realized_training_origins'].values()) == 162069
            cfg = p['training']
            expected_lr = get_warmup_cosine_lr(epoch - 1, 5, 100, cfg['lr'], cfg['min_lr'])
            assert math.isclose(h['lr'], expected_lr, rel_tol=0, abs_tol=1e-18)
            state, _, stop = selection_step(state, h['source_validation']['overall']['macro_f1'], epoch, p['validation']['early_stop'])
            assert state == h['selection']
            if stop:
                assert epoch == len(history)
        if len(history) < 100:
            assert stop, 'Run ended before registered early stopping'
        assert report['best_epoch'] == state['best_epoch']
        assert report['gradient_steps'] == 317 * len(history)
        assert sha(root / 'best.pt') == report['best_checkpoint_sha256']
        with np.load(root / 'source_validation_predictions.npz', allow_pickle=False) as stored:
            probabilities, labels = stored['probabilities'], stored['labels']
            assert probabilities.shape == (34710, 2) and np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
            assert np.array_equal(labels, metadata.label)
            assert np.array_equal(stored['compact_indices'], metadata.cache_index)
            assert np.array_equal(stored['original_indices'], metadata.original_cache_index)
            actual = stratified_metrics(labels, probabilities.argmax(1), metadata)
            assert actual == report['source_validation'] == history[report['best_epoch'] - 1]['source_validation']
        initial = torch.load(root / 'initial.pt', map_location='cpu', weights_only=True)
        best = torch.load(root / 'best.pt', map_location='cpu', weights_only=True)
        assert initial.keys() == best.keys()
        assert all(k.startswith(('encoder.', 'head.')) for k in initial)
        assert state_digest(initial) == p['initialization_anchor']['m0_mapped_state_digest'] == report['initial_state_digest']
        assert all(torch.isfinite(v).all() for v in best.values())
        assert any(not torch.equal(initial[k], v) for k, v in best.items())
        initials.append(initial)
        metric = actual['overall']
        hover = metadata[(metadata.label == 0) & (metadata.case_level2_label == 'normal') & metadata.flight_status.eq('hover')]
        # Use original crop length from earlier saved full source-val diagnosis.
        # Only write existing metrics here; stage details are computed after local recovery.
        rows[arm] = {'epochs': report['epochs'], 'best_epoch': report['best_epoch'],
            'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1'],
            'fault_recall': metric['fault_recall'], 'normal_window_fpr': metric['normal_window_fpr'],
            'original_normal_window_fpr': actual['normal_origin']['original_normal_case']['normal_window_fpr'],
            'inactive_in_fault_case_normal_fpr': actual['normal_origin']['normal_part_of_fault_case']['normal_window_fpr'],
            'original_normal_hover_fpr': actual['original_normal_by_status']['hover']['normal_window_fpr'],
            'confusion_matrix': metric['confusion_matrix'], 'gradient_steps': report['gradient_steps'],
            'best_checkpoint_sha256': report['best_checkpoint_sha256'], 'elapsed_s': report['elapsed_s']}
        reports[arm] = report
        for path in sorted(root.iterdir()):
            if path.is_file():
                files.append((path, 'runs/' + arm + '/' + path.name))
    assert all(torch.equal(initials[0][k], initials[1][k]) for k in initials[0])
    summary = {'status': 'PASS_BOTH_AUTHORIZED_M0_SOURCE_RUNS_VERIFIED',
        'verified_at': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': digest, 'seed': 42,
        'rows': rows, 'same_initial_state': True, 'selection_schedule_and_predictions_recomputed': True,
        'final_test_or_Real_evaluated': False, 'deployment_modified': False, 'unregistered_reruns': False,
        'interpretation': 'One seed; source-development only. No statistical significance or final-test/Real/deployment claim.',
        'verifier_sha256': sha(Path(__file__))}
    save(BASE / 'verified_m0_source_summary.json', summary)
    for name in ['paired_m0_summary.json', 'remote_status.json', 'authorization.json', 'execution_started.lock',
            'execute_m0.py', 'verify_and_package_m0.py', 'verified_m0_source_summary.json',
            'uniform_replacement.log', 'balanced_source_normal.log', 'execution_preparation.json',
            'remote_preparation_verification.json', 'source_preflight/preflight.json',
            'source_preflight/m0_initialization.json']:
        files.append((BASE / name, name))
    files.append((PROTOCOL, 'm0_protocol.json'))
    files.append((baseline_path, 'historical_longbudget_protocol.json'))
    manifest = {'status': 'VERIFIED_M0_RESULTS_ONLY', 'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path, name in files},
        'source_data_tensors_in_archive': False, 'final_test_or_Real_payloads': False}
    manifest_path = BASE / 'verified_m0_results_manifest.json'
    save(manifest_path, manifest)
    files.append((manifest_path, 'verified_m0_results_manifest.json'))
    archive = BASE / 'verified_m0_results.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=1) as stream:
        for path, name in files:
            stream.add(path, arcname=name, recursive=False)
    receipt = {'status': 'COMPLETE_VERIFIED_M0_RESULTS_ARCHIVE', 'path': str(archive),
        'bytes': archive.stat().st_size, 'sha256': sha(archive)}
    save(BASE / 'verified_m0_results_receipt.json', receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
