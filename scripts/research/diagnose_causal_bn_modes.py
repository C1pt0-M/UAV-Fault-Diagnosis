"""Bounded source-train BN running-versus-batch statistics diagnosis.

No training, Dropout, model/state writes, raw logs, val, test or Real reads.
Use frozen source training bundle plus prior train-normal predictions.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


def batch_statistics_copy(model):
    """Compute batch BN on a disposable model without updating any buffers."""
    disposable = copy.deepcopy(model).eval()
    found = 0
    for module in disposable.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.training = True
            module.track_running_stats = False
            found += 1
    if not found:
        raise ValueError('Expected BatchNorm1d model')
    if any(module.training for module in disposable.modules() if not isinstance(module, nn.BatchNorm1d)):
        raise ValueError('Only BatchNorm may enter training mode; Dropout must remain disabled')
    return disposable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code', 'data', 'runs', 'prior-normal', 'plan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an earlier diagnosis')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(args.code / 'packages/uav_fault'))
    sys.path.insert(0, str(args.code / 'scripts/research'))
    from train_causal_source_v1 import sha, load_protocol, source_metadata
    from train_source_normal_sampling import state_digest
    from smoke_normal_source_sampling import sampling_weights, stratified_metrics
    from uav_fault.data.causal_cache import CausalCacheDataset, VERSION
    from uav_fault.models.cnn.lstm import OnlineKGMSTCA
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from train_causal_source_v1 import IndexedSource

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def save(path, value):
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

    if not torch.cuda.is_available():
        raise RuntimeError('Original GPU required; no automatic allocation/CPU fallback')
    protocol_path = args.code / 'configs/experiments/causal_source_development_v1.json'
    protocol = load_protocol(protocol_path)
    env = {'torch': str(torch.__version__), 'numpy': np.__version__, 'torch_cuda': torch.version.cuda}
    if any(env[k] != v for k, v in protocol['formal_environment'].items()):
        raise ValueError('Frozen original execution environment mismatch')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = {str(protocol_path): sha(protocol_path), str(args.plan): sha(args.plan)}
    # Do not use prepare(): it also opens val, which is unnecessary here.
    for relative, expected in protocol['code_identities'].items():
        path = args.code / relative
        if sha(path) != expected:
            raise ValueError('Frozen code mismatch: ' + relative)
        inputs[str(path)] = expected
    for relative, expected in protocol['data_identities'].items():
        if 'val' in relative:
            continue
        path = args.data / relative
        if sha(path) != expected:
            raise ValueError('Frozen source-train identity mismatch: ' + relative)
        inputs[str(path)] = expected
    dataset = CausalCacheDataset(args.data, expected_version=VERSION, split='train', purpose='source_development')
    metadata = source_metadata(args.data, 'train', dataset)
    if len(dataset) != 162069:
        raise ValueError('Wrong current source training population')
    fixed = {}
    for arm in protocol['arms']:
        weights = sampling_weights(np.asarray(dataset.labels), metadata.normal_origin.to_numpy(), arm == 'balanced_source_normal')
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=torch.Generator().manual_seed(53))
        fixed[arm] = np.asarray(list(sampler)[:32 * 512], dtype=np.int64)
        if len(fixed[arm]) != 32 * 512:
            raise ValueError('Exactly32 full mixed batches required')
        for part in np.array_split(fixed[arm], 32):
            if set(metadata.label.iloc[part]) != {0, 1}:
                raise ValueError('Mixed source-normal/fault context required, never resample by results')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'fixed_draws_before_predictions.npz', **fixed)
    save(args.output / 'RUNNING.json', {'status': 'RUNNING_BOUNDED_BN_DIAGNOSIS', 'started_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'fixed_draws_sha256': sha(args.output / 'fixed_draws_before_predictions.npz'), 'optimizer_updates': 0})
    results = {}
    for arm, draws in fixed.items():
        run = args.runs / arm
        checkpoint = run / 'best.pt'
        report = read(run / 'report.json')
        if report['status'] != 'COMPLETE_FINITE_SOURCE_RETRAIN' or report['cpu_check'] or sha(checkpoint) != report['best_checkpoint_sha256']:
            raise ValueError('Require original completed checkpoint')
        prior_path = args.prior_normal / (arm + '_train_normal_predictions.npz')
        for path in [checkpoint, run / 'report.json', prior_path]:
            inputs[str(path)] = sha(path)
        cfg = protocol['training']
        reference = OnlineKGMSTCA(in_channels=26, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
            num_classes=2, dropout=cfg['dropout'], columns=protocol['data']['columns']).float()
        reference.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
        reference.eval().requires_grad_(False)
        original_digest = state_digest(reference.state_dict())
        batch_model = batch_statistics_copy(reference).requires_grad_(False)
        if state_digest(batch_model.state_dict()) != original_digest:
            raise RuntimeError('Disposable state differs before diagnosis')
        reference.to('cuda:0'); batch_model.to('cuda:0')
        loader = DataLoader(IndexedSource(dataset, draws), batch_size=512, shuffle=False, num_workers=0)
        probs = {'running': [], 'batch': []}; labels = []; ordered = []
        with torch.inference_mode():
            for x, y, index in loader:
                if not torch.isfinite(x).all():
                    raise ValueError('Invalid input before forward')
                x = x.to('cuda:0')
                for name, model in [('running', reference), ('batch', batch_model)]:
                    p = model(x).softmax(1).cpu().numpy()
                    if not np.isfinite(p).all():
                        raise FloatingPointError('Nonfinite diagnostic probabilities')
                    probs[name].append(p)
                labels.extend(y.tolist()); ordered.extend(index.tolist())
        if not np.array_equal(ordered, draws):
            raise ValueError('Paired draws reordered')
        linked = metadata.iloc[draws].reset_index(drop=True)
        if not np.array_equal(labels, linked.label):
            raise ValueError('Labels differ from admitted train metadata')
        probs = {k: np.concatenate(v) for k, v in probs.items()}
        normal = linked.label.to_numpy() == 0
        with np.load(prior_path, allow_pickle=False) as prior:
            positions = np.searchsorted(prior['compact_indices'], draws[normal])
            if not np.array_equal(prior['compact_indices'][positions], draws[normal]):
                raise ValueError('Prior source normal ID alignment failed')
            previous = prior['probabilities'][positions]
            delta = float(np.max(np.abs(previous - probs['running'][normal])))
            flips = int((previous.argmax(1) != probs['running'][normal].argmax(1)).sum())
            if delta > 1e-5 or flips:
                raise ValueError('Running mode differs from prior source-train inference')
        metrics = {mode: stratified_metrics(np.asarray(labels), p.argmax(1), linked) for mode, p in probs.items()}
        per_batch = []
        for i in range(32):
            section = slice(i * 512, (i + 1) * 512)
            per_batch.append({'batch': i, **{mode: stratified_metrics(np.asarray(labels)[section], p[section].argmax(1), linked.iloc[section])
                for mode, p in probs.items()}})
        reference.to('cpu'); batch_model.to('cpu')
        if state_digest(reference.state_dict()) != original_digest or state_digest(batch_model.state_dict()) != original_digest:
            raise RuntimeError('Parameters or BN buffers changed')
        if any(p.grad is not None for m in [reference, batch_model] for p in m.parameters()):
            raise RuntimeError('Diagnostic must never compute gradients')
        for mode, p in probs.items():
            linked[mode + '_fault_score'] = p[:, 1]
        linked.groupby(['case_uid', 'domain', 'flight_status', 'normal_origin'], sort=True).agg(
            draws=('cache_index', 'size'), unique_windows=('cache_index', 'nunique'),
            running_fault_score_mean=('running_fault_score', 'mean'), batch_fault_score_mean=('batch_fault_score', 'mean')
        ).to_csv(args.output / (arm + '_case_draws.csv'))
        np.savez_compressed(args.output / (arm + '_paired_predictions.npz'), **probs, labels=np.asarray(labels, dtype=np.int64),
            compact_indices=draws, original_indices=linked.original_cache_index.to_numpy(dtype=np.int64))
        results[arm] = {'draws': len(draws), 'unique_windows': int(len(np.unique(draws))), 'batches': 32,
            'checkpoint_sha256': sha(checkpoint), 'state_digest': original_digest, 'all_model_states_unchanged': True,
            'prior_running_mode_max_probability_difference': delta, 'prior_running_mode_normal_prediction_changes': flips,
            'metrics': metrics, 'per_batch': per_batch}
        save(args.output / (arm + '_summary.json'), results[arm])
        print('BN_MODE_RESULT', arm, json.dumps({mode: metrics[mode]['original_normal_by_status'].get('hover') for mode in metrics}), flush=True)
        del reference, batch_model
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError('Immutable checkpoint/input changed: ' + path)
    summary = {'status': 'COMPLETE_BOUNDED_SOURCE_TRAIN_BN_MODE_DIAGNOSIS', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'environment': env, 'results': results, 'inputs': inputs, 'script_sha256': sha(__file__),
        'optimizer_updates': 0, 'parameter_and_BN_buffer_updates': 0, 'dropout_enabled': False, 'val_test_Real_read': False,
        'decision_rule_changed': False, 'model_saved_or_deployed': False,
        'interpretation': 'Matched fixed mixed-batch source training diagnosis only. Batch-mode probabilities are context-dependent, not deployable online scores or proof of a sole root cause. No automatic BN repair.'}
    save(args.output / 'summary.json', summary)
    print('COMPLETE_BOUNDED_SOURCE_TRAIN_BN_MODE_DIAGNOSIS', flush=True)


if __name__ == '__main__':
    main()
