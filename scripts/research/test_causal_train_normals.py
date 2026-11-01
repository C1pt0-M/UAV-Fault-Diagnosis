"""Existing checkpoint eval-only train-normal versus source-val diagnosis.

No optimizer, threshold choice, new training or test/Real access. The immutable
remote training bundle supplies the same model, source reader and normalization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ARMS = ['uniform_replacement', 'balanced_source_normal']


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def normal_error_summary(metadata, predictions, fault_scores, crop_lengths):
    predictions = np.asarray(predictions)
    scores = np.asarray(fault_scores)
    if not len(metadata) or predictions.shape != (len(metadata),) or scores.shape != predictions.shape:
        raise ValueError('Normal prediction/metadata shape mismatch')
    if not np.isin(predictions, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid fixed binary predictions/probabilities')
    if not metadata.label.eq(0).all() or metadata.cache_index.duplicated().any():
        raise ValueError('Normal-only unique source samples required')
    frame = metadata.reset_index(drop=True).copy()
    frame['false_positive'] = predictions == 1
    frame['fault_score'] = scores
    pure = frame.case_level2_label.eq('normal')
    hover = pure & frame.flight_status.eq('hover')

    def stats(group):
        return {'windows': len(group), 'cases': int(group.case_uid.nunique()),
            'false_positives': int(group.false_positive.sum()),
            'false_positive_rate': float(group.false_positive.mean()) if len(group) else None}

    normal_cases = frame.loc[pure].groupby(['domain', 'flight_status', 'case_uid'], sort=True).agg(
        windows=('false_positive', 'size'), false_positives=('false_positive', 'sum'),
        fault_score_median=('fault_score', 'median'), fault_score_max=('fault_score', 'max')).reset_index()
    normal_cases['false_positive_rate'] = normal_cases.false_positives / normal_cases.windows
    quarter_rows = []
    for j in range(4):
        matching = []
        for row in frame.loc[hover].itertuples():
            n = crop_lengths.get(row.case_uid)
            if not isinstance(n, int) or n < 150 or not 150 <= row.window_end_row <= n:
                raise ValueError('Missing/invalid ORIGINAL retained crop length')
            quarter = min(3, (4 * (int(row.window_end_row) - 1)) // (n - 1))
            if quarter == j:
                matching.append(row.Index)
        quarter_rows.append({'crop_progress_quarter': j + 1, **stats(frame.loc[matching])})
    if sum(r['windows'] for r in quarter_rows) != int(hover.sum()):
        raise ValueError('Crop-quarter partition incomplete')
    result = {'all_normal': stats(frame), 'original_normal': stats(frame.loc[pure]),
        'inactive_fault_case_normal': stats(frame.loc[~pure]), 'original_normal_hover': stats(frame.loc[hover]),
        'hover_by_domain': {str(key): stats(group) for key, group in frame.loc[hover].groupby('domain')},
        'original_normal_by_mode': {str(key): stats(group) for key, group in frame.loc[pure].groupby('flight_status')},
        'normal_by_origin': {str(key): stats(group) for key, group in frame.groupby('normal_origin')},
        'hover_by_original_crop_quarter': quarter_rows,
        'original_normal_cases_with_any_false_positive': int((normal_cases.false_positives > 0).sum()),
        'hover_cases_with_any_false_positive': int(((normal_cases.flight_status == 'hover') & (normal_cases.false_positives > 0)).sum()),
        'crop_quarters_are_physical_flight_phase': False}
    false_hover = frame.loc[hover & frame.false_positive, 'fault_score'].to_numpy()
    result['hover_false_positive_scores'] = {'windows': len(false_hover),
        'median': float(np.median(false_hover)) if len(false_hover) else None,
        'min': float(false_hover.min()) if len(false_hover) else None,
        'max': float(false_hover.max()) if len(false_hover) else None}
    return result, normal_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--crop-lengths', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Independent fresh diagnosis output required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(args.code / 'packages/uav_fault'))
    sys.path.insert(0, str(args.code / 'scripts/research'))
    import torch
    from torch.utils.data import DataLoader
    from train_causal_source_v1 import load_protocol, prepare, IndexedSource
    from train_source_normal_sampling import state_digest
    from uav_fault.models.cnn.lstm import OnlineKGMSTCA

    if not torch.cuda.is_available():
        raise RuntimeError('Use original available GPU execution environment, no implicit CPU parity substitution')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol_path = args.code / 'configs/experiments/causal_source_development_v1.json'
    protocol = load_protocol(protocol_path)
    environment = {'torch': str(torch.__version__), 'numpy': np.__version__, 'torch_cuda': torch.version.cuda}
    if any(environment[k] != value for k, value in protocol['formal_environment'].items()):
        raise RuntimeError('Original GPU environment mismatch')
    datasets, frames, _, _ = prepare(args.data, protocol)
    crop_record = read(args.crop_lengths)
    crop_lengths = crop_record['rows_by_case_uid']
    if any(split not in ['train', 'val'] for split in crop_record['splits_by_case_uid'].values()):
        raise ValueError('Only source train/val crop metadata allowed')
    selected = np.flatnonzero(np.asarray(datasets['train'].labels) == 0).astype(np.int64)
    if len(selected) != 83051:
        raise ValueError('Expected all83051 current admitted source-training normal endpoints')
    train_metadata = frames['train'].iloc[selected].copy().reset_index(drop=True)
    val_metadata = frames['val'].copy()
    train_loader = DataLoader(IndexedSource(datasets['train'], selected), batch_size=512, shuffle=False, num_workers=0)
    val_loader = DataLoader(IndexedSource(datasets['val']), batch_size=512, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {str(protocol_path): sha(protocol_path), str(args.crop_lengths): sha(args.crop_lengths),
        str(args.plan): sha(args.plan), str(args.data / 'norm_stats.json'): sha(args.data / 'norm_stats.json')}
    for arm in ARMS:
        for name in ['best.pt', 'report.json', 'source_validation_predictions.npz']:
            path = args.runs / arm / name
            inputs[str(path)] = sha(path)
    save(args.output / 'RUNNING.json', {'status': 'SOURCE_EVAL_ONLY_RUNNING',
        'started_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'scope': 'All source-train normal, full source-val path check, no optimizer/test/Real', 'inputs': inputs})
    results = {}
    for arm in ARMS:
        started = time.perf_counter()
        report = read(args.runs / arm / 'report.json')
        checkpoint = args.runs / arm / 'best.pt'
        if report['status'] != 'COMPLETE_FINITE_SOURCE_RETRAIN' or report['cpu_check'] or sha(checkpoint) != report['best_checkpoint_sha256']:
            raise ValueError('Original completed selected checkpoint required')
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        config = protocol['training']
        model = OnlineKGMSTCA(in_channels=26, hidden_size=config['hidden_size'], num_layers=config['num_layers'],
            num_classes=2, dropout=config['dropout'], columns=protocol['data']['columns']).float()
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model.load_state_dict(state, strict=True)
        before = state_digest(model.state_dict())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to('cuda:0').eval()
        if any(module.training for module in model.modules()):
            raise ValueError('Inference must disable dropout and BN updates')

        def predict(loader, metadata):
            probabilities = []; labels = []; indices = []
            with torch.inference_mode():
                for i, (x, y, index) in enumerate(loader, 1):
                    if not torch.isfinite(x).all():
                        raise ValueError('Input invalid before diagnostic inference')
                    p = model(x.to('cuda:0')).softmax(1).cpu().numpy()
                    if not np.isfinite(p).all():
                        raise ValueError('Nonfinite predictions')
                    probabilities.append(p); labels.extend(y.tolist()); indices.extend(index.tolist())
                    if i % 50 == 0:
                        print('NORMAL_DIAGNOSTIC', arm, 'batch', i, flush=True)
            p = np.concatenate(probabilities); y = np.asarray(labels, dtype=np.int64)
            if indices != metadata.cache_index.tolist() or not np.array_equal(y, metadata.label):
                raise ValueError('Source prediction/metadata ordering differs')
            return p, y

        train_prob, train_labels = predict(train_loader, train_metadata)
        val_prob, val_labels = predict(val_loader, val_metadata)
        with np.load(args.runs / arm / 'source_validation_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(val_labels, saved['labels']) or not np.array_equal(val_metadata.original_cache_index, saved['original_indices']):
                raise ValueError('Saved validation population identity mismatch')
            delta = float(np.max(np.abs(val_prob - saved['probabilities'])))
            changed = int((val_prob.argmax(1) != saved['probabilities'].argmax(1)).sum())
            if delta > 1e-5 or changed:
                raise ValueError('Unchanged checkpoint eval failed saved val path parity')
        val_normal = np.asarray(val_labels) == 0
        train_summary, train_cases = normal_error_summary(train_metadata, train_prob.argmax(1), train_prob[:, 1], crop_lengths)
        val_summary, val_cases = normal_error_summary(val_metadata.loc[val_normal].copy(), val_prob[val_normal].argmax(1), val_prob[val_normal, 1], crop_lengths)
        model.to('cpu')
        after = state_digest(model.state_dict())
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Model parameters/BN buffers changed during eval diagnosis')
        if sha(checkpoint) != inputs[str(checkpoint)]:
            raise RuntimeError('Checkpoint file changed')
        np.savez_compressed(args.output / (arm + '_train_normal_predictions.npz'), probabilities=train_prob, labels=train_labels,
            compact_indices=train_metadata.cache_index.to_numpy(dtype=np.int64),
            original_indices=train_metadata.original_cache_index.to_numpy(dtype=np.int64))
        train_cases.to_csv(args.output / (arm + '_train_normal_case_errors.csv'), index=False)
        val_cases.to_csv(args.output / (arm + '_val_normal_case_errors.csv'), index=False)
        results[arm] = {'train': train_summary, 'val': val_summary,
            'unchanged_state_digest': before, 'weights_and_BN_buffers_unchanged': True,
            'checkpoint_sha256': sha(checkpoint), 'saved_val_probability_max_abs_difference': delta,
            'saved_val_changed_predictions': changed, 'elapsed_s': time.perf_counter() - started}
        save(args.output / (arm + '_summary.json'), results[arm])
        print('SOURCE_NORMAL_RESULT', arm, json.dumps({s: results[arm][s]['original_normal_hover'] for s in ['train', 'val']}), flush=True)
        del model
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError('Source checkpoint/input changed during diagnosis: ' + path)
    summary = {'status': 'COMPLETE_EXISTING_CHECKPOINT_SOURCE_TRAIN_NORMAL_DIAGNOSIS',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'data_version': protocol['data_version'],
        'device': 'cuda:0', 'environment': environment, 'train_normal_windows': 83051,
        'val_normal_windows': 17740, 'fixed_decision_rule': 'argmax normal/fault, unchanged0.5',
        'model_eval_mode': True, 'optimizer_updates': 0, 'parameter_and_BN_updates': 0,
        'test_or_Real_read': False, 'raw_logs_reread': False, 'threshold_or_checkpoint_changed': False,
        'results': results, 'inputs': inputs, 'script_sha256': sha(__file__),
        'interpretation_limit': 'Training-data performance in inference mode. Does not by itself identify channels, branches, physical flight phase or BatchNorm mechanism. Never remove valid errors or select a new threshold from this diagnosis.'}
    save(args.output / 'summary.json', summary)
    print(json.dumps({arm: {s: results[arm][s]['original_normal_hover'] for s in ['train', 'val']} for arm in ARMS}, indent=2), flush=True)


if __name__ == '__main__':
    main()
