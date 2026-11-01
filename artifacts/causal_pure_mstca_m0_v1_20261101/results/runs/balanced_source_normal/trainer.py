"""Pure26-channel MSTCA M0 using frozen source-development mechanics.

Preparation does not authorize training. Formal run additionally requires
--confirm-formal-run and the registered CUDA environment; no CPU fallback.
No test/Real access, candidate features, relation branch or pretrained weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

# Import before torch to preserve the original CUBLAS environment setup.
import train_causal_source_v1 as source
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from uav_fault.models.cnn.lstm import OnlineMSTCA, OnlineKGMSTCA
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2
from train_source_normal_sampling import state_digest, selection_step

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_NAME = 'causal_pure_mstca_m0_v1'


def load_protocol(path):
    path = Path(path)
    if source.sha(path) != path.with_suffix(path.suffix + '.sha256').read_text(encoding='utf-8').strip():
        raise ValueError('Frozen M0 protocol identity mismatch')
    p = json.loads(path.read_text(encoding='utf-8'))
    if p['protocol_name'] != PROTOCOL_NAME or p['data_version'] != source.VERSION:
        raise ValueError('Wrong M0 protocol/data version')
    if p['model']['implementation'] != 'OnlineMSTCA' or p['data']['columns'] != RFLY_SEQ_COLUMNS_V2:
        raise ValueError('M0 requires pure MSTCA and exact26-column order')
    if p['arms'] != ['uniform_replacement', 'balanced_source_normal'] or p['seeds'] != [42]:
        raise ValueError('Unregistered M0 arms/seeds')
    baseline_path = ROOT / p['baseline']['protocol_path']
    if source.sha(baseline_path) != p['baseline']['protocol_sha256']:
        raise ValueError('Historical comparison protocol identity mismatch')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    for key in ['data', 'data_identities', 'sampling', 'loss', 'validation', 'formal_environment', 'cpu_check']:
        if p[key] != baseline[key]:
            raise ValueError('Unregistered M0 comparison difference: ' + key)
    changed = {'parameters', 'initialization'}
    if set(p['training']) != set(baseline['training']):
        raise ValueError('Training configuration shape changed')
    if any(v != baseline['training'][k] for k, v in p['training'].items() if k not in changed):
        raise ValueError('M0 training budget/settings differ from comparison')
    for relative, expected in baseline['code_identities'].items():
        if p['code_identities'].get(relative) != expected:
            raise ValueError('Historical shared code pin changed: ' + relative)
    if p['training']['parameters'] != 105938:
        raise ValueError('Unregistered pure26 parameter count')
    return p


def build_model(p, seed, anchor_path):
    """Map only untrained common tensors; never import a trained best checkpoint."""
    spec = p['initialization_anchor']
    if source.sha(anchor_path) != spec['sha256']:
        raise ValueError('Untrained initialization anchor SHA mismatch')
    state = torch.load(anchor_path, map_location='cpu', weights_only=True)
    if state_digest(state) != spec['state_digest']:
        raise ValueError('Untrained initialization tensor identity mismatch')
    cfg = p['training']
    # In the registered formal environment, also reproduce all legacy tensors
    # from the seed to establish that the artifact predates optimizer updates.
    exact_environment = all(source.environment()[k] == v for k, v in p['formal_environment'].items())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if exact_environment:
            reference = OnlineKGMSTCA(in_channels=26, hidden_size=cfg['hidden_size'],
                num_layers=cfg['num_layers'], num_classes=2, dropout=cfg['dropout'], columns=p['data']['columns'])
            if state.keys() != reference.state_dict().keys() or any(not torch.equal(state[k], v) for k, v in reference.state_dict().items()):
                raise ValueError('Archived anchor is not exact fresh seeded legacy initialization')
        model = OnlineMSTCA(in_channels=26, hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'], num_classes=2, dropout=cfg['dropout']).float()
    if list(model._modules) != ['encoder', 'head'] or sum(v.numel() for v in model.parameters()) != cfg['parameters']:
        raise ValueError('M0 is not the registered pure MSTCA')
    mapping = {}
    for key, value in model.state_dict().items():
        source_key = 'raw_encoder.' + key[len('encoder.'):] if key.startswith('encoder.') else key
        if source_key not in state or state[source_key].shape != value.shape or state[source_key].dtype != value.dtype:
            raise ValueError('Common initialization key/shape mismatch: ' + key)
        mapping[key] = state[source_key].clone()
    model.load_state_dict(mapping, strict=True)
    digest = state_digest(model.state_dict())
    if digest != spec['m0_mapped_state_digest']:
        raise ValueError('M0 mapped initialization identity mismatch')
    if not all(torch.equal(model.state_dict()[key], value) for key, value in mapping.items()):
        raise ValueError('Mapped initialization not loaded exactly')
    return model, {'anchor_sha256': spec['sha256'], 'anchor_state_digest': spec['state_digest'],
        'm0_initial_state_digest': digest, 'common_encoder_and_head_exact_anchor_match': True,
        'legacy_fresh_seed_reproduction_checked_in_matching_environment': exact_environment,
        'local_environment_matches_formal': exact_environment,
        'common_state_tensors': len(mapping), 'parameters': cfg['parameters'],
        'whole_models_identical': False, 'trained_checkpoint_used': False}


def preflight(args, p):
    if args.output.exists():
        raise FileExistsError('Preserve existing M0 preparation output')
    model, init = build_model(p, args.seed, args.initial_anchor)
    source.preflight(args, p)  # Full registered data hashes, masks, split/pair checks.
    report = json.loads((args.output / 'preflight.json').read_text(encoding='utf-8'))
    report.update(status='PASS_LOCAL_M0_PREPARATION_NOT_REMOTE_CLEARANCE', model='OnlineMSTCA',
        input_shape=[26, 150], initialization=init, formal_training_started=False,
        formal_environment_matches=init['local_environment_matches_formal'])
    source.write(args.output / 'preflight.json', report)
    source.write(args.output / 'm0_initialization.json', init)
    return report


def run(args, p):
    if not args.confirm_formal_run:
        raise RuntimeError('M0_PREPARATION_ONLY: explicit formal-run confirmation required')
    if args.cpu_check:
        raise RuntimeError('M0 formal entry does not allow CPU training')
    if args.output.exists():
        raise FileExistsError('New run directory required; never overwrite')
    if args.arm not in p['arms'] or args.seed not in p['seeds']:
        raise ValueError('Unregistered M0 arm/seed')
    if not torch.cuda.is_available():
        raise RuntimeError('Formal M0 BLOCKED_NO_CUDA; no CPU fallback')
    actual = source.environment()
    for key, value in p['formal_environment'].items():
        if actual[key] != value:
            raise RuntimeError('Formal M0 environment mismatch: ' + key)
    ds, frame, weights, tables = source.prepare(args.data, p)
    cfg = p['training']
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, init = build_model(p, args.seed, args.initial_anchor)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'status': 'RUNNING_FINITE_SOURCE_RETRAIN', 'experiment': PROTOCOL_NAME,
        'model': 'OnlineMSTCA', 'parameters': cfg['parameters'], 'input_shape': [26, 150],
        'protocol_sha256': source.sha(args.protocol), 'seed': args.seed, 'arm': args.arm, 'cpu_check': False,
        'environment': actual, 'device': 'cuda:0', 'initial_state_digest': state_digest(initial),
        'initialization': init, 'data_version': source.VERSION, 'sampling': tables[args.arm],
        'source_train_windows': len(ds['train']), 'source_val_windows_full': len(ds['val']),
        'no_final_test_or_Real_reads': True, 'performance_claim': False,
        'started_at': datetime.now(timezone.utc).isoformat()}
    source.write(out / 'report.json', report)
    source.save_state(out / 'initial.pt', initial)
    for name, path in [('protocol.json', args.protocol), ('norm_stats.json', args.data / 'norm_stats.json'), ('trainer.py', Path(__file__))]:
        shutil.copyfile(path, out / name)
    device = torch.device('cuda:0')
    model.to(device)
    sampler = WeightedRandomSampler(weights[args.arm], len(ds['train']), replacement=True,
        generator=torch.Generator().manual_seed(args.seed + 11))
    trainloader = DataLoader(source.IndexedSource(ds['train']), batch_size=cfg['batch_size'], sampler=sampler,
        num_workers=0, generator=torch.Generator().manual_seed(args.seed + 101))
    metadata = frame['val']
    valloader = DataLoader(source.IndexedSource(ds['val'], metadata.cache_index.to_numpy()),
        batch_size=p['validation']['batch_size'], shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 101))
    criterion = source.FocalLoss(gamma=cfg['loss_gamma'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    selection = {'best_score': -float('inf'), 'best_epoch': None, 'significant_reference': -float('inf'), 'bad_epochs': 0}
    history = []
    try:
        for epoch in range(1, cfg['max_epochs'] + 1):
            lr = source.get_warmup_cosine_lr(epoch - 1, cfg['warmup_epochs'], cfg['max_epochs'], cfg['lr'], cfg['min_lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            losses, realized = [], Counter()
            for x, y, indices in trainloader:
                if not torch.isfinite(x).all() or not torch.isin(y, torch.tensor([0, 1])).all():
                    raise ValueError('Invalid sample before M0 forward')
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite M0 loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                realized.update(frame['train'].normal_origin.iloc[indices.numpy()].tolist())
            metrics, probs, labels = source.evaluate(model, valloader, device, metadata)
            selection, best, stop = selection_step(selection, float(metrics['overall']['macro_f1']), epoch, p['validation']['early_stop'])
            if best:
                source.save_state(out / 'best.pt', model.state_dict())
            source.save_state(out / 'last.pt', model.state_dict())
            history.append({'epoch': epoch, 'lr': lr, 'train_loss': float(np.mean(losses)),
                'gradient_steps': len(losses), 'realized_training_origins': dict(realized),
                'source_validation': metrics, 'selection': dict(selection)})
            source.write(out / 'history.json', history)
            print('CAUSAL_M0_JOB', args.arm, epoch, '/', cfg['max_epochs'], 'loss', float(np.mean(losses)),
                'source_val_macro_f1', metrics['overall']['macro_f1'], flush=True)
            if stop:
                break
        model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
        reloaded, probabilities, labels = source.evaluate(model, valloader, device, metadata)
        if reloaded != history[selection['best_epoch'] - 1]['source_validation']:
            raise RuntimeError('M0 best reload changed validation')
        if not any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters()):
            raise RuntimeError('No M0 trainable parameters changed')
        for relative, expected in p['data_identities'].items():
            if source.sha(args.data / relative) != expected:
                raise ValueError('Source data changed during M0 training')
        for relative, expected in p['code_identities'].items():
            if source.sha(ROOT / relative) != expected:
                raise ValueError('Pinned code changed during M0 training')
        if source.sha(args.protocol) != report['protocol_sha256']:
            raise ValueError('Frozen M0 protocol changed')
        np.savez_compressed(out / 'source_validation_predictions.npz', probabilities=probabilities, labels=labels,
            compact_indices=metadata.cache_index.to_numpy(dtype=np.int64), original_indices=metadata.original_cache_index.to_numpy(dtype=np.int64))
        report.update(status='COMPLETE_FINITE_SOURCE_RETRAIN', epochs=len(history),
            gradient_steps=sum(h['gradient_steps'] for h in history), best_epoch=selection['best_epoch'],
            validation_windows=len(metadata), checkpoint_reload_matches=True, parameters_changed=True,
            best_checkpoint_sha256=source.sha(out / 'best.pt'), source_validation=reloaded,
            elapsed_s=time.perf_counter() - start, finished_at=datetime.now(timezone.utc).isoformat(), performance_claim=True,
            result_scope='Source validation development only; not final-test/Real or deployment clearance')
        source.write(out / 'report.json', report)
        return report
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), elapsed_s=time.perf_counter() - start)
        source.write(out / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run'])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-anchor', type=Path, required=True)
    parser.add_argument('--arm', choices=['uniform_replacement', 'balanced_source_normal'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confirm-formal-run', action='store_true')
    args = parser.parse_args()
    args.cpu_check = False
    p = load_protocol(args.protocol)
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered M0 seed')
    if args.command == 'preflight':
        preflight(args, p)
    else:
        run(args, p)


if __name__ == '__main__':
    main()
