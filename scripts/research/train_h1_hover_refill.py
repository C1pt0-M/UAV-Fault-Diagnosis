"""H1: M0 Balanced retrain with extra train hover independent-normal windows.

Everything is the frozen M0 path (protocol, data hashes, init anchor, sampler
formula, loss, schedule, selection, val). The single change: the refill windows
from scripts/data/build_h1_hover_refill.py are appended to the train set as
normal_flight windows before Balanced weights are computed. draws per epoch
stay at the original 162069 so the optimizer budget matches M0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import train_causal_mstca_m0 as m0  # noqa: E402  (imports torch after CUBLAS env)
source = m0.source
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

H1_NAME = 'h1_hover_normal_coverage_v1'
ARM = 'balanced_source_normal'
M0_DRAWS = 162069


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


class TrainWithRefill(Dataset):
    """Original admitted train windows followed by normalized refill windows."""
    def __init__(self, base, x_raw, mean, std):
        self.base = base
        self.x = torch.from_numpy(x_raw)
        self.mean, self.std = mean, std
        self.labels = list(base.labels) + [0] * len(x_raw)

    def __len__(self):
        return len(self.base) + len(self.x)

    def __getitem__(self, index):
        if index < len(self.base):
            return self.base[index]
        return (self.x[index - len(self.base)] - self.mean) / self.std, torch.tensor(0, dtype=torch.int64)


def load_h1(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg['name'] != H1_NAME or cfg['arm'] != ARM or cfg['seed'] != 42 or cfg['draws_per_epoch'] != M0_DRAWS:
        raise ValueError('Unregistered H1 configuration')
    return cfg


def prepare(args, p, h1):
    ds, frame, _, _ = source.prepare(args.data, p)  # all frozen M0 identity checks
    refill = args.refill
    manifest = json.loads((refill / 'manifest.json').read_text(encoding='utf-8'))
    if sha(refill / 'x.npy') != h1['refill_x_sha256'] or sha(refill / 'windows.csv') != h1['refill_windows_sha256']:
        raise ValueError('Refill identity mismatch')
    if manifest['source_identities']['norm_stats.json'] != sha(args.data / 'norm_stats.json'):
        raise ValueError('Refill built against different normalization')
    x = np.load(refill / 'x.npy', mmap_mode='r')
    rows = pd.read_csv(refill / 'windows.csv', dtype={'association_group': str}, keep_default_na=False)
    if x.shape != (len(rows), 26, 150) or not np.isfinite(x).all():
        raise ValueError('Invalid refill tensor')
    train = frame['train']
    if set(rows.case_uid) - set(train[train.normal_origin == source.ORIGINAL].case_uid):
        raise ValueError('Refill outside train independent-normal cases')
    if set(rows.case_uid) & set(frame['val'].case_uid) or set(rows.association_group) & set(frame['val'].association_group):
        raise ValueError('Refill leaks into val cases/groups')
    key = set(zip(train.case_uid, train.window_start_row.astype(int)))
    if any((u, int(s)) in key for u, s in zip(rows.case_uid, rows.window_start_row)):
        raise ValueError('Refill duplicates an admitted cap64 window')
    stats = json.loads((args.data / 'norm_stats.json').read_text(encoding='utf-8'))
    mean = torch.tensor(stats['mean'], dtype=torch.float32).view(-1, 1)
    std = torch.tensor(stats['std'], dtype=torch.float32).view(-1, 1)
    combined = TrainWithRefill(ds['train'], np.array(x, dtype=np.float32), mean, std)
    extra = pd.DataFrame({'normal_origin': source.ORIGINAL, 'case_uid': rows.case_uid,
                          'window_start_row': rows.window_start_row, 'window_end_row': rows.window_end_row})
    train_frame = pd.concat([train[['normal_origin', 'case_uid', 'window_start_row', 'window_end_row']], extra], ignore_index=True)
    labels = np.asarray(combined.labels, dtype=np.int64)
    weights = source.sampling_weights(labels, train_frame.normal_origin.to_numpy(), True)
    # Keep M0's class prior: total normal mass equals the original admitted
    # normal count, so only the split inside the normal_flight pool changes.
    normal = torch.from_numpy(labels == 0)
    m0_normal_mass = float((np.asarray(ds['train'].labels) == 0).sum())
    weights[normal] *= m0_normal_mass / float(weights[normal].sum())
    table = source.probability_table(weights, labels, train_frame.normal_origin.to_numpy())
    late = ((train_frame.normal_origin == source.ORIGINAL) & train_frame.case_uid.str.contains('/hover/')
            & (train_frame.window_end_row * 0.02 > 30)).to_numpy()
    table['target_hover_gt30_normal'] = {'windows': int(late.sum()), 'probability': float(weights[torch.from_numpy(late)].sum() / weights.sum())}
    return ds, frame, train_frame, combined, weights, table


def run(args, p, h1):
    import time
    from collections import Counter
    from datetime import datetime, timezone
    import shutil
    from torch.utils.data import DataLoader, WeightedRandomSampler
    if args.output.exists():
        raise FileExistsError('New run directory required')
    cpu = args.cpu_check
    if not cpu:
        if not args.confirm_formal_run or not torch.cuda.is_available():
            raise RuntimeError('Formal H1 needs --confirm-formal-run and CUDA')
        for k, v in p['formal_environment'].items():
            if source.environment()[k] != v:
                raise RuntimeError('Formal environment mismatch: ' + k)
    ds, frame, train_frame, train, weights, table = prepare(args, p, h1)
    cfg = p['training']
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, init = m0.build_model(p, args.seed, args.initial_anchor)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = args.output; out.mkdir(parents=True)
    device = torch.device('cpu' if cpu else 'cuda:0')
    profile = p['cpu_check'] if cpu else cfg
    draws = profile['draws_per_epoch'] if cpu else h1['draws_per_epoch']
    epochs = profile['epochs'] if cpu else cfg['max_epochs']
    metadata = frame['val']
    if cpu:
        picked = []
        for _, g in metadata.groupby('case_uid', sort=False):
            picks = np.unique(np.linspace(0, len(g) - 1, min(profile['val_windows_per_case'], len(g))).round().astype(int))
            picked.extend(g.iloc[picks].cache_index.tolist())
        metadata = metadata.iloc[picked].copy().reset_index(drop=True)
    report = {'status': 'RUNNING_CPU_CHECK' if cpu else 'RUNNING_FORMAL', 'experiment': H1_NAME, 'arm': ARM, 'seed': args.seed,
              'm0_protocol_sha256': sha(args.protocol), 'h1_config_sha256': sha(args.h1_config),
              'environment': source.environment(), 'device': str(device), 'initialization': init,
              'initial_state_digest': m0.state_digest(initial), 'train_windows_total': len(train),
              'train_windows_original': len(ds['train']), 'refill_windows': len(train) - len(ds['train']),
              'draws_per_epoch': draws, 'sampling': table, 'source_val_windows': len(metadata),
              'no_final_test_or_Real_reads': True, 'started_at': datetime.now(timezone.utc).isoformat()}
    source.write(out / 'report.json', report)
    for name, path in [('m0_protocol.json', args.protocol), ('h1_config.json', args.h1_config), ('trainer.py', Path(__file__))]:
        shutil.copyfile(path, out / name)
    model.to(device)
    sampler = WeightedRandomSampler(weights, draws, replacement=True, generator=torch.Generator().manual_seed(args.seed + 11))
    trainloader = DataLoader(source.IndexedSource(train), batch_size=profile['batch_size'], sampler=sampler,
                             num_workers=0, generator=torch.Generator().manual_seed(args.seed + 101))
    valloader = DataLoader(source.IndexedSource(ds['val'], metadata.cache_index.to_numpy()),
                           batch_size=profile['validation_batch_size'] if cpu else p['validation']['batch_size'],
                           shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(args.seed + 101))
    criterion = source.FocalLoss(gamma=cfg['loss_gamma'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    selection = {'best_score': -float('inf'), 'best_epoch': None, 'significant_reference': -float('inf'), 'bad_epochs': 0}
    history = []; start = time.perf_counter()
    is_refill = np.arange(len(train)) >= len(ds['train'])
    try:
        for epoch in range(1, epochs + 1):
            lr = source.get_warmup_cosine_lr(epoch - 1, cfg['warmup_epochs'], cfg['max_epochs'], cfg['lr'], cfg['min_lr'])
            for g in optimizer.param_groups:
                g['lr'] = lr
            model.train(); losses = []; realized = Counter(); refill_draws = 0
            for x, y, idx in trainloader:
                if not torch.isfinite(x).all() or not torch.isin(y, torch.tensor([0, 1])).all():
                    raise ValueError('Invalid sample before forward')
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                realized.update(train_frame.normal_origin.iloc[idx.numpy()].tolist()); refill_draws += int(is_refill[idx.numpy()].sum())
            metrics, probs, labels = source.evaluate(model, valloader, device, metadata)
            selection, best, stop = m0.selection_step(selection, float(metrics['overall']['macro_f1']), epoch, p['validation']['early_stop'])
            if best:
                source.save_state(out / 'best.pt', model.state_dict())
            source.save_state(out / 'last.pt', model.state_dict())
            history.append({'epoch': epoch, 'lr': lr, 'train_loss': float(np.mean(losses)), 'gradient_steps': len(losses),
                            'realized_training_origins': dict(realized), 'refill_draws': refill_draws,
                            'source_validation': metrics, 'selection': dict(selection)})
            source.write(out / 'history.json', history)
            print('H1_JOB', epoch, '/', epochs, 'loss', float(np.mean(losses)), 'val_macro_f1',
                  metrics['overall']['macro_f1'], 'refill_draws', refill_draws, flush=True)
            if stop:
                break
        model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
        reloaded, probabilities, labels = source.evaluate(model, valloader, device, metadata)
        if reloaded != history[selection['best_epoch'] - 1]['source_validation']:
            raise RuntimeError('Best reload changed validation')
        for rel, expected in p['data_identities'].items():
            if source.sha(args.data / rel) != expected:
                raise ValueError('Data changed during training')
        np.savez_compressed(out / 'source_validation_predictions.npz', probabilities=probabilities, labels=labels,
                            compact_indices=metadata.cache_index.to_numpy(dtype=np.int64),
                            original_indices=metadata.original_cache_index.to_numpy(dtype=np.int64))
        report.update(status='PASS_CPU_CHECK_NOT_FORMAL' if cpu else 'COMPLETE_FORMAL', epochs=len(history),
                      gradient_steps=sum(h['gradient_steps'] for h in history), best_epoch=selection['best_epoch'],
                      best_checkpoint_sha256=source.sha(out / 'best.pt'), source_validation=reloaded,
                      elapsed_s=time.perf_counter() - start, finished_at=datetime.now(timezone.utc).isoformat())
        source.write(out / 'report.json', report)
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), elapsed_s=time.perf_counter() - start)
        source.write(out / 'report.json', report)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['run'])
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--h1-config', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--refill', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--initial-anchor', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu-check', action='store_true')
    ap.add_argument('--confirm-formal-run', action='store_true')
    args = ap.parse_args()
    p = m0.load_protocol(args.protocol)
    run(args, p, load_h1(args.h1_config))


if __name__ == '__main__':
    main()
