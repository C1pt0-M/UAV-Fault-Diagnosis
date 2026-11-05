"""M0S: frozen M0 B training with an added window-statistics branch.

Config: configs/experiments/m0s_stats_branch_v1.json. --fold 0/1/2 trains on the
other two folds and predicts the held-out fold; --fold -1 trains on full train.
The branch's last layer is zero-initialized, so step-0 output equals M0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import train_causal_mstca_m0 as m0  # imports torch after the CUBLAS env setup
source = m0.source
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

EPS = 1e-4


def fold_of(group):
    return int(hashlib.sha256(('h1cv|' + group).encode()).hexdigest(), 16) % 3


def window_stats(x):
    """x: (B, 26, 150) normalized input -> (B, 130)."""
    d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(2)
    return torch.cat([x.mean(2), torch.log(x.std(2, unbiased=False) + EPS), x.amin(2), x.amax(2), torch.log(d + EPS)], 1)


class M0S(nn.Module):
    def __init__(self, base, stats_mean, stats_std):
        super().__init__()
        self.encoder, self.head = base.encoder, base.head
        self.register_buffer('stats_mean', stats_mean)
        self.register_buffer('stats_std', stats_std)
        self.stats = nn.Sequential(nn.Linear(130, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        nn.init.zeros_(self.stats[2].weight); nn.init.zeros_(self.stats[2].bias)

    def forward(self, x):
        s = ((window_stats(x) - self.stats_mean) / self.stats_std).clamp(-10, 10)
        return self.head(self.encoder(x) + self.stats(s))


def run(args):
    cfg_m0s = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg_m0s['name'] != 'm0s_stats_branch_v1':
        raise ValueError('Unregistered config')
    p = m0.load_protocol(args.protocol)
    out = args.output
    if out.exists():
        raise FileExistsError('New run directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    for k, v in p['formal_environment'].items():
        if source.environment()[k] != v:
            raise RuntimeError('Environment mismatch: ' + k)
    ds, frame, _, _ = source.prepare(args.data, p)
    train = frame['train']
    folds = train.association_group.map(fold_of).to_numpy()
    keep = np.flatnonzero(folds != args.fold) if args.fold >= 0 else np.arange(len(train))
    held = np.flatnonzero(folds == args.fold) if args.fold >= 0 else np.array([], dtype=np.int64)
    labels = np.asarray(ds['train'].labels, dtype=np.int64)[keep]
    weights = source.sampling_weights(labels, train.normal_origin.to_numpy()[keep], True)
    cfg = p['training']
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Fixed statistics standardization from fold-train windows (independent RNG).
    rng = np.random.default_rng(4242)
    pick = np.sort(rng.choice(keep, size=min(20000, len(keep)), replace=False))
    sample = torch.stack([ds['train'][int(i)][0] for i in pick])
    s = window_stats(sample)
    s_mean, s_std = s.mean(0), s.std(0).clamp_min(1e-6)
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    base, init = m0.build_model(p, 42, args.initial_anchor)
    model = M0S(base, s_mean, s_std)  # branch init consumes RNG after the M0 build, as frozen
    extra = sum(v.numel() for v in model.stats.parameters())
    if extra != cfg_m0s['model']['extra_parameters']:
        raise ValueError('Unexpected branch size')
    out.mkdir(parents=True)
    device = torch.device('cuda:0'); model.to(device)
    sub = Subset(source.IndexedSource(ds['train']), keep.tolist())
    sampler = WeightedRandomSampler(weights, len(keep), replacement=True, generator=torch.Generator().manual_seed(42 + 11))
    trainloader = DataLoader(sub, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0,
                             generator=torch.Generator().manual_seed(42 + 101))
    meta = frame['val']
    valloader = DataLoader(source.IndexedSource(ds['val'], meta.cache_index.to_numpy()), batch_size=512, shuffle=False)
    # Step-0 equivalence check against plain M0 on a val batch.
    with torch.no_grad():
        xb = next(iter(valloader))[0][:256].to(device)
        model.eval(); base_out = model.head(model.encoder(xb)); full_out = model(xb)
        if not torch.equal(base_out, full_out):
            raise RuntimeError('Zero-init branch changed initial output')
    criterion = source.FocalLoss(gamma=cfg['loss_gamma'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sel = {'best_score': -float('inf'), 'best_epoch': None, 'significant_reference': -float('inf'), 'bad_epochs': 0}
    history = []; start = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        lr = source.get_warmup_cosine_lr(epoch - 1, cfg['warmup_epochs'], cfg['max_epochs'], cfg['lr'], cfg['min_lr'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        model.train(); losses = []
        for x, y, _ in trainloader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        metrics, _, _ = source.evaluate(model, valloader, device, meta)
        sel, best, stop = m0.selection_step(sel, float(metrics['overall']['macro_f1']), epoch, p['validation']['early_stop'])
        if best:
            source.save_state(out / 'best.pt', model.state_dict())
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_macro_f1': metrics['overall']['macro_f1'],
                        'val_hover_indep_fpr': metrics['original_normal_by_status']['hover']['normal_window_fpr']})
        print('M0S fold', args.fold, epoch, 'loss', history[-1]['train_loss'], 'val_f1', history[-1]['val_macro_f1'],
              'hover_fpr', history[-1]['val_hover_indep_fpr'], flush=True)
        if stop:
            break
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    _, vprobs, vlabels = source.evaluate(model, valloader, device, meta)
    np.savez_compressed(out / 'val_predictions.npz', probabilities=vprobs, labels=vlabels,
                        original_indices=meta.original_cache_index.to_numpy(dtype=np.int64))
    if len(held):
        model.eval(); probs = []
        with torch.no_grad():
            for x, _, _ in DataLoader(source.IndexedSource(ds['train'], held), batch_size=512, shuffle=False):
                probs.append(model(x.to(device)).softmax(1).cpu().numpy())
        np.savez_compressed(out / 'heldout_predictions.npz', probabilities=np.concatenate(probs),
                            original_indices=train.original_cache_index.to_numpy()[held])
    json.dump({'fold': args.fold, 'train_windows': len(keep), 'heldout_windows': len(held), 'best_epoch': sel['best_epoch'],
               'epochs': len(history), 'm0_initial_digest': init['m0_initial_state_digest'], 'extra_parameters': extra,
               'step0_equals_m0': True, 'history': history, 'elapsed_s': time.perf_counter() - start, 'status': 'COMPLETE'},
              open(out / 'report.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--initial-anchor', type=Path, required=True)
    ap.add_argument('--fold', type=int, choices=[-1, 0, 1, 2], required=True)
    ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
