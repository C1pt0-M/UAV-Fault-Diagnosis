"""Group-CV M0: frozen M0 B training on two folds, predict the held-out fold.

Uses the same protocol, anchor, sampler formula, loss, schedule, early stop and
source-val checkpoint selection as M0 B. The only difference: train windows
from the held-out fold are removed, and draws per epoch = fold-train size.
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ARM = 'balanced_source_normal'


def fold_of(group):
    return int(hashlib.sha256(('h1cv|' + group).encode()).hexdigest(), 16) % 3


def build_seed_model(p, seed):
    """Non-42 seeds: same procedure that produced the seed-42 anchor."""
    from uav_fault.models.cnn.lstm import OnlineMSTCA, OnlineKGMSTCA
    cfg = p['training']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        legacy = OnlineKGMSTCA(in_channels=26, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                               num_classes=2, dropout=cfg['dropout'], columns=p['data']['columns'])
        model = OnlineMSTCA(in_channels=26, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                            num_classes=2, dropout=cfg['dropout']).float()
    state = legacy.state_dict()
    mapping = {k: state['raw_encoder.' + k[len('encoder.'):] if k.startswith('encoder.') else k].clone()
               for k in model.state_dict()}
    model.load_state_dict(mapping, strict=True)
    return model, {'m0_initial_state_digest': m0.state_digest(model.state_dict()), 'seed': seed}


def run(args):
    p = m0.load_protocol(args.protocol)
    seed = args.seed
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
    keep = np.flatnonzero(folds != args.fold)
    held = np.flatnonzero(folds == args.fold)
    labels = np.asarray(ds['train'].labels, dtype=np.int64)[keep]
    origin = train.normal_origin.to_numpy()[keep]
    weights = source.sampling_weights(labels, origin, True)
    cfg = p['training']
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if seed == 42:
        model, init = m0.build_model(p, 42, args.initial_anchor)
    else:
        model, init = build_seed_model(p, seed)
    out.mkdir(parents=True)
    device = torch.device('cuda:0'); model.to(device)
    sub = Subset(source.IndexedSource(ds['train']), keep.tolist())
    sampler = WeightedRandomSampler(weights, len(keep), replacement=True, generator=torch.Generator().manual_seed(seed + 11))
    trainloader = DataLoader(sub, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0,
                             generator=torch.Generator().manual_seed(seed + 101))
    meta = frame['val']
    valloader = DataLoader(source.IndexedSource(ds['val'], meta.cache_index.to_numpy()), batch_size=512, shuffle=False)
    heldloader = DataLoader(source.IndexedSource(ds['train'], held), batch_size=512, shuffle=False)
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
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_macro_f1': metrics['overall']['macro_f1']})
        print('M0_CV fold', args.fold, epoch, 'loss', history[-1]['train_loss'], 'val_f1', history[-1]['val_macro_f1'], flush=True)
        if stop:
            break
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    model.eval(); probs = []
    with torch.no_grad():
        for x, _, _ in heldloader:
            probs.append(model(x.to(device)).softmax(1).cpu().numpy())
    probs = np.concatenate(probs)
    np.savez_compressed(out / 'heldout_predictions.npz', probabilities=probs,
                        original_indices=train.original_cache_index.to_numpy()[held])
    json.dump({'fold': args.fold, 'seed': seed, 'train_windows': len(keep), 'heldout_windows': len(held), 'best_epoch': sel['best_epoch'],
               'epochs': len(history), 'initial_digest': init['m0_initial_state_digest'], 'history': history,
               'elapsed_s': time.perf_counter() - start, 'status': 'COMPLETE'}, open(out / 'report.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--initial-anchor', type=Path, required=True)
    ap.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
