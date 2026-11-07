"""V confirmation runner; CPU smoke uses TRAIN STAND_IN, never real val.

Training delegates every optimizer epoch to frozen T, and CE/initialization/
cache handling to frozen S/T. Only logits persistence and V gates are new.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_t_selection as t
from train_t_selection import s
import evaluate_s_ablation as m
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROTOCOL_SHA = '63f26e5bc57bb642903e3f2d0998efb5c982d14507e2f1b32e7e5ffb3baa4778'
DEFAULT_PROTOCOL = ROOT / 'configs/experiments/v_confirm_fixed100_mstca_v1.json'
OUTPUT_ROOT = ROOT / 'artifacts/v_confirm_fixed100_mstca_v1'
SMOKE_ROOT = OUTPUT_ROOT / 'smoke'
ORIGIN = ROOT / 'artifacts/t_selection_mstca_v1/evaluation_cv/evaluation.json'
LOCAL_ORIGIN = ROOT / 'artifacts/t_selection_mstca_v1/server_r1/evaluation_cv/evaluation.json'
IMPLEMENTATION_FILES = ('scripts/research/train_v_confirm.py', 'scripts/research/evaluate_v_confirm.py',
                        'scripts/research/preflight_v_confirm.py', 'tests/test_v_confirm.py')
SEEDS = t.SEEDS


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    if s.sha(path) != PROTOCOL_SHA or path.with_suffix('.json.sha256').read_text(encoding='utf-8').strip() != PROTOCOL_SHA:
        raise ValueError('Frozen V protocol/lock mismatch')
    protocol = s.read_json(path)
    pins = {name: t.normalized_sha(ROOT / name) for name in protocol['code_identities']['files']}
    if pins != protocol['code_identities']['files']:
        raise ValueError('V normalized frozen S/T code identity mismatch')
    return protocol, pins


def implementation_manifest():
    return {'protocol_sha256': PROTOCOL_SHA, 'hash_rule': 'CRLF->LF then SHA256',
            'files': {name: t.normalized_sha(ROOT / name) for name in IMPLEMENTATION_FILES}}


def origin_gate(path, expected_sha):
    if not path or s.sha(path) != expected_sha:
        raise ValueError('T phase-1 evaluation SHA mismatch: refuse before data access')
    result = s.read_json(path)
    if (result.get('status') != 'COMPLETE' or result.get('phase') != 'cv'
            or result.get('protocol_sha256') != t.PROTOCOL_SHA or result.get('synthetic_demo')
            or result.get('performance_claim') is not True or result.get('population') != 162069
            or len(result.get('run_report_sha256', {})) != 15):
        raise ValueError('T origin must be a complete formal phase-1 evaluation')
    return {'path': str(Path(path).resolve()), 'sha256': expected_sha,
            't_verdicts_unchanged': result['decisions'], 't_adoption_unchanged': result['adopted_rule']}


def origin_path(args):
    supplied = getattr(args, 't_phase1_evaluation', None)
    return Path(supplied) if supplied else (LOCAL_ORIGIN if args.cpu_smoke and not ORIGIN.exists() else ORIGIN)


def write_report(path, value):
    """Unique atomic payloads keep an old valid report on I/O failure.

    Retry only a transient file replacement, never a training/prediction pass.
    Every real retry is retained in io_events. Python tempfile RNG is separate
    from both Torch and NumPy training RNGs.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    for attempt in range(3):
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            value.setdefault('io_events', []).append({'operation': 'atomic_report_replace',
                'attempt': attempt + 1, 'error': str(exc)})
            if attempt == 2:
                raise
            time.sleep(.1)


def logits_to_probability(logits, device='cpu'):
    logits = np.asarray(logits)
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != 2 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected finite float32 logits [n,2]')
    values = []
    with torch.inference_mode():
        for start in range(0, len(logits), 512):
            batch = torch.as_tensor(logits[start:start + 512], device=device)
            values.append(batch.double().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def infer_logits(model, loader, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for x, _ in loader:
            logits = model(x.to(device))
            if logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise FloatingPointError('Invalid raw float32 model logits')
            result.append(logits.cpu().numpy())
    return np.concatenate(result)


def save_logits(path, logits, dataset, epochs, scope):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, logits=logits, original_cache_index=dataset.ids.numpy(),
            labels=dataset.y.numpy(), epochs=np.asarray(epochs, dtype=np.int64), population_scope=np.asarray(scope))


def selections(losses):
    return {'V0': t.replay_t0(losses), 'V1': {'epoch': 100, 'available': len(losses) == 100}}


def predict_selected(model, loader, dataset, selected, output, device, *, stand_in=False):
    epochs = {selected['V0']['epoch']}
    if selected['V1']['available']:
        epochs.add(selected['V1']['epoch'])
    artifacts = {}
    scope = 'STAND_IN' if stand_in else 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION'
    for epoch in sorted(epochs):
        model.load_state_dict(torch.load(Path(output) / f'checkpoints/epoch_{epoch}.pt', map_location=device, weights_only=True))
        logits = infer_logits(model, loader, device)
        path = Path(output) / f'{"stand_in" if stand_in else "val"}_logits_epoch_{epoch}.npz'
        save_logits(path, logits, dataset, [epoch], scope)
        artifacts[str(epoch)] = {'file': path.name, 'sha256': s.sha(path)}
    return artifacts


def reproduction_gate(seed, history, selected, v0_fp, reference_root=None):
    root = ROOT / 'artifacts/s_ablation_mstca_v1/val' if reference_root is None else Path(reference_root)
    reference = root / f'S0_seed{seed}_fold0/report.json'
    old = s.read_json(reference)
    if (old.get('status') != 'COMPLETE' or old.get('phase') != 'val' or old.get('arm') != 'S0'
            or old.get('seed') != seed or old.get('fold') != 0 or old.get('protocol_sha256') != s.PROTOCOL_SHA):
        raise ValueError('Wrong S0 phase-2 reproduction reference')
    best = old['best_epoch']
    if type(best) is not int or not 1 <= best <= len(history) or history[best - 1]['state_digest'] != old['checkpoint_state_digests']['best']:
        raise RuntimeError('V/S0 checkpoint digest mismatch: STOP')
    if selected['V0']['epoch'] != best:
        raise RuntimeError('V0/S0 selected epoch mismatch: STOP')
    if v0_fp != {42: 785, 43: 428, 44: 741}[seed]:
        raise RuntimeError('V0/S0 val FP@R97 mismatch: STOP')
    return {'status': 'PASS', 's0_report_sha256': s.sha(reference), 'best_epoch': best, 'fp_r97': v0_fp}


def require_formal_slot(args, manifest_sha):
    parent = OUTPUT_ROOT / 'val'
    if Path(args.output).resolve() != (parent / f'seed{args.seed}').resolve():
        raise ValueError('Formal V output must use its canonical seed slot')
    for seed in SEEDS[:SEEDS.index(args.seed)]:
        path = parent / f'seed{seed}/report.json'
        old = s.read_json(path) if path.exists() else {}
        if (old.get('status') != 'COMPLETE' or old.get('protocol_sha256') != PROTOCOL_SHA
                or old.get('seed') != seed or old.get('epochs') != 100
                or old.get('implementation_manifest_sha256') != manifest_sha):
            raise RuntimeError('Preceding V seed incomplete/different implementation')


def train_trajectory(model, trainloader, inner, output, epochs, device, seed, report):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = s.FocalLoss(gamma=2., alpha=None)
    ce_loader = t.inference_loader(inner, seed)
    logits_loader = t.inference_loader(inner, seed)
    (Path(output) / 'checkpoints').mkdir(exist_ok=False)
    history = []
    inner_logits = np.lib.format.open_memmap(Path(output) / 'inner_logits_progress.npy', mode='w+',
        dtype=np.float32, shape=(epochs, len(inner), 2))
    report['inner_logits_progress_file'] = 'inner_logits_progress.npy'
    for epoch in range(1, epochs + 1):
        lr, loss, steps = t.train_epoch(model, trainloader, optimizer, criterion, device, epoch)
        model.eval()
        s.save_state(Path(output) / f'checkpoints/epoch_{epoch}.pt', model.state_dict())
        digest = s._state_digest(model.state_dict())
        ce = s.inner_ce(model, ce_loader, device)
        inner_logits[epoch - 1] = infer_logits(model, logits_loader, device)
        inner_logits.flush()  # Persist this epoch before publishing its completed report entry.
        history.append({'epoch': epoch, 'lr': lr, 'fit_focal_loss': loss, 'gradient_steps': steps,
                        'inner_class_balanced_ce': ce, 'state_digest': digest})
        report.update(epochs=epoch, optimizer_steps=sum(row['gradient_steps'] for row in history), history=history)
        write_report(Path(output) / 'report.json', report)
    return history, np.asarray(inner_logits, dtype=np.float32)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh V output directory required')
    if type(args.seed) is not int or args.seed not in SEEDS:
        raise ValueError('Unregistered V seed')
    protocol, pins = load_protocol(args.protocol)
    origin = origin_gate(origin_path(args), protocol['origin']['t_phase1_evaluation_sha256'])
    actual = t.environment()
    if args.cpu_smoke:
        if not output.resolve().is_relative_to(SMOKE_ROOT.resolve()):
            raise ValueError('Local STAND_IN smoke must stay under V smoke root')
    else:
        t.require_formal_environment(actual)
    manifest = implementation_manifest()
    manifest_sha = getattr(args, 'implementation_manifest_sha256', None)
    if not args.cpu_smoke:
        path = getattr(args, 'implementation_manifest', None)
        if not path or not manifest_sha or s.sha(path) != manifest_sha or s.read_json(path) != manifest:
            raise ValueError('Formal V run needs the sealed implementation manifest')
        require_formal_slot(args, manifest_sha)
    identities = t.verify_data(args.data, local=args.cpu_smoke)
    frame = s.read_ledger(args.data)
    s.fold_sizes(frame, s.read_json(s.DEFAULT_PROTOCOL))
    positions, _ = s.split_ledger(frame, 0, 'val')
    full_counts = {role: len(positions[role]) for role in ('fit', 'inner')}
    if args.cpu_smoke:
        positions['fit'] = s.take_smoke(frame, positions['fit'], 512)
        positions['inner'] = s.take_smoke(frame, positions['inner'], 256)
        full_fit, _ = s.split_ledger(frame, 0, 'val')
        pool = np.setdiff1d(full_fit['fit'], positions['fit'])
        stand_indices = s.take_smoke(frame, pool, 256)
    fit = s.CacheWindows(args.data, frame, positions['fit'])
    inner = s.CacheWindows(args.data, frame, positions['inner'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    before = torch.random.get_rng_state().clone()
    model = s.build_variant('mstca', 26, 2, .2)
    receipt = s.initialize_variant(model, 'mstca', args.seed, policy='module_seed_v1')
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError('Factory consumed global CPU RNG')
    if receipt['full_state_digest'] != t.golden_digests()[('mstca', args.seed)]:
        raise RuntimeError('Golden mismatch: STOP; never regenerate or change CPU dispatch')
    from smoke_normal_source_sampling import sampling_weights, ORIGINAL, PREFAULT, FAULT
    selected = frame.iloc[positions['fit']]
    origins = np.where(selected.qualified_y == 1, FAULT,
                       np.where(selected.normal_origin == 'normal_flight', ORIGINAL, PREFAULT))
    sampler = WeightedRandomSampler(sampling_weights(selected.qualified_y.to_numpy(), origins, True),
        len(fit), replacement=True, generator=torch.Generator().manual_seed(args.seed + 11))
    trainloader = DataLoader(fit, batch_size=512, sampler=sampler, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + 101))
    output.mkdir(parents=True, exist_ok=False)
    s.write_json(output / 'factory_receipt.json', receipt)
    s.write_json(output / 'implementation_manifest.json', manifest)
    population = {role: frame.iloc[positions[role]].original_cache_index.tolist() for role in ('fit', 'inner')}
    if args.cpu_smoke:
        population['STAND_IN'] = frame.iloc[stand_indices].original_cache_index.tolist()
    s.write_json(output / 'population.json', population)
    report = {'status': 'RUNNING', 'phase': 'val', 'seed': args.seed, 'cpu_smoke': args.cpu_smoke,
        'prediction_scope': 'STAND_IN' if args.cpu_smoke else 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION',
        'performance_claim': False, 'protocol_sha256': PROTOCOL_SHA, 'code_sha256': pins,
        'origin': origin, 'environment': actual, 'device': str(device), 'data_identities': identities,
        'implementation_files': manifest['files'],
        'implementation_manifest_sha256': manifest_sha or s.sha(output / 'implementation_manifest.json'),
        'initial_digest': receipt['full_state_digest'], 'param_count': receipt['param_count'], 'macs': receipt['macs'],
        'full_population_counts': full_counts, 'used_window_counts': {k: len(positions[k]) for k in ('fit', 'inner')},
        'epochs': 0, 'optimizer_steps': 0, 'history': [], 'io_events': [],
        'val_predicted_during_training': False, 'started_at': datetime.now(timezone.utc).isoformat()}
    s.write_json(output / 'report.json', report)
    started = time.perf_counter()
    try:
        model.to(device)
        epochs = 2 if args.cpu_smoke else 100
        history, inner_logits = train_trajectory(model, trainloader, inner, output, epochs, device, args.seed, report)
        chosen = selections([row['inner_class_balanced_ce'] for row in history])
        s.write_json(output / 'selections.json', chosen)
        save_logits(output / 'inner_logits.npz', inner_logits, inner, range(1, epochs + 1), 'SOURCE_TRAIN_INNER')
        if args.cpu_smoke:
            confirmation = s.CacheWindows(args.data, frame, stand_indices)
            confirmation_n = len(stand_indices)
        else:
            val_frame = s.read_ledger(args.data, 'val')
            if len(val_frame) != 34710 or int(val_frame.qualified_y.eq(0).sum()) != 17740:
                raise ValueError('Wrong full val population')
            confirmation = s.CacheWindows(args.data, val_frame, np.arange(len(val_frame)), split='val')
            confirmation_n = len(val_frame)
        artifacts = predict_selected(model, t.inference_loader(confirmation, args.seed), confirmation,
                                     chosen, output, device, stand_in=args.cpu_smoke)
        if not args.cpu_smoke and args.seed <= 44:
            epoch = chosen['V0']['epoch']
            with np.load(output / artifacts[str(epoch)]['file'], allow_pickle=False) as saved:
                probabilities = logits_to_probability(saved['logits'], device)
            report['reproduction'] = reproduction_gate(args.seed, history, chosen,
                m.fp_at_recall(confirmation.y.numpy(), probabilities)['fp'])
        else:
            report['reproduction'] = {'status': 'NOT_APPLICABLE_STAND_IN_OR_NEW_SEED'}
        if load_protocol(args.protocol)[1] != pins or implementation_manifest() != manifest:
            raise ValueError('Code/protocol changed during V run')
        origin_gate(origin['path'], protocol['origin']['t_phase1_evaluation_sha256'])
        report.update(status='COMPLETE_CPU_SMOKE_STAND_IN_NOT_FORMAL' if args.cpu_smoke else 'COMPLETE',
            prediction_artifacts=artifacts, confirmation_window_count=confirmation_n,
            val_prediction_passes=0 if args.cpu_smoke else len(artifacts),
            stand_in_prediction_passes=len(artifacts) if args.cpu_smoke else 0,
            inner_logits_sha256=s.sha(output / 'inner_logits.npz'),
            checkpoint_state_digests={str(row['epoch']): row['state_digest'] for row in history},
            wall_time_s=time.perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat())
        write_report(output / 'report.json', report)
        return report
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), wall_time_s=time.perf_counter() - started)
        write_report(output / 'report.json', report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    result.add_argument('--seed', type=int, choices=SEEDS, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--cpu-smoke', action='store_true')
    result.add_argument('--t-phase1-evaluation', type=Path)
    result.add_argument('--implementation-manifest', type=Path)
    result.add_argument('--implementation-manifest-sha256')
    return result


if __name__ == '__main__':
    r = run(parser().parse_args())
    print({k: r[k] for k in ('status', 'seed', 'epochs', 'prediction_scope')})
