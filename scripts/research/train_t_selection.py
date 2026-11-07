"""Frozen T trajectories. Local execution is CPU-smoke CV only; never opens val.

S helpers are imported, never patched. Formal CV hashes val identity bytes only;
local mode skips all three val files. No selection consumes held-out predictions.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_s_ablation as s
import evaluate_s_ablation as metrics
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROTOCOL_SHA = 'f434552dc74e45333202d313d3a1810b7be3676ab21c4ac1f2badccc2b644ca9'
DEFAULT_PROTOCOL = ROOT / 'configs/experiments/t_selection_mstca_v1.json'
OUTPUT_ROOT = ROOT / 'artifacts/t_selection_mstca_v1'
SMOKE_ROOT = OUTPUT_ROOT / 'smoke'
IMPLEMENTATION_FILES = ('scripts/research/train_t_selection.py',
    'scripts/research/evaluate_t_selection.py', 'scripts/research/preflight_t_selection.py',
    'tests/test_t_selection.py')
SEEDS = (42, 43, 44, 45, 46)
SMOKE_LIMITS = {'fit': 512, 'inner': 256, 'heldout': 256}
VAL_FILES = {'val_cache_complete.json', 'cache/val.pt', 'ledgers/val_included_windows.csv'}


def normalized_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    actual = s.sha(path)
    if actual != PROTOCOL_SHA or path.with_suffix('.json.sha256').read_text(encoding='utf-8').strip() != actual:
        raise ValueError('Frozen T protocol/lock mismatch')
    protocol = s.read_json(path)
    pins = {name: normalized_sha(ROOT / name) for name in protocol['code_identities']['files']}
    if pins != protocol['code_identities']['files']:
        raise ValueError('Normalized frozen code identity mismatch')
    return protocol, pins


def implementation_manifest():
    return {'protocol_sha256': PROTOCOL_SHA, 'hash_rule': 'CRLF->LF then SHA256',
            'files': {name: normalized_sha(ROOT / name) for name in IMPLEMENTATION_FILES}}


def golden_digests():
    result = s.golden_digests()
    tree = ast.parse((ROOT / 'tests/test_t_selection.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NEW_GOLDEN_DIGESTS' for t in node.targets):
            extra = ast.literal_eval(node.value)
            if set(extra) != {('mstca', 45), ('mstca', 46)} or set(extra) & set(result):
                raise ValueError('Only two new MSTCA goldens allowed')
            return result | extra
    raise ValueError('Two pinned local CPU goldens missing')


def environment():
    result = s.environment()
    result['aten_cpu_capability'] = os.environ.get('ATEN_CPU_CAPABILITY')
    result['effective_cpu_capability'] = torch.backends.cpu.get_cpu_capability()
    return result


def require_formal_environment(actual=None):
    actual = environment() if actual is None else actual
    s.require_formal_environment(actual)
    if actual.get('aten_cpu_capability') != 'default':
        raise RuntimeError('Formal environment requires ATEN_CPU_CAPABILITY=default; never setdefault')


def verify_data(root, *, local=False):
    identities = s.read_json(s.BASELINE)['data_identities']
    matched, skipped = {}, []
    for name, expected in identities.items():
        if local and name in VAL_FILES:
            skipped.append(name)
            continue
        actual = s.sha(Path(root) / name)
        if actual != expected:
            raise ValueError('Frozen data identity mismatch: ' + name)
        matched[name] = actual
    return {'matched_sha256': matched, 'skipped_sha256': skipped,
            'val_access': 'NONE_NOT_EVEN_HASHED' if local else 'SHA256_BYTES_ONLY_UNTIL_AUTHORIZED_PHASE2'}


def replay_t0(losses):
    if not 1 <= len(losses) <= 100:
        raise ValueError('T0 needs 1..100 CE observations')
    state = {'best_loss': math.inf, 'best_epoch': None, 'reference': math.inf, 'bad_epochs': 0}
    stop_epoch = None
    for epoch, loss in enumerate(losses, 1):
        state, _, stop = s.selection_step(state, float(loss), epoch)
        if stop:
            stop_epoch = epoch
            break
    return {'epoch': state['best_epoch'], 'stop_epoch': stop_epoch,
            'observed_through': stop_epoch or len(losses)}


def select_epochs(losses, inner_probabilities, labels):
    probabilities = np.asarray(inner_probabilities)
    if probabilities.ndim != 2 or len(probabilities) != len(losses):
        raise ValueError('Inner trajectories/CE history mismatch')
    fps = [metrics.fp_at_recall(labels, row)['fp'] for row in probabilities]
    if any(fp is None for fp in fps):
        raise ValueError('T2 inner FP@R97 is undefined')
    t2 = min(range(len(fps)), key=lambda i: (fps[i], -i)) + 1
    return {'T0': replay_t0(losses), 'T1': {'epoch': 100, 'available': len(losses) == 100},
            'T2': {'epoch': t2, 'inner_fp_at_r97': fps[t2 - 1], 'tie_break': 'later'}}


def reproduction_gate(seed, fold, history, selections, report_root=None):
    root = ROOT / 'artifacts/s_ablation_mstca_v1/cv' if report_root is None else Path(report_root)
    old = s.read_json(root / f'S0_seed{seed}_fold{fold}/report.json')
    if (old.get('status') != 'COMPLETE' or old.get('phase') != 'cv' or old.get('protocol_sha256') != s.PROTOCOL_SHA
            or old.get('seed') != seed or old.get('fold') != fold or old.get('arm') != 'S0'):
        raise ValueError('S0 reproduction reference is not a matching COMPLETE run')
    best = old['best_epoch']
    if type(best) is not int or not 1 <= best <= len(history):
        raise RuntimeError('S0 best epoch outside available trajectory: STOP')
    if history[best - 1]['state_digest'] != old['checkpoint_state_digests']['best']:
        raise RuntimeError('S0 reproduction checkpoint digest mismatch: STOP')
    if selections['T0']['epoch'] != best:
        raise RuntimeError('S0 reproduction selected epoch mismatch: STOP')
    return {'status': 'PASS', 's0_report_sha256': s.sha(root / f'S0_seed{seed}_fold{fold}/report.json'),
            'best_epoch': best}


def phase1_gate(path, expected_sha, adopted_rule):
    if not path or not expected_sha:
        raise RuntimeError('Phase val needs phase-1 verdicts and SHA256 before data access')
    if s.sha(path) != expected_sha:
        raise ValueError('Phase-1 verdict SHA mismatch')
    result = s.read_json(path)
    if (result.get('status') != 'COMPLETE' or result.get('phase') != 'cv'
            or result.get('protocol_sha256') != PROTOCOL_SHA or result.get('synthetic_demo')
            or result.get('performance_claim') is not True or result.get('population') != 162069
            or len(result.get('run_report_sha256', {})) != 15
            or result.get('evaluation_reproduction', {}).get('status') != 'PASS'):
        raise ValueError('Phase-1 verdict must be complete same-protocol 15-run CV, never smoke')
    if result.get('adopted_rule') != adopted_rule or adopted_rule not in ('T0', 'T1', 'T2'):
        raise ValueError('Adopted rule must match the SHA-pinned phase-1 verdict')
    if set(result.get('decisions', {})) != {'T1', 'T2'}:
        raise ValueError('Missing phase-1 decisions')
    from evaluate_t_selection import adopt
    expected = adopt(result['decisions'], result['comparisons']['T2_vs_T1']['aggregate_ratio_ci'])
    if expected != adopted_rule:
        raise ValueError('Phase-1 adoption differs from the frozen mechanical rule')
    return result


def require_formal_slot(args, manifest_sha):
    slots = [(seed, fold) for seed in SEEDS for fold in (range(3) if args.phase == 'cv' else [0])]
    current = (args.seed, args.fold)
    parent = OUTPUT_ROOT / args.phase
    if Path(args.output).resolve() != (parent / f'seed{args.seed}_fold{args.fold}').resolve():
        raise ValueError('Formal output must use the canonical T run slot')
    for seed, fold in slots[:slots.index(current)]:
        path = parent / f'seed{seed}_fold{fold}/report.json'
        previous = s.read_json(path) if path.exists() else {}
        if (previous.get('status') != 'COMPLETE' or previous.get('protocol_sha256') != PROTOCOL_SHA
                or previous.get('phase') != args.phase or previous.get('seed') != seed
                or previous.get('fold') != fold or previous.get('epochs') != 100
                or previous.get('implementation_manifest_sha256') != manifest_sha):
            raise RuntimeError('Required preceding T run incomplete/different implementation')


def inference_loader(dataset, seed):
    return DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed + 101))


def predict(model, loader, device):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for x, _ in loader:
            probabilities.append(model(x.to(device)).double().softmax(1)[:, 1].cpu().numpy().astype(np.float32))
    result = np.concatenate(probabilities)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise FloatingPointError('Invalid fault probabilities')
    return result


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    lr = s.get_warmup_cosine_lr(epoch - 1, 5, 100, 3e-4, 1e-6)
    for group in optimizer.param_groups:
        group['lr'] = lr
    model.train()
    losses = []
    for x, y in loader:
        if not torch.isfinite(x).all() or not torch.isin(y, torch.tensor([0, 1])).all():
            raise ValueError('Invalid sample before forward/BN')
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fit focal loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return lr, float(np.mean(losses)), len(losses)


def trajectory(model, trainloader, innerloader, inner_prediction_loader, heldloader,
               output, epochs, device, *, record_predictions=True, progress=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = s.FocalLoss(gamma=2., alpha=None)
    history, inner, held = [], [], []
    (Path(output) / 'checkpoints').mkdir(exist_ok=False)
    for epoch in range(1, epochs + 1):
        lr, loss, steps = train_epoch(model, trainloader, optimizer, criterion, device, epoch)
        model.eval()
        s.save_state(Path(output) / f'checkpoints/epoch_{epoch}.pt', model.state_dict())
        digest = s._state_digest(model.state_dict())
        ce = s.inner_ce(model, innerloader, device)
        if record_predictions:
            inner.append(predict(model, inner_prediction_loader, device))
            if heldloader is not None:
                held.append(predict(model, heldloader, device))
        history.append({'epoch': epoch, 'lr': lr, 'fit_focal_loss': loss,
                        'inner_class_balanced_ce': ce, 'gradient_steps': steps, 'state_digest': digest})
        if progress is not None:
            progress(history)
    return history, np.asarray(inner, dtype=np.float32), np.asarray(held, dtype=np.float32)


def save_predictions(path, probabilities, dataset, *, epochs=None):
    epochs = np.arange(1, len(probabilities) + 1, dtype=np.int64) if epochs is None else np.asarray(epochs, dtype=np.int64)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, probabilities=probabilities,
            original_cache_index=dataset.ids.numpy(), labels=dataset.y.numpy(), epochs=epochs)


def predict_selected_val(model, loader, dataset, selections, adopted_rule, output, device):
    chosen = {selections['T0']['epoch'], selections[adopted_rule]['epoch']}
    mapping = {}
    for epoch in sorted(chosen):
        model.load_state_dict(torch.load(Path(output) / f'checkpoints/epoch_{epoch}.pt', map_location=device, weights_only=True))
        probability = predict(model, loader, device)
        path = Path(output) / f'val_epoch_{epoch}.npz'
        save_predictions(path, probability[None, :], dataset, epochs=[epoch])
        mapping[str(epoch)] = s.sha(path)
    return mapping


def run(args, *, probe_epochs=None):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    if args.phase == 'val':
        phase1_gate(args.phase1_verdicts, args.phase1_verdicts_sha256, args.adopted_rule)
        if args.cpu_smoke or args.fold != 0:
            raise ValueError('Val is formal only, fold 0, never CPU smoke')
    if type(args.seed) is not int or type(args.fold) is not int or args.seed not in SEEDS or args.fold not in (0, 1, 2) or args.phase not in ('cv', 'val'):
        raise ValueError('Unregistered T seed/fold/phase')
    if probe_epochs is not None and (probe_epochs != 2 or args.seed != 42 or args.fold != 0 or args.phase != 'cv' or args.cpu_smoke):
        raise ValueError('Determinism probe is CUDA seed42/fold0 CV, exactly two epochs')
    actual = environment()
    if not args.cpu_smoke:
        require_formal_environment(actual)
    elif not output.resolve().is_relative_to(SMOKE_ROOT.resolve()):
        raise ValueError('CPU smoke must stay under T smoke root')
    protocol, pins = load_protocol(args.protocol)
    manifest = implementation_manifest()
    manifest_sha = getattr(args, 'implementation_manifest_sha256', None)
    if not args.cpu_smoke:
        path = getattr(args, 'implementation_manifest', None)
        if not path or not manifest_sha or s.sha(path) != manifest_sha or s.read_json(path) != manifest:
            raise ValueError('Formal run requires the SHA-pinned implementation manifest')
        if probe_epochs is None:
            require_formal_slot(args, manifest_sha)
    identities = verify_data(args.data, local=args.cpu_smoke)
    inherited = s.read_json(s.DEFAULT_PROTOCOL)
    frame = s.read_ledger(args.data)
    sizes = s.fold_sizes(frame, inherited)
    positions, _ = s.split_ledger(frame, args.fold, args.phase)
    full_counts = {key: len(value) for key, value in positions.items()}
    if args.cpu_smoke:
        positions = {key: s.take_smoke(frame, value, SMOKE_LIMITS[key]) for key, value in positions.items()}
    fit = s.CacheWindows(args.data, frame, positions['fit'])
    inner = s.CacheWindows(args.data, frame, positions['inner'])
    held = s.CacheWindows(args.data, frame, positions['heldout']) if args.phase == 'cv' else None
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
        raise RuntimeError('Factory changed global CPU RNG')
    if receipt['full_state_digest'] != golden_digests()[('mstca', args.seed)]:
        raise RuntimeError('Golden initialization mismatch: STOP, never regenerate')
    # Historical sampler has setdefault side effects; formal environment was already asserted.
    from smoke_normal_source_sampling import sampling_weights, ORIGINAL, PREFAULT, FAULT
    selected = frame.iloc[positions['fit']]
    origins = np.where(selected.qualified_y == 1, FAULT,
                       np.where(selected.normal_origin == 'normal_flight', ORIGINAL, PREFAULT))
    weights = sampling_weights(selected.qualified_y.to_numpy(), origins, True)
    sampler = WeightedRandomSampler(weights, len(fit), replacement=True,
        generator=torch.Generator().manual_seed(args.seed + 11))
    trainloader = DataLoader(fit, batch_size=512, sampler=sampler, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 101))
    output.mkdir(parents=True, exist_ok=False)
    s.write_json(output / 'factory_receipt.json', receipt)
    s.write_json(output / 'implementation_manifest.json', manifest)
    s.write_json(output / 'population.json', {key: frame.iloc[value].original_cache_index.tolist() for key, value in positions.items()})
    report = {'status': 'RUNNING', 'phase': args.phase, 'variant': 'mstca', 'seed': args.seed, 'fold': args.fold,
        'cpu_smoke': args.cpu_smoke, 'determinism_probe': probe_epochs is not None,
        'protocol_sha256': PROTOCOL_SHA, 'code_sha256': pins, 'implementation_files': manifest['files'],
        'implementation_manifest_sha256': manifest_sha or s.sha(output / 'implementation_manifest.json'),
        'environment': actual, 'device': str(device), 'data_root': str(Path(args.data).resolve()),
        'initial_digest': receipt['full_state_digest'],
        'data_identities': identities, 'fold_sizes': sizes, 'full_population_counts': full_counts,
        'used_window_counts': {key: len(value) for key, value in positions.items()},
        'param_count': receipt['param_count'], 'macs': receipt['macs'], 'epochs': 0, 'optimizer_steps': 0,
        'phase1_verdicts_sha256': getattr(args, 'phase1_verdicts_sha256', None),
        'history': [], 'started_at': datetime.now(timezone.utc).isoformat()}
    s.write_json(output / 'report.json', report)
    started = time.perf_counter()
    try:
        model.to(device)
        def progress(history):
            report.update(history=history, epochs=len(history),
                          optimizer_steps=sum(row['gradient_steps'] for row in history))
            s.write_json(output / 'report.json', report, update=True)
        epochs = 2 if args.cpu_smoke else (probe_epochs or 100)
        history, inner_probs, held_probs = trajectory(model, trainloader,
            inference_loader(inner, args.seed), inference_loader(inner, args.seed),
            inference_loader(held, args.seed) if held is not None else None,
            output, epochs, device, progress=progress)
        selections = select_epochs([row['inner_class_balanced_ce'] for row in history], inner_probs, inner.y.numpy())
        s.write_json(output / 'selections.json', selections)
        save_predictions(output / 'inner_predictions.npz', inner_probs, inner)
        predictions = {'inner': s.sha(output / 'inner_predictions.npz')}
        if held is not None:
            save_predictions(output / 'heldout_predictions.npz', held_probs, held)
            predictions['heldout'] = s.sha(output / 'heldout_predictions.npz')
        if args.phase == 'cv' and args.seed <= 44 and not args.cpu_smoke and probe_epochs is None:
            report['reproduction'] = reproduction_gate(args.seed, args.fold, history, selections)
        else:
            report['reproduction'] = {'status': 'NOT_APPLICABLE_VAL_NEW_SEED_OR_NONFORMAL'}
        if args.phase == 'val':
            val_frame = s.read_ledger(args.data, 'val')
            if len(val_frame) != 34710:
                raise ValueError('Wrong source val population')
            val_set = s.CacheWindows(args.data, val_frame, np.arange(len(val_frame)), split='val')
            mapping = predict_selected_val(model, inference_loader(val_set, args.seed), val_set,
                selections, args.adopted_rule, output, device)
            report.update(adopted_rule=args.adopted_rule, val_prediction_sha256=mapping,
                          val_prediction_passes=len(mapping), val_window_count=len(val_frame))
        if load_protocol(args.protocol)[1] != pins or implementation_manifest() != manifest:
            raise ValueError('Code/protocol changed during run')
        status = 'COMPLETE_CPU_SMOKE_NOT_FORMAL' if args.cpu_smoke else ('COMPLETE_DETERMINISM_PROBE' if probe_epochs is not None else 'COMPLETE')
        report.update(status=status, prediction_sha256=predictions,
            checkpoint_state_digests={str(row['epoch']): row['state_digest'] for row in history},
            wall_time_s=time.perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat())
        s.write_json(output / 'report.json', report, update=True)
        return report
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), wall_time_s=time.perf_counter() - started)
        s.write_json(output / 'report.json', report, update=True)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    result.add_argument('--seed', type=int, choices=SEEDS, required=True)
    result.add_argument('--fold', type=int, choices=(0, 1, 2), required=True)
    result.add_argument('--phase', choices=('cv', 'val'), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--cpu-smoke', action='store_true')
    result.add_argument('--implementation-manifest', type=Path)
    result.add_argument('--implementation-manifest-sha256')
    result.add_argument('--phase1-verdicts', type=Path)
    result.add_argument('--phase1-verdicts-sha256')
    result.add_argument('--adopted-rule', choices=('T0', 'T1', 'T2'))
    return result


if __name__ == '__main__':
    result = run(parser().parse_args())
    print({key: result[key] for key in ('status', 'seed', 'fold', 'epochs')})
