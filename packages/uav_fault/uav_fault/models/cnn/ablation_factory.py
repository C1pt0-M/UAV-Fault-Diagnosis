"""Four CPU research baselines with independent semantic-module initialization.

Importable with only packages/uav_fault on sys.path; no research-script imports.
No training/data access or KG construction here.
MACs count Conv/Linear only, for one 150-frame window (one multiply-add = one MAC).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .lstm import OnlineCNN1D, OnlineMSTCA

VARIANTS = ('mstca', 'mstca_no_se', 'mstca_single_k7', 'cnn1d_matched')
POLICY = 'module_seed_v1'
PURE_M0_DIGEST = 'a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646'
TARGET_PARAMETERS = 105938
_MSTCA_NAMES = (
    'encoder.branch_s.0', 'encoder.branch_s.1',
    'encoder.branch_m.0', 'encoder.branch_m.1',
    'encoder.branch_l.0', 'encoder.branch_l.1',
    'encoder.se.fc.0', 'encoder.se.fc.2',
    'encoder.block2.0', 'encoder.block2.1',
    'encoder.block3.0', 'encoder.block3.1', 'head.1',
)
_CNN_NAMES = ('features.0', 'features.1', 'features.4', 'features.5',
              'features.8', 'features.9', 'head.1')
_CONV_LENGTHS = {'encoder.branch_s.0': 150, 'encoder.branch_m.0': 150,
                 'encoder.branch_l.0': 150, 'encoder.block2.0': 75,
                 'encoder.block3.0': 37, 'features.0': 150,
                 'features.4': 75, 'features.8': 37}


def _state_digest(state):
    """Byte-identical to train_source_normal_sampling.state_digest; no script dependency."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cnn1d_candidates():
    """Enumerate monotone integer widths 1 <= c1 <= c2 <= c3=128, Base26/2.

    Same k7/pool/k5/pool/k3/GAP/head flow as OnlineCNN1D. Rank by absolute
    parameter error, then lower parameter count, then lexicographic widths.
    No models or random numbers are needed for the enumeration.
    """
    rows = []
    for c1 in range(1, 129):
        for c2 in range(c1, 129):
            parameters = 5*c1*c2 + 185*c1 + 387*c2 + 642
            macs = 26*c1*7*150 + c1*c2*5*75 + c2*128*3*37 + 128*2
            rows.append({'widths': [c1, c2, 128], 'param_count': parameters,
                         'macs': macs, 'absolute_error': abs(parameters-TARGET_PARAMETERS)})
    return sorted(rows, key=lambda r: (r['absolute_error'], r['param_count'], r['widths']))


def _check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {VARIANTS}')


def build_variant(variant, in_channels=26, num_classes=2, dropout=0.2):
    """Construct only; call initialize_variant before using the model.

    Default constructors run inside fork_rng, without torch.manual_seed, so
    construction leaves the caller's CPU RNG unchanged. Initialization below
    overwrites every parameter/buffer using local generators or BN constants.
    """
    _check_variant(variant)
    if type(in_channels) is not int or in_channels <= 0:
        raise ValueError('in_channels must be a positive integer')
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError('num_classes must be an integer >= 2')
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        raise ValueError('dropout must be finite and between 0 and 1')
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        if variant == 'cnn1d_matched':
            chosen = cnn1d_candidates()[0]
            if chosen['absolute_error'] > .02 * TARGET_PARAMETERS:
                raise ValueError('No CNN width candidate meets the 2% parameter tolerance')
            c1, c2, _ = chosen['widths']
            model = OnlineCNN1D(in_channels=in_channels, hidden_size=c1,
                                num_classes=num_classes, dropout=dropout)
            model.features[4] = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
            model.features[5] = nn.BatchNorm1d(c2)
            model.features[8] = nn.Conv1d(c2, 128, kernel_size=3, padding=1)
            model.features[9] = nn.BatchNorm1d(128)
            model.head[1] = nn.Linear(128, num_classes)
        else:
            model = OnlineMSTCA(in_channels=in_channels, hidden_size=64,
                                num_classes=num_classes, dropout=dropout)
            if variant == 'mstca_no_se':
                model.encoder.se = nn.Identity()
            elif variant == 'mstca_single_k7':
                model.encoder.branch_s[0] = nn.Conv1d(in_channels, 24, kernel_size=7, padding=3)
                model.encoder.branch_l[0] = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
    return model


def _semantic_modules(model, variant):
    names = _CNN_NAMES if variant == 'cnn1d_matched' else _MSTCA_NAMES
    if variant == 'mstca_no_se':
        names = tuple(n for n in names if not n.startswith('encoder.se.'))
    actual = {name: module for name, module in model.named_modules()
              if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d))}
    if set(actual) != set(names):
        raise ValueError('Model semantic modules do not match the declared variant')
    return {name: actual[name] for name in names}


def _layout(module):
    return {key: (tuple(tensor.shape), tensor.dtype) for key, tensor in module.state_dict().items()}


def _macs(modules):
    total = 0
    for name, module in modules.items():
        if isinstance(module, nn.Conv1d):
            total += (_CONV_LENGTHS[name] * module.out_channels *
                      (module.in_channels // module.groups) * module.kernel_size[0])
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features
    return total


def initialize_variant(model, variant, seed, policy=POLICY):
    """Initialize every state tensor; sharing is by semantic *module* + layout.

    shape in the seed payload is str(tuple(weight.shape)), e.g. '(24, 26, 7)'.
    The low 63 bits of the SHA256 integer form the sub-seed. Conv/Linear bias
    uses the same local generator after weight initialization, so changing a
    kernel also changes its bias even when that bias's own shape is unchanged.
    """
    _check_variant(variant)
    if policy != POLICY:
        raise ValueError(f'Unsupported initialization policy: {policy!r}')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if any(v.device.type != 'cpu' or v.dtype not in (torch.float32, torch.int64)
           for v in model.state_dict().values()):
        raise ValueError('Factory initialization requires CPU float32 parameters and int64 BN counters')
    modules = _semantic_modules(model, variant)
    first = modules['features.0' if variant == 'cnn1d_matched' else 'encoder.branch_s.0']
    # Validate the supplied model before changing it, including variant-specific shapes.
    expected = build_variant(variant, first.in_channels, model.head[1].out_features)
    if _layout(model) != _layout(expected):
        raise ValueError('Model state layout does not match the declared variant')
    baseline = expected if variant == 'mstca' else build_variant('mstca', first.in_channels, model.head[1].out_features)
    reference = _semantic_modules(baseline, 'mstca')
    shared_modules = {name for name, module in modules.items()
                      if name in reference and type(module) is type(reference[name])
                      and _layout(module) == _layout(reference[name])}
    for name, module in modules.items():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_parameters()
            continue
        shape = tuple(module.weight.shape)
        payload = f'{policy}|{seed}|{name}|{shape}'
        sub_seed = int(hashlib.sha256(payload.encode('utf-8')).hexdigest(), 16) & ((1 << 63)-1)
        generator = torch.Generator(device='cpu').manual_seed(sub_seed)
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
        if module.bias is not None:
            fan_in = module.weight.shape[1] * math.prod(module.weight.shape[2:])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    state = model.state_dict()
    shared_keys = sorted(k for k in state if k.rsplit('.', 1)[0] in shared_modules)
    specific_keys = sorted(set(state) - set(shared_keys))
    return {'variant': variant, 'seed': seed, 'policy': policy,
            'param_count': sum(v.numel() for v in model.parameters()), 'macs': _macs(modules),
            'shared_keys': shared_keys, 'variant_specific_keys': specific_keys,
            'full_state_digest': _state_digest(state),
            'shared_subset_digest': _state_digest({k: state[k] for k in shared_keys})}


def load_pure_m0_anchor(path: str | Path):
    """Historical seed42 compatibility only; never a module_seed_v1 initializer."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k, v in state.items()):
        raise ValueError('Pure M0 anchor must be a tensor state dictionary')
    if _state_digest(state) != PURE_M0_DIGEST:
        raise ValueError('Pure M0 anchor digest mismatch')
    model = build_variant('mstca')
    if len(state) != 41 or _layout(model) != {k: (tuple(v.shape), v.dtype) for k,v in state.items()}:
        raise ValueError('Pure M0 anchor layout mismatch')
    model.load_state_dict(state, strict=True)
    return model
