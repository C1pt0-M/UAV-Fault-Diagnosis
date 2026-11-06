"""CPU/synthetic regressions for the new factory; no optimizer or dataset reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
from train_source_normal_sampling import state_digest
from uav_fault.models.cnn.ablation_factory import (
    POLICY, PURE_M0_DIGEST, VARIANTS, _state_digest, build_variant, cnn1d_candidates,
    initialize_variant, load_pure_m0_anchor,
)
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2
from uav_fault.online.fault_detector import FaultDetector
from build_pure_m0_anchor import DEFAULT_OUTPUT, DEFAULT_PROTOCOL, OUTPUT_ROOT, convert_anchor, sha256

EXPECTED = {'mstca': (105938, 6764928, 41, 0),
            'mstca_no_se': (103810, 6762880, 37, 0),
            'mstca_single_k7': (105106, 6640128, 37, 4),
            'cnn1d_matched': (105929, 7097374, 2, 21)}
PURE_ANCHOR = DEFAULT_OUTPUT/'pure_m0_seed42.pt'

# Generated with PyTorch 2.8.0+cpu. Keep these fixed to detect cross-version
# initialization drift in the formal 2.5.1+cu124 environment; do not skip or
# automatically regenerate the goldens when the installed version changes.
GOLDEN_DIGESTS = {
    ('mstca', 42): 'c86a31c99c3858e6b0d96262fb92dac4c472cb8376c0356df8a3c8e35fec521d',
    ('mstca', 43): '227dc429536953fae4a0d8d63069617ed09abe39a4842e60cde7b375738686f5',
    ('mstca', 44): 'f53dd1b285f7586146402a7cdcfcee2c2dde3b8886346c66c64961727388a6bd',
    ('mstca_no_se', 42): '40cdc82c75161257d4ec2f1d3174f475d825b2bce7401c49a7bac27e38e4c297',
    ('mstca_no_se', 43): '869af3abf771b0889110dcdce320c9946c491f885935f038833839489be740e0',
    ('mstca_no_se', 44): '819fe00c9175e3aaf387c61272b29eb15cfecca1329cc3b8ac2a441ba63dcac6',
    ('mstca_single_k7', 42): '2f4cf7a1eb6d917158be670044481951ac2e533579dccc2d4ef84a109b406509',
    ('mstca_single_k7', 43): '3e343db34359531698597b7c6d9cf25d12af8fc63c76edcecd909e4ff3916729',
    ('mstca_single_k7', 44): '653bc50a62a4d6a9a61d5f8dafd89ebf71cee042562f4e5b22c005690a9d4d28',
    ('cnn1d_matched', 42): '61c67e76cdf337df0bd6eb05bc0f375e84a207d859878e0ded2381dc30fbf48a',
    ('cnn1d_matched', 43): '8860d857fc60d2ebc35c1dd5081762af725de88a5b2ea193ce68549002f97041',
    ('cnn1d_matched', 44): '2c96ed1bb3b595622e5a0b4ce5fc4a6e8a4f85ed16b547a5c9be12fdc6af6076',
}


@pytest.fixture(scope='module', autouse=True)
def cpu_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def synthetic(batch=32):
    return torch.randn(batch, 26, 150, generator=torch.Generator().manual_seed(20261106))


def initialized(variant, seed=42):
    model = build_variant(variant)
    receipt = initialize_variant(model, variant, seed)
    model.eval()
    return model, receipt


def test_factory_import_with_only_package_path():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT/"packages/uav_fault")!r}); '
        'from uav_fault.models.cnn.ablation_factory import build_variant, initialize_variant; '
        'initialize_variant(build_variant("mstca"), "mstca", 42); '
        'assert "train_source_normal_sampling" not in sys.modules; '
        'assert "smoke_normal_source_sampling" not in sys.modules'
    )
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('case', ['empty', 'bn_buffers', 'mixed_noncontiguous', 'full_model'])
def test_local_digest_matches_historical_helper(case):
    if case == 'empty':
        state = {}
    elif case == 'bn_buffers':
        bn = nn.BatchNorm1d(3)
        bn.running_mean.copy_(torch.tensor([1., -2., 3.]))
        bn.running_var.copy_(torch.tensor([.5, 2., 4.]))
        bn.num_batches_tracked.fill_(17)
        state = bn.state_dict()
    elif case == 'mixed_noncontiguous':
        state = {'z_weight': torch.arange(12, dtype=torch.float64).reshape(3,4).T.requires_grad_(),
                 'running_mean': torch.tensor([-.25, .75], dtype=torch.float32),
                 'num_batches_tracked': torch.tensor(123, dtype=torch.int64)}
    else:
        model, _ = initialized('mstca')
        state = model.state_dict()
    assert _state_digest(state) == state_digest(state)
    assert _state_digest(dict(reversed(list(state.items())))) == state_digest(state)


@pytest.mark.parametrize('variant,seed', list(GOLDEN_DIGESTS))
def test_full_state_digest_golden(variant, seed):
    _, receipt = initialized(variant, seed)
    expected = GOLDEN_DIGESTS[(variant, seed)]
    assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
    assert receipt['full_state_digest'] == expected


@pytest.mark.parametrize('variant', VARIANTS)
def test_shapes_counts_and_independent_mac_hooks(variant):
    model, receipt = initialized(variant)
    if variant == 'mstca':
        assert type(model) is OnlineMSTCA
    counted = []
    def count(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            counted.append(output.numel() * module.in_channels // module.groups * module.kernel_size[0])
        else:
            counted.append(output.numel() * module.in_features)
    handles = [m.register_forward_hook(count) for m in model.modules() if isinstance(m,(nn.Conv1d,nn.Linear))]
    try:
        with torch.inference_mode():
            one = model(synthetic(1))
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        many = model(synthetic(32))
    assert one.shape == (1,2) and many.shape == (32,2)
    assert torch.isfinite(one).all() and torch.isfinite(many).all()
    params, macs, shared, specific = EXPECTED[variant]
    assert sum(v.numel() for v in model.parameters()) == receipt['param_count'] == params
    assert sum(counted) == receipt['macs'] == macs
    assert len(receipt['shared_keys']) == shared
    assert len(receipt['variant_specific_keys']) == specific
    assert all(v.device.type == 'cpu' for v in model.state_dict().values())


@pytest.mark.parametrize('variant', VARIANTS)
def test_repeatability_and_distinct_seeds(variant):
    first, a = initialized(variant)
    second, b = initialized(variant)
    _, other = initialized(variant,43)
    assert a == b
    assert a['full_state_digest'] != other['full_state_digest']
    with torch.inference_mode():
        assert torch.equal(first(synthetic()), second(synthetic()))


@pytest.mark.parametrize('variant', VARIANTS)
def test_global_cpu_rng_unchanged_and_no_torch_manual_seed(variant, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Factory must not call torch.manual_seed')
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    before = torch.random.get_rng_state().clone()
    model = build_variant(variant)
    assert torch.equal(before, torch.random.get_rng_state())
    initialize_variant(model, variant, 42)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('seed', [42,43,44])
def test_public_semantics_and_receipt_key_partition(seed):
    baseline, base_receipt = initialized('mstca',seed)
    base = baseline.state_dict()
    for variant in VARIANTS:
        model, receipt = initialized(variant,seed)
        state = model.state_dict()
        if variant == 'mstca_no_se':
            assert set(state) == {k for k in base if not k.startswith('encoder.se.')}
            expected_specific = set()
        elif variant == 'mstca_single_k7':
            assert set(state) == set(base)
            expected_specific = {f'encoder.branch_{branch}.0.{part}'
                                 for branch in ('s','l') for part in ('weight','bias')}
            assert model.encoder.branch_s[0].kernel_size == (7,)
            assert model.encoder.branch_m[0].kernel_size == (7,)
            assert model.encoder.branch_l[0].kernel_size == (7,)
        elif variant == 'cnn1d_matched':
            expected_specific = set(state)-{'head.1.weight','head.1.bias'}
            assert model.features[8].out_channels == model.head[1].in_features == 128
        else:
            expected_specific = set()
            assert receipt == base_receipt
        expected_shared = set(state)-expected_specific
        assert receipt['shared_keys'] == sorted(expected_shared)
        assert receipt['variant_specific_keys'] == sorted(expected_specific)
        assert not expected_shared & expected_specific
        assert all(torch.equal(state[k],base[k]) for k in expected_shared)
        if variant == 'mstca_single_k7':
            # Bias shapes are unchanged, but the Conv module's seed includes its new weight shape.
            assert all(not torch.equal(state[k],base[k]) for k in expected_specific)
            assert not torch.equal(state['encoder.branch_s.0.weight'],state['encoder.branch_m.0.weight'])
        assert receipt['full_state_digest'] == state_digest(state)
        assert receipt['shared_subset_digest'] == state_digest({k:state[k] for k in expected_shared})
        assert receipt['policy'] == POLICY and receipt['seed'] == seed and receipt['variant'] == variant


def test_named_local_seed_and_default_distribution_exactly():
    model, _ = initialized('mstca_single_k7',44)
    name = 'encoder.branch_s.0'
    weight = model.encoder.branch_s[0].weight
    payload = f'module_seed_v1|44|{name}|{tuple(weight.shape)}'
    sub_seed = int(hashlib.sha256(payload.encode()).hexdigest(),16) & ((1<<63)-1)
    gen = torch.Generator().manual_seed(sub_seed)
    expected_weight = torch.empty_like(weight)
    nn.init.kaiming_uniform_(expected_weight,a=5**.5,generator=gen)
    expected_bias = torch.empty_like(model.encoder.branch_s[0].bias)
    bound = (26*7)**-.5
    nn.init.uniform_(expected_bias,-bound,bound,generator=gen)
    assert torch.equal(weight,expected_weight)
    assert torch.equal(model.encoder.branch_s[0].bias,expected_bias)
    for module in model.modules():
        if isinstance(module,nn.BatchNorm1d):
            assert torch.equal(module.weight,torch.ones_like(module.weight))
            assert torch.equal(module.bias,torch.zeros_like(module.bias))
            assert torch.equal(module.running_mean,torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var,torch.ones_like(module.running_var))
            assert module.num_batches_tracked.item() == 0


def test_cnn_static_enumeration_and_tie_break():
    candidates = cnn1d_candidates()
    assert len(candidates) == 128*129//2
    assert candidates[0] == {'widths':[74,121,128],'param_count':105929,'macs':7097374,'absolute_error':9}
    assert candidates[1]['widths'] == [80,115,128] and candidates[1]['absolute_error'] == 9
    assert candidates[0]['absolute_error']/105938 < .02
    assert all(1 <= r['widths'][0] <= r['widths'][1] <= r['widths'][2] == 128 for r in candidates)


def test_pure_anchor_compatibility():
    if not PURE_ANCHOR.exists():
        pytest.skip('Run build_pure_m0_anchor.py first; historical pure anchor is absent')
    before = torch.random.get_rng_state().clone()
    model = load_pure_m0_anchor(PURE_ANCHOR)
    assert torch.equal(before,torch.random.get_rng_state())
    assert state_digest(model.state_dict()) == PURE_M0_DIGEST
    _, new = initialized('mstca',42)
    assert new['full_state_digest'] != PURE_M0_DIGEST
    receipt = json.loads((DEFAULT_OUTPUT/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['pure_state_digest'] == PURE_M0_DIGEST and receipt['tensor_count'] == 41
    assert receipt['new_file_sha256'] == sha256(PURE_ANCHOR)
    for target, origin in receipt['mapping'].items():
        assert origin == ('raw_encoder.'+target[len('encoder.'):] if target.startswith('encoder.') else target)


def test_pure_anchor_tamper_rejected(tmp_path):
    model, _ = initialized('mstca')
    path = tmp_path/'not_historical.pt'
    with path.open('xb') as stream:
        torch.save(model.state_dict(),stream)
    with pytest.raises(ValueError,match='digest mismatch'):
        load_pure_m0_anchor(path)


def test_conversion_refuses_overwrite(tmp_path):
    if not tmp_path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        pytest.skip('Conversion output tests require --basetemp under the task artifact root')
    # tmp_path exists: conversion must reject it before touching source/output files.
    with pytest.raises(FileExistsError):
        convert_anchor(DEFAULT_PROTOCOL,tmp_path)


@pytest.mark.parametrize('field', ['sha256','state_digest','m0_mapped_state_digest'])
def test_conversion_rejects_bad_source_identity_or_mapping(tmp_path, field):
    if not tmp_path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        pytest.skip('Conversion output tests require --basetemp under the task artifact root')
    p = json.loads(DEFAULT_PROTOCOL.read_text(encoding='utf-8'))
    p['initialization_anchor'][field] = '0'*64
    protocol = tmp_path/'bad.json'
    with protocol.open('x',encoding='utf-8') as stream:
        json.dump(p,stream)
    with protocol.with_suffix('.json.sha256').open('x',encoding='utf-8') as stream:
        stream.write(sha256(protocol))
    output = tmp_path/'must_not_exist'
    with pytest.raises(ValueError):
        convert_anchor(protocol,output)
    assert not output.exists()


@pytest.mark.parametrize('seed', [-1,True,'42'])
def test_invalid_seed_rejected_without_mutation(seed):
    model = build_variant('mstca')
    before = state_digest(model.state_dict())
    with pytest.raises(ValueError,match='seed'):
        initialize_variant(model,'mstca',seed)
    assert state_digest(model.state_dict()) == before


def test_bad_variant_policy_and_layout_rejected():
    with pytest.raises(ValueError,match='Unknown variant'):
        build_variant('kg_mstca')
    model = build_variant('mstca_single_k7')
    before = state_digest(model.state_dict())
    with pytest.raises(ValueError,match='policy'):
        initialize_variant(model,'mstca_single_k7',42,policy='legacy')
    with pytest.raises(ValueError,match='layout'):
        initialize_variant(model,'mstca',42)
    assert state_digest(model.state_dict()) == before


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='P3: batch baro_alt_relative uses file origin, not each window origin')
def test_p3_changing_altitude_batch_inputs_equal_single_window_inputs():
    # Real feature-building and scheduling methods, with a tensor capture in place
    # of the model. No weights, config files, datasets, or deployed assets needed.
    detector = FaultDetector.__new__(FaultDetector)
    detector.columns = list(RFLY_SEQ_COLUMNS_V2)
    detector.mean = np.zeros(26,dtype=np.float32)
    detector.std = np.ones(26,dtype=np.float32)
    detector.window_size = 150
    detector.exclude_indices = []
    detector._sparse_cols = set()
    detector._last_sparse = {}
    records = [dict.fromkeys(detector.columns,0.) for _ in range(170)]
    for i,row in enumerate(records):
        row['baro_alt_relative'] = 1000+i*.02
    captured = []
    def capture(batch):
        captured.extend(x.copy() for x in batch)
        return [{} for _ in batch]
    detector._predict_batch_tensor = capture
    predictions = detector.predict_records_batch(records,every_n=10)
    batch_inputs = list(captured)
    assert [end for end,_ in predictions] == [150,160,170]
    for i,(end,_) in enumerate(predictions):
        captured.clear()
        detector.predict(records[end-150:end])
        np.testing.assert_array_equal(batch_inputs[i],captured[0])
