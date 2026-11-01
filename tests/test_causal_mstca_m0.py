from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_causal_mstca_m0 as m0
from uav_fault.models.cnn.lstm import OnlineKGMSTCA
from train_source_normal_sampling import state_digest

PROTOCOL = ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json'


def unlocked():
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def write_locked(path, p):
    path.write_text(json.dumps(p, ensure_ascii=False), encoding='utf-8')
    path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding='utf-8')
    return path


def make_anchor(tmp_path):
    p = unlocked()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = OnlineKGMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2,
            dropout=.2, columns=p['data']['columns'])
    state = reference.state_dict()
    path = tmp_path / 'untrained.pt'
    torch.save(state, path)
    expected = {'encoder.' + k[len('raw_encoder.'):]: v.clone() for k, v in state.items() if k.startswith('raw_encoder.')}
    expected.update({k: v.clone() for k, v in state.items() if k.startswith('head.')})
    p['initialization_anchor'].update(sha256=m0.source.sha(path), state_digest=state_digest(state),
        m0_mapped_state_digest=state_digest(expected))
    return p, path, expected


def test_frozen_m0_protocol_reuses_latest26_population_and_budget():
    p = m0.load_protocol(PROTOCOL)
    assert p['model']['implementation'] == 'OnlineMSTCA'
    assert p['training']['parameters'] == 105938
    assert p['data']['train_windows'] == 162069 and p['data']['val_windows'] == 34710
    assert p['training']['max_epochs'] == 100 and p['training']['warmup_epochs'] == 5
    assert p['validation']['early_stop'] == {'min_epochs': 30, 'patience': 20, 'significant_delta': .002,
        'rule': 'Reuse selection_step separate significant_reference; strict improvements always checkpoint, patience counts only from min_epochs'}
    assert all('test' not in name.lower() and 'real' not in name.lower() for name in p['data_identities'])
    for relative, expected in p['code_identities'].items():
        assert m0.source.sha(ROOT / relative) == expected


def test_protocol_hash_lock_rejects_mutation(tmp_path):
    path = write_locked(tmp_path / 'm0.json', unlocked())
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='protocol identity'): m0.load_protocol(path)


@pytest.mark.parametrize('field,value', [('max_epochs', 30), ('warmup_epochs', 3), ('parameters', 156491)])
def test_relocked_wrong_budget_or_model_size_still_rejected(tmp_path, field, value):
    p = unlocked(); p['training'][field] = value
    with pytest.raises(ValueError): m0.load_protocol(write_locked(tmp_path / 'm0.json', p))


def test_relocked_legacy31_or_data_identity_bypass_rejected(tmp_path):
    p = unlocked(); p['data']['columns'].append('baro_pressure')
    with pytest.raises(ValueError, match='exact26'): m0.load_protocol(write_locked(tmp_path / '31.json', p))
    p = unlocked(); p['data_identities']['cache/train.pt'] = '0' * 64
    with pytest.raises(ValueError, match='data_identities'): m0.load_protocol(write_locked(tmp_path / 'data.json', p))


def test_relocked_shared_legacy_code_pin_bypass_rejected(tmp_path):
    p = unlocked(); p['code_identities']['scripts/research/train_causal_source_v1.py'] = '0' * 64
    with pytest.raises(ValueError, match='shared code pin'): m0.load_protocol(write_locked(tmp_path / 'code.json', p))


def test_exact_common_encoder_head_mapping_without_relation_forward(tmp_path, monkeypatch):
    p, anchor, expected = make_anchor(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Relation feature function must not run for M0')
    monkeypatch.setattr(OnlineKGMSTCA, '_relation_features', forbidden)
    torch.manual_seed(123); before = torch.random.get_rng_state().clone()
    model, receipt = m0.build_model(p, 42, anchor)
    assert torch.equal(before, torch.random.get_rng_state())
    assert list(model._modules) == ['encoder', 'head']
    assert sum(v.numel() for v in model.parameters()) == 105938
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())
    assert receipt['common_encoder_and_head_exact_anchor_match'] and not receipt['trained_checkpoint_used']
    model.eval()
    with torch.inference_mode():
        result = model(torch.zeros(2, 26, 150))
    assert result.shape == (2, 2) and torch.isfinite(result).all()


def test_matching_environment_reproduces_seed_and_rejects_updated_anchor(tmp_path, monkeypatch):
    p, anchor, expected = make_anchor(tmp_path)
    monkeypatch.setattr(m0.source, 'environment', lambda: dict(p['formal_environment']))
    model, receipt = m0.build_model(p, 42, anchor)
    assert receipt['legacy_fresh_seed_reproduction_checked_in_matching_environment']
    assert state_digest(model.state_dict()) == state_digest(expected)
    changed = torch.load(anchor, map_location='cpu', weights_only=True)
    changed['head.1.bias'][0] += .01
    torch.save(changed, anchor)
    p['initialization_anchor'].update(sha256=m0.source.sha(anchor), state_digest=state_digest(changed))
    with pytest.raises(ValueError, match='not exact fresh seeded'):
        m0.build_model(p, 42, anchor)


def test_building_two_arms_starts_from_identical_mapped_state(tmp_path):
    p, anchor, expected = make_anchor(tmp_path)
    a, _ = m0.build_model(p, 42, anchor)
    torch.manual_seed(789)
    b, _ = m0.build_model(p, 42, anchor)
    assert state_digest(a.state_dict()) == state_digest(b.state_dict()) == state_digest(expected)


def test_changed_anchor_bytes_and_bad_mapped_digest_rejected(tmp_path):
    p, anchor, expected = make_anchor(tmp_path)
    bad = deepcopy(p); bad['initialization_anchor']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='anchor SHA'): m0.build_model(bad, 42, anchor)
    bad = deepcopy(p); bad['initialization_anchor']['m0_mapped_state_digest'] = '0' * 64
    with pytest.raises(ValueError, match='mapped initialization'): m0.build_model(bad, 42, anchor)


def test_preparation_does_not_authorize_formal_run_or_access_data(tmp_path, monkeypatch):
    monkeypatch.setattr(m0.source, 'prepare', lambda *args: pytest.fail('Data must not be read'))
    args = SimpleNamespace(confirm_formal_run=False, cpu_check=False, output=tmp_path / 'unused')
    with pytest.raises(RuntimeError, match='PREPARATION_ONLY'): m0.run(args, {})
    assert not args.output.exists()


def test_formal_no_cuda_rejected_before_data_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(m0.source, 'prepare', lambda *args: pytest.fail('Data must not be read'))
    p = unlocked(); args = SimpleNamespace(confirm_formal_run=True, cpu_check=False,
        output=tmp_path / 'unused', arm='uniform_replacement', seed=42)
    with pytest.raises(RuntimeError, match='BLOCKED_NO_CUDA'): m0.run(args, p)
    assert not args.output.exists()


def test_formal_wrong_environment_rejected_before_data(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(m0.source, 'environment', lambda: {'torch': 'unexpected', 'numpy': 'unexpected', 'torch_cuda': 'unexpected'})
    monkeypatch.setattr(m0.source, 'prepare', lambda *args: pytest.fail('Data must not be read'))
    args = SimpleNamespace(confirm_formal_run=True, cpu_check=False, output=tmp_path / 'unused',
        arm='balanced_source_normal', seed=42)
    with pytest.raises(RuntimeError, match='environment mismatch'): m0.run(args, unlocked())
    assert not args.output.exists()


def test_no_cpu_training_mode_even_with_confirmation(tmp_path):
    args = SimpleNamespace(confirm_formal_run=True, cpu_check=True, output=tmp_path / 'unused')
    with pytest.raises(RuntimeError, match='does not allow CPU training'): m0.run(args, {})
