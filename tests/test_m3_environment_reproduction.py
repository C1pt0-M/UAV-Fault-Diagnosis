"""A failed formal-environment gate cannot proceed to data or training."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
import reproduce_m3_environment as gate
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2
from uav_fault.models.cnn.lstm import OnlineKGMSTCA
from train_source_normal_sampling import state_digest


def test_adapter_exact_identity_failure_precedes_any_source_data_load(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original = OnlineKGMSTCA(in_channels=26, hidden_size=64, num_layers=1,
                                num_classes=2, dropout=.2, columns=RFLY_SEQ_COLUMNS_V2)
    state = original.state_dict()
    anchor = tmp_path/'untrained.pt'
    torch.save(state, anchor)
    common = {'encoder.'+k[len('raw_encoder.'):]:v for k,v in state.items() if k.startswith('raw_encoder.')}
    common.update({k:v for k,v in state.items() if k.startswith('head.')})
    config = dict(formal_environment=dict(torch=str(torch.__version__), numpy=np.__version__, torch_cuda=torch.version.cuda),
                  code_identities={}, data=dict(base_columns=RFLY_SEQ_COLUMNS_V2),
                  initialization=dict(source=dict(sha256=gate.sha(anchor), state_digest=state_digest(state)),
                                      shared_state_digest=state_digest(common),
                                      adapter_initial_state_digest_local='0'*64))
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(config), encoding='utf-8')
    protocol.with_suffix('.json.sha256').write_text(gate.sha(protocol), encoding='utf-8')
    loaded = []
    original_load = torch.load
    def tracked_load(path, *args, **kwargs):
        loaded.append(Path(path))
        assert Path(path) == anchor
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked_load)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **k: pytest.fail('No optimizer before passing gates'))
    args = SimpleNamespace(protocol=protocol, initial_anchor=anchor, data=tmp_path/'absent', features=tmp_path/'absent')
    with pytest.raises(ValueError, match='Adapter initialization differs from frozen local digest'):
        gate.reproduce(args)
    assert loaded == [anchor]
    assert not (tmp_path/'runs').exists()


def test_failure_receipt_is_written_without_formal_jobs(tmp_path, monkeypatch):
    def fail(args):
        raise ValueError('Adapter initialization differs from frozen local digest')
    monkeypatch.setattr(gate, 'reproduce', fail)
    output = tmp_path/'failed.json'
    argv = ['gate', '--protocol', 'unused', '--data', 'unused', '--features', 'unused',
            '--initial-anchor', 'unused', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='Adapter initialization differs'):
        gate.main()
    receipt = json.loads(output.read_text(encoding='utf-8'))
    assert receipt['status'] == 'FAILED_M3_ENVIRONMENT_REPRODUCTION_NO_TRAINING'
    assert receipt['optimizer_steps'] == receipt['formal_jobs_started'] == 0
    with pytest.raises(FileExistsError):
        gate.main()


def test_unregistered_formal_environment_rejected_before_checkpoint_load(tmp_path, monkeypatch):
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(formal_environment=dict(torch='unexpected', numpy='unexpected', torch_cuda='unexpected'))))
    protocol.with_suffix('.json.sha256').write_text(gate.sha(protocol))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('No checkpoint before environment passes'))
    with pytest.raises(ValueError, match='Formal environment mismatch'):
        gate.reproduce(SimpleNamespace(protocol=protocol))
