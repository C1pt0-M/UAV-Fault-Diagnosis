import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from uav_fault.models import train_online_detector

from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V1
from uav_fault.models.train_online_detector import train


class Args:
    model = 'cnn1d'
    cache_dir = ''
    output_dir = ''
    epochs = 2
    batch_size = 4
    lr = 3e-4
    weight_decay = 1e-4
    gamma = 2.0
    hidden_size = 16
    num_layers = 1
    dropout = 0.2
    num_workers = 0
    patience = 1
    min_epochs = 1
    min_delta = 0.002
    warmup = 5
    seed = 42
    max_train_samples = 0


def _make_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Canonical channel names: validate_cache_contract checks the columns against
    # the declared feature_schema, because a cache with invented names would feed
    # all-zero channels to the model at runtime.
    columns = list(RFLY_SEQ_COLUMNS_V1)
    norm_stats = {
        'mean': [0.0] * len(columns),
        'std': [1.0] * len(columns),
        'columns': columns,
        'window_size': 150,
        'target_hz': 50.0,
        'stride': 10,
        'feature_schema': 'rfly_seq_columns_v1',
        'protocol_name': 'rflymad_online_real3_binary_v1',
        'label_space': 'real3',
        'source_fault_classes': ['motor', 'sensor'],
    }
    (cache_dir / 'norm_stats.json').write_text(json.dumps(norm_stats), encoding='utf-8')

    def _save(name: str, labels: list[int]):
        x = torch.zeros((len(labels), len(columns), 150), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.int64)
        torch.save({'x': x, 'y': y}, cache_dir / f'{name}.pt')

    _save('train', [0, 0, 0, 1, 1, 1])
    _save('val', [0, 0, 0])
    _save('test', [1, 1, 0])


def test_train_online_detector_always_writes_valid_best_pt(tmp_path: Path):
    cache_dir = tmp_path / 'cache'
    output_dir = tmp_path / 'model'
    _make_cache(cache_dir)

    args = Args()
    args.cache_dir = str(cache_dir)
    args.output_dir = str(output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)

    best_path = output_dir / 'best.pt'
    assert best_path.exists()
    state_dict = torch.load(best_path, map_location='cpu', weights_only=True)
    assert isinstance(state_dict, dict)
    assert state_dict

    config = json.loads((output_dir / 'config.json').read_text(encoding='utf-8'))
    assert config['num_classes'] == 2
    assert config['label_names'] == ['normal', 'fault']
    assert config['protocol_name'] == 'rflymad_online_real3_binary_v1'
    assert config['label_space'] == 'real3'
    assert config['source_fault_classes'] == ['motor', 'sensor']
    for name in ('epochs', 'seed', 'weight_decay', 'warmup', 'patience',
                 'min_epochs', 'min_delta', 'max_train_samples', 'num_workers'):
        assert config[name] == getattr(args, name)
    assert 1 <= config['best_epoch'] <= config['epochs_trained']
    history = json.loads((output_dir / 'history.json').read_text(encoding='utf-8'))
    assert len(history) == config['epochs_trained']
    assert history[config['best_epoch'] - 1]['val_macro_f1'] == config['best_val_f1']
    provenance = json.loads((output_dir / 'provenance.json').read_text(encoding='utf-8'))
    assert set(provenance['cache_files']) == {'norm_stats.json', 'train.pt', 'val.pt', 'test.pt'}
    for name, identity in provenance['cache_files'].items():
        assert identity['sha256'] == hashlib.sha256((cache_dir / name).read_bytes()).hexdigest()
        assert identity['bytes'] == (cache_dir / name).stat().st_size
    assert provenance['source_sha256']['trainer'] == hashlib.sha256(
        Path(train_online_detector.__file__).read_bytes()
    ).hexdigest()
    assert provenance['python']
    assert provenance['torch']

    copied_norm_stats = json.loads((output_dir / 'norm_stats.json').read_text(encoding='utf-8'))
    assert copied_norm_stats['window_size'] == 150
    assert copied_norm_stats['stride'] == 10
    assert copied_norm_stats['feature_schema'] == 'rfly_seq_columns_v1'
    assert len(copied_norm_stats['columns']) == 31

    metrics_test = json.loads((output_dir / 'metrics_test.json').read_text(encoding='utf-8'))
    assert 'f1_fault' in metrics_test


@pytest.mark.parametrize('output', [None, ''])
def test_train_requires_explicit_output_directory(output):
    args = Args()
    args.output_dir = output
    with pytest.raises(ValueError, match='output'):
        train(args)


@pytest.mark.parametrize('occupied', [False, True])
def test_train_rejects_existing_directory_without_touching_it(tmp_path, occupied):
    output = tmp_path / 'existing'
    output.mkdir()
    if occupied:
        (output / 'best.pt').write_bytes(b'existing experiment')
    args = Args()
    args.cache_dir = str(tmp_path / 'missing-cache')
    args.output_dir = str(output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        train(args)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_cli_requires_output_directory():
    result = subprocess.run(
        [sys.executable, '-m', 'uav_fault.models.train_online_detector'],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert '--output-dir' in result.stderr


def test_interrupted_training_keeps_first_best_and_history(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    output = tmp_path / 'run'
    _make_cache(cache)
    args = Args()
    args.cache_dir = str(cache)
    args.output_dir = str(output)
    calls = 0

    def evaluate_then_interrupt(*unused):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return {'macro_f1': 0.0, 'recall': 0.0}

    monkeypatch.setattr(train_online_detector, 'evaluate', evaluate_then_interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(args)
    assert torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    history = json.loads((output / 'history.json').read_text(encoding='utf-8'))
    assert config['best_epoch'] == config['epochs_trained'] == 1
    assert config['best_val_f1'] == 0.0
    assert len(history) == 1
    assert history[0]['epoch'] == 1
    assert (output / 'norm_stats.json').is_file()
    assert (output / 'provenance.json').is_file()


def test_failed_checkpoint_write_preserves_previous_best(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    output = tmp_path / 'run'
    _make_cache(cache)
    args = Args()
    args.cache_dir = str(cache)
    args.output_dir = str(output)
    scores = iter([0.2, 0.6])
    monkeypatch.setattr(train_online_detector, 'evaluate',
                        lambda *unused: {'macro_f1': next(scores), 'recall': 0.0})
    save = torch.save
    previous = None

    def fail_second_save(payload, path):
        nonlocal previous
        if previous is None:
            save(payload, path)
            previous = Path(path).read_bytes()
        else:
            Path(path).write_bytes(b'partial checkpoint')
            raise OSError('simulated disk failure')

    monkeypatch.setattr(torch, 'save', fail_second_save)
    with pytest.raises(OSError, match='simulated disk failure'):
        train(args)
    assert (output / 'best.pt').read_bytes() == previous
    assert torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    assert config['best_epoch'] == config['epochs_trained'] == 1
