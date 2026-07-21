import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    columns = [f'feat_{i}' for i in range(31)]
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

    copied_norm_stats = json.loads((output_dir / 'norm_stats.json').read_text(encoding='utf-8'))
    assert copied_norm_stats['window_size'] == 150
    assert copied_norm_stats['stride'] == 10
    assert copied_norm_stats['feature_schema'] == 'rfly_seq_columns_v1'
    assert len(copied_norm_stats['columns']) == 31

    metrics_test = json.loads((output_dir / 'metrics_test.json').read_text(encoding='utf-8'))
    assert 'f1_fault' in metrics_test
