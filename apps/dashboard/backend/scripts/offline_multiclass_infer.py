#!/usr/bin/env python3
"""
离线多分类单文件推理脚本。
由 dashboard backend 通过子进程调用（conda 环境）执行。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='offline multiclass inference for single csv')
    parser.add_argument('--repo-root', required=True, help='项目根目录')
    parser.add_argument('--csv-path', required=True, help='输入csv路径')
    parser.add_argument('--model-dir', required=True, help='模型目录（包含best.pt/config.json）')
    parser.add_argument('--norm-stats', default='', help='可选norm_stats.json路径')
    parser.add_argument('--output-json', required=True, help='推理结果输出路径')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return parser.parse_args()


def add_import_paths(repo_root: Path):
    package_root = repo_root / 'packages' / 'uav_fault'
    path_text = str(package_root)
    if package_root.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)


def create_model(model_name: str, in_channels: int, num_classes: int):
    from uav_fault.models.cnn.cnn_lstm_v2_normalproto import CNNLSTMv2NormalProtoClassifier

    if model_name == 'cnn_lstm_v2_normalproto':
        return CNNLSTMv2NormalProtoClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            target_steps=512,
            raw_proj_dim=32,
            branch_channels=32,
            cnn_out_channels=128,
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.3,
        )
    raise ValueError(f'不支持的模型类型: {model_name}')


def load_norm_stats(norm_candidates: list[Path | None], columns: list[str], *, model_cfg: dict | None = None):
    from uav_fault.model_contract import offline_norm_contract_errors

    for path in norm_candidates:
        if path is None or not path.exists():
            continue
        try:
            with path.open('r', encoding='utf-8') as fp:
                stats = json.load(fp)
        except (OSError, ValueError) as exc:
            raise ValueError(f'无法读取离线归一化文件 {path}: {exc}') from exc
        errors = offline_norm_contract_errors(stats, columns, config=model_cfg)
        if errors:
            raise ValueError(f'无效离线归一化文件 {path}: {"; ".join(errors)}')
        mean = np.asarray(stats['mean'], dtype=np.float32)
        std = np.asarray(stats['std'], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std, str(path)
    return None, None, ''


def run_inference(args):
    import torch

    repo_root = Path(args.repo_root).resolve()
    csv_path = Path(args.csv_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_json = Path(args.output_json).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f'csv不存在: {csv_path}')
    if not model_dir.exists():
        raise FileNotFoundError(f'模型目录不存在: {model_dir}')

    add_import_paths(repo_root)
    from uav_fault.data.build_sequence_cache import SEQ_COLUMNS, SEQ_LEN, load_and_resample
    from uav_fault.model_contract import offline_input_contract_errors, offline_label_names

    config_path = model_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'缺少config.json: {config_path}')
    with config_path.open('r', encoding='utf-8') as fp:
        model_cfg = json.load(fp)

    errors = offline_input_contract_errors(model_cfg)
    if errors:
        raise ValueError(f'无效离线模型输入合同 {model_dir}: {"; ".join(errors)}')
    model_name = model_cfg.get('model', model_dir.name)
    in_channels = model_cfg['in_channels']
    num_classes = model_cfg.get('num_classes', 5)
    label_list = offline_label_names(model_cfg, num_classes)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')

    raw = load_and_resample(str(csv_path), seq_len=SEQ_LEN)  # (T, C)
    x = torch.from_numpy(raw.T.astype(np.float32))  # (C, T)

    norm_candidates = [
        model_dir / 'norm_stats.json',
        Path(args.norm_stats) if args.norm_stats else None,
        repo_root / 'data' / 'processed' / 'seq_cache' / 'norm_stats.json',
        repo_root / 'data' / 'processed' / 'online_cache_trainmix_labeled_50hz' / 'norm_stats.json',
    ]
    norm_mean, norm_std, norm_source = load_norm_stats(norm_candidates, SEQ_COLUMNS, model_cfg=model_cfg)
    if norm_mean is not None and norm_std is not None:
        mean = torch.from_numpy(norm_mean).unsqueeze(1)
        std = torch.from_numpy(norm_std).unsqueeze(1)
        x = (x - mean) / std
    else:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        x = (x - mean) / std
        norm_source = 'sample_zscore'

    model = create_model(model_name, in_channels=in_channels, num_classes=num_classes).to(device)
    try:
        state_dict = torch.load(model_dir / 'best.pt', map_location=device, weights_only=True)
    except TypeError:
        # 保留与在线加载器一致的旧 Torch 兼容路径。
        state_dict = torch.load(model_dir / 'best.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    probs_map = {label_list[i]: float(probs[i]) for i in range(num_classes)}

    rank_idx = np.argsort(-probs)
    topk = [
        {
            'index': int(i),
            'label': label_list[int(i)],
            'prob': float(probs[int(i)]),
        }
        for i in rank_idx[: min(3, len(rank_idx))]
    ]

    result = {
        'prediction': {
            'index': pred_idx,
            'label': label_list[pred_idx],
            'confidence': float(probs[pred_idx]),
        },
        'probs': probs_map,
        'labels': label_list,
        'topk': topk,
        'debug': {
            'model_name': model_name,
            'model_dir': str(model_dir),
            'csv_path': str(csv_path),
            'seq_len': int(SEQ_LEN),
            'in_channels': in_channels,
            'num_classes': num_classes,
            'norm_source': norm_source,
            'device': str(device),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    return result


def main():
    args = parse_args()
    try:
        run_inference(args)
        return 0
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
