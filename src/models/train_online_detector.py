"""
在线故障检测模型训练脚本（二分类: normal vs fault）。

支持多种模型架构: BiLSTM / 1D-CNN / Transformer。
数据从 stacked .pt 文件加载（每个 split 一个文件），支持内存级随机访问。

用法:
  cd UAV-Fault-Diagnosis
  python src/models/train_online_detector.py --model bilstm
  python src/models/train_online_detector.py --model cnn1d
  python src/models/train_online_detector.py --model transformer
"""
import argparse
import json
import copy
import math
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn.focal_loss import FocalLoss
from cnn.lstm import OnlineBiLSTM, OnlineCNN1D, OnlineTransformer, OnlineMSTCA



def validate_cache_contract(cache_dir: Path) -> dict:
    norm_stats_path = cache_dir / 'norm_stats.json'
    with open(norm_stats_path) as f:
        stats = json.load(f)

    columns = stats.get('columns')
    mean = stats.get('mean')
    std = stats.get('std')
    if not isinstance(columns, list) or not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError('norm_stats.json 缺少 columns/mean/std 列表')
    if not (len(columns) == len(mean) == len(std)):
        raise ValueError('norm_stats.json 中 columns/mean/std 长度不一致')

    expected_cfg = {
        'target_hz': 50.0,
        'window_size': 150,
        'stride': 10,
        'feature_schema': 'rfly_seq_columns_v1',
    }
    for key, expected in expected_cfg.items():
        actual = stats.get(key)
        if actual != expected:
            raise ValueError(f'norm_stats.json 合同不匹配: {key}={actual!r}, 期望 {expected!r}')
    if len(columns) != 31:
        raise ValueError(f'norm_stats.json 合同不匹配: columns 数量={len(columns)}, 期望 31')

    for split_name in ('train', 'val', 'test'):
        pt_path = cache_dir / f'{split_name}.pt'
        data = torch.load(pt_path, weights_only=True)
        if 'x' not in data or 'y' not in data:
            raise ValueError(f'{pt_path} 缺少 x/y')
        x = data['x']
        y = data['y']
        if x.ndim != 3:
            raise ValueError(f'{pt_path} 的 x 维度异常: 期望 3，实际 {x.ndim}')
        if y.ndim != 1:
            raise ValueError(f'{pt_path} 的 y 维度异常: 期望 1，实际 {y.ndim}')
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'{pt_path} 的 x/y 样本数不一致')
        if x.shape[1] != len(columns):
            raise ValueError(f'{pt_path} 的通道数 {x.shape[1]} 与 columns 数量 {len(columns)} 不一致')
        labels = set(torch.unique(y).cpu().tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f'{pt_path} 的 y 含非法标签: {sorted(labels)}')

    return stats
LABEL_NAMES = ['normal', 'fault']

MODEL_REGISTRY = {
    'bilstm': OnlineBiLSTM,
    'cnn1d': OnlineCNN1D,
    'transformer': OnlineTransformer,
    'mstca': OnlineMSTCA,
}

class OnlineWindowDataset(Dataset):
    """从 stacked .pt 文件加载在线检测窗口数据。"""

    def __init__(self, pt_path: str, norm_stats_path: str,
                 augment: bool = False, max_samples: int = 0):
        data = torch.load(pt_path, weights_only=True)
        x = data['x'].float()  # (N, C, T)
        y = data['y'].long()   # (N,)
        del data  # 释放引用

        # 分层下采样（保持类别比例）
        if max_samples > 0 and max_samples < len(y):
            rng = np.random.RandomState(42)
            idx0 = np.where(y.numpy() == 0)[0]
            idx1 = np.where(y.numpy() == 1)[0]
            ratio = len(idx1) / len(y)
            n1 = int(max_samples * ratio)
            n0 = max_samples - n1
            sel0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
            sel1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
            sel = np.sort(np.concatenate([sel0, sel1]))
            x = x[sel]
            y = y[sel]

        with open(norm_stats_path) as f:
            stats = json.load(f)
        mean = torch.tensor(stats['mean'], dtype=torch.float32).view(1, -1, 1)  # (1, C, 1)
        std = torch.tensor(stats['std'], dtype=torch.float32).view(1, -1, 1)
        # in-place z-score 归一化，避免创建新 tensor 导致 OOM
        x.sub_(mean).div_(std)
        self.x = x
        self.y = y
        self.augment = augment
        self.labels = self.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]  # (C, T)
        y = self.y[idx]
        if self.augment:
            x = self._augment(x.clone())
        return x, y

    @staticmethod
    def _augment(x):
        C, T = x.shape
        # 幅值抖动 (±3%)
        if np.random.random() < 0.35:
            x = x * (1.0 + np.random.uniform(-0.03, 0.03))
        # 加性高斯噪声
        if np.random.random() < 0.35:
            x = x + torch.randn_like(x) * 0.01
        return x


# ── 评估 ──────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        'f1_fault': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'classification_report': classification_report(
            all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
        ),
    }


def print_metrics(metrics, split_name):
    print(f"\n{'='*50}")
    print(f"  {split_name} 评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 (fault):   {metrics['f1_fault']:.4f}")

    report = metrics['classification_report']
    print(f"\n{'':>10s} precision    recall  f1-score   support")
    for name in LABEL_NAMES:
        r = report[name]
        print(f"{name:>10s}     {r['precision']:.2f}      {r['recall']:.2f}      "
              f"{r['f1-score']:.2f}       {int(r['support'])}")

    cm = np.array(metrics['confusion_matrix'])
    print(f"\n混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':>10s} {'normal':>8s} {'fault':>8s}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>10s} {cm[i, 0]:>8d} {cm[i, 1]:>8d}")


# ── 学习率调度 ─────────────────────────────────────────

def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── 可视化 ─────────────────────────────────────────────

def plot_results(output_dir, metrics_test, train_losses, val_f1s):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cm = np.array(metrics_test['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Online Detector Training Loss')

    axes[1].plot(val_f1s)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Online Detector Validation Macro F1')

    axes[2].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Test Confusion Matrix')
    axes[2].set_xticks(range(len(LABEL_NAMES)))
    axes[2].set_yticks(range(len(LABEL_NAMES)))
    axes[2].set_xticklabels(LABEL_NAMES)
    axes[2].set_yticklabels(LABEL_NAMES)
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[2].text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                         ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()
    print(f"可视化已保存: {output_dir / 'results.png'}")


# ── 训练主函数 ─────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_stats_path = cache_dir / 'norm_stats.json'
    stats = validate_cache_contract(cache_dir)

    print("加载数据集...")
    train_ds = OnlineWindowDataset(str(cache_dir / 'train.pt'), str(norm_stats_path),
                                   augment=True, max_samples=args.max_train_samples)
    val_ds = OnlineWindowDataset(str(cache_dir / 'val.pt'), str(norm_stats_path), augment=False)
    test_ds = OnlineWindowDataset(str(cache_dir / 'test.pt'), str(norm_stats_path), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=device.type == 'cuda')

    print(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")

    # 统计标签分布
    train_labels = np.array(train_ds.labels)
    n_normal = (train_labels == 0).sum()
    n_fault = (train_labels == 1).sum()
    print(f"训练集标签分布: normal={n_normal}, fault={n_fault} "
          f"(fault比例={n_fault / len(train_labels) * 100:.1f}%)")

    in_channels = len(stats['columns'])
    num_classes = 2
    window_size = int(stats.get('window_size') or train_ds.x.shape[-1])
    feature_schema = stats.get('feature_schema', 'unknown')
    print(f"通道数: {in_channels}, 类别数: {num_classes}")

    # 构建模型
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    print(f"模型: {args.model} ({model_cls.__name__})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    criterion = FocalLoss(gamma=args.gamma).to(device)
    print(f"FocalLoss: gamma={args.gamma}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses = []
    val_f1s = []

    print(f"\n开始训练 (epochs={args.epochs}, warmup={args.warmup}, patience={args.patience})...")

    for epoch in range(args.epochs):
        lr = get_warmup_cosine_lr(epoch, args.warmup, args.epochs, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['macro_f1']
        val_f1s.append(val_f1)

        ep = epoch + 1
        if ep % 5 == 0 or ep <= 5:
            print(f"  Epoch {ep:3d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"val_f1={val_f1:.4f} | val_recall={val_metrics['recall']:.4f} | lr={lr:.2e}")

        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {ep} (best val_f1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print_metrics(val_metrics, 'Validation (best)')
    print_metrics(test_metrics, 'Test')

    # 保存
    torch.save(best_state, output_dir / 'best.pt')
    with open(output_dir / 'metrics_val.json', 'w') as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'metrics_test.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    config = {
        'model': args.model,
        'task': 'online_binary',
        'feature_schema': feature_schema,
        'window_size': window_size,
        'feature_count': in_channels,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'label_names': LABEL_NAMES,
        'epochs_trained': len(train_losses),
        'best_val_f1': best_val_f1,
        'gamma': args.gamma,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_params': n_params,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 复制 norm_stats.json 到模型目录，便于推理引擎直接加载
    shutil.copy2(norm_stats_path, output_dir / 'norm_stats.json')

    print(f"\n模型和结果已保存到: {output_dir}")
    plot_results(output_dir, test_metrics, train_losses, val_f1s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在线故障检测模型训练（二分类）')
    parser.add_argument('--model', default='bilstm', choices=['bilstm', 'cnn1d', 'transformer', 'mstca'],
                        help='模型架构: bilstm, cnn1d, transformer')
    parser.add_argument('--cache-dir', default='data/processed/online_cache_rfly')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (默认: models/online_detector_<model>)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1,
                        help='LSTM/Transformer 层数 (CNN 忽略此参数)')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=0.002)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-samples', type=int, default=0,
                        help='训练集最大样本数 (分层下采样), 0=不限制')
    args = parser.parse_args()

    # 自动设置输出目录
    if args.output_dir is None:
        args.output_dir = f'models/online_detector_{args.model}'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
