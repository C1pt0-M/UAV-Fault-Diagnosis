"""
1D-CNN + FocalLoss 训练脚本。
用法: python -m uav_fault.models.train_cnn_focal
"""
import argparse
import json
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

from uav_fault.models.cnn.resnet1d import ResNet1d
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.cnn.dataset import SequenceDataset

LABEL_NAMES = ['normal', 'battery_failure', 'motor_failure', 'prop_imbalance', 'thrust_loss']


def evaluate(model, loader, device):
    """评估模型，返回指标字典。"""
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

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def print_metrics(metrics, split_name):
    """打印评估结果。"""
    print(f"\n{'='*50}")
    print(f"  {split_name} 评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    report = metrics['classification_report']
    print(f"\n{'':>17s} precision    recall  f1-score   support")
    for name in LABEL_NAMES:
        r = report[name]
        print(f"{name:>17s}     {r['precision']:.2f}      {r['recall']:.2f}      {r['f1-score']:.2f}       {int(r['support'])}")

    cm = np.array(metrics['confusion_matrix'])
    import pandas as pd
    print(f"\n混淆矩阵:")
    print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES))


def plot_results(output_dir, metrics_test, train_losses, val_f1s):
    """生成训练曲线和混淆矩阵可视化。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过可视化")
        return

    cm = np.array(metrics_test['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')

    axes[1].plot(val_f1s)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Validation Macro F1')

    axes[2].imshow(cm_norm, cmap='Blues')
    axes[2].set_title('Test Confusion Matrix')
    axes[2].set_xticks(range(len(LABEL_NAMES)))
    axes[2].set_yticks(range(len(LABEL_NAMES)))
    axes[2].set_xticklabels(LABEL_NAMES, rotation=45, ha='right', fontsize=7)
    axes[2].set_yticklabels(LABEL_NAMES, fontsize=7)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[2].text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                         ha='center', va='center', color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()
    print(f"可视化已保存: {output_dir / 'results.png'}")


def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Warmup + Cosine Annealing 学习率。"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_alpha(train_labels, use_alpha: bool, thrust_alpha_boost: float):
    """构建 focal alpha；频次权重可选，同时单独提升 thrust_loss。"""
    alpha = np.ones(len(LABEL_NAMES), dtype=np.float32)

    if use_alpha:
        counts = np.zeros(len(LABEL_NAMES), dtype=np.float32)
        for y in train_labels:
            counts[y] += 1
        inv = 1.0 / np.sqrt(counts + 1e-6)
        inv = inv / inv.mean()
        alpha *= inv

    thrust_idx = LABEL_NAMES.index('thrust_loss')
    alpha[thrust_idx] *= thrust_alpha_boost

    alpha = alpha / alpha.mean()
    if np.allclose(alpha, np.ones_like(alpha), atol=1e-3):
        return None
    return alpha.tolist()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_stats_path = cache_dir / 'norm_stats.json'

    # 数据集（训练集开启增强）
    train_ds = SequenceDataset(str(cache_dir / 'train'), str(norm_stats_path), augment=True)
    val_ds = SequenceDataset(str(cache_dir / 'val'), str(norm_stats_path), augment=False)
    test_ds = SequenceDataset(str(cache_dir / 'test'), str(norm_stats_path), augment=False)

    num_workers = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=num_workers > 0,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0,
                            pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=num_workers > 0,
                             pin_memory=device.type == 'cuda')

    print(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")

    # 读取通道数
    with open(norm_stats_path) as f:
        stats = json.load(f)
    in_channels = len(stats['columns'])
    print(f"通道数: {in_channels}, 类别数: {len(LABEL_NAMES)}")

    # 模型
    model = ResNet1d(
        in_channels=in_channels, num_classes=len(LABEL_NAMES), base_channels=args.base_channels
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # Loss
    alpha = build_alpha(train_ds.labels, use_alpha=args.use_alpha,
                        thrust_alpha_boost=args.thrust_alpha_boost)
    if alpha is None:
        print("  alpha: None")
    else:
        print(f"  alpha: {[f'{a:.3f}' for a in alpha]}")
    criterion = FocalLoss(alpha=alpha, gamma=args.gamma).to(device)
    print(f"FocalLoss: gamma={args.gamma}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    best_val_f1 = 0.0
    best_val_score = -1.0
    best_state = None
    patience_counter = 0
    train_losses = []
    val_f1s = []

    print(f"\n开始训练 (epochs={args.epochs}, warmup={args.warmup}, patience={args.patience}, min_epochs={args.min_epochs})...")

    for epoch in range(args.epochs):
        # 手动设置学习率 (warmup + cosine)
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

        # 验证
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['macro_f1']
        thrust_recall = val_metrics['classification_report']['thrust_loss']['recall']
        val_score = args.monitor_macro_weight * val_f1 + (1.0 - args.monitor_macro_weight) * thrust_recall
        val_f1s.append(val_f1)

        ep = epoch + 1
        print(
            f"  Epoch {ep:3d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"val_f1={val_f1:.4f} | thrust_recall={thrust_recall:.4f} | "
            f"val_score={val_score:.4f} | lr={lr:.2e}"
        )

        # Early stopping (min_epochs 之后才开始计数)
        if val_score > best_val_score + args.min_delta:
            best_val_score = val_score
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {ep} (best val_f1={best_val_f1:.4f}, best val_score={best_val_score:.4f})")
                break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评估
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
        'in_channels': in_channels,
        'num_classes': len(LABEL_NAMES),
        'epochs_trained': len(train_losses),
        'best_val_f1': best_val_f1,
        'best_val_score': best_val_score,
        'gamma': args.gamma,
        'use_alpha': args.use_alpha,
        'thrust_alpha_boost': args.thrust_alpha_boost,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'base_channels': args.base_channels,
        'monitor_macro_weight': args.monitor_macro_weight,
        'weight_decay': args.weight_decay,
        'n_params': n_params,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n模型和结果已保存到: {output_dir}")

    plot_results(output_dir, test_metrics, train_losses, val_f1s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1D-CNN + FocalLoss 训练')
    parser.add_argument('--cache-dir', default='data/processed/seq_cache')
    parser.add_argument('--output-dir', default='assets/models/cnn_focal')
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--use-alpha', action='store_true', default=False)
    parser.add_argument('--thrust-alpha-boost', type=float, default=1.35)
    parser.add_argument('--base-channels', type=int, default=48)
    parser.add_argument('--monitor-macro-weight', type=float, default=0.7)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-epochs', type=int, default=60)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
