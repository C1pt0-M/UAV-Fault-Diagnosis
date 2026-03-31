"""
特征组消融实验。
在 LSTM 上测试去掉不同通道组对分类性能的影响。

用法: python src/models/run_ablation.py
"""
import argparse
import json
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

from cnn.dataset import SequenceDataset
from cnn.focal_loss import FocalLoss
from cnn.lstm import BiLSTMClassifier

LABEL_NAMES = ['normal', 'battery_failure', 'motor_failure', 'prop_imbalance', 'thrust_loss']

# 43 通道的完整列表（和 build_sequence_cache.py 一致）
ALL_COLUMNS = [
    'motor1', 'motor2', 'motor3', 'motor4',
    'roll', 'pitch', 'yaw',
    'rollspeed', 'pitchspeed', 'yawspeed',
    'vx', 'vy', 'vz',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'battery_voltage_v',
    # 控制器内部状态
    'hover_thrust', 'accel_innov',
    'rollspeed_integ', 'pitchspeed_integ', 'yawspeed_integ',
    'thrust_z',
    'torque_roll',
    # 振动指标
    'accel_vibration_metric', 'gyro_vibration_metric',
    # 角速率设定值与跟踪误差
    'rates_sp_roll', 'rates_sp_pitch', 'rates_sp_yaw',
    'rate_err_roll', 'rate_err_pitch', 'rate_err_yaw',
    # ESC 转速
    'esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4',
    # 派生通道
    'motor_cross_mean', 'motor_cross_std', 'acc_norm', 'motor_effort',
]

# 定义特征组（通道名 -> 索引）
FEATURE_GROUPS = {
    'vibration': ['accel_vibration_metric', 'gyro_vibration_metric'],
    'esc_rpm': ['esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4'],
    'rate_error': ['rate_err_roll', 'rate_err_pitch', 'rate_err_yaw'],
    'rate_setpoint': ['rates_sp_roll', 'rates_sp_pitch', 'rates_sp_yaw'],
    'controller': ['hover_thrust', 'accel_innov', 'rollspeed_integ',
                    'pitchspeed_integ', 'yawspeed_integ', 'thrust_z', 'torque_roll'],
    'derived': ['motor_cross_mean', 'motor_cross_std', 'acc_norm', 'motor_effort'],
}

# 消融实验列表: (实验名, 要去掉的特征组列表)
ABLATION_EXPERIMENTS = [
    ('full_43ch', []),                                          # 基线：全部 43 通道
    ('drop_vibration', ['vibration']),                          # 去掉振动指标
    ('drop_esc_rpm', ['esc_rpm']),                              # 去掉 ESC 转速
    ('drop_rate_error', ['rate_error']),                        # 去掉角速率跟踪误差
    ('drop_rate_setpoint', ['rate_setpoint']),                  # 去掉角速率设定值
    ('drop_controller', ['controller']),                        # 去掉控制器内部状态
    ('drop_derived', ['derived']),                              # 去掉派生通道
    ('drop_all_new', ['vibration', 'esc_rpm', 'rate_error',    # 去掉所有新增通道（回到旧版）
                       'rate_setpoint', 'controller', 'derived']),
    ('only_base20', ['vibration', 'esc_rpm', 'rate_error',     # 仅保留最初 20 个基础传感器
                      'rate_setpoint', 'controller', 'derived']),
]


def get_channel_indices(drop_groups):
    """返回保留的通道索引。"""
    drop_cols = set()
    for group_name in drop_groups:
        drop_cols.update(FEATURE_GROUPS[group_name])

    indices = [i for i, col in enumerate(ALL_COLUMNS) if col not in drop_cols]
    return indices


class ChannelSubsetDataset(Dataset):
    """包装 SequenceDataset，只保留指定通道。"""

    def __init__(self, base_dataset, channel_indices):
        self.base = base_dataset
        self.indices = channel_indices
        self.labels = base_dataset.labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x[self.indices], y  # 只选指定通道


def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


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

    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'normal_recall': report['normal']['recall'],
        'thrust_recall': report['thrust_loss']['recall'],
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }


def train_one_experiment(exp_name, channel_indices, cache_dir, args):
    """训练单个消融实验并返回结果。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = len(channel_indices)
    kept_cols = [ALL_COLUMNS[i] for i in channel_indices]

    print(f"\n{'='*60}")
    print(f"  实验: {exp_name} ({n_channels} 通道)")
    print(f"{'='*60}")

    norm_stats_path = cache_dir / 'norm_stats.json'

    train_ds = ChannelSubsetDataset(
        SequenceDataset(str(cache_dir / 'train'), str(norm_stats_path), augment=True),
        channel_indices)
    val_ds = ChannelSubsetDataset(
        SequenceDataset(str(cache_dir / 'val'), str(norm_stats_path), augment=False),
        channel_indices)
    test_ds = ChannelSubsetDataset(
        SequenceDataset(str(cache_dir / 'test'), str(norm_stats_path), augment=False),
        channel_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, persistent_workers=True)

    model = BiLSTMClassifier(
        in_channels=n_channels, hidden_size=128, num_layers=2,
        num_classes=len(LABEL_NAMES), dropout=0.3,
    ).to(device)

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

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
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['macro_f1']

        ep = epoch + 1
        if ep % 10 == 0 or ep <= 5:
            print(f"  Epoch {ep:3d}/{args.epochs} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 0.002:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {ep} (best val_f1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)

    print(f"\n  Test: Acc={test_metrics['accuracy']:.4f}  "
          f"Macro F1={test_metrics['macro_f1']:.4f}  "
          f"normal_recall={test_metrics['normal_recall']:.3f}  "
          f"thrust_recall={test_metrics['thrust_recall']:.3f}")

    return {
        'experiment': exp_name,
        'n_channels': n_channels,
        'accuracy': test_metrics['accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'weighted_f1': test_metrics['weighted_f1'],
        'normal_recall': test_metrics['normal_recall'],
        'thrust_recall': test_metrics['thrust_recall'],
        'best_val_f1': best_val_f1,
        'epochs_trained': ep,
    }


def plot_ablation(results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [r['experiment'] for r in results]
    macro_f1 = [r['macro_f1'] for r in results]
    normal_r = [r['normal_recall'] for r in results]
    thrust_r = [r['thrust_recall'] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, macro_f1, width, label='Macro F1', color='#2196F3')
    ax.bar(x, normal_r, width, label='Normal Recall', color='#4CAF50')
    ax.bar(x + width, thrust_r, width, label='Thrust Recall', color='#FF5722')

    ax.set_ylabel('Score')
    ax.set_title('Feature Group Ablation (LSTM)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=6, padding=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_results.png', dpi=150)
    plt.close()
    print(f"\n可视化已保存: {output_dir / 'ablation_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='特征组消融实验')
    parser.add_argument('--cache-dir', default='data/processed/seq_cache')
    parser.add_argument('--output-dir', default='models/ablation')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--min-epochs', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for exp_name, drop_groups in ABLATION_EXPERIMENTS:
        channel_indices = get_channel_indices(drop_groups)
        result = train_one_experiment(exp_name, channel_indices, cache_dir, args)
        all_results.append(result)

    # 打印汇总
    print(f"\n\n{'='*80}")
    print("  消融实验汇总 (测试集)")
    print(f"{'='*80}")
    print(f"{'实验':<20s} {'通道':>4s} {'Accuracy':>10s} {'Macro F1':>10s} "
          f"{'Normal R':>10s} {'Thrust R':>10s}")
    print('-' * 70)
    for r in all_results:
        print(f"{r['experiment']:<20s} {r['n_channels']:>4d} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['normal_recall']:>10.3f} {r['thrust_recall']:>10.3f}")

    # 保存结果
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    import csv
    with open(output_dir / 'ablation_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    plot_ablation(all_results, output_dir)
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
