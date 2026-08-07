"""
LSTM / Transformer 时序分类训练脚本。
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from uav_fault.models.cnn.dataset import SequenceDataset
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.cnn.lstm import BiLSTMClassifier
from uav_fault.models.cnn.transformer import TransformerClassifier
from uav_fault.models.cnn.textcnn import TextCNN1d
from uav_fault.models.cnn.cnn_lstm import CNNLSTMClassifier

LABEL_NAMES = ['normal', 'motor', 'propeller', 'sensor']


def evaluate(model, loader, device, label_names=LABEL_NAMES):
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
    present_labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    target_names = [label_names[i] for i in present_labels]
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'labels': present_labels,
        'label_names': target_names,
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=present_labels).tolist(),
        'classification_report': classification_report(
            all_labels,
            all_preds,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def print_metrics(metrics, split_name):
    print(f"\n{'='*50}")
    print(f"  {split_name} 评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    report = metrics['classification_report']
    label_names = metrics.get('label_names', LABEL_NAMES)
    print(f"\n{'':>12s} precision    recall  f1-score   support")
    for name in label_names:
        r = report[name]
        print(f"{name:>12s}     {r['precision']:.2f}      {r['recall']:.2f}      {r['f1-score']:.2f}       {int(r['support'])}")
    cm = np.array(metrics['confusion_matrix'])
    import pandas as pd
    print('\n混淆矩阵:')
    print(pd.DataFrame(cm, index=label_names, columns=label_names))


def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def plot_results(output_dir, metrics_test, train_losses, val_f1s, model_name):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array(metrics_test['confusion_matrix'])
    label_names = metrics_test.get('label_names', LABEL_NAMES)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} Training Loss')
    axes[1].plot(val_f1s)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title(f'{model_name} Validation Macro F1')
    axes[2].imshow(cm_norm, cmap='Blues')
    axes[2].set_title(f'{model_name} Test Confusion Matrix')
    axes[2].set_xticks(range(len(label_names)))
    axes[2].set_yticks(range(len(label_names)))
    axes[2].set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(label_names, fontsize=8)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[2].text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})', ha='center', va='center', color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()
    print(f'可视化已保存: {output_dir / "results.png"}')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'模型: {args.model}')
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = cache_dir / 'norm_stats.json'

    train_ds = SequenceDataset(str(cache_dir / 'train'), str(norm_stats_path), augment=True)
    val_ds = SequenceDataset(str(cache_dir / 'val'), str(norm_stats_path), augment=False)
    test_ds = SequenceDataset(str(cache_dir / 'test'), str(norm_stats_path), augment=False)
    ext_test_dir = cache_dir / 'external_test'
    ext_test_ds = SequenceDataset(str(ext_test_dir), str(norm_stats_path), augment=False) if ext_test_dir.exists() else None

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
    ext_test_loader = DataLoader(ext_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, persistent_workers=num_workers > 0,
                                 pin_memory=device.type == 'cuda') if ext_test_ds is not None else None

    print(f'训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}')
    if ext_test_ds is not None:
        print(f'外部测试集: {len(ext_test_ds)}')
    with open(norm_stats_path) as f:
        stats = json.load(f)
    in_channels = len(stats['columns'])
    print(f'通道数: {in_channels}, 类别数: {len(LABEL_NAMES)}')

    if args.model == 'lstm':
        model = BiLSTMClassifier(in_channels=in_channels, hidden_size=128, num_layers=2,
                                 num_classes=len(LABEL_NAMES), dropout=0.3).to(device)
    elif args.model == 'transformer':
        model = TransformerClassifier(in_channels=in_channels, d_model=128, nhead=4,
                                      num_layers=4, num_classes=len(LABEL_NAMES), patch_size=8,
                                      dropout=0.2).to(device)
    elif args.model == 'textcnn':
        model = TextCNN1d(in_channels=in_channels, num_classes=len(LABEL_NAMES),
                          num_filters=128, kernel_sizes=(3, 5, 7, 11, 21), dropout=0.3).to(device)
    elif args.model == 'cnn_lstm':
        model = CNNLSTMClassifier(in_channels=in_channels, cnn_channels=(64, 128, 128),
                                  lstm_hidden=128, lstm_layers=2,
                                  num_classes=len(LABEL_NAMES), dropout=0.3).to(device)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数量: {n_params:,}')
    criterion = FocalLoss(gamma=args.gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    train_losses, val_f1s = [], []
    print(f'\n开始训练 (epochs={args.epochs}, warmup={args.warmup}, patience={args.patience})...')

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
        print(f'  Epoch {ep:3d}/{args.epochs} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} | lr={lr:.2e}')
        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {ep} (best val_f1={best_val_f1:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    ext_test_metrics = evaluate(model, ext_test_loader, device) if ext_test_loader is not None else None
    print_metrics(val_metrics, 'Validation (best)')
    print_metrics(test_metrics, 'Test')
    if ext_test_metrics is not None:
        print_metrics(ext_test_metrics, 'External Test')

    torch.save(best_state, output_dir / 'best.pt')
    with open(output_dir / 'metrics_val.json', 'w') as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'metrics_test.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    if ext_test_metrics is not None:
        with open(output_dir / 'metrics_external_test.json', 'w') as f:
            json.dump(ext_test_metrics, f, indent=2, ensure_ascii=False)
    config = {
        'model': args.model,
        'in_channels': in_channels,
        'num_classes': len(LABEL_NAMES),
        'label_names': LABEL_NAMES,
        'epochs_trained': len(train_losses),
        'best_val_f1': best_val_f1,
        'gamma': args.gamma,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_params': n_params,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'\n模型和结果已保存到: {output_dir}')
    plot_results(output_dir, test_metrics, train_losses, val_f1s, args.model.upper())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM/Transformer 时序分类训练')
    parser.add_argument('--model', choices=['lstm', 'transformer', 'textcnn', 'cnn_lstm'], required=True)
    parser.add_argument('--cache-dir', default='data/processed/seq_cache')
    parser.add_argument('--output-dir', default='assets/models/seq_model')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--min-epochs', type=int, default=40)
    parser.add_argument('--min-delta', type=float, default=0.002)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
