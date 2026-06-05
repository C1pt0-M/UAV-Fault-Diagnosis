#!/usr/bin/env python3
"""
RflyMAD 跨数据集验证脚本
在 RflyMAD 数据上测试现有的 CNN1D 在线检测模型
"""

import sys
sys.path.append('/home/jack/UAV-Fault-Diagnosis/src')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from validation.load_rflymad import load_rflymad_dataset
from models.cnn.lstm import OnlineCNN1D

def load_model(model_dir):
    """加载训练好的模型"""
    model_dir = Path(model_dir)

    # 读取配置
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # 读取归一化统计
    with open(model_dir / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)

    # 创建模型
    model = OnlineCNN1D(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        hidden_size=config.get('hidden_size', 64),
        dropout=config.get('dropout', 0.2)
    )

    # 加载权重
    checkpoint = torch.load(model_dir / 'best.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"✓ 模型加载成功: {model_dir.name}")
    print(f"  - 输入通道: {config['in_channels']}")
    print(f"  - 训练集最佳 F1: {config.get('best_val_f1', 'N/A')}")

    return model, norm_stats

def normalize_data(X, norm_stats, channel_mapping):
    """
    归一化数据

    参数:
        X: (N, 12, 150) - RflyMAD 12通道数据
        norm_stats: 模型训练时的归一化统计 (33通道)
        channel_mapping: RflyMAD 12通道到模型33通道的映射

    返回:
        X_norm: (N, 33, 150) - 归一化后的33通道数据
    """
    N, _, T = X.shape
    X_norm = np.zeros((N, 33, T))

    means = np.array(norm_stats['mean'])
    stds = np.array(norm_stats['std'])

    # 映射 12 通道到 33 通道
    for rfly_idx, model_idx in channel_mapping.items():
        X_norm[:, model_idx, :] = (X[:, rfly_idx, :] - means[model_idx]) / (stds[model_idx] + 1e-8)

    # 其他通道填充为 0 (已经是归一化后的值)

    return X_norm

def create_channel_mapping():
    """
    创建 RflyMAD 12通道到模型33通道的映射

    RflyMAD 12通道顺序: [roll, pitch, yaw, gx, gy, gz, vx, vy, vz, ax, ay, az]
    模型 33通道顺序: [motor1-4, roll, pitch, yaw, rollspeed, pitchspeed, yawspeed,
                     vx, vy, vz, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ...]
    """
    mapping = {
        # RflyMAD idx -> Model idx
        0: 4,   # roll -> roll (idx 4)
        1: 5,   # pitch -> pitch
        2: 6,   # yaw -> yaw
        3: 16,  # gx -> gyro_x (idx 16)
        4: 17,  # gy -> gyro_y
        5: 18,  # gz -> gyro_z
        6: 10,  # vx -> vx (idx 10)
        7: 11,  # vy -> vy
        8: 12,  # vz -> vz
        9: 13,  # ax -> accel_x (idx 13)
        10: 14, # ay -> accel_y
        11: 15, # az -> accel_z
    }

    # 注意: rollspeed/pitchspeed/yawspeed 在模型中是 idx 7-9
    # 但 RflyMAD 的 gyro 是原始陀螺仪，不完全等同于角速度
    # 这里我们把 gyro 映射到 gyro_x/y/z (16-18)
    # 如果需要更精确，可以把 gyro 也映射到 rollspeed 等

    return mapping

def evaluate_model(model, X, y, device='cpu'):
    """评估模型"""
    model = model.to(device)
    model.eval()

    # 转换为 tensor
    X_tensor = torch.FloatTensor(X).to(device)

    # 批量推理
    batch_size = 256
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # 计算指标
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='binary')

    print(f"\n{'='*60}")
    print(f"跨数据集验证结果 (RflyMAD)")
    print(f"{'='*60}")
    print(f"准确率: {acc:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fault'], digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print(f"\n混淆矩阵:")
    print(f"              预测")
    print(f"         Normal  Fault")
    print(f"Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Fault    {cm[1,0]:6d}  {cm[1,1]:6d}")

    return {
        'accuracy': acc,
        'f1': f1,
        'y_true': y,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'confusion_matrix': cm
    }

def plot_results(results, save_path):
    """绘制结果图表"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 混淆矩阵
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Fault'],
                yticklabels=['Normal', 'Fault'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # ROC 曲线 (简化版)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_prob'][:, 1])
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 结果图表已保存: {save_path}")

def main():
    print("="*60)
    print("RflyMAD 跨数据集验证")
    print("="*60)

    # 1. 加载 RflyMAD 数据
    print("\n[1/4] 加载 RflyMAD 数据...")
    data_dir = "/home/jack/RflyMAD_processed"
    X, y, file_labels = load_rflymad_dataset(data_dir, window_size=150, stride=50)

    # 2. 加载模型
    print("\n[2/4] 加载训练好的模型...")
    model_dir = "/home/jack/UAV-Fault-Diagnosis/models/online_detector_CNN"
    model, norm_stats = load_model(model_dir)

    # 3. 数据预处理
    print("\n[3/4] 数据预处理...")
    channel_mapping = create_channel_mapping()
    X_norm = normalize_data(X, norm_stats, channel_mapping)
    print(f"  - 归一化完成: {X_norm.shape}")

    # 4. 评估
    print("\n[4/4] 模型评估...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  - 使用设备: {device}")

    results = evaluate_model(model, X_norm, y, device)

    # 5. 保存结果
    output_dir = Path("/home/jack/UAV-Fault-Diagnosis/results/rflymad_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存数值结果
    results_file = output_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'f1': float(results['f1']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'num_samples': len(y),
            'num_normal': int(np.sum(y == 0)),
            'num_fault': int(np.sum(y == 1))
        }, f, indent=2)
    print(f"\n✓ 结果已保存: {results_file}")

    # 绘制图表
    plot_results(results, output_dir / "validation_plots.png")

    print(f"\n{'='*60}")
    print("验证完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
