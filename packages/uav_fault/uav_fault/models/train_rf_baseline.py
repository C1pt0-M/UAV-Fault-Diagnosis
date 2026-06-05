"""
随机森林 baseline 训练脚本
用法: python -m uav_fault.models.train_rf_baseline
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
import joblib


def load_data(data_dir: str):
    """加载 train/val/test 数据，返回特征矩阵和标签。"""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')

    feature_cols = [c for c in train_df.columns if c not in ('filename', 'label')]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_val = le.transform(val_df['label'])
    y_test = le.transform(test_df['label'])

    # 中位数填充缺失值（仅在 train 上拟合）
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, imputer, feature_cols


def evaluate(model, X, y_true, le, split_name: str) -> dict:
    """评估模型并打印报告。"""
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{'='*50}")
    print(f"  {split_name} 评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=le.classes_)}")

    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=le.classes_, output_dict=True
        ),
    }


def save_results(output_dir: Path, model, le, imputer, feature_cols,
                 metrics_val, metrics_test, top_features):
    """保存模型和评估结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    joblib.dump(model, output_dir / 'rf_model.joblib')
    joblib.dump(le, output_dir / 'label_encoder.joblib')
    joblib.dump(imputer, output_dir / 'imputer.joblib')

    # 特征名
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f)

    # 指标
    with open(output_dir / 'metrics_val.json', 'w') as f:
        json.dump(metrics_val, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'metrics_test.json', 'w') as f:
        json.dump(metrics_test, f, indent=2, ensure_ascii=False)

    # 特征重要性
    top_features.to_csv(output_dir / 'feature_importance_top30.csv', index=False)

    print(f"\n模型和结果已保存到: {output_dir}")


def get_feature_importance(model, X_val, y_val, feature_cols, n_top=30):
    """计算 Gini 重要性和 Permutation 重要性。"""
    # Gini importance
    gini_imp = pd.DataFrame({
        'feature': feature_cols,
        'gini_importance': model.feature_importances_,
    }).sort_values('gini_importance', ascending=False)

    # Permutation importance (在 val 集上，避免对 test 集的间接依赖)
    print("\n正在计算 permutation importance（可能需要几分钟）...")
    perm_result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1,
        scoring='f1_macro'
    )
    perm_imp = pd.DataFrame({
        'feature': feature_cols,
        'perm_importance_mean': perm_result.importances_mean,
        'perm_importance_std': perm_result.importances_std,
    }).sort_values('perm_importance_mean', ascending=False)

    # 合并
    merged = gini_imp.merge(perm_imp, on='feature')
    merged = merged.sort_values('perm_importance_mean', ascending=False).head(n_top)

    print(f"\nTop {n_top} 特征重要性:")
    print(merged.to_string(index=False))

    return merged


def plot_results(output_dir: Path, metrics_test, top_features, le):
    """生成可视化图表。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过可视化")
        return

    # 混淆矩阵
    cm = np.array(metrics_test['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(cm_norm, cmap='Blues')
    ax1.set_title('Confusion Matrix (Normalized)')
    ax1.set_xticks(range(len(le.classes_)))
    ax1.set_yticks(range(len(le.classes_)))
    ax1.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(le.classes_, fontsize=8)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax1.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                     ha='center', va='center', color=color, fontsize=8)

    # 特征重要性
    top = top_features.head(20)
    ax2.barh(range(len(top)), top['perm_importance_mean'].values, color='steelblue')
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top['feature'].values, fontsize=7)
    ax2.set_xlabel('Permutation Importance (macro F1)')
    ax2.set_title('Top 20 Feature Importance')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    plt.close()
    print(f"可视化已保存: {output_dir / 'results.png'}")


def main():
    parser = argparse.ArgumentParser(description='随机森林 baseline 训练')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='assets/models/rf_baseline')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("加载数据...")
    X_train, X_val, X_test, y_train, y_val, y_test, le, imputer, feature_cols = \
        load_data(args.data_dir)

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"类别: {list(le.classes_)}")

    # 训练随机森林
    print("\n训练随机森林...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 评估
    metrics_val = evaluate(rf, X_val, y_val, le, 'Validation')
    metrics_test = evaluate(rf, X_test, y_test, le, 'Test')

    # 特征重要性 (在 val 集上计算)
    top_features = get_feature_importance(rf, X_val, y_val, feature_cols)

    # 保存
    output_dir = Path(args.output_dir)
    save_results(output_dir, rf, le, imputer, feature_cols,
                 metrics_val, metrics_test, top_features)

    # 可视化
    plot_results(output_dir, metrics_test, top_features, le)


if __name__ == '__main__':
    main()
