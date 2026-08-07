"""
XGBoost 训练脚本（表格特征）。
用法: python -m uav_fault.models.train_xgboost
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
import joblib


def load_data(data_dir: str):
    """加载 train/val/test 数据。"""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')

    feature_cols = [c for c in train_df.columns if c not in ('filename', 'label')]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_val = le.transform(val_df['label'])
    y_test = le.transform(test_df['label'])

    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, imputer, feature_cols


def evaluate(model, X, y_true, le, split_name: str) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description='XGBoost 训练')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='assets/models/xgboost')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from xgboost import XGBClassifier

    print("加载数据...")
    X_train, X_val, X_test, y_train, y_val, y_test, le, imputer, feature_cols = \
        load_data(args.data_dir)

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"类别: {list(le.classes_)}")

    # 合并 train+val 用于 early stopping
    print("\n训练 XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    print(f"最佳迭代: {xgb.best_iteration}")

    # 评估
    metrics_val = evaluate(xgb, X_val, y_val, le, 'Validation')
    metrics_test = evaluate(xgb, X_test, y_test, le, 'Test')

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb, output_dir / 'xgb_model.joblib')
    joblib.dump(le, output_dir / 'label_encoder.joblib')
    joblib.dump(imputer, output_dir / 'imputer.joblib')

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    with open(output_dir / 'metrics_val.json', 'w') as f:
        json.dump(metrics_val, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'metrics_test.json', 'w') as f:
        json.dump(metrics_test, f, indent=2, ensure_ascii=False)

    # 特征重要性 (top 30)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb.feature_importances_,
    }).sort_values('importance', ascending=False).head(30)
    importance.to_csv(output_dir / 'feature_importance_top30.csv', index=False)
    print(f"\nTop 10 特征:")
    print(importance.head(10).to_string(index=False))

    # 可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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

        top = importance.head(20)
        ax2.barh(range(len(top)), top['importance'].values, color='steelblue')
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top['feature'].values, fontsize=7)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 20 Feature Importance')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_dir / 'results.png', dpi=150)
        plt.close()
        print(f"可视化已保存: {output_dir / 'results.png'}")
    except ImportError:
        pass

    print(f"\n模型和结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
