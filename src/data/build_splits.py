"""
RflyMAD 四分类划分：SIL+HIL -> train/val/test，Real -> external_test。
输出: data/processed/splits.json + train.csv / val.csv / test.csv / external_test.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_splits(manifest_path: str, output_dir: str, seed: int = 42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {'filename', 'platform', 'label', 'group_id'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'manifest 缺少字段: {sorted(missing)}')

    dev_df = df[df['platform'].isin(['SIL', 'HIL'])].reset_index(drop=True)
    real_df = df[df['platform'] == 'Real'].reset_index(drop=True)
    if len(dev_df) == 0:
        raise RuntimeError('没有 SIL/HIL 样本，无法划分 train/val/test')

    labels = dev_df['label']
    groups = dev_df['group_id'].astype(str)

    n_splits_test = 7
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(dev_df, labels, groups))

    train_val_df = dev_df.iloc[train_val_idx].reset_index(drop=True)
    train_val_labels = train_val_df['label']
    train_val_groups = train_val_df['group_id'].astype(str)

    n_splits_val = 6
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_val.split(train_val_df, train_val_labels, train_val_groups))

    train_df = train_val_df.iloc[train_sub_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_sub_idx].reset_index(drop=True)
    test_df = dev_df.iloc[test_idx].reset_index(drop=True)
    external_df = real_df.copy()

    split_frames = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'external_test': external_df,
    }
    split_groups = {k: set(v['group_id'].astype(str)) for k, v in split_frames.items()}

    assert split_groups['train'].isdisjoint(split_groups['val']), 'group 泄漏: train ∩ val 非空'
    assert split_groups['train'].isdisjoint(split_groups['test']), 'group 泄漏: train ∩ test 非空'
    assert split_groups['val'].isdisjoint(split_groups['test']), 'group 泄漏: val ∩ test 非空'
    assert split_groups['external_test'].isdisjoint(split_groups['train']), 'group 泄漏: external ∩ train 非空'
    assert split_groups['external_test'].isdisjoint(split_groups['val']), 'group 泄漏: external ∩ val 非空'
    assert split_groups['external_test'].isdisjoint(split_groups['test']), 'group 泄漏: external ∩ test 非空'

    splits = {k: v['filename'].tolist() for k, v in split_frames.items()}
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)

    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f'{split_name}.csv', index=False)

    print('划分结果:')
    for split_name, split_df in split_frames.items():
        print(f"\n  {split_name}: {len(split_df)} 样本")
        if len(split_df) == 0:
            continue
        print(f"    平台: {split_df['platform'].value_counts().sort_index().to_dict()}")
        print(f"    标签: {split_df['label'].value_counts().sort_index().to_dict()}")

    print(f"\n文件已保存到: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RflyMAD 四分类分层分组划分')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build_splits(args.manifest, args.output_dir, args.seed)
