import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


REQUIRED_COLUMNS = {
    "sample_id",
    "filename",
    "platform",
    "label_main_6class",
    "label_main_6class_id",
    "is_in_main_6class",
    "label_binary",
    "label_binary_id",
    "has_required_online_columns",
    "quality_flag",
}


def _build_protocol(seed: int) -> dict:
    return {
        "protocol_name": "rflymad_main_v1",
        "manifest_version": "rflymad_manifest_v2",
        "seed": seed,
        "split_basis": {
            "train_val_test_domains": ["SIL", "HIL"],
            "external_test_domains": ["Real"],
        },
        "tasks": {
            "main_6class": {
                "label_column": "label_main_6class",
                "eligible_filter": "is_in_main_6class == 1 and quality_flag != 'bad'",
                "stratify_column": "label_main_6class",
                "group_column": "sample_id",
            },
            "online_binary": {
                "label_column": "label_binary",
                "eligible_filter": "has_required_online_columns == 1 and quality_flag != 'bad'",
                "stratify_column": "label_binary",
                "group_column": "sample_id",
            },
        },
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "usage_tags": {
            "replay_demo_from": ["test"],
            "archive_from": ["test", "external_test"],
        },
    }


def _safe_n_splits(labels: pd.Series, preferred: int, minimum: int = 2) -> int:
    counts = labels.value_counts()
    min_count = int(counts.min())
    return max(minimum, min(preferred, min_count))


def _simple_train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = max(1, len(df) // 2)
    train_df = df.iloc[:midpoint].reset_index(drop=True)
    val_df = df.iloc[midpoint:].reset_index(drop=True)
    return train_df, val_df


def build_splits_v2(manifest_path, output_dir, seed: int = 42):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    dev_df = df[
        df["platform"].isin(["SIL", "HIL"])
        & (df["is_in_main_6class"] == 1)
        & (df["quality_flag"] != "bad")
    ].reset_index(drop=True)
    external_df = df[df["platform"] == "Real"].reset_index(drop=True)

    if len(dev_df) < 3:
        raise RuntimeError("SIL/HIL 主任务样本不足，无法划分 train/val/test")

    labels = dev_df["label_main_6class"].astype(str)
    n_splits_test = _safe_n_splits(labels, preferred=5)
    skf_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(skf_test.split(dev_df, labels))

    train_val_df = dev_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = dev_df.iloc[test_idx].reset_index(drop=True)

    train_val_labels = train_val_df["label_main_6class"].astype(str)
    if int(train_val_labels.value_counts().min()) < 2:
        train_df, val_df = _simple_train_val_split(train_val_df)
    else:
        n_splits_val = _safe_n_splits(train_val_labels, preferred=5)
        skf_val = StratifiedKFold(
            n_splits=n_splits_val, shuffle=True, random_state=seed + 1
        )
        train_idx, val_idx = next(skf_val.split(train_val_df, train_val_labels))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "external_test": external_df,
    }

    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    splits_json = {
        split_name: split_df["filename"].tolist()
        for split_name, split_df in split_frames.items()
    }
    (output_dir / "splits_v2.json").write_text(
        json.dumps(splits_json, ensure_ascii=False, indent=2)
    )
    (output_dir / "protocol_main_v1.json").write_text(
        json.dumps(_build_protocol(seed), ensure_ascii=False, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description="构建 RflyMAD v2 数据划分")
    parser.add_argument("--manifest", default="data/RflyMAD/manifest/manifest_v2.csv")
    parser.add_argument("--output-dir", default="data/RflyMAD/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_splits_v2(args.manifest, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
