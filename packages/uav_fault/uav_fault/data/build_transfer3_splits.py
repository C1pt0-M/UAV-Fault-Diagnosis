import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


KEEP_LABELS = ["normal", "motor", "sensor"]
LABEL_3CLASS = {"normal": "normal", "motor": "motor", "sensor": "sensor"}
REQUIRED_COLUMNS = {
    "sample_id",
    "filename",
    "platform",
    "flight_mode",
    "label_main_6class",
    "quality_flag",
}


def _prepare_main3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_main_3class"] = out["label_main_6class"].map(LABEL_3CLASS)
    out["is_in_main_3class"] = out["label_main_3class"].notna().astype(int)
    out["label_main_3class_id"] = out["label_main_3class"].map(
        {"normal": 0, "motor": 1, "sensor": 2}
    )
    return out


def _filter_3class(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["quality_flag"] != "bad")
        & (df["label_main_6class"].isin(KEEP_LABELS))
    ].reset_index(drop=True)


def _safe_n_splits(labels: pd.Series, preferred: int, minimum: int = 2) -> int:
    counts = labels.value_counts()
    min_count = int(counts.min())
    return max(minimum, min(preferred, min_count))


def _simple_train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = max(1, len(df) // 2)
    return (
        df.iloc[:midpoint].reset_index(drop=True),
        df.iloc[midpoint:].reset_index(drop=True),
    )


def _stratify_key_real(df: pd.DataFrame) -> pd.Series:
    return (
        df["label_main_3class"].astype(str).fillna("unknown")
        + "__"
        + df["flight_mode"].astype(str).fillna("unknown")
    )


def _fallback_real(df: pd.DataFrame) -> pd.Series:
    return df["label_main_3class"].astype(str).fillna("unknown")


def _safe_real_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = _stratify_key_real(df)
    counts = stratify.value_counts()
    if counts.min() < 2:
        stratify = _fallback_real(df)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, stratify))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def _protocol(seed: int, holdout_ratio: float, train_ratio: float, val_ratio: float) -> dict:
    return {
        "protocol_name": "rflymad_transfer3_v1",
        "manifest_version": "rflymad_manifest_v2",
        "seed": seed,
        "task": {
            "label_column": "label_main_3class",
            "labels": KEEP_LABELS,
            "train_val_test_domains": ["SIL", "HIL"],
            "real_domains": ["Real"],
        },
        "real_protocol": {
            "real_adapt_train": train_ratio,
            "real_adapt_val": val_ratio,
            "real_holdout_test": holdout_ratio,
        },
        "dev_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    }


def build_transfer3_splits(
    manifest_path: str,
    output_dir: str,
    holdout_ratio: float = 0.70,
    real_train_ratio: float = 0.20,
    real_val_ratio: float = 0.10,
    seed: int = 42,
):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺少字段: {sorted(missing)}")

    df = _prepare_main3(df)
    df = _filter_3class(df)

    dev_df = df[df["platform"].isin(["SIL", "HIL"])].reset_index(drop=True)
    real_df = df[df["platform"] == "Real"].reset_index(drop=True)

    labels = dev_df["label_main_3class"].astype(str)
    n_splits_test = _safe_n_splits(labels, preferred=5)
    skf_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(skf_test.split(dev_df, labels))
    train_val_df = dev_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = dev_df.iloc[test_idx].reset_index(drop=True)

    train_val_labels = train_val_df["label_main_3class"].astype(str)
    if int(train_val_labels.value_counts().min()) < 2:
        train_df, val_df = _simple_train_val_split(train_val_df)
    else:
        n_splits_val = _safe_n_splits(train_val_labels, preferred=5)
        skf_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
        train_idx, val_idx = next(skf_val.split(train_val_df, train_val_labels))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    adapt_df, holdout_df = _safe_real_split(real_df, test_size=holdout_ratio, seed=seed + 2)
    adapt_val_ratio = real_val_ratio / (real_train_ratio + real_val_ratio)
    real_adapt_train_df, real_adapt_val_df = _safe_real_split(
        adapt_df,
        test_size=adapt_val_ratio,
        seed=seed + 3,
    )

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "external_test": holdout_df,
        "real_adapt_train": real_adapt_train_df,
        "real_adapt_val": real_adapt_val_df,
        "real_holdout_test": holdout_df,
    }

    for name, frame in split_frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    summary = {
        "counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "label_counts": {
            name: frame["label_main_3class"].value_counts().sort_index().to_dict()
            for name, frame in split_frames.items()
        },
        "platform_counts": {
            name: frame["platform"].value_counts().sort_index().to_dict()
            for name, frame in split_frames.items()
        },
    }
    (output_dir / "protocol_transfer3_v1.json").write_text(
        json.dumps(_protocol(seed, holdout_ratio, real_train_ratio, real_val_ratio), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary_transfer3_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="构建 transfer 3class 数据切分")
    parser.add_argument("--manifest", default="data/RflyMAD/manifest/manifest_v2.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-ratio", type=float, default=0.70)
    parser.add_argument("--real-train-ratio", type=float, default=0.20)
    parser.add_argument("--real-val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_transfer3_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        holdout_ratio=args.holdout_ratio,
        real_train_ratio=args.real_train_ratio,
        real_val_ratio=args.real_val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
