import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


REQUIRED_COLUMNS = {
    "sample_id",
    "filename",
    "platform",
    "flight_mode",
    "label_main_6class",
    "is_in_main_6class",
    "quality_flag",
}


def _filter_real_main6(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["platform"] == "Real")
        & (df["is_in_main_6class"] == 1)
        & (df["quality_flag"] != "bad")
    ].reset_index(drop=True)


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    # 优先兼顾主类标签与飞行模式，避免 Real 小样本切分过度偏向单一 mode。
    return (
        df["label_main_6class"].astype(str).fillna("unknown")
        + "__"
        + df["flight_mode"].astype(str).fillna("unknown")
    )


def _fallback_key(df: pd.DataFrame) -> pd.Series:
    return df["label_main_6class"].astype(str).fillna("unknown")


def _safe_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = stratify.value_counts()
    if counts.min() < 2:
        stratify = _fallback_key(df)
        counts = stratify.value_counts()
    if counts.min() < 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(
            splitter.split(df, df["label_main_6class"].astype(str))
        )
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, stratify))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def build_real_adapt_splits(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.20,
    val_ratio: float = 0.10,
    holdout_ratio: float = 0.70,
    seed: int = 42,
):
    ratio_sum = train_ratio + val_ratio + holdout_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"切分比例之和必须为 1.0，当前为 {ratio_sum}")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺少字段: {sorted(missing)}")

    real_df = _filter_real_main6(df)
    if len(real_df) < 10:
        raise RuntimeError("Real main_6class 样本太少，无法进行稳定切分")

    stratify = _stratify_key(real_df)
    adapt_df, holdout_df = _safe_split(
        real_df,
        test_size=holdout_ratio,
        seed=seed,
        stratify=stratify,
    )

    adapt_val_ratio = val_ratio / (train_ratio + val_ratio)
    adapt_train_df, adapt_val_df = _safe_split(
        adapt_df,
        test_size=adapt_val_ratio,
        seed=seed + 1,
        stratify=_stratify_key(adapt_df),
    )

    split_frames = {
        "real_adapt_train": adapt_train_df,
        "real_adapt_val": adapt_val_df,
        "real_holdout_test": holdout_df,
    }

    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    summary = {
        "protocol_name": "rflymad_real_adapt_v1",
        "manifest_path": str(manifest_path),
        "seed": seed,
        "ratios": {
            "real_adapt_train": train_ratio,
            "real_adapt_val": val_ratio,
            "real_holdout_test": holdout_ratio,
        },
        "counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "label_counts": {
            name: frame["label_main_6class"].value_counts().sort_index().to_dict()
            for name, frame in split_frames.items()
        },
        "flight_mode_counts": {
            name: frame["flight_mode"].value_counts().sort_index().to_dict()
            for name, frame in split_frames.items()
        },
    }
    (output_dir / "real_adapt_protocol_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="构建 Real few-shot adaptation 切分")
    parser.add_argument("--manifest", default="data/RflyMAD/manifest/manifest_v2.csv")
    parser.add_argument("--output-dir", default="data/RflyMAD/splits_real_adapt")
    parser.add_argument("--train-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--holdout-ratio", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_real_adapt_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
