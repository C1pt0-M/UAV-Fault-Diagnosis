import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "filename",
    "platform",
    "flight_mode",
    "label_binary",
    "has_required_online_columns",
    "quality_flag",
}


def _filter_online_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["has_required_online_columns"] == 1)
        & (df["quality_flag"] != "bad")
        & (df["label_binary"].isin(["normal", "fault"]))
    ].reset_index(drop=True)


def _pick_real_stratify_key(df: pd.DataFrame) -> pd.Series:
    key = (
        df["label_binary"].astype(str).fillna("unknown")
        + "__"
        + df["flight_mode"].astype(str).fillna("unknown")
    )
    counts = key.value_counts()
    if not counts.empty and int(counts.min()) >= 3:
        return key
    return df["label_binary"].astype(str).fillna("unknown")


def _split_group_indices(indices: list[int], holdout_ratio: float, val_share: float, rng: np.random.RandomState):
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_holdout = int(round(n * holdout_ratio))
    if n >= 2:
        n_holdout = min(max(n_holdout, 1), n - 1)
    else:
        n_holdout = 0

    holdout = shuffled[:n_holdout]
    remain = shuffled[n_holdout:]
    if not remain:
        return [], [], holdout

    n_val = int(round(len(remain) * val_share))
    if len(remain) >= 2:
        n_val = min(max(n_val, 0), len(remain) - 1)
    else:
        n_val = 0

    val = remain[:n_val]
    train = remain[n_val:]
    return train, val, holdout


def _split_real_df(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    holdout_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    stratify = _pick_real_stratify_key(df)
    val_share = val_ratio / (train_ratio + val_ratio)

    train_idx: list[int] = []
    val_idx: list[int] = []
    holdout_idx: list[int] = []
    for _, group in df.groupby(stratify, sort=True):
        group_indices = group.index.tolist()
        t_idx, v_idx, h_idx = _split_group_indices(
            group_indices,
            holdout_ratio=holdout_ratio,
            val_share=val_share,
            rng=rng,
        )
        train_idx.extend(t_idx)
        val_idx.extend(v_idx)
        holdout_idx.extend(h_idx)

    return (
        df.loc[sorted(train_idx)].reset_index(drop=True),
        df.loc[sorted(val_idx)].reset_index(drop=True),
        df.loc[sorted(holdout_idx)].reset_index(drop=True),
    )


def build_online_a_layer_splits(
    manifest_path: str,
    dev_splits_path: str,
    output_dir: str,
    *,
    seed: int = 42,
    real_train_ratio: float = 0.20,
    real_val_ratio: float = 0.10,
    real_holdout_ratio: float = 0.70,
):
    manifest_path = Path(manifest_path)
    dev_splits_path = Path(dev_splits_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    manifest = _filter_online_eligible(manifest)

    with dev_splits_path.open("r", encoding="utf-8") as fp:
        dev_splits = json.load(fp)

    dev_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        filenames = set(dev_splits.get(split_name, []))
        dev_frames[split_name] = manifest[
            manifest["platform"].isin(["SIL", "HIL"]) & manifest["filename"].isin(filenames)
        ].reset_index(drop=True)
        dev_frames[split_name].to_csv(output_dir / f"{split_name}.csv", index=False)

    real_df = manifest[manifest["platform"] == "Real"].reset_index(drop=True)
    real_adapt_train_df, real_adapt_val_df, real_holdout_test_df = _split_real_df(
        real_df,
        train_ratio=real_train_ratio,
        val_ratio=real_val_ratio,
        holdout_ratio=real_holdout_ratio,
        seed=seed,
    )
    real_adapt_train_df.to_csv(output_dir / "real_adapt_train.csv", index=False)
    real_adapt_val_df.to_csv(output_dir / "real_adapt_val.csv", index=False)
    real_holdout_test_df.to_csv(output_dir / "real_holdout_test.csv", index=False)

    baseline_splits = {
        "train": dev_frames["train"]["filename"].tolist(),
        "val": dev_frames["val"]["filename"].tolist(),
        "test": dev_frames["test"]["filename"].tolist(),
        "external_test": real_holdout_test_df["filename"].tolist(),
    }
    fewshot_splits = {
        "train": dev_frames["train"]["filename"].tolist() + real_adapt_train_df["filename"].tolist(),
        "val": dev_frames["val"]["filename"].tolist() + real_adapt_val_df["filename"].tolist(),
        "test": dev_frames["test"]["filename"].tolist(),
        "external_test": real_holdout_test_df["filename"].tolist(),
    }

    (output_dir / "splits_online_a_baseline_v1.json").write_text(
        json.dumps(baseline_splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "splits_online_a_fewshot_v1.json").write_text(
        json.dumps(fewshot_splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "protocol_name": "rflymad_online_a_v1",
        "seed": seed,
        "real_ratios": {
            "real_adapt_train": real_train_ratio,
            "real_adapt_val": real_val_ratio,
            "real_holdout_test": real_holdout_ratio,
        },
        "dev_counts": {name: int(len(frame)) for name, frame in dev_frames.items()},
        "real_counts": {
            "real_adapt_train": int(len(real_adapt_train_df)),
            "real_adapt_val": int(len(real_adapt_val_df)),
            "real_holdout_test": int(len(real_holdout_test_df)),
        },
        "real_label_counts": {
            "real_adapt_train": real_adapt_train_df["label_binary"].value_counts().to_dict(),
            "real_adapt_val": real_adapt_val_df["label_binary"].value_counts().to_dict(),
            "real_holdout_test": real_holdout_test_df["label_binary"].value_counts().to_dict(),
        },
    }
    (output_dir / "protocol_online_a_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build online A-layer baseline/fewshot splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.20)
    parser.add_argument("--real-val-ratio", type=float, default=0.10)
    parser.add_argument("--real-holdout-ratio", type=float, default=0.70)
    args = parser.parse_args()
    build_online_a_layer_splits(
        manifest_path=args.manifest,
        dev_splits_path=args.dev_splits,
        output_dir=args.output_dir,
        seed=args.seed,
        real_train_ratio=args.real_train_ratio,
        real_val_ratio=args.real_val_ratio,
        real_holdout_ratio=args.real_holdout_ratio,
    )


if __name__ == "__main__":
    main()
