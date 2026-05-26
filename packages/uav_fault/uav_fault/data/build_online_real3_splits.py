from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_LABELS = ["normal", "motor", "sensor"]
FAULT_LABELS = ["motor", "sensor"]
LABEL_3CLASS = {"normal": "normal", "motor": "motor", "sensor": "sensor"}
LABEL_3CLASS_ID = {"normal": 0, "motor": 1, "sensor": 2}
REQUIRED_COLUMNS = {
    "sample_id",
    "filename",
    "platform",
    "flight_mode",
    "label_binary",
    "label_main_6class",
    "quality_flag",
    "has_required_online_columns",
}


def _prepare_real3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_main_3class"] = out["label_main_6class"].map(LABEL_3CLASS)
    out["label_main_3class_id"] = out["label_main_3class"].map(LABEL_3CLASS_ID)
    out["is_in_main_3class"] = out["label_main_3class"].notna().astype(int)
    out["label_binary"] = out["label_main_3class"].map(
        {"normal": "normal", "motor": "fault", "sensor": "fault"}
    )
    out["label_binary_id"] = out["label_binary"].map({"normal": 0, "fault": 1})
    return out


def _filter_real3_online(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["quality_flag"] != "bad")
        & (df["has_required_online_columns"] == 1)
        & (df["label_main_6class"].isin(KEEP_LABELS))
    ].reset_index(drop=True)


def _simple_train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = max(1, len(df) // 2)
    return (
        df.iloc[:midpoint].reset_index(drop=True),
        df.iloc[midpoint:].reset_index(drop=True),
    )


def _pick_real_stratify_key(df: pd.DataFrame) -> pd.Series:
    key = (
        df["label_main_3class"].astype(str).fillna("unknown")
        + "__"
        + df["flight_mode"].astype(str).fillna("unknown")
    )
    counts = key.value_counts()
    if not counts.empty and int(counts.min()) >= 3:
        return key
    return df["label_main_3class"].astype(str).fillna("unknown")


def _take_group_holdout(
    indices: list[int],
    *,
    ratio: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_items = len(shuffled)
    if n_items <= 1:
        return shuffled, []

    n_holdout = int(round(n_items * ratio))
    n_holdout = min(max(n_holdout, 1), n_items - 1)
    return shuffled[n_holdout:], shuffled[:n_holdout]


def _split_group_indices(
    indices: list[int],
    *,
    holdout_ratio: float,
    val_share: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int], list[int]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_items = len(shuffled)
    n_holdout = int(round(n_items * holdout_ratio))
    if n_items >= 2:
        n_holdout = min(max(n_holdout, 1), n_items - 1)
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
    real_train_ratio: float,
    real_val_ratio: float,
    real_holdout_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if real_train_ratio <= 0 or real_val_ratio < 0 or real_holdout_ratio <= 0:
        raise ValueError("real split ratios must be positive except real_val_ratio may be zero")
    if abs((real_train_ratio + real_val_ratio + real_holdout_ratio) - 1.0) > 1e-6:
        raise ValueError("real_train_ratio + real_val_ratio + real_holdout_ratio must equal 1.0")

    rng = np.random.RandomState(seed)
    stratify = _pick_real_stratify_key(df)
    val_share = real_val_ratio / (real_train_ratio + real_val_ratio)

    train_idx: list[int] = []
    val_idx: list[int] = []
    holdout_idx: list[int] = []
    for _, group in df.groupby(stratify, sort=True):
        t_idx, v_idx, h_idx = _split_group_indices(
            group.index.tolist(),
            holdout_ratio=real_holdout_ratio,
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


def _split_dev_df(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_val_idx: list[int] = []
    test_idx: list[int] = []
    for _, group in df.groupby(df["label_main_3class"].astype(str), sort=True):
        remain, holdout = _take_group_holdout(
            group.index.tolist(),
            ratio=0.20,
            rng=rng,
        )
        train_val_idx.extend(remain)
        test_idx.extend(holdout)

    train_val_df = df.loc[sorted(train_val_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)

    train_val_labels = train_val_df["label_main_3class"].astype(str)
    if train_val_df.empty or int(train_val_labels.value_counts().min()) < 2:
        train_df, val_df = _simple_train_val_split(train_val_df)
    else:
        rng_val = np.random.RandomState(seed + 1)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for _, group in train_val_df.groupby(train_val_labels, sort=True):
            remain, holdout = _take_group_holdout(
                group.index.tolist(),
                ratio=0.20,
                rng=rng_val,
            )
            train_idx.extend(remain)
            val_idx.extend(holdout)
        train_df = train_val_df.loc[sorted(train_idx)].reset_index(drop=True)
        val_df = train_val_df.loc[sorted(val_idx)].reset_index(drop=True)

    return train_df, val_df, test_df


def _counts_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def _summarize_split_frames(split_frames: dict[str, pd.DataFrame]) -> dict:
    return {
        "counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "label_counts": {
            name: _counts_by(frame, "label_binary")
            for name, frame in split_frames.items()
        },
        "source_fault_type_counts": {
            name: _counts_by(frame, "label_main_3class")
            for name, frame in split_frames.items()
        },
        "platform_counts": {
            name: _counts_by(frame, "platform")
            for name, frame in split_frames.items()
        },
    }


def _protocol(
    *,
    seed: int,
    real_train_ratio: float,
    real_val_ratio: float,
    real_holdout_ratio: float,
    split_frames: dict[str, pd.DataFrame],
) -> dict:
    return {
        "protocol_name": "rflymad_online_real3_binary_v1",
        "manifest_version": "rflymad_manifest_v2",
        "seed": int(seed),
        "task": "online_binary",
        "label_space": "real3",
        "label_column": "label_binary",
        "label_names": ["normal", "fault"],
        "source_fault_column": "label_main_3class",
        "source_fault_classes": FAULT_LABELS,
        "included_main_classes": KEEP_LABELS,
        "train_val_test_domains": ["SIL", "HIL"],
        "real_domains": ["Real"],
        "real_protocol": {
            "real_adapt_train": float(real_train_ratio),
            "real_adapt_val": float(real_val_ratio),
            "real_holdout_test": float(real_holdout_ratio),
        },
        **_summarize_split_frames(split_frames),
    }


def build_online_real3_splits(
    manifest_path: str,
    output_dir: str,
    *,
    seed: int = 42,
    real_train_ratio: float = 0.20,
    real_val_ratio: float = 0.10,
    real_holdout_ratio: float = 0.70,
) -> None:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    df = _prepare_real3(df)
    df = _filter_real3_online(df)

    dev_df = df[df["platform"].isin(["SIL", "HIL"])].reset_index(drop=True)
    real_df = df[df["platform"] == "Real"].reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("no SIL/HIL rows left after Real3 online filtering")
    if real_df.empty:
        raise ValueError("no Real rows left after Real3 online filtering")

    train_df, val_df, test_df = _split_dev_df(dev_df, seed=seed)
    real_adapt_train_df, real_adapt_val_df, real_holdout_test_df = _split_real_df(
        real_df,
        real_train_ratio=real_train_ratio,
        real_val_ratio=real_val_ratio,
        real_holdout_ratio=real_holdout_ratio,
        seed=seed + 2,
    )

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "real_adapt_train": real_adapt_train_df,
        "real_adapt_val": real_adapt_val_df,
        "real_holdout_test": real_holdout_test_df,
    }
    for name, frame in split_frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    baseline_splits = {
        "train": train_df["filename"].tolist(),
        "val": val_df["filename"].tolist(),
        "test": test_df["filename"].tolist(),
        "external_test": real_holdout_test_df["filename"].tolist(),
    }
    fewshot_splits = {
        "train": train_df["filename"].tolist() + real_adapt_train_df["filename"].tolist(),
        "val": val_df["filename"].tolist() + real_adapt_val_df["filename"].tolist(),
        "test": test_df["filename"].tolist(),
        "external_test": real_holdout_test_df["filename"].tolist(),
    }

    (output_dir / "splits_online_real3_binary_baseline_v1.json").write_text(
        json.dumps(baseline_splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "splits_online_real3_binary_v1.json").write_text(
        json.dumps(fewshot_splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    protocol = _protocol(
        seed=seed,
        real_train_ratio=real_train_ratio,
        real_val_ratio=real_val_ratio,
        real_holdout_ratio=real_holdout_ratio,
        split_frames=split_frames,
    )
    (output_dir / "protocol_online_real3_binary_v1.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build online Real3 binary splits for normal vs motor/sensor fault detection."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.20)
    parser.add_argument("--real-val-ratio", type=float, default=0.10)
    parser.add_argument("--real-holdout-ratio", type=float, default=0.70)
    args = parser.parse_args()
    build_online_real3_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        real_train_ratio=args.real_train_ratio,
        real_val_ratio=args.real_val_ratio,
        real_holdout_ratio=args.real_holdout_ratio,
    )


if __name__ == "__main__":
    main()
