from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UAV_FAULT_ROOT = REPO_ROOT / "packages" / "uav_fault"
if str(UAV_FAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(UAV_FAULT_ROOT))

from uav_fault.data.make_manifest_v2 import build_manifest_v2


DEFAULT_SPLITS = ("train", "val", "test", "external_test")


def _split_files(processed_root: Path, split_names: tuple[str, ...]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for split in split_names:
        split_dir = processed_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"split directory not found: {split_dir}")
        files = sorted(path.name for path in split_dir.glob("Case_*.csv") if path.is_file())
        if not files:
            raise ValueError(f"split directory has no Case_*.csv files: {split_dir}")
        splits[split] = files
    return splits


def _summarize(manifest: pd.DataFrame, splits: dict[str, list[str]]) -> dict:
    filename_to_split = {
        filename: split
        for split, filenames in splits.items()
        for filename in filenames
    }
    df = manifest.copy()
    df["split"] = df["filename"].map(filename_to_split)
    missing_split = df["split"].isna()
    if bool(missing_split.any()):
        missing = df.loc[missing_split, "filename"].head(10).tolist()
        raise ValueError(f"manifest contains files not present in split dirs: {missing}")

    return {
        "total_cases": int(len(df)),
        "split_counts": {
            split: int((df["split"] == split).sum())
            for split in splits
        },
        "label_counts": {
            split: {
                str(k): int(v)
                for k, v in df.loc[df["split"] == split, "label_main_6class"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            for split in splits
        },
        "platform_counts": {
            split: {
                str(k): int(v)
                for k, v in df.loc[df["split"] == split, "platform"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            for split in splits
        },
        "quality_counts": {
            str(k): int(v)
            for k, v in df["quality_flag"].value_counts(dropna=False).sort_index().items()
        },
        "has_required_online_columns_counts": {
            str(k): int(v)
            for k, v in df["has_required_online_columns"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
    }


def materialize_processed_splits(
    *,
    processed_root: Path,
    output_dir: Path,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    manifest_output: Path | None = None,
    manifest_summary_output: Path | None = None,
    splits_output: Path | None = None,
    summary_output: Path | None = None,
    reuse_manifest: bool = False,
) -> dict:
    processed_root = processed_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = manifest_output or (output_dir / "manifest_from_processed_splits.csv")
    manifest_summary_output = manifest_summary_output or (
        output_dir / "manifest_from_processed_splits.summary.json"
    )
    splits_output = splits_output or (output_dir / "splits_from_processed_splits.json")
    summary_output = summary_output or (output_dir / "processed_splits.summary.json")

    splits = _split_files(processed_root, split_names)
    if not reuse_manifest or not manifest_output.exists():
        build_manifest_v2(
            csv_dir=[processed_root / split for split in split_names],
            output_path=manifest_output,
            summary_output_path=manifest_summary_output,
            schema_version="rflymad_manifest_v2",
            build_version="processed_splits",
        )

    manifest = pd.read_csv(manifest_output)
    if manifest["filename"].duplicated().any():
        duplicates = manifest.loc[manifest["filename"].duplicated(), "filename"].head(10).tolist()
        raise ValueError(f"duplicate filenames in manifest: {duplicates}")

    split_filenames = {filename for filenames in splits.values() for filename in filenames}
    manifest_filenames = set(manifest["filename"].astype(str))
    missing_manifest = sorted(split_filenames - manifest_filenames)
    if missing_manifest:
        raise ValueError(f"split files missing from manifest: {missing_manifest[:10]}")

    splits_output.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize(manifest, splits)
    summary.update(
        {
            "processed_root": str(processed_root),
            "manifest": str(manifest_output),
            "splits": str(splits_output),
        }
    )
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize manifest/splits from an existing processed_RflyMAD split tree."
    )
    parser.add_argument("--processed-root", default="data/processed_RflyMAD")
    parser.add_argument("--output-dir", default="runtime/training_full")
    parser.add_argument("--split-names", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--manifest-output", default="")
    parser.add_argument("--manifest-summary-output", default="")
    parser.add_argument("--splits-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Reuse --manifest-output if it already exists; useful after a slow full CSV scan.",
    )
    args = parser.parse_args()

    summary = materialize_processed_splits(
        processed_root=Path(args.processed_root),
        output_dir=Path(args.output_dir),
        split_names=tuple(args.split_names),
        manifest_output=Path(args.manifest_output) if args.manifest_output else None,
        manifest_summary_output=Path(args.manifest_summary_output)
        if args.manifest_summary_output
        else None,
        splits_output=Path(args.splits_output) if args.splits_output else None,
        summary_output=Path(args.summary_output) if args.summary_output else None,
        reuse_manifest=args.reuse_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
