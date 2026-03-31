import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_int_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    valid = text.str.fullmatch(r"-?\d+")
    out = pd.Series([None] * len(series), index=series.index, dtype="object")
    out.loc[valid.fillna(False)] = text.loc[valid.fillna(False)].map(int)
    return out


def clean_case_mapping(input_path: Path, existing_only: bool = False) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_path, dtype=str)
    if df.empty:
        raise RuntimeError(f"case_mapping 为空: {input_path}")
    if "processed_case_id" not in df.columns:
        raise RuntimeError(f"case_mapping 缺少 processed_case_id: {input_path}")

    stats = {
        "input_rows": int(len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_processed_case_id_rows": int(
            df.duplicated(subset=["processed_case_id"]).sum()
        ),
    }

    df = df.drop_duplicates().copy()
    df["processed_case_id"] = _parse_int_series(df["processed_case_id"])
    invalid_case_id_rows = int(df["processed_case_id"].isna().sum())
    df = df[df["processed_case_id"].notna()].copy()

    missing_processed_csv_rows = 0
    if existing_only and "processed_csv_path" in df.columns:
        exists_mask = df["processed_csv_path"].map(
            lambda p: Path(str(p)).exists() if pd.notna(p) and str(p).strip() else False
        )
        missing_processed_csv_rows = int((~exists_mask).sum())
        df = df[exists_mask].copy()

    df = df.drop_duplicates(subset=["processed_case_id"], keep="last").copy()
    df = df.sort_values("processed_case_id").reset_index(drop=True)

    stats.update(
        {
            "invalid_processed_case_id_rows": invalid_case_id_rows,
            "missing_processed_csv_rows": missing_processed_csv_rows,
            "output_rows": int(len(df)),
            "unique_processed_case_id_count": int(df["processed_case_id"].nunique()),
        }
    )
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 case_mapping.csv")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned, stats = clean_case_mapping(input_path, existing_only=args.existing_only)
    cleaned.to_csv(output_path, index=False)

    if summary_path is not None:
        summary_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"WROTE={output_path}")
    print(f"ROWS={len(cleaned)}")
    print(f"UNIQUE_CASES={stats['unique_processed_case_id_count']}")


if __name__ == "__main__":
    main()
