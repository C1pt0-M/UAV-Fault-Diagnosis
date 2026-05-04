import argparse
from pathlib import Path

import pandas as pd


def apply_os_listdir_bucket_order(df: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    df = df.copy()
    if "bucket_case_num" in df.columns:
        df = df.drop(columns=["bucket_case_num"])

    bucket_nums = []
    for row in df.itertuples(index=False):
        platform = row.platform
        flight = row.flight_mode
        fault = row.fault_type_canonical
        case_dir = row.case_dir
        bucket = raw_root / flight / fault
        if not bucket.exists():
            bucket_nums.append(None)
            continue
        names = [
            name
            for name in __import__("os").listdir(bucket)
            if (bucket / name).is_dir()
        ]
        try:
            bucket_nums.append(names.index(case_dir))
        except ValueError:
            bucket_nums.append(None)
    df["bucket_case_num"] = bucket_nums
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按原始目录 os.listdir 顺序修正 SHIL TestInfo bucket 顺序"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    raw_root = Path(args.raw_root)
    df = pd.read_csv(inp)
    fixed = apply_os_listdir_bucket_order(df, raw_root)
    fixed.to_csv(out, index=False)
    print(f"WROTE={out}")
    print(f"ROWS={len(fixed)}")


if __name__ == "__main__":
    main()
