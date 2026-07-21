import importlib.util
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "data" / "build_online_cache.py"
spec = importlib.util.spec_from_file_location("build_online_cache", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
build_online_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_online_cache)


def _sample_csv(path: Path, phase: str) -> None:
    pd.DataFrame(
        {
            "time_s": [index * 0.02 for index in range(6)],
            "flight_phase": [phase] * 6,
        }
    ).to_csv(path, index=False)


def test_extra_normal_can_use_a_separate_window_cap(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    output_dir = tmp_path / "cache"
    source_csv = csv_dir / "source_fault.csv"
    extra_csv = extra_dir / "nf_buaa_case.csv"
    _sample_csv(source_csv, "fault")
    _sample_csv(extra_csv, "normal")

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        '{"train": ["source_fault.csv"], "val": ["source_fault.csv"], "test": ["source_fault.csv"]}',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"filename": "source_fault.csv", "label_binary": "fault"},
        ]
    ).to_csv(manifest_path, index=False)

    build_online_cache.build_cache(
        str(csv_dir),
        str(splits_path),
        str(output_dir),
        window_size=4,
        stride=1,
        target_hz=0.0,
        extra_normal_dir=str(extra_dir),
        extra_normal_glob="nf_*.csv",
        extra_normal_split="train",
        max_windows_per_file=1,
        extra_normal_max_windows_per_file=3,
        manifest_path=str(manifest_path),
    )

    train = torch.load(output_dir / "train.pt", weights_only=True)
    stats = (output_dir / "norm_stats.json").read_text(encoding="utf-8")

    assert train["x"].shape[0] == 4
    assert train["y"].tolist() == [1, 0, 0, 0]
    assert '"extra_normal_max_windows_per_file": 3' in stats
