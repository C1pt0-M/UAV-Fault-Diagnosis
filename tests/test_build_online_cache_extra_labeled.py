import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "data" / "build_online_cache.py"
spec = importlib.util.spec_from_file_location("build_online_cache_extra_labeled", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_case(path: Path, phases: list[str]) -> None:
    rows = len(phases)
    pd.DataFrame(
        {
            "time_s": [index * 0.02 for index in range(rows)],
            "flight_phase": phases,
            "_actuator_outputs_0_output[0]": [1200.0] * rows,
            "_actuator_outputs_0_output[1]": [1200.0] * rows,
            "_actuator_outputs_0_output[2]": [1200.0] * rows,
            "_actuator_outputs_0_output[3]": [1200.0] * rows,
            "_sensor_combined_0_accelerometer_m_s2[2]": [-9.8] * rows,
        }
    ).to_csv(path, index=False)


def test_extra_labeled_manifest_balances_train_and_excludes_untouched(tmp_path: Path) -> None:
    base = tmp_path / "base"
    domain = tmp_path / "domain"
    base.mkdir()
    domain.mkdir()
    _write_case(base / "base.csv", ["normal"] * 30)
    _write_case(domain / "adapt.csv", ["normal"] * 20 + ["fault"] * 20)
    _write_case(domain / "heldout.csv", ["normal"] * 20 + ["fault"] * 20)

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps({"train": ["base.csv"], "val": ["base.csv"], "test": ["base.csv"]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"filename": "base.csv", "label_binary": "normal"}]).to_csv(
        manifest_path, index=False
    )
    domain_manifest = tmp_path / "domain_split.csv"
    pd.DataFrame(
        [
            {"file": "adapt.csv", "split": "train"},
            {"file": "heldout.csv", "split": "untouched_test"},
        ]
    ).to_csv(domain_manifest, index=False)

    output = tmp_path / "cache"
    module.build_cache(
        str(base),
        str(splits_path),
        str(output),
        window_size=4,
        stride=1,
        target_hz=0.0,
        max_windows_per_file=1,
        extra_labeled_root=str(domain),
        extra_labeled_manifest=str(domain_manifest),
        extra_labeled_max_windows_per_file=6,
        manifest_path=str(manifest_path),
    )

    train = torch.load(output / "train.pt", weights_only=True)
    assert train["y"].tolist().count(0) == 4
    assert train["y"].tolist().count(1) == 3
    assert train["x"].shape[0] == 7
    stats = json.loads((output / "norm_stats.json").read_text(encoding="utf-8"))
    assert stats["extra_labeled_split_counts"] == {"train": 1, "untouched_test": 1}
    assert len(stats["extra_labeled_files"]) == 2
