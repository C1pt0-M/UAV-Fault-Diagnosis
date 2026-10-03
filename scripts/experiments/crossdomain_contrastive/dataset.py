"""Window dataset for cross-domain contrastive training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_MAP = {"normal": 0, "motor": 1, "sensor": 2}
DOMAIN_MAP = {"SIL": 0, "HIL": 1, "Real": 2, "REAL": 2}


def _load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(path.read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _load_contract(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class WindowDataset(Dataset):
    """Load windows from manifest with on-the-fly feature extraction."""

    def __init__(
        self,
        manifest_path: Path,
        contract_path: Path,
        norm_stats_path: Path,
        split: Optional[str | list[str]] = None,
        *,
        max_windows_per_case: int = 0,
        label_field: str = "level2_label",
        seed: int = 42,
        prefiltered: bool = False,
    ):
        self.contract = _load_contract(contract_path)
        self.mean, self.std = _load_norm_stats(norm_stats_path)
        self.features = self.contract["features"]
        self.window_size = self.contract["window_size_frames"]
        self.baro_idx = next(
            f["index"] for f in self.features
            if f.get("preprocessing") == "window_relative_subtract_first"
        )

        # Load manifest
        manifest = pd.read_csv(manifest_path, dtype=str).fillna("")

        if not prefiltered:
            if split is not None:
                if isinstance(split, str):
                    split = [split]
                manifest = manifest[manifest["split"].isin(split)].copy()
            # Filter to closed-set labels only
            manifest = manifest[manifest[label_field].isin(LABEL_MAP.keys())].copy()

            if max_windows_per_case > 0:
                rng = np.random.default_rng(seed)
                groups = manifest.groupby("case_uid")
                sampled = []
                for _, group in groups:
                    if len(group) <= max_windows_per_case:
                        sampled.append(group)
                    else:
                        sampled.append(group.sample(n=max_windows_per_case, random_state=int(rng.integers(1e9))))
                manifest = pd.concat(sampled, ignore_index=True)

        self.records = manifest[
            ["case_uid", "domain", "source_csv", "window_start_row",
             "window_end_row", label_field]
        ].copy()
        self.records["window_start_row"] = self.records["window_start_row"].astype(int)
        self.records["window_end_row"] = self.records["window_end_row"].astype(int)
        self.label_field = label_field

        # Cache source column mappings per domain
        self._col_maps: dict[str, dict[str, int]] = {}
        for domain in ("SIL", "HIL", "Real"):
            col_map = {}
            for feat in self.features:
                src = feat["source_columns"][domain]
                if src not in ("derived", "derived_from_euler_rad"):
                    col_map[src] = feat["index"]
            self._col_maps[domain] = col_map
        # Alias uppercase
        self._col_maps["REAL"] = self._col_maps["Real"]

        # CSV cache
        self._csv_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_csv_features(self, csv_path: str, domain: str) -> np.ndarray:
        """Load and convert a full CSV to feature matrix (cached)."""
        if csv_path in self._csv_cache:
            return self._csv_cache[csv_path]

        col_map = self._col_maps[domain]
        df = pd.read_csv(csv_path, usecols=list(col_map.keys()), low_memory=False)
        n_rows = len(df)
        n_features = len(self.features)
        output = np.zeros((n_rows, n_features), dtype=np.float32)

        for src_col, feat_idx in col_map.items():
            output[:, feat_idx] = pd.to_numeric(df[src_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        # Derived: acc_norm
        feat_idx_map = {f["canonical_name"]: f["index"] for f in self.features}
        ax = output[:, feat_idx_map["accel_x"]]
        ay = output[:, feat_idx_map["accel_y"]]
        az = output[:, feat_idx_map["accel_z"]]
        output[:, feat_idx_map["acc_norm"]] = np.sqrt(ax**2 + ay**2 + az**2)

        self._csv_cache[csv_path] = output
        # Keep cache bounded
        if len(self._csv_cache) > 200:
            oldest = next(iter(self._csv_cache))
            del self._csv_cache[oldest]

        return output

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        csv_path = row["source_csv"]
        domain = row["domain"]
        start = row["window_start_row"]
        end = row["window_end_row"]

        features = self._load_csv_features(csv_path, domain)
        window = features[start:end].copy()  # (150, 26)

        # baro_alt relative preprocessing
        window[:, self.baro_idx] -= window[0, self.baro_idx]

        # z-score normalize
        window = (window - self.mean) / self.std

        label = LABEL_MAP[row[self.label_field]]
        domain_id = DOMAIN_MAP[domain]

        return {
            "x": torch.from_numpy(window),
            "label": label,
            "domain": domain_id,
        }
