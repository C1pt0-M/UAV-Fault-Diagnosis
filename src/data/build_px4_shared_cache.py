from __future__ import annotations

import numpy as np

from features.shared_feature_builder import build_sequence_features
from features.shared_schema import TRAINING_FEATURES_V1


def build_shared_feature_rows(rows: list[dict]) -> list[dict[str, float]]:
    derived_rows = build_sequence_features(rows)
    out: list[dict[str, float]] = []
    for raw, derived in zip(rows, derived_rows):
        merged = {k: float(raw.get(k, 0.0)) for k in raw.keys()}
        merged.update({k: float(v) for k, v in derived.items()})
        out.append(merged)
    return out


def resample_feature_matrix(
    rows: list[dict[str, float]],
    columns: list[str] | None = None,
    target_len: int = 2048,
) -> np.ndarray:
    columns = columns or list(TRAINING_FEATURES_V1)
    if not rows:
        return np.zeros((target_len, len(columns)), dtype=np.float32)

    src_len = len(rows)
    x_src = np.linspace(0.0, 1.0, src_len)
    x_dst = np.linspace(0.0, 1.0, target_len)
    matrix = np.zeros((target_len, len(columns)), dtype=np.float32)
    for idx, col in enumerate(columns):
        series = np.array([float(row.get(col, 0.0)) for row in rows], dtype=np.float32)
        matrix[:, idx] = np.interp(x_dst, x_src, series)
    return matrix
