"""Contract-driven window extraction for the 26-channel feature contract v1.

Why this module exists: three places need "read a processed RflyMAD case CSV and
produce the 26 contract channels for a row range" -- the topology graph builder, the
online cache builder, and the correlation pilot. The pilot and the graph builder each
carry their own copy, so this module is the shared implementation for new consumers
instead of a fourth copy.

Difference from the pilot's copy, deliberately: the pilot skips the window-relative
altitude preprocessing because Pearson correlation is invariant to a constant shift.
Anything that feeds a model needs the real preprocessing, so it is applied here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_PATH = Path(r"D:\UAV-Data\feature_contract_v1.json")

# Preprocessing tags used by the contract.
_PREP_WINDOW_RELATIVE = "window_relative_subtract_first"
_PREP_DERIVED = "derived"

# Channels the contract derives rather than maps from a source column.
DERIVED_ACC_NORM = "acc_norm"


def load_contract(path: Path | str = CONTRACT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract_columns(contract: dict) -> list[str]:
    """Canonical channel order, which must match RFLY_SEQ_COLUMNS_V2."""
    return [item["canonical_name"] for item in contract["features"]]


def resolve_domain(contract: dict, domain: str) -> str:
    """Map a manifest domain label onto the contract's key, case-insensitively.

    The window manifest writes REAL while the feature contract writes Real, so an
    exact lookup silently fails for every Real case.
    """
    keys = set()
    for item in contract["features"]:
        keys.update(item["source_columns"].keys())
    if domain in keys:
        return domain
    lowered = {str(key).lower(): key for key in keys}
    resolved = lowered.get(str(domain).lower())
    if resolved is None:
        raise KeyError(f"unknown domain {domain!r}; contract has {sorted(keys)}")
    return resolved


def source_columns_for_domain(contract: dict, domain: str) -> dict[str, str]:
    """canonical name -> source column name in the processed CSV for this domain."""
    resolved = resolve_domain(contract, domain)
    mapping: dict[str, str] = {}
    for item in contract["features"]:
        source = item["source_columns"].get(resolved)
        if source in (None, _PREP_DERIVED):
            continue
        mapping[item["canonical_name"]] = source
    return mapping


def read_case_features(
    path: Path | str,
    domain: str,
    contract: dict,
) -> tuple[np.ndarray, list[str]]:
    """Read one processed case CSV into a (frames, 26) float64 matrix.

    Returns the matrix in contract order plus the column names, so callers can assert
    the order rather than assume it.
    """
    path = Path(path)
    columns = contract_columns(contract)
    index = {name: i for i, name in enumerate(columns)}
    feature_meta = {item["canonical_name"]: item for item in contract["features"]}

    mapping = source_columns_for_domain(contract, domain)
    frame = pd.read_csv(path, usecols=list(set(mapping.values())), low_memory=False)
    output = np.empty((len(frame), len(columns)), dtype=np.float64)

    for canonical, source in mapping.items():
        output[:, index[canonical]] = pd.to_numeric(
            frame[source], errors="coerce"
        ).to_numpy(dtype=np.float64)

    # acc_norm is derived from the accelerometer.
    ax = output[:, index["accel_x"]]
    ay = output[:, index["accel_y"]]
    az = output[:, index["accel_z"]]
    output[:, index[DERIVED_ACC_NORM]] = np.sqrt(ax * ax + ay * ay + az * az)

    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    _ = feature_meta  # kept for callers that need per-channel metadata
    return output, columns


def apply_window_relative(
    matrix: np.ndarray, columns: list[str], start: int, end: int
) -> np.ndarray:
    """Slice [start:end) and apply the contract's window-relative preprocessing.

    Currently only baro_alt_relative: subtract the value at the first frame of the
    window, so the altitude baseline does not encode the launch site or weather.
    """
    window = matrix[start:end].copy()
    relative = [
        item
        for item in ("baro_alt_relative",)
        if item in columns
    ]
    for name in relative:
        idx = columns.index(name)
        window[:, idx] = window[:, idx] - window[0, idx]
    return window


def window_batch(
    matrix: np.ndarray,
    columns: list[str],
    spans: list[tuple[int, int]],
) -> np.ndarray:
    """Stack several windows from one case into (n, channels, frames)."""
    if not spans:
        return np.empty((0, len(columns), 0), dtype=np.float32)
    frames = spans[0][1] - spans[0][0]
    out = np.empty((len(spans), len(columns), frames), dtype=np.float32)
    for i, (start, end) in enumerate(spans):
        out[i] = apply_window_relative(matrix, columns, start, end).T
    return out
