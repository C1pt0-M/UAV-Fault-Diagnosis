"""Acceptance check: does a trained online detector false-alarm on held-out live sessions?

The historical acceptance criterion ("all 40 RflySim normal sessions have low false-alarm
rates") was checked on sessions that were also in the train split, so it measured
training-set behaviour. This script evaluates the sessions that were explicitly held out
(live_holdout.pt, produced by build_online_cache_from_windows.py), which is the only way
to tell whether the live-domain problem is actually fixed.

Usage
-----
    python scripts/eval/eval_live_holdout.py --model-dir assets/models/online_detector_kg_mstca_rfly
    python scripts/eval/eval_live_holdout.py --model-dir ... --cache-dir D:\\UAV-Data\\online_cache_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "uav_fault"))

from uav_fault.models.cnn.lstm import (  # noqa: E402
    OnlineBiLSTM,
    OnlineCNN1D,
    OnlineKGMSTCA,
    OnlineMSTCA,
    OnlineTransformer,
)

MODEL_REGISTRY = {
    "bilstm": OnlineBiLSTM,
    "cnn1d": OnlineCNN1D,
    "transformer": OnlineTransformer,
    "mstca": OnlineMSTCA,
    "kg_mstca": OnlineKGMSTCA,
}

DEFAULT_CACHE = Path(r"D:\UAV-Data\online_cache_v2")


def build_model(model_dir: Path, columns: list[str], device: torch.device) -> torch.nn.Module:
    with open(model_dir / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    name = config.get("model") or config.get("architecture") or "kg_mstca"
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    kwargs = dict(
        in_channels=len(columns),
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 1),
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.2),
    )
    if "columns" in _signature_params(MODEL_REGISTRY[name]):
        kwargs["columns"] = list(columns)
    model = MODEL_REGISTRY[name](**kwargs)
    state = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _signature_params(cls) -> list[str]:
    import inspect

    return list(inspect.signature(cls).parameters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--holdout", default="live_holdout.pt")
    parser.add_argument("--threshold", type=float, default=None,
                        help="fault probability cut-off; defaults to the model's own value")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cache_dir = Path(args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cache_dir / "norm_stats.json", encoding="utf-8") as handle:
        stats = json.load(handle)
    columns = list(stats["columns"])
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1)

    payload = torch.load(cache_dir / args.holdout, weights_only=True)
    x = payload["x"].float()
    groups = payload.get("groups") or ["all"] * x.shape[0]
    x = (x - mean) / std

    model = build_model(model_dir, columns, device)
    threshold = args.threshold
    if threshold is None:
        threshold = float(stats.get("fault_threshold", 0.5))

    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.batch_size):
            batch = x[start:start + args.batch_size].to(device)
            logits = model(batch)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    probs_arr = np.asarray(probs)
    flagged = probs_arr >= threshold

    print("=" * 78)
    print("HELD-OUT LIVE SESSION FALSE-ALARM CHECK")
    print("=" * 78)
    print(f"  model      : {model_dir}")
    print(f"  holdout    : {cache_dir / args.holdout}  ({x.shape[0]} windows)")
    print(f"  threshold  : {threshold}")
    print()
    print(f"  false-alarm rate (windows flagged fault): {flagged.mean() * 100:.2f}%")
    print(f"  windows flagged                         : {int(flagged.sum())} / {len(flagged)}")
    print(f"  fault probability  mean={probs_arr.mean():.3f} "
          f"median={np.median(probs_arr):.3f} "
          f"p95={np.percentile(probs_arr, 95):.3f} "
          f"max={probs_arr.max():.3f}")

    print()
    print("  per flight profile:")
    print(f"    {'maneuver':12s}{'windows':>9s}{'flagged':>9s}{'rate':>9s}{'mean p':>9s}")
    per: dict[str, list[float]] = defaultdict(list)
    for group, prob in zip(groups, probs):
        per[group].append(prob)
    for group in sorted(per):
        values = np.asarray(per[group])
        flag = values >= threshold
        print(f"    {group:12s}{len(values):9d}{int(flag.sum()):9d}"
              f"{flag.mean() * 100:8.2f}%{values.mean():9.3f}")

    # The historical failure was fault_prob pinned near 1, so report that explicitly.
    near_one = (probs_arr > 0.9).mean()
    print()
    print(f"  windows with p(fault) > 0.9: {near_one * 100:.2f}%  "
          f"(the historical live-path failure looked like ~100%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
