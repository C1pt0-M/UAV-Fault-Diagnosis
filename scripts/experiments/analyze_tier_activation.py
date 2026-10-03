"""Compare how the two prior tiers behave when a fault occurs.

Question: the tier-1 couplings are strong during normal hover, and the tier-2
couplings are silent during normal hover. Which group actually carries the fault
signal?

Uses the Delta_A matrices already produced by topology_change_pilot.py
(A_post - A_pre around fault onset; for normal cases two adjacent windows).

Result (SIL train, 50 cases per class):
  hover_observable (measurable during hover) barely moves under motor fault
  (1.08x normal jitter) and even shrinks under sensor fault (0.40x).
  hover_silent (hidden by the hover regime) moves the most
  (motor 1.91x, sensor 0.97x).
The couplings that are loud during normal flight are therefore NOT the ones that
announce a fault; the fault manifests in the normally quiet couplings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PILOT_DIR = Path(r"D:\UAV-Data\experiments\topology_change_pilot")
GRAPH_DIR = Path(r"D:\UAV-Data\experiments\topology_graph")
OUTPUT_PATH = GRAPH_DIR / "edge_activation.json"


def main() -> int:
    names = json.loads((GRAPH_DIR / "feature_names.json").read_text(encoding="utf-8"))
    observable = np.load(GRAPH_DIR / "adjacency_hover_observable.npy")
    silent = np.load(GRAPH_DIR / "adjacency_hover_silent.npy")

    n = len(names)
    iu = np.triu_indices(n, k=1)
    masks = {
        "hover_observable": observable[iu] > 0,
        "hover_silent": silent[iu] > 0,
        "not_in_prior": ~((observable[iu] > 0) | (silent[iu] > 0)),
    }

    delta = {
        label: np.load(PILOT_DIR / f"mean_delta_a_{label}.npy")[iu]
        for label in ("normal", "motor", "sensor")
    }

    print("=" * 92)
    print("MEAN |Delta_A| PER EDGE GROUP  (how much each coupling changed pre -> post)")
    print("=" * 92)
    print(
        f"{'group':24s}{'pairs':>7s}{'normal':>11s}{'motor':>11s}{'sensor':>11s}"
        f"{'motor/normal':>14s}{'sensor/normal':>15s}"
    )
    report: dict[str, dict] = {}
    for name, mask in masks.items():
        nm = float(np.abs(delta["normal"][mask]).mean())
        mo = float(np.abs(delta["motor"][mask]).mean())
        se = float(np.abs(delta["sensor"][mask]).mean())
        report[name] = {
            "pairs": int(mask.sum()),
            "normal_mean_abs_delta": nm,
            "motor_mean_abs_delta": mo,
            "sensor_mean_abs_delta": se,
            "motor_over_normal": mo / nm,
            "sensor_over_normal": se / nm,
        }
        print(
            f"{name:24s}{int(mask.sum()):7d}{nm:11.4f}{mo:11.4f}{se:11.4f}"
            f"{mo / nm:14.2f}{se / nm:15.2f}"
        )

    print()
    print("=" * 92)
    print("EDGES THAT 'WAKE UP' UNDER FAULT (change >= 2x that edge's normal jitter)")
    print("=" * 92)
    wake: dict[str, dict] = {}
    for label in ("motor", "sensor"):
        ratio = np.abs(delta[label]) / np.maximum(np.abs(delta["normal"]), 1e-6)
        wake[label] = {}
        for name, mask in masks.items():
            woke = int((ratio[mask] >= 2.0).sum())
            total = int(mask.sum())
            wake[label][name] = {"woke": woke, "total": total, "fraction": woke / max(total, 1)}
            print(
                f"  {label:7s} {name:24s} {woke:3d}/{total:3d}  "
                f"{woke / max(total, 1) * 100:5.1f}%"
            )

    for label in ("motor", "sensor"):
        print()
        print("=" * 92)
        print(f"TOP 10 CHANGED EDGES UNDER {label.upper()} FAULT")
        print("=" * 92)
        order = np.argsort(-np.abs(delta[label]))
        for k in order[:10]:
            i, j = iu[0][k], iu[1][k]
            tag = next(
                (key for key, mask in masks.items() if mask[k]), "not_in_prior"
            )
            print(
                f"  {names[i]:20s} <-> {names[j]:20s} "
                f"normal={abs(delta['normal'][k]):.4f} "
                f"{label}={abs(delta[label][k]):.4f} "
                f"ratio={abs(delta[label][k]) / max(abs(delta['normal'][k]), 1e-6):7.1f}x  [{tag}]"
            )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "edge_activation_v1",
                "pilot_dir": str(PILOT_DIR),
                "groups": report,
                "wake_up": wake,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
