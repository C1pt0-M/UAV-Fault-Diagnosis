from __future__ import annotations

import argparse
import json
from pathlib import Path


# RflySim default GPS origin (Beijing). Mission altitudes remain relative to this home.
HOME_LAT = 40.1540302
HOME_LON = 116.2593683
HOME_AMSL = 50.0


def _item(command: int, seq: int, lat: float, lon: float, alt: float, param1: float = 0.0) -> dict:
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": alt,
        "AltitudeMode": 0,
        "autoContinue": True,
        "command": command,
        "doJumpId": seq,
        "frame": 3,
        "params": [param1, 0, 0, None, lat, lon, alt],
        "type": "SimpleItem",
    }


def _plan(points: list[tuple[int, float, float, float, float]]) -> dict:
    items = [
        _item(command, idx, lat, lon, alt, param1)
        for idx, (command, lat, lon, alt, param1) in enumerate(points, start=1)
    ]
    return {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 5,
            "firmwareType": 12,
            "hoverSpeed": 3,
            "items": items,
            "plannedHomePosition": [HOME_LAT, HOME_LON, HOME_AMSL],
            "vehicleType": 2,
            "version": 2,
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }


def mission_templates() -> dict[str, dict]:
    north = HOME_LAT + 0.00009
    south = HOME_LAT - 0.00009
    east = HOME_LON + 0.000132
    west = HOME_LON - 0.000132
    return {
        "normal_hover_5m_60s.plan": _plan(
            [
                (22, HOME_LAT, HOME_LON, 5, 0),
                (19, HOME_LAT, HOME_LON, 5, 60),
                (21, HOME_LAT, HOME_LON, 0, 0),
            ]
        ),
        "normal_vertical_3_10m.plan": _plan(
            [
                (22, HOME_LAT, HOME_LON, 3, 0),
                (16, HOME_LAT, HOME_LON, 10, 10),
                (16, HOME_LAT, HOME_LON, 5, 10),
                (16, HOME_LAT, HOME_LON, 10, 10),
                (16, HOME_LAT, HOME_LON, 3, 10),
                (21, HOME_LAT, HOME_LON, 0, 0),
            ]
        ),
        "normal_box_10m.plan": _plan(
            [
                (22, HOME_LAT, HOME_LON, 5, 0),
                (16, north, HOME_LON, 10, 5),
                (16, north, east, 10, 5),
                (16, south, east, 10, 5),
                (16, south, west, 10, 5),
                (16, HOME_LAT, HOME_LON, 10, 5),
                (21, HOME_LAT, HOME_LON, 0, 0),
            ]
        ),
        "normal_mixed_5_15m.plan": _plan(
            [
                (22, HOME_LAT, HOME_LON, 5, 0),
                (16, north, HOME_LON, 5, 8),
                (16, north, east, 10, 8),
                (16, HOME_LAT, east, 15, 8),
                (16, south, HOME_LON, 8, 8),
                (16, HOME_LAT, west, 5, 8),
                (16, HOME_LAT, HOME_LON, 5, 8),
                (21, HOME_LAT, HOME_LON, 0, 0),
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic QGC normal-flight missions")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Documents" / "QGroundControl" / "Missions"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in mission_templates().items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
