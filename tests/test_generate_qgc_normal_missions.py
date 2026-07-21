import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "data" / "generate_qgc_normal_missions.py"
SPEC = importlib.util.spec_from_file_location("generate_qgc_normal_missions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
missions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(missions)


def test_qgc_mission_templates_have_takeoff_and_land():
    templates = missions.mission_templates()

    assert len(templates) == 4
    for filename, plan in templates.items():
        assert filename.endswith(".plan")
        assert plan["fileType"] == "Plan"
        items = plan["mission"]["items"]
        assert items[0]["command"] == 22
        assert items[-1]["command"] == 21
        assert [item["doJumpId"] for item in items] == list(range(1, len(items) + 1))
        json.dumps(plan)


def test_qgc_mission_home_matches_rflysim_origin():
    templates = missions.mission_templates()

    for plan in templates.values():
        assert plan["mission"]["plannedHomePosition"] == [
            missions.HOME_LAT,
            missions.HOME_LON,
            missions.HOME_AMSL,
        ]
        assert missions.HOME_LAT == 40.1540302
        assert missions.HOME_LON == 116.2593683
