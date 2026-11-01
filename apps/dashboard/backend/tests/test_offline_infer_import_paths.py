import importlib.util
import sys
from pathlib import Path


def _load_offline_multiclass_infer():
    backend_root = Path(__file__).resolve().parents[1]
    script_path = backend_root / "scripts" / "offline_multiclass_infer.py"
    spec = importlib.util.spec_from_file_location(
        "offline_multiclass_infer_for_test",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_infer_adds_packaged_uav_fault_paths(monkeypatch):
    module = _load_offline_multiclass_infer()
    repo_root = Path(__file__).resolve().parents[4]
    package_path = str(repo_root / "packages" / "uav_fault")

    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != package_path])

    module.add_import_paths(repo_root)

    assert package_path in sys.path
    module.add_import_paths(repo_root)
    assert sys.path.count(package_path) == 1
