from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_is_the_python_dependency_source_of_truth() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["dependencies"]
    assert {"dashboard", "training", "training-autodl", "telemetry", "test"} <= set(
        project["optional-dependencies"]
    )

    autodl_requirements = (REPO_ROOT / "requirements-autodl-train.txt").read_text(
        encoding="utf-8"
    )
    assert ".[training-autodl]" in autodl_requirements
    for relative_path in (
        "requirements.txt",
        "requirements-dev.txt",
        "apps/dashboard/backend/requirements.txt",
    ):
        assert not (REPO_ROOT / relative_path).exists()


def test_pre_monorepo_shims_are_removed() -> None:
    for relative_path in ("uav_dashboard_api", "uav_fault", "rfly_bridge"):
        package_dir = REPO_ROOT / relative_path
        assert not (package_dir / "__init__.py").exists()

    stale_script_names = {
        "annotate_fault_activity.py",
        "build_competition_metrics.py",
        "compare_offline_results.py",
        "compare_online_a_layer.py",
        "compare_online_backbones.py",
        "compare_transfer3_a_layer.py",
        "compare_transfer3_backbones.py",
        "db_view.py",
        "deep_audit_dataset.py",
        "eval_telemetry_live.py",
        "phase_a_input_gate.py",
        "plot_competition_metrics.py",
        "plot_misclassified_thrust_loss.py",
        "prepare_trainmix_dataset.py",
        "run_online_backbone_suite.py",
        "transfer3_protocol_audit.py",
    }
    assert not stale_script_names & {
        path.name for path in (REPO_ROOT / "scripts").glob("*.py")
    }


def test_active_entrypoints_do_not_reference_pre_monorepo_src() -> None:
    active_paths = (
        REPO_ROOT / "packages/rfly_bridge/rfly_bridge/rfly_live_bridge.py",
        REPO_ROOT / "packages/rfly_bridge/rfly_bridge/run_buaa_realtime_bridge.py",
        REPO_ROOT / "scripts/eval/transfer3_protocol_audit.py",
    )
    for path in active_paths:
        assert ' / "src"' not in path.read_text(encoding="utf-8")


def test_runtime_modules_do_not_depend_on_competition_paths() -> None:
    active_paths = (
        REPO_ROOT / "apps/dashboard/backend/.env.example",
        REPO_ROOT / "apps/dashboard/backend/uav_dashboard_api/config.py",
        REPO_ROOT
        / "apps/dashboard/backend/uav_dashboard_api/services/telemetry_backends.py",
        REPO_ROOT
        / "apps/dashboard/backend/uav_dashboard_api/services/rfly_diagnostics.py",
    )
    for path in active_paths:
        assert "competition/" not in path.read_text(encoding="utf-8")


def test_active_tests_do_not_use_pre_monorepo_paths() -> None:
    stale_fragments = (
        'PROJECT_ROOT / "src"',
        'PROJECT_ROOT / "models"',
        'PROJECT_ROOT / "dashboard"',
    )
    for test_path in (REPO_ROOT / "tests").glob("test_*.py"):
        if test_path.name == Path(__file__).name or test_path.name.startswith("test_sim_factory_"):
            continue
        content = test_path.read_text(encoding="utf-8")
        for fragment in stale_fragments:
            assert fragment not in content, f"{test_path.name} still uses {fragment}"


def test_gitignore_does_not_hide_project_facts() -> None:
    content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/tests/*" not in content
    assert "\nCONTEXT.md\n" not in f"\n{content}\n"
    assert "\nrequirements-autodl-train.txt\n" not in f"\n{content}\n"
    assert "!docs/project_status_and_windows_migration.md" in content


def test_removed_sim_factory_suite_is_explicitly_excluded() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = metadata["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--ignore-glob=tests/test_sim_factory_*.py" in addopts
    assert "/tests/test_sim_factory_*.py" in (
        REPO_ROOT / ".gitignore"
    ).read_text(encoding="utf-8")
