from __future__ import annotations

import hashlib
import json
import re
import subprocess
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


def test_env_example_documents_every_config_env_var() -> None:
    backend = REPO_ROOT / "apps" / "dashboard" / "backend"
    config_source = (backend / "uav_dashboard_api" / "config.py").read_text(
        encoding="utf-8"
    )
    documented = {
        line.split("=", 1)[0].strip()
        for line in (backend / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    read_by_config = set(
        re.findall(
            r"^\s{4}[A-Z][A-Z0-9_]+\s*=\s*"
            r"(?:_get_env(?:_bool|_list)?|os\.getenv)\(\s*'([A-Z][A-Z0-9_]+)'",
            config_source,
            re.MULTILINE,
        )
    )

    assert read_by_config
    assert read_by_config <= documented, sorted(read_by_config - documented)


def test_gitignore_does_not_hide_project_facts() -> None:
    content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/tests/*" not in content
    assert "\nCONTEXT.md\n" not in f"\n{content}\n"
    assert "\nrequirements-autodl-train.txt\n" not in f"\n{content}\n"
    assert "!docs/project_status_and_windows_migration.md" in content


def test_frozen_adaptation_protocol_is_visible_and_keeps_original_bytes() -> None:
    root = "artifacts/normal_adaptation_audit_20261029"
    protocol_path = f"{root}/normal_adaptation_protocol_v1.json"
    roles_path = f"{root}/real_case_roles_v1.csv"
    for path in (protocol_path, roles_path, "docs/实时正常数据适配协议_v1.md"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path], cwd=REPO_ROOT,
        )
        assert result.returncode == 1, path
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "artifacts/experiment/best.pt"],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    protocol = json.loads((REPO_ROOT / protocol_path).read_text(encoding="utf-8"))
    assert hashlib.sha256((REPO_ROOT / roles_path).read_bytes()).hexdigest() == protocol["role_manifest_sha256"]
    for path in (protocol_path, roles_path):
        result = subprocess.run(
            ["git", "check-attr", "text", "--", path], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip().endswith(": text: unset"), path


def test_removed_sim_factory_suite_is_explicitly_excluded() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = metadata["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--ignore-glob=tests/test_sim_factory_*.py" in addopts
    assert "/tests/test_sim_factory_*.py" in (
        REPO_ROOT / ".gitignore"
    ).read_text(encoding="utf-8")
