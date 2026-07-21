from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_TARGETS = (
    "apps/dashboard/backend/uav_dashboard_api",
    "packages/uav_fault/uav_fault",
    "packages/rfly_bridge/rfly_bridge",
    "scripts",
    "tests",
)
PYTEST_SUITES = (
    ("tests",),
    ("apps/dashboard/backend/tests",),
    (
        "packages/uav_fault/uav_fault/online",
        "packages/uav_fault/uav_fault/telemetry",
        "packages/rfly_bridge/rfly_bridge",
    ),
)


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(f"[verify] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical project verification suite")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _run([sys.executable, "-m", "compileall", "-q", *PYTHON_TARGETS])
    for suite in PYTEST_SUITES:
        _run([sys.executable, "-m", "pytest", "-q", *suite])

    if not args.skip_frontend:
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm is required unless --skip-frontend is set")
        frontend = REPO_ROOT / "apps" / "dashboard" / "frontend"
        _run([npm, "test", "--", "--run"], cwd=frontend)
        if not args.skip_build:
            _run([npm, "run", "build"], cwd=frontend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
