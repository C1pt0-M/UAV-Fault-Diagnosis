import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
BACKEND_README = REPO_ROOT / 'apps' / 'dashboard' / 'backend' / 'README.md'
FRONTEND_README = REPO_ROOT / 'apps' / 'dashboard' / 'frontend' / 'README.md'


class DashboardStartupScriptsTest(unittest.TestCase):
    def test_windows_dashboard_startup_scripts_exist(self):
        expected = [
            'start_dashboard_backend.ps1',
            'start_dashboard_frontend.ps1',
            'start_dashboard.ps1',
        ]

        for script_name in expected:
            with self.subTest(script=script_name):
                self.assertTrue(
                    (SCRIPTS_DIR / script_name).exists(),
                    f'expected startup script to exist: scripts/{script_name}',
                )

    def test_legacy_dashboard_bash_startup_scripts_are_removed(self):
        legacy = [
            'start_dashboard.sh',
            'start_dashboard_backend.sh',
            'start_dashboard_frontend.sh',
        ]

        for script_name in legacy:
            with self.subTest(script=script_name):
                self.assertFalse(
                    (SCRIPTS_DIR / script_name).exists(),
                    f'expected legacy dashboard bash script to be removed: scripts/{script_name}',
                )

    def test_backend_powershell_script_prefers_windows_venv_python(self):
        script = (SCRIPTS_DIR / 'start_dashboard_backend.ps1').read_text(encoding='utf-8')
        self.assertIn('.venv\\Scripts\\python.exe', script)
        self.assertIn('run.py', script)

    def test_frontend_powershell_script_runs_frontend_command(self):
        script = (SCRIPTS_DIR / 'start_dashboard_frontend.ps1').read_text(encoding='utf-8')
        self.assertIn('npm run frontend', script)

    def test_main_powershell_script_launches_backend_and_frontend_windows(self):
        script = (SCRIPTS_DIR / 'start_dashboard.ps1').read_text(encoding='utf-8')
        self.assertIn('Start-Process', script)
        self.assertIn('start_dashboard_backend.ps1', script)
        self.assertIn('start_dashboard_frontend.ps1', script)

    def test_readme_uses_windows_dashboard_startup_commands(self):
        root_readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
        backend_readme = BACKEND_README.read_text(encoding='utf-8')
        frontend_readme = FRONTEND_README.read_text(encoding='utf-8')

        self.assertIn(
            r'powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard.ps1',
            root_readme,
        )
        self.assertIn('scripts/start_dashboard_backend.ps1', root_readme)
        self.assertIn('scripts/start_dashboard_frontend.ps1', root_readme)
        self.assertNotIn('bash scripts/start_dashboard.sh', root_readme)
        self.assertNotIn('bash scripts/start_dashboard_backend.sh', root_readme)
        self.assertNotIn('bash scripts/start_dashboard_frontend.sh', root_readme)
        self.assertIn('start_dashboard_backend.ps1', backend_readme)
        self.assertIn('start_dashboard_frontend.ps1', frontend_readme)


if __name__ == '__main__':
    unittest.main()
