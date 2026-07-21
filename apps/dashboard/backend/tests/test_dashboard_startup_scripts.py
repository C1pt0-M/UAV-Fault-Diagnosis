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
        self.assertIn('-m uav_dashboard_api', script)

    def test_backend_powershell_script_exposes_monorepo_packages_to_python(self):
        script = (SCRIPTS_DIR / 'start_dashboard_backend.ps1').read_text(encoding='utf-8')

        self.assertIn('packages\\\\uav_fault', script)
        self.assertIn('packages\\\\rfly_bridge', script)
        self.assertIn('$env:PYTHONPATH', script)
        self.assertIn('[IO.Path]::PathSeparator', script)

    def test_duplicate_backend_run_script_is_removed(self):
        self.assertFalse((REPO_ROOT / 'apps' / 'dashboard' / 'backend' / 'run.py').exists())

    def test_application_factory_does_not_override_sigint(self):
        app_factory = (
            REPO_ROOT
            / 'apps'
            / 'dashboard'
            / 'backend'
            / 'uav_dashboard_api'
            / '__init__.py'
        ).read_text(encoding='utf-8')

        self.assertNotIn('signal.signal(signal.SIGINT', app_factory)

    def test_frontend_powershell_script_runs_frontend_command(self):
        script = (SCRIPTS_DIR / 'start_dashboard_frontend.ps1').read_text(encoding='utf-8')
        self.assertIn('npm run frontend', script)

    def test_main_powershell_script_launches_backend_and_frontend_windows(self):
        script = (SCRIPTS_DIR / 'start_dashboard.ps1').read_text(encoding='utf-8')
        self.assertIn('Start-Process', script)
        self.assertIn('start_dashboard_backend.ps1', script)
        self.assertIn('start_dashboard_frontend.ps1', script)

    def test_rfly_launcher_preserves_tcp_simulator_and_mavlink_udp(self):
        powershell_script = (SCRIPTS_DIR / 'start_rflysim_sitl_qgc.ps1').read_text(
            encoding='utf-8'
        )
        wsl_script = (SCRIPTS_DIR / 'rflysim_sitl_qgc.sh').read_text(
            encoding='utf-8'
        )

        self.assertNotIn('export PX4_SIM_HOST_ADDR=', powershell_script)
        self.assertIn('export RFLY_SIM_HOST_IP=', powershell_script)
        self.assertIn('rfly_wsl_udp_relay.py', powershell_script)
        self.assertIn('export RFLY_SIM_FORWARD_PORT=', powershell_script)
        self.assertIn('Test-TcpListenPort', powershell_script)
        self.assertIn('cat ${rcs_romfs_path}/rcS', wsl_script)
        self.assertNotIn('skip px4-rc.simulator', wsl_script)
        self.assertIn('-t $RFLY_SIM_HOST_IP', wsl_script)
        self.assertIn('$udp_rfly_remote_port', wsl_script)

    def test_rfly_stop_script_cleans_all_px4_and_coptersim_variants(self):
        script = (SCRIPTS_DIR / 'stop_rflysim_sitl.ps1').read_text(encoding='utf-8')

        self.assertIn('CopterSimNoUI', script)
        self.assertIn("pkill -9 -f '[p]x4'", script)

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
