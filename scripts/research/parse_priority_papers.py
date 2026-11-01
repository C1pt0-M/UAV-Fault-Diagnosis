"""Read three user-supplied papers with the configured MinerU skill.

Use a Python download adapter because the skill's curl stderr decoding fails
on this Windows environment. Do not print credentials or signed result URLs.
"""
import concurrent.futures
import importlib.util
from pathlib import Path
import tempfile
import zipfile
import subprocess
import shutil
import re

import requests

ROOT = Path(__file__).resolve().parents[2]
SKILL = Path.home() / ".agents/skills/mineru-ocr/scripts/mineru_parse.py"
JOBS = {
    "aeroguard": "AeroGuard_Towards_Real-Time_UAV_Fault_Detection_With_Hybrid_Models.pdf",
    "imu_prediction": "IMU-Only_UAV_Motor_Fault_Prediction_Before_Failure_A_Multi-Model_Multi-Horizon_Machine_Learning_Benchmark_on_the_RflyMAD_Dataset.pdf",
    "crossdomain_incremental": "Quadrotor_UAV_Actuators_Fine-Grained_Fault_Diagnosis_With_Cross-Domain_Incremental_Learning.pdf",
}


def download_result(url, output_dir, retries=4, timeout=300):
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".zip", delete=False) as target:
        target_path = Path(target.name)
        try:
            with requests.get(url, stream=True, timeout=(30, timeout)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Result download HTTP {response.status_code}")
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
        except requests.RequestException:
            pass
    if target_path.stat().st_size == 0:
        curl = shutil.which("curl.exe") or shutil.which("curl")
        result = subprocess.run([curl, "--location", "--fail", "--silent", "--show-error",
            "--connect-timeout", "20", "--max-time", "90", "--output", str(target_path), url],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            detail = re.sub(r"https?://\S+", "[result URL]", result.stderr)
            raise RuntimeError(f"Result download curl exit {result.returncode}: {detail[:400]}")
    with zipfile.ZipFile(target_path) as archive:
        for member in archive.namelist():
            if not (directory / member).resolve().is_relative_to(directory):
                raise RuntimeError("Result archive contains a path outside the output directory")
    return str(target_path)


def parse_one(pair):
    name, filename = pair
    output = ROOT / "docs/research/2026-10-07/fulltext" / name
    if (output / "full.md").exists():
        return name + ": existing extraction retained"
    spec = importlib.util.spec_from_file_location("mineru_" + name, SKILL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.download_with_curl = download_result
    module.parse_file(str(ROOT / "docs/论文" / filename), output_dir=str(output), language="en")
    return name + ": extracted"


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(parse_one, JOBS.items()):
            print(result, flush=True)
