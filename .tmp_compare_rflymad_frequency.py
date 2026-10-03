from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT = Path(r"D:/UAV-Data/processing_logs/sil-50hz-20260808-155358/failed_cases.log")
OFFICIAL_ROOT = Path(r"D:/RflyMAD/Data_processing_tools")
SOURCE_ROOT = Path(r"D:/RflyMAD")
WORK_ROOT = Path(r"D:/UAV-Data/.tmp_rflymad_frequency_compare")


def failed_sources() -> list[Path]:
    result = []
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 2)
        if len(fields) >= 2 and fields[1].startswith("D:"):
            result.append(Path(fields[1]))
    return result


def load_reader():
    deps = ("fileprocess", "timetrans", "data_extractor", "get_parse")
    for name in deps:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "official_frequency_compare", OFFICIAL_ROOT / "Rflytool_main.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SHIL_data_reader
    finally:
        sys.path.pop(0)
        for name in deps:
            sys.modules.pop(name, None)


def prepare_case(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "TestInfo.csv", destination / "TestInfo.csv")
    shutil.copytree(source / "Log", destination / "Log")
    shutil.copytree(source / "TrueData", destination / "TrueData")
    log_dir = destination / "Log"
    ulg_files = sorted(log_dir.glob("*.ulg"))
    if not ulg_files:
        raise FileNotFoundError(f"No ULog: {source}")
    for ulg in ulg_files:
        completed = subprocess.run(
            ["ulog2csv", ulg.name], cwd=log_dir, capture_output=True, text=True, timeout=300
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr or completed.stdout)


def run_one(reader, source: Path, frequency: int, root: Path) -> tuple[str, str, int | None, str]:
    case_root = root / str(frequency) / source.name
    prepare_case(source, case_root)
    output = root / str(frequency) / f"{source.name}.csv"
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(OFFICIAL_ROOT)
        reader(str(case_root), 1, frequency, str(output))
    except Exception as exc:
        return source.name, str(frequency), None, repr(exc)
    finally:
        os.chdir(old_cwd)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    return source.name, str(frequency), rows, ""


def main() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)
    reader = load_reader()
    sources = failed_sources()
    results = []
    for index, source in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {source}", flush=True)
        for frequency in (20, 50):
            results.append(run_one(reader, source, frequency, WORK_ROOT))
            print(results[-1], flush=True)
    with (WORK_ROOT / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "frequency_hz", "data_rows", "error"))
        writer.writerows(results)


if __name__ == "__main__":
    main()
