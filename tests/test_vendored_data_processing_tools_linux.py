from pathlib import Path

import pytest


pytestmark = pytest.mark.external


def test_vendored_tool_uses_platform_independent_paths_and_no_windows_shell_loop():
    tool_root = Path("vendor/Data_processing_tools_linux")
    assert tool_root.exists()

    fileprocess = (tool_root / "fileprocess.py").read_text()
    rflytool = (tool_root / "Rflytool_main.py").read_text()
    read_contents = (tool_root / "read_contents.py").read_text()

    assert "for %i in (*)" not in fileprocess
    assert "for %i in (*)" not in read_contents

    assert "subprocess.run([" in fileprocess or "subprocess.run(" in fileprocess
    assert "subprocess.run([" in read_contents or "subprocess.run(" in read_contents

    assert "datapath + '\\\\' + PX4_file_folder" not in rflytool
    assert "os.path.join(datapath, PX4_file_folder)" in rflytool

    assert "target_path = self.data_file_path + '//' + path" not in fileprocess
    assert "os.path.join(self.data_file_path, path)" in fileprocess

    assert (
        "target_file_path = PX4_path + '\\\\' + listOfCSVFiles[i]" not in read_contents
    )
    assert "os.path.join(PX4_path, listOfCSVFiles[i])" in read_contents

    data_extractor = (tool_root / "data_extractor.py").read_text()
    assert "'./data_SIL_PX4.json'" not in data_extractor
    assert 'os.path.join(BASE_DIR, "data_SIL_PX4.json")' in data_extractor
