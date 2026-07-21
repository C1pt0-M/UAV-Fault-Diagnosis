from pathlib import Path
import re

import pytest


pytestmark = pytest.mark.external


def test_vendored_rflytool_has_min_length_alignment_before_concatenate():
    content = Path("vendor/Data_processing_tools_linux/Rflytool_main.py").read_text()

    assert "def align_row_count" in content
    assert "min_rows = min(arr.shape[0] for arr in arrays)" in content
    assert "return [arr[:min_rows] for arr in arrays]" in content

    concat_sites = re.findall(
        r"np\.concatenate\(\(all_data, new_data\), axis=1\)", content
    )
    assert len(concat_sites) >= 2
    assert "all_data, new_data = align_row_count(all_data, new_data)" in content
    assert (
        "gtd_time_seq_arr, all_data = align_row_count(gtd_time_seq_arr, all_data)"
        in content
    )
    assert (
        "px4_time_seq_arr, all_data = align_row_count(px4_time_seq_arr, all_data)"
        in content
    )
    assert (
        "data_index_arr, all_data = align_row_count(data_index_arr, all_data)"
        in content
    )
