from pathlib import Path

import pytest

from uav_dashboard_api.utils.pdf_fonts import collect_cjk_font_candidates
from uav_dashboard_api.utils.reportlab_pdf_renderer import render_pdf_file, resolve_font_name

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


def test_collect_cjk_font_candidates_returns_existing_paths():
    candidates = collect_cjk_font_candidates()
    assert candidates
    assert all(Path(item).is_file() for item in candidates)


def test_render_pdf_file_embeds_chinese_text(tmp_path: Path):
    font_name = resolve_font_name(collect_cjk_font_candidates())
    if font_name == "Helvetica":
        pytest.skip("当前环境未找到可用的中文字体")

    output = tmp_path / "chinese-report.pdf"
    lines = [
        "基于数据驱动的无人机故障检测系统",
        "离线诊断报告",
        "生成时间：2026-05-27 15:43:20",
        "",
        "一、诊断结论",
        "本次离线诊断识别为电机故障，当前属于高风险状态。",
        "- 质量等级：良好",
    ]
    render_pdf_file(lines, output)

    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")

    if PdfReader is None:
        return

    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    assert "离线诊断报告" in text or "诊断结论" in text
    assert "????" not in text
