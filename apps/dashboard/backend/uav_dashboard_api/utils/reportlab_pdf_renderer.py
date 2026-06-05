from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from uav_dashboard_api.utils.pdf_fonts import collect_cjk_font_candidates

REGISTERED_FONT_NAME = "UAVReportFont"

SECTION_HEADING_PREFIXES = (
    "一、",
    "二、",
    "三、",
    "四、",
    "五、",
    "六、",
    "七、",
    "八、",
    "九、",
)


def register_cjk_font(font_path: Path) -> str:
    if REGISTERED_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return REGISTERED_FONT_NAME

    suffix = font_path.suffix.lower()
    if suffix == ".ttc":
        pdfmetrics.registerFont(
            TTFont(REGISTERED_FONT_NAME, str(font_path), subfontIndex=0)
        )
    else:
        pdfmetrics.registerFont(TTFont(REGISTERED_FONT_NAME, str(font_path)))
    return REGISTERED_FONT_NAME


def resolve_font_name(font_candidates: list[str] | None = None) -> str:
    candidates = list(font_candidates or []) + collect_cjk_font_candidates()
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        normalized = str(Path(raw))
        if normalized in seen:
            continue
        seen.add(normalized)
        path = Path(normalized)
        if not path.is_file():
            continue
        try:
            return register_cjk_font(path)
        except Exception:
            continue
    return "Helvetica"


def build_styles(font_name: str):
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "UAVTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "UAVSubtitle",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2563eb"),
            wordWrap="CJK",
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "UAVHeading",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "UAVBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#334155"),
            wordWrap="CJK",
        ),
        "meta": ParagraphStyle(
            "UAVMeta",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            wordWrap="CJK",
        ),
        "emphasis": ParagraphStyle(
            "UAVEmphasis",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=11,
            leading=17,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "UAVBullet",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#334155"),
            leftIndent=10,
            wordWrap="CJK",
        ),
    }


def paragraph(text: str, style):
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def build_story(lines: list[str], styles: dict):
    story = []
    for index, line in enumerate(lines):
        if not line:
            story.append(Spacer(1, 6))
            continue

        if index == 0:
            story.append(paragraph(line, styles["title"]))
            continue
        if index == 1:
            story.append(paragraph(line, styles["subtitle"]))
            continue
        if line.startswith(SECTION_HEADING_PREFIXES):
            story.append(paragraph(line, styles["heading"]))
            continue
        if line.startswith(("生成时间：", "任务 ID：", "资产文件：", "诊断模型：")):
            story.append(paragraph(line, styles["meta"]))
            continue
        if line.startswith(("风险等级：", "模型置信度：")):
            story.append(paragraph(line, styles["emphasis"]))
            continue
        if line.startswith("- ") or line.startswith("  "):
            story.append(paragraph(line, styles["bullet"]))
            continue

        story.append(paragraph(line, styles["body"]))
    return story


def render_pdf_file(
    lines: list[str],
    output_path: Path | str,
    *,
    font_candidates: list[str] | None = None,
) -> str:
    """Render diagnosis report lines to PDF. Returns resolved font name."""
    output = Path(output_path)
    font_name = resolve_font_name(font_candidates)
    if font_name == "Helvetica":
        raise RuntimeError(
            "未找到可用的中文字体，请安装微软雅黑/黑体，或在 Linux 上安装 Noto CJK 字体。"
        )

    styles = build_styles(font_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="UAV Diagnosis Report",
        author="UAV Fault Diagnosis System",
    )
    doc.build(build_story(lines, styles))
    return font_name


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: reportlab_pdf_renderer.py <payload_json> <output_pdf>")

    payload_path = Path(argv[1])
    output_path = Path(argv[2])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lines = payload.get("lines") or []
    render_pdf_file(
        lines,
        output_path,
        font_candidates=payload.get("font_candidates"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
