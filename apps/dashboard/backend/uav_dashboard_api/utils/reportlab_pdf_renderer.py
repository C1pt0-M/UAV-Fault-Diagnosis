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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from uav_dashboard_api.utils.pdf_fonts import resolve_pdf_font_name


def resolve_font_name(font_candidates: list[str] | None) -> str:
    return resolve_pdf_font_name(font_candidates)


def build_styles(font_name: str):
    styles = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'UAVTitle',
            parent=styles['Title'],
            fontName=font_name,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0f172a'),
            wordWrap='CJK',
            spaceAfter=4,
        ),
        'subtitle': ParagraphStyle(
            'UAVSubtitle',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2563eb'),
            wordWrap='CJK',
            spaceAfter=12,
        ),
        'heading': ParagraphStyle(
            'UAVHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=13,
            leading=20,
            textColor=colors.HexColor('#0f172a'),
            wordWrap='CJK',
            spaceBefore=6,
            spaceAfter=4,
        ),
        'body': ParagraphStyle(
            'UAVBody',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor('#334155'),
            wordWrap='CJK',
        ),
        'meta': ParagraphStyle(
            'UAVMeta',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor('#64748b'),
            wordWrap='CJK',
        ),
        'emphasis': ParagraphStyle(
            'UAVEmphasis',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=11,
            leading=17,
            textColor=colors.HexColor('#111827'),
            wordWrap='CJK',
        ),
        'bullet': ParagraphStyle(
            'UAVBullet',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor('#334155'),
            leftIndent=10,
            wordWrap='CJK',
        ),
    }


def paragraph(text: str, style):
    return Paragraph(escape(text).replace('\n', '<br/>'), style)


def build_story(lines: list[str], styles: dict):
    story = []
    for index, line in enumerate(lines):
        if not line:
            story.append(Spacer(1, 6))
            continue

        if index == 0:
            story.append(paragraph(line, styles['title']))
            continue
        if index == 1:
            story.append(paragraph(line, styles['subtitle']))
            continue
        if line.startswith(
            ('一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、')
        ):
            story.append(paragraph(line, styles['heading']))
            continue
        if line.startswith(('生成时间：', '任务 ID：', '资产文件：', '诊断模型：')):
            story.append(paragraph(line, styles['meta']))
            continue
        if line.startswith(('风险等级：', '模型置信度：', '报告可信度：', '数据质量等级：')):
            story.append(paragraph(line, styles['emphasis']))
            continue
        if line.startswith('- ') or line.startswith('  '):
            story.append(paragraph(line, styles['bullet']))
            continue

        story.append(paragraph(line, styles['body']))
    return story


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit('usage: reportlab_pdf_renderer.py <payload_json> <output_pdf>')

    payload_path = Path(argv[1])
    output_path = Path(argv[2])
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    lines = payload.get('lines') or []
    font_name = resolve_font_name(payload.get('font_candidates'))
    styles = build_styles(font_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title='UAV Diagnosis Report',
        author='基于数据驱动的无人机故障智能诊断系统',
    )
    doc.build(build_story(lines, styles))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
