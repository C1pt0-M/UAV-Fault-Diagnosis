from __future__ import annotations

import os
import sys
from pathlib import Path


LINUX_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

WINDOWS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simfang.ttf",
]

MACOS_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _windows_font_dir() -> Path | None:
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if not windir:
        return None
    fonts_dir = Path(windir) / "Fonts"
    return fonts_dir if fonts_dir.is_dir() else None


def _expand_windows_candidates() -> list[str]:
    fonts_dir = _windows_font_dir()
    if fonts_dir is None:
        return list(WINDOWS_FONT_CANDIDATES)

    names = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "simfang.ttf",
    ]
    candidates = [str(fonts_dir / name) for name in names]
    candidates.extend(WINDOWS_FONT_CANDIDATES)
    return candidates


def build_pdf_font_candidates() -> list[str]:
    platform_candidates: list[str]
    if sys.platform.startswith("win"):
        platform_candidates = _expand_windows_candidates()
    elif sys.platform == "darwin":
        platform_candidates = list(MACOS_FONT_CANDIDATES)
    else:
        platform_candidates = list(LINUX_FONT_CANDIDATES)

    seen: set[str] = set()
    ordered: list[str] = []
    for raw in platform_candidates + LINUX_FONT_CANDIDATES:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def resolve_pdf_font_name(font_candidates: list[str] | None) -> str:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Helvetica"

    candidates: list[str] = []
    for raw in list(font_candidates or []) + build_pdf_font_candidates():
        text = str(raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    for raw in candidates:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("UAVReportFont", str(path)))
            return "UAVReportFont"
        except Exception:
            continue
    return "Helvetica"
