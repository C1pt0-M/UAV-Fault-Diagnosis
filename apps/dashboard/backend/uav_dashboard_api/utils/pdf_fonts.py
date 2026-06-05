from __future__ import annotations

import os
import platform
from pathlib import Path


def collect_cjk_font_candidates() -> list[str]:
    """Collect OS-specific CJK font paths for PDF rendering (first match wins)."""
    candidates: list[str] = []
    system = platform.system()

    if system == "Windows":
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend(
            [
                fonts_dir / "msyh.ttc",
                fonts_dir / "msyhbd.ttc",
                fonts_dir / "simhei.ttf",
                fonts_dir / "simsun.ttc",
                fonts_dir / "simsunb.ttf",
                fonts_dir / "msyhl.ttc",
            ]
        )
    elif system == "Darwin":
        candidates.extend(
            [
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/STHeiti Light.ttc"),
                Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
                Path("/Library/Fonts/Arial Unicode.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            ]
        )

    bundled = Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.otf"
    candidates.append(bundled)

    output: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        path = Path(item)
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            output.append(str(path))
    return output
