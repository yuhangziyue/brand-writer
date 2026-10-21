"""T1 第一步：PyMuPDF 抽文字层与版面坐标，逐页渲染 PNG。

文字层是**唯一可被引用的事实源**。多模态模型只负责配对与归纳，
它说的任何数字和专名都要回到这里核验（见 crosscheck.py）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ragwriter.models import Block, Page

# 章节分隔页：整页只有一个两位数编号，字号极大（本 deck 为 96pt）。
# 第 7 页目录页的编号是 60pt，不能用「有两位数」判定，否则目录会被当成分隔页。
DIVIDER_MIN_FONT = 90
DIVIDER_NO_RE = re.compile(r"^\s*(\d{2})\s*$")

HEADER_ZONE_RATIO = 0.25  # 页眉区域占页高的比例，用于定位标题
RENDER_DPI = 110  # 960x540 的幻灯片渲染到约 1467x825，够前端看清又不至于太大


def _page_blocks(page: pymupdf.Page) -> list[Block]:
    out: list[Block] = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:  # 0 = 文本块
            continue
        spans = [s for line in b["lines"] for s in line["spans"]]
        text = "".join(s["text"] for s in spans).strip()
        if not text:
            continue
        x0, y0, x1, y1 = b["bbox"]
        out.append(
            Block(text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=max(s["size"] for s in spans))
        )
    return out


def _title_of(blocks: list[Block], page_height: float) -> str:
    """标题 = 页眉区域内字号最大的那块，并列时取最靠上的。

    不能直接用「全页最大字号」：第 33 页正文里有 40pt 的数字标注「+27%」，
    比 32pt 的真标题还大；第 7 页目录的栏目编号是 60pt。页眉区域限制在页面
    上四分之一，本 deck 所有正文页的标题块 y0 都落在 8~34 之间。
    """
    if not blocks:
        return ""
    header = [b for b in blocks if b.y0 < page_height * HEADER_ZONE_RATIO]
    cands = header or blocks
    top_size = max(b.size for b in cands)
    picked = sorted((b for b in cands if b.size >= top_size - 0.1), key=lambda b: b.y0)
    return picked[0].text


def _is_divider(blocks: list[Block]) -> tuple[bool, str]:
    """分隔页判定，返回 (是否分隔页, 章节名)。

    章节名取编号块下方最近的那块文本，如第 8 页的「GEO背景介绍」。
    """
    big = [b for b in blocks if b.size >= DIVIDER_MIN_FONT]
    for b in big:
        m = DIVIDER_NO_RE.match(b.text)
        if not m:
            continue
        below = [o for o in blocks if o is not b and o.y0 >= b.y0]
        below.sort(key=lambda o: (o.y0, o.x0))
        name = below[0].text if below else ""
        return True, f"{m.group(1)} {name}".strip()
    return False, ""


def extract_pages(pdf_path: str | Path, render_dir: str | Path | None = None) -> list[Page]:
    """抽取全部页面。render_dir 给定时同时渲染 PNG。"""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    if render_dir is not None:
        render_dir = Path(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)

    page_height = float(doc[0].rect.height) if doc.page_count else 540.0
    raw: list[tuple[int, list[Block], str]] = []
    for i, pg in enumerate(doc, start=1):
        raw.append((i, _page_blocks(pg), pg.get_text().strip()))

    # 先扫一遍定章节边界，再回填每页所属章节
    chapter_at: dict[int, str] = {}
    for page_no, blocks, _ in raw:
        ok, name = _is_divider(blocks)
        if ok:
            chapter_at[page_no] = name

    pages: list[Page] = []
    current = "封面与总览"
    for page_no, blocks, text in raw:
        if page_no in chapter_at:
            current = chapter_at[page_no]
        image_path = ""
        if render_dir is not None:
            image_path = str(Path(render_dir) / f"p{page_no:02d}.png")
            if not Path(image_path).exists():
                doc[page_no - 1].get_pixmap(dpi=RENDER_DPI).save(image_path)
        pages.append(
            Page(
                page_no=page_no,
                chapter=current,
                title=_title_of(blocks, page_height),
                text=text,
                blocks=blocks,
                image_path=image_path,
                is_divider=page_no in chapter_at,
            )
        )
    doc.close()
    return pages
