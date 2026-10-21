"""T1 主路：页卡结构化。

把一页变成「要点 + 数字配对」。真正要解决的是**数字挂到哪个标签上**，
不是认字——34 个关键数字里 31 个文字层本来就抽得到。

两种输入模式，默认 blocks：

- **blocks（默认）**：把该页文本块按 PDF 内容流顺序连同字号交给模型。
  取证结论：不排序时数值与其标签在本 deck 上 30/30 全部紧邻，
  分组信息本来就在块序里。`sort=True` 才会把多栏内容按行横扫打散，
  那是调用方式造成的问题，不是 PDF 的问题。
- **vision（可选增强）**：额外送渲染图。需要 provider 真支持图片输入——
  开发期用的中转虽然在模型清单里声明了 image 模态，实测 prompt_tokens 只有 58，
  图被静默丢弃，且模型在看不到图时会**编造**截图里的数字。
  所以这条路径默认关闭，开启前必须先验证 prompt_tokens 是否随图增长。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ragwriter.llm.client import LLMClient
from ragwriter.models import Page

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "page_card.md"

PAGE_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "page_topic": {"type": "string", "description": "这一页讲什么，一句话"},
        "points": {
            "type": "array",
            "description": "本页要点，每条一句话，尽量使用文字层原文措辞",
            "items": {"type": "string"},
        },
        "metrics": {
            "type": "array",
            "description": "数字与其描述对象的配对",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "label": {"type": "string"},
                    "source_note": {"type": "string"},
                    "certain": {"type": "boolean"},
                },
                "required": ["value", "label", "source_note", "certain"],
                "additionalProperties": False,
            },
        },
        "image_only_items": {
            "type": "array",
            "description": "只在图里出现、文字层找不到的内容",
            "items": {"type": "string"},
        },
    },
    "required": ["page_topic", "points", "metrics", "image_only_items"],
    "additionalProperties": False,
}


class CardMetric(BaseModel):
    value: str
    label: str
    source_note: str = ""
    certain: bool = True


class PageCard(BaseModel):
    page_no: int
    page_topic: str = ""
    points: list[str] = Field(default_factory=list)
    metrics: list[CardMetric] = Field(default_factory=list)
    image_only_items: list[str] = Field(default_factory=list)


def _load_prompt() -> tuple[str, str, str]:
    """从 Markdown 里取 System 与两个 User 模板，保证线上用的就是仓库里那份原文。"""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    system = text.split("## System", 1)[1].split("## User（blocks 模式）", 1)[0].strip()
    rest = text.split("## User（blocks 模式）", 1)[1]
    user_blocks, user_vision = rest.split("## User（vision 模式）", 1)
    return system, user_blocks.strip(), user_vision.strip()


def render_blocks(page: Page) -> str:
    """把文本块按内容流顺序渲染成带序号与字号的清单。

    字号保留一位小数：模型要靠它区分「28pt 的放大数值」与「12pt 的出处注记」。
    """
    return "\n".join(
        f"#{i:02d} ({b.size:.1f}) {b.text}" for i, b in enumerate(page.blocks, start=1)
    )


def build_page_card(client: LLMClient, page: Page, *, use_vision: bool = False) -> PageCard:
    system, user_blocks, user_vision = _load_prompt()
    if use_vision:
        user = user_vision.replace("{page_no}", str(page.page_no)).replace(
            "{page_text}", page.text
        )
        raw = client.complete_with_image(
            system,
            user,
            page.image_path,
            json_schema=PAGE_CARD_SCHEMA,
            max_tokens=3000,
            tag=f"page_card:p{page.page_no:02d}",
        )
    else:
        user = user_blocks.replace("{page_no}", str(page.page_no)).replace(
            "{page_blocks}", render_blocks(page)
        )
        raw = client.complete(
            system,
            user,
            json_schema=PAGE_CARD_SCHEMA,
            max_tokens=3000,
            tag=f"page_card:p{page.page_no:02d}",
        )
    return PageCard(page_no=page.page_no, **json.loads(raw))
