"""写作规划：主题 + 体裁 → 大纲（子问题 + 资料不足列表）。

规划只看品牌档案的 catalog（全部页标题），不看全文：
让模型「知道资料里有什么」，但「具体写什么」必须经过每个子问题的检索（05-architecture 第 3.1 节）。

输出走 strict JSON schema，解析后再做一道确定性清理：
页码提示只保留目录里真实存在的页、空小节丢掉——模型给的页码只是检索的提示，不能把不存在的页带下去。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ragwriter.contracts import Genre, Plan, PlanSection
from ragwriter.generate import prompts

PROMPT = "planner"

# strict 模式要求：每层 additionalProperties=false，且全部字段列入 required
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "lede_points", "sections", "insufficient", "ending_points"],
    "properties": {
        "title": {"type": "string"},
        "lede_points": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["heading", "sub_question", "expected_pages_hint"],
                "properties": {
                    "heading": {"type": "string"},
                    "sub_question": {"type": "string"},
                    "expected_pages_hint": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
        "insufficient": {"type": "array", "items": {"type": "string"}},
        "ending_points": {"type": "array", "items": {"type": "string"}},
    },
}


# 体裁的小节数上限。超出就截断：每节都要检索 + 写作 + 逐句校验，节数直接决定成本与时延。
# 下限不补——宁可少一节，也不让程序凑出一个没有依据的小节
MAX_SECTIONS: dict[str, int] = {"blog": 5, "faq": 8, "brand_intro": 4, "product_intro": 5}


class PlanError(ValueError):
    """规划输出无法使用（非 JSON、没有任何小节）。编排层据此发 error 事件。"""


def build_messages(topic: str, genre: Genre, profile: dict, constraints: Sequence[str] = ()) -> tuple[str, str]:
    system = prompts.render(PROMPT, "System", genre_spec=prompts.genre_spec(genre, "规划"))
    user = prompts.render(
        PROMPT, "User",
        topic=topic,
        genre=genre,
        constraints=prompts.constraints_text(constraints),
        catalog=prompts.catalog_text(profile),
    )
    return system, user


def plan(client, topic: str, genre: Genre, profile: dict, constraints: Sequence[str] = ()) -> Plan:
    system, user = build_messages(topic, genre, profile, constraints)
    # 真跑里同一主题偶发返回空小节列表（D02 产品介绍，重跑即正常），重试一次再报错
    for attempt in range(2):
        raw = client.complete(system, user, json_schema=PLAN_SCHEMA, max_tokens=2048, tag="planner")
        try:
            result = parse_plan(raw, valid_pages=prompts.catalog_pages(profile))
            break
        except PlanError:
            if attempt == 1:
                raise
    result.sections = result.sections[: MAX_SECTIONS.get(genre, len(result.sections))]
    return result


def parse_plan(raw: str, *, valid_pages: set[int] | None = None) -> Plan:
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as exc:
        raise PlanError(f"规划输出不是合法 JSON：{raw[:200]}") from exc

    sections: list[PlanSection] = []
    for s in data.get("sections") or []:
        heading = str(s.get("heading", "")).strip()
        sub_q = str(s.get("sub_question", "")).strip()
        if not sub_q:
            continue
        hint = []
        for p in s.get("expected_pages_hint") or []:
            try:
                p = int(p)
            except (TypeError, ValueError):
                continue
            if (valid_pages is None or not valid_pages or p in valid_pages) and p not in hint:
                hint.append(p)
        sections.append(PlanSection(heading=heading or sub_q, sub_question=sub_q, expected_pages_hint=hint))
    if not sections:
        raise PlanError("规划没有给出任何可检索的小节")

    return Plan(
        title=str(data.get("title", "")).strip(),
        lede_points=_str_list(data.get("lede_points")),
        sections=sections,
        insufficient=_str_list(data.get("insufficient")),
        ending_points=_str_list(data.get("ending_points")),
    )


def _str_list(v: Any) -> list[str]:
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def _strip_fence(raw: str) -> str:
    """strict 模式下不该出现代码围栏，但退化 provider 可能忽略 response_format。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    return m.group(1) if m else raw.strip()
