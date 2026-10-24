"""写作流程第 ① 步：主题审核。

10-23 子曰定：每次写作都先审核主题、与用户确认，再检索。主题是否有效、精确主题是什么、
写哪些要点、产品介绍选哪个产品——全部由模型按 `prompts/topic_check.md` 判断，程序只调用与展示，
不用阈值或正则去判「主题好不好」。

只给页标题目录，不给正文：这一步只回答「资料大概能不能写」，拿不到任何数字，
所以确认卡片里不可能出现未经检索的事实。
"""

from __future__ import annotations

import json
import re
from typing import Any, get_args

from ragwriter.contracts import Genre, TopicCheck
from ragwriter.generate import prompts

PROMPT = "topic_check"
GENRES: tuple[str, ...] = get_args(Genre)

TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["valid", "genre", "topic", "angle", "outline_points", "premise_gaps", "alternatives",
                 "reason", "confirm_text"],
    "properties": {
        "valid": {"type": "boolean"},
        "genre": {"type": "string", "enum": list(GENRES)},
        "topic": {"type": "string"},
        "angle": {"type": "string"},
        "outline_points": {"type": "array", "items": {"type": "string"}},
        "premise_gaps": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "confirm_text": {"type": "string"},
    },
}

# 确认卡片是给用户看的「我打算怎么写」，不是事实陈述；但模型偶尔会把目录标题里的数字带进来。
# 数字一律剥掉——这一步看不到正文，任何数字都没有经过检索与校验
_DIGITS = re.compile(r"[\d％%]+(?:\.\d+)?\s*(?:倍|亿|万|个月|年)?")


def _entities_text(profile: dict) -> str:
    ents = profile.get("entities") or []
    out = []
    for e in ents:
        if isinstance(e, dict):
            out.append("、".join(str(v) for v in e.values() if v))
        else:
            out.append(str(e))
    return "；".join(out) or "万悉科技（Trendee）"


def _forbidden_text(profile: dict) -> str:
    forbidden = (profile.get("rules") or {}).get("forbidden") or []
    lines = []
    for f in forbidden:
        if isinstance(f, dict):
            lines.append(next((str(f[k]) for k in ("rule", "text", "desc") if k in f), json.dumps(f, ensure_ascii=False)))
        else:
            lines.append(str(f))
    return "\n".join(f"- {x}" for x in lines) or "- 无"


def _clean(s: str) -> str:
    return _DIGITS.sub("", s).strip()


def check_topic(client, request: str, *, genre: str | None, topic: str | None, profile: dict,
                slots: dict | None = None, feedback: str = "") -> TopicCheck:
    """审核一次写作请求。genre / topic 是用户指定的（写作台）或意图识别抽出的，可为空。"""
    system = prompts.render(PROMPT, "System")
    user = prompts.render(
        PROMPT, "User",
        request=request or "（无）",
        genre=genre or "（未指定，请判断）",
        topic=topic or "（未指定，请从请求中理解）",
        feedback=feedback or "（无）",
        entities=_entities_text(profile),
        catalog=prompts.catalog_text(profile),
        forbidden=_forbidden_text(profile),
    )
    out = client.complete_json(system, user, json_schema=TOPIC_SCHEMA, max_tokens=1200, tag="topic_check")
    g = out.get("genre") if out.get("genre") in GENRES else (genre if genre in GENRES else "blog")
    if genre in GENRES:
        g = genre  # 写作台指定的体裁不让模型改；要改体裁由用户在确认时提出
    return TopicCheck(
        valid=bool(out.get("valid")),
        genre=g,
        topic=(out.get("topic") or topic or request or "").strip(),
        angle=_clean(str(out.get("angle") or "")),
        outline_points=[_clean(str(x)) for x in out.get("outline_points") or [] if str(x).strip()][:8],
        premise_gaps=[str(x).strip() for x in out.get("premise_gaps") or [] if str(x).strip()][:3],
        alternatives=[_clean(str(x)) for x in out.get("alternatives") or [] if str(x).strip()][:3],
        reason=str(out.get("reason") or "").strip(),
        confirm_text=str(out.get("confirm_text") or "").strip(),
    )


def brief_text(brief: dict | None) -> list[str]:
    """把用户确认过的写作要点转成规划节点的附加要求（走 constraints 通道，不改规划签名）。"""
    if not brief:
        return []
    out = []
    if brief.get("angle"):
        out.append(f"已与用户确认的中心观点 / 切入点：{brief['angle']}")
    if brief.get("outline_points"):
        out.append("已与用户确认的写作要点（按此组织小节）：" + "；".join(brief["outline_points"]))
    if brief.get("premise_gaps"):
        out.append("已告知用户资料里没有、文中不写的内容：" + "；".join(brief["premise_gaps"]))
    return out
