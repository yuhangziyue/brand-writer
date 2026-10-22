"""问答的查询改写：把追问补全成可以独立检索的问题。

最大的风险不是改写得不好，而是**把答案写进问题**：模型从记忆里补上「15倍」，
检索命中正确页，表面一切正常，但这个数字从来没经过检索。
提示词禁止这样做；程序侧再兜一道确定性检查——改写结果里出现了输入中没有的数字，就退回原消息。
退回原消息最坏是检索召回差一点，不会引入未经检索的事实。
"""

from __future__ import annotations

import json
import re

from ragwriter.generate import prompts

PROMPT = "query_rewrite"

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def rewrite_query(client, message: str, slots: dict, recent_user_turns: list[str]) -> str:
    slots_json = json.dumps(slots or {}, ensure_ascii=False, indent=2, default=str)
    recent = recent_user_turns[-3:]
    system = prompts.render(PROMPT, "System")
    user = prompts.render(
        PROMPT, "User",
        slots=slots_json,
        recent="\n".join(f"- {t}" for t in recent) or "- 无",
        message=message,
    )
    raw = client.complete(system, user, max_tokens=200, tag="query_rewrite")
    out = _first_line(raw)
    if not out:
        return message
    known = set(_NUM_RE.findall(" ".join([message, slots_json, *recent])))
    if any(n not in known for n in _NUM_RE.findall(out)):
        return message
    return out


def _first_line(raw: str) -> str:
    text = re.sub(r"```\w*", "", raw or "").strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(改写后|独立问题|问题)[:：]\s*", "", line)
    return line.strip("`「」“”\"'")
