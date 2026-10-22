"""写作：逐节流式写作、导语/结尾、问答作答、单句改写。

四个函数只负责「组提示词 + 调模型」，不做校验——校验在 verify/，编排在 agent/。
这样每个函数的输入都是确定的：同一份 ContextPack 进来，送进模型的文字逐字可复现，回放与排查都靠这一点。

提示词正文全部在 prompts/*.md；品牌档案按节点切片（10-agent-architecture 第 2.2 节）：
写作与改写只看 profile["rules"]（错字修正表、禁用规则），不看目录与实体表。

资料不足（产品评审 M1）：写作节点不把「品宣资料未涉及」写进正文，而是在输出最后一行写
`GAPS: [...]`；split_gaps() 把它剥下来，正文进校验，缺口进校验报告的资料不足列表。
问答节点不走这条，正文里直接说「品宣资料未涉及」。
"""

from __future__ import annotations

import json
import re
from typing import Iterator, Literal, Sequence

from ragwriter.contracts import CheckResult, ContextPack, Genre, Plan, PlanSection, Snippet
from ragwriter.generate import prompts
from ragwriter.generate.context import render_snippet

SECTION_PROMPT = "section_writer"
ANSWER_PROMPT = "answer"
REWRITE_PROMPT = "rewriter"

CANNOT_REWRITE = "无法改写"

_BOOKEND = {"lede": "导语", "ending": "结尾"}

_GAPS_RE = re.compile(r"^\s*`?GAPS\s*[:：]\s*(.*?)`?\s*$", re.IGNORECASE)


def stream_section(
    client,
    section: PlanSection,
    pack: ContextPack,
    *,
    genre: Genre,
    profile: dict,
    constraints: Sequence[str] = (),
) -> Iterator[str]:
    system = prompts.render(
        SECTION_PROMPT, "System",
        profile_rules=prompts.profile_rules_text(profile),
        genre_spec=prompts.genre_spec(genre, "写作"),
    )
    user = prompts.render(
        SECTION_PROMPT, "User",
        context=pack.rendered,
        genre=genre,
        heading=section.heading,
        sub_question=section.sub_question,
        constraints=prompts.constraints_text(constraints),
    )
    return client.stream(system, user, max_tokens=1200, tag="section_writer")


def stream_bookend(
    client,
    kind: Literal["lede", "ending"],
    plan: Plan,
    verified_sections: list[str],
    *,
    genre: Genre,
    profile: dict,
) -> Iterator[str]:
    """导语 / 结尾。输入只有已通过校验的句子（带原 S 编号），提示词禁止引入新事实。

    写法随体裁变：Blog 导语结论先行；FAQ 基本不要导语结尾（是否调用由编排层决定）。
    输出末行同样带 GAPS，调用方用 split_gaps 剥离。
    """
    kind_name = _BOOKEND[kind]
    points = plan.lede_points if kind == "lede" else plan.ending_points
    system = prompts.render(
        SECTION_PROMPT, "System（导语与结尾）",
        kind_name=kind_name,
        kind_spec=prompts.genre_spec(genre, kind_name),
        profile_rules=prompts.profile_rules_text(profile),
    )
    user = prompts.render(
        SECTION_PROMPT, "User（导语与结尾）",
        kind_name=kind_name,
        title=plan.title,
        points="\n".join(f"- {p}" for p in points) or "- 无",
        verified=_verified_text(plan, verified_sections),
    )
    return client.stream(system, user, max_tokens=600, tag=f"bookend_{kind}")


def stream_answer(client, question: str, pack: ContextPack, *, profile: dict) -> Iterator[str]:
    system = prompts.render(ANSWER_PROMPT, "System", profile_rules=prompts.profile_rules_text(profile))
    user = prompts.render(ANSWER_PROMPT, "User", context=pack.rendered, question=question)
    return client.stream(system, user, max_tokens=600, tag="answer")


def rewrite_sentence(
    client,
    sentence: str,
    failures: list[CheckResult],
    snippets: list[Snippet],
    *,
    profile: dict,
) -> str:
    """单句改写，非流式。返回改写后的一句；模型判定无法在片段内改对时返回空串，由编排层删除该句。"""
    failed = [f for f in failures if not f.ok] or failures
    system = prompts.render(REWRITE_PROMPT, "System", profile_rules=prompts.profile_rules_text(profile))
    user = prompts.render(
        REWRITE_PROMPT, "User",
        sentence=sentence,
        failures="\n".join(f"- {f.check}：{f.reason or '未通过'}" for f in failed) or "- 无",
        snippets="\n".join(render_snippet(s) for s in snippets) or "（无可引用片段）",
    )
    raw = client.complete(system, user, max_tokens=400, tag="rewriter")
    return clean_rewrite(raw)


def split_gaps(text: str) -> tuple[str, list[str]]:
    """把写作节点输出末尾的 `GAPS: [...]` 行剥下来，返回（正文, 缺口列表）。

    容错：GAPS 行可能不在最后一行（模型偶尔在其后补一句），所以扫全部行；
    数组不是合法 JSON 时按顿号 / 逗号切分兜底——缺口描述丢了不致命，混进正文才致命。
    """
    body: list[str] = []
    gaps: list[str] = []
    for line in text.splitlines():
        m = _GAPS_RE.match(line)
        if not m:
            body.append(line)
            continue
        raw = m.group(1).strip()
        try:
            items = json.loads(raw)
            items = items if isinstance(items, list) else [items]
        except json.JSONDecodeError:
            items = re.split(r"[、，,；;]", raw.strip("[]"))
        gaps += [s for s in (str(i).strip().strip("\"'“”「」") for i in items) if s]
    return "\n".join(body).strip(), gaps


def clean_rewrite(raw: str) -> str:
    """模型偶尔会加引号、前缀或围栏；只取第一句非空行。"""
    text = re.sub(r"```\w*", "", raw).strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(改写后|修改后|改写)[:：]\s*", "", line)
    line = line.strip("「」“”\"'")
    if not line or line.startswith(CANNOT_REWRITE):
        return ""
    return line


def _verified_text(plan: Plan, verified_sections: list[str]) -> str:
    blocks = []
    for i, text in enumerate(verified_sections):
        heading = plan.sections[i].heading if i < len(plan.sections) else ""
        body = text.strip() or "（本节没有通过校验的句子）"
        blocks.append(f"### 第{i + 1}节 {heading}".rstrip() + f"\n{body}")
    return "\n\n".join(blocks) or "（没有已通过校验的句子）"
