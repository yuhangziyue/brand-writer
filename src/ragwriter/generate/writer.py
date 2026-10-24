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
POLISH_PROMPT = "polish"
GENRE_NAME = {"blog": "官网 Blog", "faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}
TONE = {
    "blog": "专业、克制的行业洞察文章，有观点但不喊口号。",
    "faq": "简洁直接，像官网帮助中心的回答。",
    "brand_intro": "正式、可信的第三人称说明文。",
    "product_intro": "清楚、具体的产品说明，讲功能不堆形容词。",
}
_BLOCK = re.compile(r"^<<<\s*(LEDE|ENDING|SECTION\s+(\d+))\s*>>>\s*$", re.M)

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
    thesis: str = "",
    outline: Sequence[str] = (),
    prev_headings: Sequence[str] = (),
) -> Iterator[str]:
    """写一节。thesis / outline / prev_headings 是全文脉络（10-23）：没有它们，各节只看得到自己的片段，
    写出来是一组互不相干的小答案，读不出中心观点。它们只用于衔接，提示词里写明不是事实来源。"""
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
        purpose=section.purpose or "（未指定）",
        thesis=thesis or "（本体裁无中心论点）",
        outline="；".join(outline) or "（无）",
        prev_headings="；".join(prev_headings) or "（这是第一节）",
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


# 回答形状 → 作答要求。意图识别判断「问的是什么」，这里决定「答成什么样」，不答没问的东西
SHAPE_RULES = {
    "number": "用户问的是一个数字：只用一句话给出数字、它描述的对象和出处口径，不展开背景。",
    "yes_no": "用户在核实一个说法：第一句先明确回答「对」或「不对」（「部分对」也可以），再用一句给出原文依据；说法有误时指出原文的正确对应关系。",
    "list": "用户要的是列举：用「- 」开头的条目逐项列出，每条一行、带引用，不写开场白和总结。",
    "short": "用一到两句话直接回答，不补充用户没问的内容。",
    "explain": "用户想理解一件事：三到五句，先给结论，再按原文的逻辑解释，只用与问题直接相关的片段。",
}


def stream_answer(client, question: str, pack: ContextPack, *, profile: dict,
                  shape: str = "short") -> Iterator[str]:
    system = prompts.render(ANSWER_PROMPT, "System", profile_rules=prompts.profile_rules_text(profile),
                            shape_rule=SHAPE_RULES.get(shape, SHAPE_RULES["short"]))
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


def polish_article(client, blocks: dict[str, str], *, genre: Genre, profile: dict, thesis: str = "",
                   headings: dict[str, str] | None = None) -> dict[str, str]:
    """通篇润色（10-23 子曰：要模型参与写作过程，让语句通顺）。

    blocks：{"LEDE": 文本, "SECTION 1": 文本, …, "ENDING": 文本}，文本都是已逐句过校验、带 [S编号] 的定稿。
    返回同样键的润色稿；模型漏掉的块不返回（调用方保留原文）。是否采用由调用方逐块重新校验后决定。
    """
    headings = headings or {}
    parts = []
    for key, text in blocks.items():
        head = f"（小标题：{headings[key]}）\n" if headings.get(key) else ""
        parts.append(f"<<<{key}>>>\n{head}{text}")
    system = prompts.render(
        POLISH_PROMPT, "System",
        genre_name=GENRE_NAME.get(genre, "文章"),
        thesis_line=f"，围绕中心论点「{thesis}」" if thesis else "",
        tone=TONE.get(genre, TONE["blog"]),
        profile_rules=prompts.profile_rules_text(profile),
    )
    user = prompts.render(POLISH_PROMPT, "User", article="\n\n".join(parts))
    raw = client.complete(system, user, max_tokens=4000, tag="polish")
    return parse_blocks(raw)


def parse_blocks(raw: str) -> dict[str, str]:
    marks = list(_BLOCK.finditer(raw))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        key = "LEDE" if m.group(1) == "LEDE" else "ENDING" if m.group(1) == "ENDING" else f"SECTION {m.group(2)}"
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        body = raw[m.end():end].strip()
        body = re.sub(r"^（小标题：[^）]*）\s*", "", body)  # 模型把提示用的小标题行抄回来时去掉
        if body:
            out[key] = body
    return out
