"""提示词加载器。

提示词是交付物，放在 `prompts/*.md`，评审直接读文件；代码里不内嵌任何提示词正文。
文件按「## 段名」切段（`## System`、`## User`、`## 体裁：blog`……），文件头的说明不送模型。

占位符一律用 `{{name}}` 双花括号：提示词里大量出现 JSON 示例和 `{}`，
单花括号的 str.format 会把它们当占位符。

`{{rules}}` 是保留占位符，注入 `prompts/_rules.md` 的原文——
四条硬约束全局只有这一份，各节点引用，不复制（10-agent-architecture 第 2.1 节）。
第 ④ 条「资料不足怎么办」按节点分三种写法（写作不写进正文而列 GAPS、问答明说、改写放弃），
由 RULES_MODE 按提示词文件名选段拼装，其余条款各节点完全相同。

体裁结构放在 `prompts/genres/*.md`，由 genre_spec() 按体裁取段注入 planner 与 section_writer；
brand_intro / product_intro 共用 intro.md，只换「## 节目：xxx」段。

替换是单遍的：变量值里即使含有 `{{...}}`（检索片段、用户输入都可能有）也不会被再次展开，
避免用户输入篡改提示词结构。
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
RULES_NAME = "_rules"
RulesMode = Literal["write", "qa", "rewrite"]
_GAP_SECTION: dict[str, str] = {"write": "资料不足（写作）", "qa": "资料不足（问答）", "rewrite": "资料不足（改写）"}
# 提示词文件 → 第 ④ 条用哪种写法；未列出的写作类节点默认 write
RULES_MODE: dict[str, str] = {"answer": "qa", "rewriter": "rewrite"}

GENRE_FILE: dict[str, str] = {
    "blog": "genres/blog",
    "faq": "genres/faq",
    "brand_intro": "genres/intro",
    "product_intro": "genres/intro",
}

_SECTION_RE = re.compile(r"^## (?!#)(.+?)\s*$")
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


@lru_cache(maxsize=None)
def load_sections(name: str) -> dict[str, str]:
    """读取 prompts/{name}.md，按二级标题切成 {段名: 正文}。

    代码块里的「## 」不算标题——提示词里可能放 Markdown 示例。
    """
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _SECTION_RE.match(line)
        if m:
            current = m.group(1)
            if current in sections:
                raise ValueError(f"prompts/{name}.md 段名重复：{current}")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def rules_text(mode: RulesMode = "write") -> str:
    """_rules.md 拼装：①②③ + 本节点的 ④ + 附加规则。"""
    secs = load_sections(RULES_NAME)
    return "\n\n".join([secs["规则"], secs[_GAP_SECTION[mode]], secs["附加规则"]])


def genre_spec(genre: str, part: str) -> str:
    """体裁文件的一段（规划 / 写作 / 导语 / 结尾）。intro.md 的 {{program}} 换成该体裁的节目单。"""
    try:
        name = GENRE_FILE[genre]
    except KeyError:
        raise KeyError(f"未知体裁：{genre}") from None
    program = load_sections(name).get(f"节目：{genre}")
    return render(name, part, **({"program": program} if program is not None else {}))


def section(name: str, heading: str) -> str:
    try:
        return load_sections(name)[heading]
    except KeyError:
        raise KeyError(f"prompts/{name}.md 里没有「## {heading}」段") from None


def render(name: str, part: str, /, **variables: Any) -> str:
    """取一段并替换占位符。缺变量直接报错，不静默留下 `{{x}}` 送进模型。

    name / part 是仅限位置参数：提示词里有 `{{heading}}` 这样的变量，不能和形参撞名。
    """
    template = section(name, part)
    values = {k: _to_text(v) for k, v in variables.items()}
    if "rules" in _VAR_RE.findall(template):
        values.setdefault("rules", rules_text(RULES_MODE.get(name, "write")))  # type: ignore[arg-type]

    missing = sorted({v for v in _VAR_RE.findall(template) if v not in values})
    if missing:
        raise KeyError(f"prompts/{name}.md「{part}」缺少变量：{missing}")
    return _VAR_RE.sub(lambda m: values[m.group(1)], template)


def prompt_version(name: str) -> str:
    """提示词版本 = 文件内容 + _rules.md + 体裁文件的哈希前 8 位。

    运行日志与校验报告记它，回放时能确认「这次产出用的是哪一版约束」。
    """
    h = hashlib.sha256()
    for n in (name, RULES_NAME, *sorted(set(GENRE_FILE.values()))):
        h.update((PROMPT_DIR / f"{n}.md").read_bytes())
    return h.hexdigest()[:8]


def _to_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


# ── 品牌档案切片（10-agent-architecture 第 2.2 节）────────────────
# 规划看 catalog，写作 / 改写看 rules。档案由建库产物抽取，字段形态以宽松方式读取：
# 条目可能是字符串、{wrong, right} 一类的字典或二元组，这里统一渲染成一行一条。

def constraints_text(constraints: Any) -> str:
    items = [str(c).strip() for c in (constraints or ()) if str(c).strip()]
    return "；".join(items) if items else "无"


def catalog_text(profile: dict) -> str:
    """「第9页 · 章节 · 标题」一行一页。"""
    lines = []
    for entry in profile.get("catalog") or []:
        if isinstance(entry, dict):
            no = entry.get("page_no", entry.get("page", entry.get("no")))
            parts = [f"第{no}页" if no is not None else ""]
            parts += [str(entry[k]) for k in ("chapter", "title") if entry.get(k)]
            lines.append(" · ".join(p for p in parts if p))
        else:
            lines.append(str(entry))
    return "\n".join(lines) if lines else "（品牌档案缺少目录）"


def catalog_pages(profile: dict) -> set[int]:
    pages: set[int] = set()
    for entry in profile.get("catalog") or []:
        no = None
        if isinstance(entry, dict):
            no = entry.get("page_no", entry.get("page", entry.get("no")))
        else:
            m = re.search(r"第\s*(\d+)\s*页|^(\d+)", str(entry))
            if m:
                no = m.group(1) or m.group(2)
        try:
            pages.add(int(no))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return pages


def _correction_line(item: Any) -> str:
    if isinstance(item, dict):
        wrong = next((item[k] for k in ("wrong", "from", "original", "src") if k in item), None)
        right = next((item[k] for k in ("right", "to", "fixed", "dst") if k in item), None)
        if wrong is not None and right is not None:
            pages = item.get("pages") or item.get("page") or []
            if not isinstance(pages, list):
                pages = [pages]
            where = f"（第{'、'.join(map(str, pages))}页）" if pages else ""
            return f"「{wrong}」→「{right}」{where}"
        return "；".join(f"{k}：{v}" for k, v in item.items())
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"「{item[0]}」→「{item[1]}」"
    return str(item)


def _rule_line(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("rule", "text", "desc", "description"):
            if k in item:
                return str(item[k])
        return "；".join(f"{k}：{v}" for k, v in item.items())
    return str(item)


def profile_rules_text(profile: dict) -> str:
    """写作类节点看到的品牌档案切片：错字修正表 + 禁用规则。只作约束，不可引用。"""
    rules = profile.get("rules") or {}
    corrections = rules.get("corrections") or []
    if isinstance(corrections, dict):
        corrections = list(corrections.items())
    forbidden = rules.get("forbidden") or []

    out = ["### 品牌档案 · 写作规则（只作约束，不是事实来源，不可引用）", "", "错字修正表（只有这些可以修正）："]
    out += [f"- {_correction_line(c)}" for c in corrections] or ["- 无"]
    out += ["", "禁用规则（即使片段里出现也不得使用）："]
    out += [f"- {_rule_line(f)}" for f in forbidden] or ["- 无"]
    return "\n".join(out)
