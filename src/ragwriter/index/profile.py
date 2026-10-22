"""L1 长期上下文：品牌档案（10-agent-architecture.md 第 2.2 节）。

档案只作导航，**不可被引用**：意图识别看 entities，规划看 catalog，写作看 rules。

规则（错字修正、禁用项）**从 docs/structured-content.md 程序抽取**，不在代码里手抄第二份。
手抄的话，文档改了代码不会跟着改，两份规则会悄悄漂移；而规则漏一条，
写作节点就可能把「思维连」原样写进正文，或把第 31 页的客户站点名写出来。
所以抽不到时直接报错，不静默给空——空规则表比报错危险得多。
"""

from __future__ import annotations

import re
from pathlib import Path

from ragwriter.contracts import USABLE_STATUSES
from ragwriter.models import Page

STRUCTURED_DOC = "structured-content.md"

# 实体表是少数几个小常量。别名来自第 2 页（中英两版人名并列出现），
# 量小且稳定，从文档里抽反而要写一堆脆弱的正则
ENTITIES: list[dict] = [
    {"name": "万悉科技", "aliases": ["Trendee", "万悉"], "type": "company"},
    {"name": "毛慧娜", "aliases": ["Huina Mao"], "type": "person", "role": "创始人 & CEO", "page_no": 2},
    {"name": "帅鑫", "aliases": ["Xin Shuai"], "type": "person", "role": "联合创始人 & CTO", "page_no": 2},
    {"name": "LLM-原生GEO", "aliases": ["LLM原生GEO"], "type": "product"},
]

_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")


def _clean(md: str) -> str:
    """去掉 markdown 强调、反引号、引用符和标记 emoji，留可读的纯文本。"""
    s = re.sub(r"\*\*|`|[\U0001F300-\U0001FAFF]", "", md)
    s = re.sub(r"^\s*>\s?", "", s, flags=re.M)
    return re.sub(r"\s+", " ", s).strip()


# 匿名约束原文里举了被禁用的具体值（站点名、会话数、收入）。档案会进写作提示词，
# 把这些值写进「禁止使用 X」反而等于把 X 递给了模型——只留类别名，抹掉具体值
_REDACT = re.compile(r"(站点名|会话数|收入)\s*[\w$.,]+")


def _section(doc: str, heading_re: str) -> str:
    """取某个标题下、到下一个同级或更高级标题前的正文。"""
    m = re.search(heading_re, doc, flags=re.M)
    if not m:
        raise ValueError(f"{STRUCTURED_DOC} 里找不到标题：{heading_re}")
    level = len(re.match(r"#+", doc[m.start():]).group(0))
    rest = doc[m.end():]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.M)
    return rest[: nxt.start()] if nxt else rest


def _blockquotes(section: str) -> list[str]:
    """段落级的引用块（连续的 > 行合成一段）。"""
    out, cur = [], []
    for line in section.splitlines():
        if line.lstrip().startswith(">"):
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out


def _pages_of(cell: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", cell)]


def _parse_defects(doc: str) -> tuple[list[dict], list[dict]]:
    """「附：源文件缺陷汇总」表 → (corrections, forbidden)。

    分类看「处理规则」列：含「不得」的是禁用；含「修正为 / 理顺为」的是错字修正，
    原文取「问题」列引号里的词，改法取「处理规则」列引号里的词。其余行
    （如第 6 页「照原样记录」）只是说明，写作节点用不到，不进档案。
    """
    sec = _section(doc, r"^## 附：源文件缺陷汇总")
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in sec.splitlines()
        if line.strip().startswith("|")
    ]
    # 跳过表头与分隔行
    rows = [r for r in rows if len(r) >= 3 and not set(r[0]) <= set("-: ") and r[0] != "页"]
    corrections, forbidden = [], []
    for page_cell, problem, rule in rows[:]:
        pages = _pages_of(page_cell)
        if not pages:
            continue
        if "不得" in rule:
            forbidden.append({"pages": pages, "rule": _clean(f"{problem}：{rule}")})
        elif re.search(r"修正为|理顺为", rule):
            wrong = _QUOTED.findall(problem)
            right = _QUOTED.findall(rule)
            if not wrong or not right:
                raise ValueError(f"缺陷表这一行抽不出原文/改法：{page_cell} | {problem} | {rule}")
            corrections.append(
                {"pages": pages, "wrong": wrong[0], "right": right[0], "note": _clean(rule)}
            )
    return corrections, forbidden


def _page_constraint(doc: str, page_no: int, marker: str) -> dict:
    """某页小节里含 marker 的引用块 → 一条禁用规则。"""
    sec = _section(doc, rf"^### p{page_no} · ")
    for q in _blockquotes(sec):
        if marker in q:
            return {"pages": [page_no], "rule": _REDACT.sub(r"\1", _clean(q))}
    raise ValueError(f"{STRUCTURED_DOC} 第 {page_no} 页没有含「{marker}」的约束说明")


def build_profile(pages: list[Page], docs_dir: Path) -> dict:
    """生成品牌档案。约 900 token，整份可以直接放进规划节点的提示词。"""
    path = Path(docs_dir) / STRUCTURED_DOC
    if not path.exists():
        raise FileNotFoundError(f"品牌档案的规则来源不存在：{path}")
    doc = path.read_text(encoding="utf-8")

    corrections, forbidden = _parse_defects(doc)
    # 第 31 页匿名约束、第 30 页「出海仅此一处」不在缺陷表里，写在对应页的引用块中
    forbidden.append(_page_constraint(doc, 31, "匿名约束"))
    forbidden.append(_page_constraint(doc, 30, "仅此一处"))

    # 每类至少要有东西：缺陷表格式一改，宁可建库失败也不给写作节点一张空规则表
    if not corrections:
        raise ValueError("缺陷汇总表里没抽到任何错字修正")
    if len(forbidden) < 3:
        raise ValueError(f"禁用规则只抽到 {len(forbidden)} 条，缺陷表或页约束格式可能变了")

    catalog = [
        {"page_no": p.page_no, "chapter": p.chapter, "title": p.title}
        for p in sorted(pages, key=lambda p: p.page_no)
    ]
    return {
        "entities": ENTITIES,
        "catalog": catalog,
        "rules": {
            "corrections": corrections,
            "forbidden": forbidden,
            "status_policy": {
                "usable": list(USABLE_STATUSES),
                "excluded": ["image_only", "unverified"],
                "image_structural": "关系只在页面图示里，引用时写「据第N页图示」",
                "image_only": "数字或正文只在截图里，文字层无法核验，默认不进上下文",
                "unverified": "两条解析路径冲突，待 V3 区分词判定，默认不进上下文",
            },
        },
    }
