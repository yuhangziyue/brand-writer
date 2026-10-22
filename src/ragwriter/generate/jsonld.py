"""GEO 原生输出：FAQ → schema.org FAQPage JSON-LD。

FAQ 的每组问答本来就是 AI 搜索最容易整段引用的单元，再配上结构化标记，
就是品宣第 24 页说的「让内容符合 AI 的理解与引用标准」——这是系统自己的产出也在践行 GEO。

纯模板、零模型调用：输入是已经通过校验的问答，这里只做格式转换，不可能引入新事实。
[S编号] 角标是给审核人看的内部引用，发布到页面的结构化数据里要去掉。
"""

from __future__ import annotations

import re
from typing import Any

_CITE_RE = re.compile(r"\s*\[S\d+\]")


def strip_cites(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", _CITE_RE.sub("", text)).strip()


def faq_jsonld(qa_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """生成 FAQPage。问题或答案去角标后为空的组跳过——空答案的 Question 是无效标记。"""
    entities = []
    for q, a in qa_pairs:
        q, a = strip_cites(q), strip_cites(a)
        if not q or not a:
            continue
        entities.append({
            "@type": "Question",
            "name": q,
            "acceptedAnswer": {"@type": "Answer", "text": a},
        })
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entities}
