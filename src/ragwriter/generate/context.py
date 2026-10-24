"""上下文构建：一个子问题的检索结果 → 编号片段包（ContextPack）。

步骤（05-architecture 第 3.3 节）：
1. 命中子块按相关度排序，丢掉不可用状态——检索层理应已过滤，这里再兜一道，
   因为 image_only 进了上下文就可能被写进官网，代价远大于多一次判断。
2. 按页分组：同一页的命中合成一组，每页只回填一次（父页 = 引用单位）。
3. 回填：取所属页的其它可用事实，补齐命中子块缺的语境
   （第 9 页的「ChatGPT：61.66亿」单看不知道是什么，回填同页的「AI网站月访问量 Top 6」才完整）。
   预算 max_snippets 先给命中，再轮流给各页回填——不让一张事实密集的页把第二相关的页挤掉。
4. 页组排序：最相关放最前，次相关放最后，其余放中间，缓解长上下文的「中间遗忘」。
5. 按最终顺序连续编号 S{sid_start..}；同一次写作的各节用不同起点，S 编号在全文内唯一。
"""

from __future__ import annotations

from typing import Callable

from ragwriter.contracts import USABLE_STATUSES, ContextPack, Hit, RetrievalResult, Snippet
from ragwriter.models import Fact

EMPTY_NOTE = "（本子问题没有检索到可用片段）"


def build_context(
    sub_question: str,
    result: RetrievalResult,
    page_facts: Callable[[int], list[Fact]],
    *,
    sid_start: int = 1,
    max_snippets: int = 10,
) -> ContextPack:
    groups = _select(result.hits, page_facts, max_snippets)
    ordered = _edges_first(groups)

    snippets: list[Snippet] = []
    for facts in ordered:
        for f in facts:
            snippets.append(Snippet(
                sid=f"S{sid_start + len(snippets)}",
                fact_id=f.fact_id,
                page_no=f.page_no,
                text=f.text,
                status=f.status,
                metric=f.metric,
            ))
    titles = {f.page_no: _title_of(f.prefix) for facts in ordered for f in facts}
    return ContextPack(sub_question=sub_question, snippets=snippets,
                       rendered=render(sub_question, snippets, titles=titles))


def _title_of(prefix: str) -> str:
    """prefix 形如「第2页 · 封面与总览 · 我们的愿景：让好的品牌进入AI的深度记忆」，取第三段。"""
    parts = prefix.split(" · ", 2)
    return parts[2].strip() if len(parts) == 3 else ""


def render(sub_question: str, snippets: list[Snippet], *, titles: dict[int, str] | None = None) -> str:
    """按页分组，每组前加一行页标题。

    检索时事实的索引文本含 prefix（页码 · 章节 · 页标题），送进模型的却只有正文：
    第 2 页命中的是「万悉科技 (Trendee)」，模型看不到这页标题就是「我们的愿景：……」。
    把页标题带进上下文，模型才知道每条片段在讲什么（10-23 业务实测：问愿景答未涉及）。
    """
    lines = [f"## 子问题：{sub_question}"]
    last = None
    for s in snippets:
        if titles and s.page_no != last and titles.get(s.page_no):
            lines.append(f"### 第{s.page_no}页《{titles[s.page_no]}》")
        last = s.page_no
        lines.append(render_snippet(s))
    if not snippets:
        lines.append(EMPTY_NOTE)
    return "\n".join(lines)


def render_snippet(s: Snippet) -> str:
    """一行一个片段：「[S4 | 第9页] 文本」。

    image_structural 标「图示」：结构关系只在页面图示里，提示词要求引用时写「据第 N 页图示」。
    metric 一律显式带出处口径，没有口径也写「未注明」——空着不写，模型会自己补一个出处。
    """
    tag = f"{s.sid} | 第{s.page_no}页" + (" · 图示" if s.status == "image_structural" else "")
    text = s.text
    if s.metric is not None:
        text += f"〔出处口径：{s.metric.source_note or '未注明'}〕"
    return f"[{tag}] {text}"


# ── 内部 ──────────────────────────────────────────────────────

def _score(h: Hit) -> float:
    return h.rerank_score if h.rerank_score is not None else h.rrf_score


def _select(hits: list[Hit], page_facts: Callable[[int], list[Fact]], budget: int) -> list[list[Fact]]:
    """返回按相关度排序的页组，每组 = 该页入选的事实（命中在前，回填在后）。"""
    groups: dict[int, list[Fact]] = {}  # 插入顺序 = 页的相关度顺序
    seen_ids: set[str] = set()
    seen_text: set[tuple[int, str]] = set()  # 同页同文（如第 5 页三次「品牌Brands」）只留一条
    total = 0

    def take(f: Fact) -> bool:
        key = (f.page_no, "".join(f.text.split()))
        if f.status not in USABLE_STATUSES or f.fact_id in seen_ids or key in seen_text:
            return False
        if f.kind == "point" and len(key[1]) < 6:
            # 版面碎片（「中国经营报」「部分媒体」「行业认可」）不进上下文：没有可引用的信息，
            # 却在 8–10 条名额里占一半（10-23 实测「万悉是做什么的」）。数字配对不受影响（kind=metric）
            return False
        seen_ids.add(f.fact_id)
        seen_text.add(key)
        groups.setdefault(f.page_no, []).append(f)
        return True

    for h in sorted(hits, key=_score, reverse=True):  # sorted 稳定：同分保持检索层顺序
        if total >= budget:
            break
        total += take(h.fact)

    # 每页只取一次回填候选；轮流各页取一条，直到预算用完或各页取尽
    queues = {p: list(page_facts(p)) for p in groups}
    while total < budget and any(queues.values()):
        for p in list(groups):
            q = queues[p]
            while q and total < budget:
                if take(q.pop(0)):
                    total += 1
                    break
    return list(groups.values())


def _edges_first(groups: list[list[Fact]]) -> list[list[Fact]]:
    """[1,2,3,4,5] → [1,3,5,4,2]：第 1 相关在最前，第 2 相关在最后。"""
    front = groups[0::2]
    back = groups[1::2]
    return front + back[::-1]
