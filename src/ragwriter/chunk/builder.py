"""T2 切分：父页 / 子事实，外加确定性上下文前缀。

父 = 一页。页既是主题单位，也是引用单位——引用溯源要落到页，
读者点一下就能回到原页核对。

子 = 一条要点或一个数字配对。检索命中的是子，送进上下文时回填父页。

前缀由页面结构直接拼出，**不调模型**。Anthropic 的 Contextual Retrieval
是让模型给每个 chunk 写一段上下文说明，这里不需要：
幻灯片的「第几页 · 属于哪章 · 页标题」本来就是确定的，拼出来就是最好的前缀，
既不花钱也不会漂移。
"""

from __future__ import annotations

from ragwriter.models import Fact, Metric, Page

MAX_FACT_CHARS = 400


def page_prefix(page: Page) -> str:
    return f"第{page.page_no}页 · {page.chapter} · {page.title}"


def _metric_text(m: Metric) -> str:
    base = f"{m.label}：{m.value}" if m.label else m.value
    return f"{base}（{m.source_note}）" if m.source_note else base


def build_facts(
    page: Page,
    checked: dict,
    *,
    skip_dividers: bool = True,
) -> list[Fact]:
    """把一页的校验结果变成子 chunk。

    checked 来自 crosscheck.check_page_card：{"metrics": [(Metric, verdict)],
    "points": [(str, verdict)]}。verdict 前缀即状态。
    """
    if skip_dividers and page.is_divider:
        return []  # 纯章节页没有信息量，进了索引只会制造噪声

    prefix = page_prefix(page)
    facts: list[Fact] = []
    seq = 0

    def _status(verdict: str) -> str:
        return verdict.split(":", 1)[0]  # "verified:双路一致" -> "verified"

    for metric, verdict in checked.get("metrics", []):
        seq += 1
        facts.append(
            Fact(
                fact_id=f"p{page.page_no:02d}-f{seq:02d}",
                page_no=page.page_no,
                kind="metric",
                text=_metric_text(metric)[:MAX_FACT_CHARS],
                prefix=prefix,
                status=_status(verdict),
                metric=metric,
            )
        )

    for text, verdict in checked.get("points", []):
        text = text.strip()
        if len(text) < 4:
            continue  # 「痛点」「解决」这类版面装饰词，单独成块没有检索价值
        seq += 1
        facts.append(
            Fact(
                fact_id=f"p{page.page_no:02d}-f{seq:02d}",
                page_no=page.page_no,
                kind="point",
                text=text[:MAX_FACT_CHARS],
                prefix=prefix,
                status=_status(verdict),
            )
        )
    return facts


def attach_discriminators(facts: list[Fact]) -> None:
    """为同页的数字配对标签算「区分词」，供生成期的 V3 闸使用。

    同一页上「AI驱动流量增长」与「AI驱动订单增长」的共有词是
    「AI」「驱动」「增长」，各自的区分词就是「流量」与「订单」。
    句子里说 15 倍却只出现「流量」，就是数字对、对象错。
    """
    by_page: dict[int, list[Fact]] = {}
    for f in facts:
        if f.kind == "metric" and f.metric and f.metric.label:
            by_page.setdefault(f.page_no, []).append(f)

    for siblings in by_page.values():
        if len(siblings) < 2:
            continue
        charsets = [set(f.metric.label) for f in siblings]  # type: ignore[union-attr]
        for i, f in enumerate(siblings):
            others: set[str] = set()
            for j, cs in enumerate(charsets):
                if i != j:
                    others |= cs
            unique = charsets[i] - others
            label = f.metric.label  # type: ignore[union-attr]
            # 按原标签顺序把区分字连成词片段，便于在句子里做子串判断
            frags, cur = [], ""
            for ch in label:
                if ch in unique:
                    cur += ch
                elif cur:
                    frags.append(cur)
                    cur = ""
            if cur:
                frags.append(cur)
            f.metric.discriminators = [x for x in frags if len(x) >= 1]  # type: ignore[union-attr]
