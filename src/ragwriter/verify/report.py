"""校验报告：随文章一起输出，字段对齐 05 第 4.3 节。

报告是给人复核的：评审人不必信任系统，只要拿着数字清单逐条去原页对。
所以每个数字都要带「所在句 / 引用页 / 原文片段 / 是否一致」，而不是只给一个通过率。

各闸拦截次数要把改写、删除过程中的失败都算上——终稿里的句子大多已经过了，
只看终稿会显得闸门从没拦过东西，恰好掩盖了它最重要的工作。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ragwriter.contracts import SentenceVerdict, Snippet
from ragwriter.verify.checks import (
    snippet_material,
    is_factual,
    number_tokens_detail,
    token_in_text,
)
from ragwriter.verify.sentences import strip_cites

CHECKS = ("V1", "V2", "V3", "V4", "V5")


def _failed_checks(rec: dict) -> list[str]:
    """改写/删除记录里的失败检查。兼容 list 与「V2,V3」字符串，中英文键名都认。"""
    v = rec.get("failed_checks", rec.get("失败检查", []))
    if isinstance(v, str):
        v = [x.strip() for x in v.replace("，", ",").split(",")]
    return [x for x in v if x in CHECKS]


def _number_rows(v: SentenceVerdict, citation_map: dict[str, Snippet] | None) -> list[dict]:
    rows = []
    v2 = next((r for r in v.results if r.check == "V2"), None)
    cited = [citation_map[c] for c in v.sentence.cites if citation_map and c in citation_map]
    # 「据第 5 页图示」的页码不是事实数字，和 V2 同一口径剔除（否则报告虚报「数字不一致」）
    from ragwriter.verify.checks import _drop_page_refs
    body = _drop_page_refs(v.sentence.text, {s.page_no for s in cited})
    for tok in number_tokens_detail(body):
        if citation_map is None:
            # 没给映射表：页码与原文无从查起，一致性只能沿用 V2 的整句结论
            rows.append({
                "number": tok.raw, "sentence": v.sentence.text, "cites": v.sentence.cites,
                "pages": [], "snippet": "",
                "consistent": bool(v2 and v2.ok and v.sentence.cites),
            })
            continue
        hit = next((s for s in cited if token_in_text(tok, snippet_material(s))), None)
        rows.append({
            "number": tok.raw,
            "sentence": v.sentence.text,
            "cites": v.sentence.cites,
            "pages": sorted({s.page_no for s in cited}),
            "snippet": (hit or (cited[0] if cited else None)).text if cited else "",
            "consistent": hit is not None,
        })
    return rows


def build_report(
    verdicts: list[SentenceVerdict],
    *,
    rewrites: list[dict],
    deletions: list[dict],
    insufficient: list[dict],
    corrections: list[dict],
    citation_map: dict[str, Snippet] | None = None,
) -> dict[str, Any]:
    """汇总一篇文章的校验结果。

    verdicts      终稿里每一句的**最终**判定（被删除的句子不在这里）
    rewrites      [{"original", "failed_checks": ["V3"], "reason", "rewritten", "round"}]
                  每轮改写一条：一句改两轮就是两条
    deletions     [{"original", "failed_checks", "reason"}]  两轮改写仍不过、最终删除
    insufficient  [{"topic", "signal"}]  资料不足的子话题与触发信号
    corrections   [{"original", "corrected"}]  错字修正
    citation_map  可选。给了才能在数字清单里填引用页与原文片段；
                  多节文章的 sid 是请求内唯一的，传合并后的一张表即可

    拦截次数 = 终稿仍未通过的检查 + 每条改写记录的失败检查 + 每条删除记录的失败检查，
    即「每一次校验失败」都计一次。
    """
    gate = Counter({c: 0 for c in CHECKS})
    for v in verdicts:
        for r in v.results:
            if not r.ok:
                gate[r.check] += 1
    for rec in [*rewrites, *deletions]:
        for c in _failed_checks(rec):
            gate[c] += 1

    numbers = [row for v in verdicts for row in _number_rows(v, citation_map)]
    return {
        "sentence_total": len(verdicts),
        "factual_sentences": sum(1 for v in verdicts if is_factual(strip_cites(v.sentence.text))),
        "cited_sentences": sum(1 for v in verdicts if v.sentence.cites),
        "passed_sentences": sum(1 for v in verdicts if v.ok),
        "numbers": numbers,
        "numbers_inconsistent": sum(1 for r in numbers if not r["consistent"]),
        "rewrites": list(rewrites),
        "deletions": list(deletions),
        "insufficient": list(insufficient),
        "corrections": list(corrections),
        "gate_blocks": dict(gate),
        "failures": [
            {"sentence": v.sentence.text, "check": r.check, "reason": r.reason}
            for v in verdicts for r in v.results if not r.ok
        ],
    }
