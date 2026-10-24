"""qa 链路：查询改写 → 双路检索（改写后 + 原话）→ 充分性判断 → 短答 → 逐句校验改写。

查询改写只在有历史轮次时调用：首轮问题本来就是独立问题，省一次调用；
「那流量呢？」这种指代句一定有上一轮，必然触发改写。
保留用户原话作第二路查询，是防止改写节点把问题改偏——两路 RRF 合并，原意不会丢。
"""

from __future__ import annotations

import re

from typing import Iterator

from ragwriter.agent.handlers import (
    TurnContext,
    citations_for,
    applied_corrections,
    done_event,
    ev,
    insufficient_all_text,
    nearest_pages,
    note_retrieval,
    retrieval_payload,
    stream_text,
    verify_and_fix,
)
from ragwriter.contracts import Event


# 口头填充词。「介绍一下品宣里那个首饰供应商的案例」稠密余弦 0.576 < τ，
# 去掉填充词后「首饰供应商的案例」即判充分（10-22 真跑 D16#0 误拒）
_FILLER = re.compile(r"(请问|请|帮我|麻烦|介绍一下|讲一下|说说|聊聊|品宣里的?|资料里的?|那个|这个|一下)")


def strip_filler(q: str) -> str:
    return _FILLER.sub("", q).strip(" ，。？?")


def _pin_single_route_tops(d, queries: list[str], merged):
    """每路查询在 BM25 或稠密里排第一的事实，保底排到最前。

    10-23 实测「万悉是做什么的」：改写「万悉科技 核心产品 致力于」在 BM25 里把答案 p02-f13 排第 1，
    稠密却排第 13；两路等权 RRF 之后它被「万悉科技 (Trendee)」这类两路都沾边的泛条目压下去，
    五路查询再合并一次就掉出了上下文名额。改写本来就是冲着某条原文去的，它的单路第一最可能是答案。
    """
    pinned, seen = [], set()
    for q in queries:
        try:
            r = d.search(d.store, q, embedder=d.embedder)
        except Exception:  # noqa: BLE001 — 保底只是加分项，单路失败不影响合并结果
            continue
        for h in r.hits:
            if (h.bm25_rank == 1 or h.dense_rank == 1) and h.fact.fact_id not in seen:
                seen.add(h.fact.fact_id)
                pinned.append(h)
    if not pinned:
        return merged
    # 🔴 分数也要抬上去：下游 build_context 按 rrf_score 重排。单路 Hit 只有一个查询的分（≈0.03），
    # 合并结果是五路累加（≈0.16）——直接塞单路 Hit，保底条目重排后反而沉底、被名额挤掉
    # （10-23 评估 D06「AI 驱动订单增长多少」：第一名 p10-f02 就这样从上下文里消失）
    by_id = {h.fact.fact_id: h for h in merged.hits}
    top = max((h.rrf_score for h in merged.hits), default=0.0)
    boosted = [by_id.get(h.fact.fact_id, h).model_copy(update={"rrf_score": top + 1e-3 * (len(pinned) - i)})
               for i, h in enumerate(pinned)]
    rest = [h for h in merged.hits if h.fact.fact_id not in seen]
    return merged.model_copy(update={"hits": boosted + rest})


def handle(ctx: TurnContext) -> Iterator[Event]:
    d = ctx.deps
    message = ctx.message.strip()
    query = (ctx.intent.query or message).strip()
    recent = list(ctx.window.user_turns)
    rewritten = query
    # 模型做意图识别时已经把问题补全成独立问题，再调一次查询改写是重复劳动；规则兜底时才需要
    if recent and ctx.intent.source != "model":
        try:
            rewritten = (d.rewrite_query(d.client, message, ctx.slots, recent) or query).strip()
        except Exception as exc:  # noqa: BLE001 — 改写失败退回原问题，检索照常进行
            yield ev("error", stage="rewrite_query", reason=str(exc), fallback=False)
            rewritten = query
    ctx.intent.query = rewritten

    queries = list(dict.fromkeys(q for q in (rewritten, *ctx.intent.search_queries, message,
                                            strip_filler(rewritten)) if q))
    result = note_retrieval(ctx, _pin_single_route_tops(d, queries,
                                                        d.search_many(d.store, queries, embedder=d.embedder)))
    near = nearest_pages([result])
    focus = ctx.intent.focus or rewritten

    # 🔴 10-23 改：问答不再用检索分数在生成前拒答。检索相似度衡量的是「话题相关」而不是「能不能答」：
    # 「15倍那个是流量吧」两路第一名都是第 10 页正确事实，却因口语化余弦 0.568 < τ 被拒；
    # 反过来「两路一致即充分」又会放过员工数、办公地址这类陷阱（标定集 4/13）。
    # 能不能答交给读过片段的作答模型判断，它说「品宣资料未涉及」即拒答（双信号第二路），
    # 编没编由 V1–V5 拦。写作链路不受影响，仍按子问题逐节判 τ。
    if not result.hits:
        yield ev("retrieval", **retrieval_payload(result, section=1, kind="answer"),
                 queries=queries, rewritten=rewritten)
        signal = result.signal or "below_threshold"
        yield ev("insufficient", section=1, topic=rewritten, signal=signal, nearest_pages=near)
        text = insufficient_all_text(focus, near, verb="回答")
        yield ev("delta", section=1, kind="notice", text=text)
        ctx.outcome.update(kind="refuse", outline=[f"拒答：{rewritten}"], pages=near, final_text=text)
        yield done_event(ctx, final_text=text, refused=True, nearest=near, query=rewritten,
                         refusal_kind="insufficient_all",
                         boundary=[{"topic": rewritten, "source": "retrieval", "nearest_pages": near}],
                         report={"insufficient": [{"topic": rewritten, "signal": signal}]})
        return

    pack = d.build_context(rewritten, result, ctx.page_facts, sid_start=1,
                           max_snippets=int(ctx.options.get("max_snippets", 16 if ctx.intent.answer_shape == "list" else 10)))
    # ↑ 列举要整页列全：第 30 页 10 个服务行业，10 条名额只答出 6 个（10-23 实测）
    yield ev("retrieval", **retrieval_payload(result, section=1, kind="answer",
                                              sids={s.sid: s.fact_id for s in pack.snippets}),
             queries=queries, rewritten=rewritten,
             snippets=[{"sid": s.sid, "fact_id": s.fact_id, "page_no": s.page_no} for s in pack.snippets])

    draft = yield from stream_text(ctx, d.stream_answer(d.client, rewritten, pack, profile=ctx.profile,
                                                        shape=ctx.intent.answer_shape),
                                   section=1, kind="answer")
    block = yield from verify_and_fix(ctx, draft, section=1, kind="answer", snippets=pack.snippets)
    by_sid = {s.sid: s for s in pack.snippets}
    cites = citations_for(block.sentences, by_sid)
    yield ev("section_final", section=1, kind="answer", heading="", text=block.text,
             sentences=[s.model_dump() for s in block.sentences], citations=cites, skipped=False)

    final_text = block.text
    if ctx.intent.answer_shape == "list" and len(block.sentences) > 1:
        # 校验按句拆开再拼回会丢掉条目符号；列举类按条目重排，引用留在每条末尾
        final_text = "\n".join(f"- {s.text}" for s in block.sentences)
    refused = False
    if not block.sentences:
        # 每句都被删光：说明片段撑不起任何一句答案，按「资料不足」收口，而不是给空白
        final_text = insufficient_all_text(focus, near, verb="回答")
        refused = True
    elif "品宣资料未涉及" in final_text:
        # 双信号拒答的第二路（05 第 3.2 节）：检索判充分，但作答模型读完片段明说资料没有。
        # 真跑：价格、员工数、竞品对比都是这样——正文是对的，只是没打上拒答标记
        refused = True
    report = d.build_report(block.verdicts, rewrites=block.rewrites, deletions=block.deletions,
                            insufficient=[], corrections=applied_corrections(ctx.profile, final_text, pack.snippets),
                            citation_map=by_sid)
    body = {"question": message, "query": rewritten,
            "sentences": [s.model_dump() for s in block.sentences]}
    artifact_id = ctx.memory.save_artifact(session_id=ctx.session_id, turn_id=ctx.turn_id, genre="qa",
                                           topic=rewritten, body=body, citations=cites, verify=report)
    pages = sorted({c["page_no"] for c in cites.values()})
    ctx.outcome.update(kind="qa", artifact_id=artifact_id, outline=[f"问：{rewritten}"], pages=pages,
                       final_text=final_text)
    yield done_event(ctx, final_text=final_text, artifact_id=artifact_id, citations=cites, report=report,
                     refused=refused, nearest=near if refused else [], query=rewritten,
                     refusal_kind="insufficient_all" if refused else None)

