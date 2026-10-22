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


def handle(ctx: TurnContext) -> Iterator[Event]:
    d = ctx.deps
    message = ctx.message.strip()
    query = (ctx.intent.query or message).strip()
    recent = list(ctx.window.user_turns)
    rewritten = query
    if recent:
        try:
            rewritten = (d.rewrite_query(d.client, message, ctx.slots, recent) or query).strip()
        except Exception as exc:  # noqa: BLE001 — 改写失败退回原问题，检索照常进行
            yield ev("error", stage="rewrite_query", reason=str(exc), fallback=False)
            rewritten = query
    ctx.intent.query = rewritten

    queries = list(dict.fromkeys(q for q in (rewritten, message, strip_filler(rewritten)) if q))
    result = note_retrieval(ctx, d.search_many(d.store, queries, embedder=d.embedder))
    near = nearest_pages([result])

    if not result.sufficient:
        yield ev("retrieval", **retrieval_payload(result, section=1, kind="answer"),
                 queries=queries, rewritten=rewritten)
        signal = result.signal or "below_threshold"
        yield ev("insufficient", section=1, topic=rewritten, signal=signal, nearest_pages=near)
        text = insufficient_all_text(rewritten, near, verb="回答")
        yield ev("delta", section=1, kind="notice", text=text)
        ctx.outcome.update(kind="refuse", outline=[f"拒答：{rewritten}"], pages=near, final_text=text)
        yield done_event(ctx, final_text=text, refused=True, nearest=near, query=rewritten,
                         refusal_kind="insufficient_all",
                         boundary=[{"topic": rewritten, "source": "retrieval", "nearest_pages": near}],
                         report={"insufficient": [{"topic": rewritten, "signal": signal}]})
        return

    pack = d.build_context(rewritten, result, ctx.page_facts, sid_start=1,
                           max_snippets=int(ctx.options.get("max_snippets", 8)))
    yield ev("retrieval", **retrieval_payload(result, section=1, kind="answer",
                                              sids={s.sid: s.fact_id for s in pack.snippets}),
             queries=queries, rewritten=rewritten,
             snippets=[{"sid": s.sid, "fact_id": s.fact_id, "page_no": s.page_no} for s in pack.snippets])

    draft = yield from stream_text(ctx, d.stream_answer(d.client, rewritten, pack, profile=ctx.profile),
                                   section=1, kind="answer")
    block = yield from verify_and_fix(ctx, draft, section=1, kind="answer", snippets=pack.snippets)
    by_sid = {s.sid: s for s in pack.snippets}
    cites = citations_for(block.sentences, by_sid)
    yield ev("section_final", section=1, kind="answer", heading="", text=block.text,
             sentences=[s.model_dump() for s in block.sentences], citations=cites, skipped=False)

    final_text = block.text
    refused = False
    if not block.sentences:
        # 每句都被删光：说明片段撑不起任何一句答案，按「资料不足」收口，而不是给空白
        final_text = insufficient_all_text(rewritten, near, verb="回答")
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

