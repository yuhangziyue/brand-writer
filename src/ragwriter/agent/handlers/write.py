"""write 链路：规划 → 逐节检索 → 充分性判断 → 逐节流式写作 → 逐句校验改写 → 导语结尾 → 报告。

检索放在写作之前一次做完，而不是写一节检索一节：
「全部小节资料不足 ⇒ 不生成文章」必须在吐出第一个字之前就判定，
否则前端已经显示了半篇草稿再撤回，比直接拒答更糟。代价只是 retrieval 事件集中在前面。

资料边界（评审 M1）：资料不足**不写进正文**。整节检索不足 ⇒ 该节不写；节内缺口由写作节点在末尾
给 `GAPS: [...]`，程序剥离。三处来源（plan / retrieval / writer）汇总进 done.boundary，
前端渲染成文末「资料边界卡」——正文里只留有出处的句子。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator

from ragwriter.agent.handlers import (
    TurnContext,
    VerifiedBlock,
    applied_corrections,
    citations_for,
    done_event,
    ev,
    insufficient_all_text,
    nearest_pages,
    retrieval_payload,
    note_retrieval,
    split_gaps_fallback,
    stream_text,
    trim_leak,
    verify_and_fix,
)
from ragwriter.contracts import ContextPack, Event, Plan, RetrievalResult, Sentence, Snippet

GENRE_LABEL = {"blog": "官网 Blog", "faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}


def handle(ctx: TurnContext) -> Iterator[Event]:
    d = ctx.deps
    genre = ctx.intent.genre or "blog"
    topic = ctx.intent.topic or ""
    constraints = tuple(dict.fromkeys([*(ctx.slots.get("constraints") or []), *ctx.intent.constraints]))
    max_snippets = int(ctx.options.get("max_snippets", 14))  # 10 时第 9 页全球数据排第 11/12 被截，见 10-22 真跑
    split_gaps = getattr(d, "split_gaps", None) or split_gaps_fallback

    # ① 规划
    plan: Plan = d.plan(d.client, topic, genre, ctx.profile, constraints)
    yield ev("plan", title=plan.title, genre=genre, topic=topic,
             outline=[s.heading for s in plan.sections],
             sub_questions=[s.sub_question for s in plan.sections],
             expected_pages=[s.expected_pages_hint for s in plan.sections],
             insufficient=list(plan.insufficient),
             lede_points=list(plan.lede_points), ending_points=list(plan.ending_points))

    boundary: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []  # 给 build_report：{topic, signal}
    for item in plan.insufficient:
        boundary.append({"topic": item, "source": "plan", "nearest_pages": []})
        insufficient.append({"topic": item, "signal": "planner"})
        yield ev("insufficient", section=None, topic=item, signal="planner", source="plan", nearest_pages=[])

    # ② 逐节检索 + 充分性判断 + 上下文（sid 跨节连续，全文共用一张映射表）
    packs: list[ContextPack | None] = []
    results: list[RetrievalResult] = []
    next_sid = 1
    for i, sec in enumerate(plan.sections, 1):
        result: RetrievalResult = note_retrieval(ctx, d.search(d.store, sec.sub_question, embedder=d.embedder))
        results.append(result)
        if not result.sufficient:
            yield ev("retrieval", **retrieval_payload(result, section=i, kind="body"))
            near = nearest_pages([result])
            signal = result.signal or "below_threshold"
            boundary.append({"topic": sec.sub_question, "source": "retrieval", "nearest_pages": near,
                             "section": i, "heading": sec.heading})
            insufficient.append({"topic": sec.sub_question, "signal": signal})
            yield ev("insufficient", section=i, topic=sec.sub_question, signal=signal, source="retrieval",
                     nearest_pages=near)
            packs.append(None)
            continue
        pack: ContextPack = d.build_context(sec.sub_question, result, ctx.page_facts,
                                            sid_start=next_sid, max_snippets=max_snippets)
        next_sid += len(pack.snippets)
        sids = {s.sid: s.fact_id for s in pack.snippets}
        yield ev("retrieval", **retrieval_payload(result, section=i, kind="body", sids=sids),
                 snippets=[{"sid": s.sid, "fact_id": s.fact_id, "page_no": s.page_no} for s in pack.snippets])
        packs.append(pack)

    if not plan.sections or all(p is None for p in packs):
        near = nearest_pages(results)
        text = insufficient_all_text(topic, near, verb="写作")
        yield ev("delta", section=0, kind="notice", text=text)
        ctx.outcome.update(kind="refuse", outline=["资料不足，未生成"], pages=near, final_text=text)
        yield done_event(ctx, final_text=text, refused=True, nearest=near, refusal_kind="insufficient_all",
                         boundary=boundary, report={"insufficient": insufficient},
                         reason="all_sections_insufficient")
        return

    # ③ 逐节写作 + 校验
    all_snippets: dict[str, Snippet] = {}
    verdicts: list = []
    rewrites: list = []
    deletions: list = []
    sections_out: list[dict[str, Any]] = []
    inject = ctx.options.get("inject_fault") == "v3_swap"
    for i, (sec, pack) in enumerate(zip(plan.sections, packs), 1):
        if pack is None:
            # 整节不写，也不调模型：杜绝模型「好心」补事实；前端据 skipped 把大纲里这一节收起
            sections_out.append({"section": i, "heading": sec.heading, "text": "", "sentences": [],
                                 "citations": {}, "skipped": True, "pages": []})
            yield ev("section_final", section=i, kind="body", heading=sec.heading, text="",
                     sentences=[], citations={}, skipped=True)
            continue
        for s in pack.snippets:
            all_snippets[s.sid] = s
        chunks: Iterable[str] = d.stream_section(d.client, sec, pack, genre=genre, profile=ctx.profile,
                                                 constraints=constraints)
        if inject:
            chunks = _inject_v3_swap(ctx, chunks, pack)
        raw = yield from stream_text(ctx, chunks, section=i, kind="body")
        if ctx.injected:
            # 只注入一次（第一个有同页数字配对的小节）：演示一次拦截就够，多了像系统本身不可靠
            inject = False
        draft, gaps = split_gaps(raw)
        yield from trim_leak(ctx, draft, section=i, kind="body")
        pack_pages = sorted({s.page_no for s in pack.snippets})
        for g in gaps:
            boundary.append({"topic": g, "source": "writer", "nearest_pages": pack_pages[:3],
                             "section": i, "heading": sec.heading})
            insufficient.append({"topic": g, "signal": "writer"})
            yield ev("insufficient", section=i, topic=g, signal="writer", source="writer",
                     nearest_pages=pack_pages[:3])
        block: VerifiedBlock = yield from verify_and_fix(ctx, draft, section=i, kind="body",
                                                         snippets=pack.snippets)
        verdicts += block.verdicts
        rewrites += block.rewrites
        deletions += block.deletions
        cites = citations_for(block.sentences, all_snippets)
        sec_out = {"section": i, "heading": sec.heading, "text": block.text,
                   "sentences": [s.model_dump() for s in block.sentences], "citations": cites,
                   "skipped": False, "pages": sorted({c["page_no"] for c in cites.values()})}
        sections_out.append(sec_out)
        yield ev("section_final", section=i, kind="body", heading=sec.heading, text=block.text,
                 sentences=sec_out["sentences"], citations=cites, skipped=False)

    # ④ 导语与结尾：只能基于已过校验的小节，同样逐句校验
    snippet_list = list(all_snippets.values())
    verified_texts = [s["text"] for s in sections_out]  # 与 plan.sections 一一对齐，跳过的节是空串
    bookends: dict[str, VerifiedBlock] = {}
    for kind, sec_no in (("lede", 0), ("ending", len(plan.sections) + 1)):
        try:
            chunks = d.stream_bookend(d.client, kind, plan, verified_texts, genre=genre, profile=ctx.profile)
            raw = yield from stream_text(ctx, chunks, section=sec_no, kind=kind)
        except Exception as exc:  # noqa: BLE001 — 导语结尾失败不拖垮正文，正文已逐节定稿
            yield ev("error", stage=kind, reason=str(exc), fallback=False)
            continue
        draft, _ = split_gaps(raw)
        yield from trim_leak(ctx, draft, section=sec_no, kind=kind)
        if not draft.strip():
            continue
        block = yield from verify_and_fix(ctx, draft, section=sec_no, kind=kind, snippets=snippet_list)
        verdicts += block.verdicts
        rewrites += block.rewrites
        deletions += block.deletions
        bookends[kind] = block
        yield ev("section_final", section=sec_no, kind=kind, heading="", text=block.text,
                 sentences=[s.model_dump() for s in block.sentences],
                 citations=citations_for(block.sentences, all_snippets), skipped=False)

    # ⑤ 汇总
    parts = [f"# {plan.title}"]
    if "lede" in bookends and bookends["lede"].text:
        parts.append(bookends["lede"].text)
    for s in sections_out:
        if not s["skipped"] and s["text"]:
            parts.append(f"## {s['heading']}\n\n{s['text']}")
    if "ending" in bookends and bookends["ending"].text:
        parts.append(bookends["ending"].text)
    final_text = "\n\n".join(parts)

    all_sentences = [s for b in bookends.values() for s in b.sentences]
    for s in sections_out:
        all_sentences += [Sentence(**x) for x in s["sentences"]]
    citations = citations_for(all_sentences, all_snippets)
    report = d.build_report(verdicts, rewrites=rewrites, deletions=deletions, insufficient=insufficient,
                            corrections=applied_corrections(ctx.profile, final_text, snippet_list),
                            citation_map=all_snippets)

    body = {
        "title": plan.title,
        "lede": [s.model_dump() for s in bookends["lede"].sentences] if "lede" in bookends else [],
        "sections": [{"heading": s["heading"], "sentences": s["sentences"], "skipped": s["skipped"]}
                     for s in sections_out],
        "ending": [s.model_dump() for s in bookends["ending"].sentences] if "ending" in bookends else [],
        "boundary": boundary,
    }
    artifact_id = ctx.memory.save_artifact(session_id=ctx.session_id, turn_id=ctx.turn_id, genre=genre,
                                           topic=topic, body=body, citations=citations, verify=report)
    outline = []
    for s in sections_out:
        pages = "、".join(str(p) for p in s["pages"])
        outline.append(s["heading"] + (f"（第{pages}页）" if pages else "（资料不足，未写）"))
    ref_pages = sorted({c["page_no"] for c in citations.values()})
    ctx.outcome.update(kind="write", artifact_id=artifact_id, outline=outline, pages=ref_pages,
                       final_text=final_text)
    yield done_event(ctx, final_text=final_text, artifact_id=artifact_id, citations=citations, report=report,
                     refusal_kind="partial" if boundary else None, boundary=boundary,
                     title=plan.title, genre=genre, topic=topic,
                     summary={"sentences": len(verdicts), "rewrites": len(rewrites),
                              "deletions": len(deletions), "boundary": len(boundary)})


# ── 演示注入（评审 M2）────────────────────────────────────
def _metric_pair(ctx: TurnContext, pack: ContextPack) -> tuple[Any, Any, str] | None:
    """在本节所引的页里找一对同页、不同值的数字配对，返回 (a, b, a 所在片段的 sid)。"""
    by_page: dict[int, str] = {}
    for s in pack.snippets:
        by_page.setdefault(s.page_no, s.sid)
    for page, sid in by_page.items():
        metrics = [m for m in ctx.page_metrics(page) if m is not None and m.value and m.label]
        for a in metrics:
            for b in metrics:
                if a.value != b.value and a.label != b.label:
                    return a, b, sid
    return None


def _inject_v3_swap(ctx: TurnContext, chunks: Iterable[str], pack: ContextPack) -> Iterator[str]:
    """把模型输出里的一个数字配对对调（如「15倍」挂到「流量」上），专门演示 V3 拦截。

    为了做对调必须先拿到整节文本，所以注入时本节的流式是「攒完再切片」；默认关闭，只用于演示。
    草稿里没出现这对数字时，追加一句「{a 的标签}{b 的值}[a 的片段]」——这正是 A5 的构造。
    """
    text = "".join(chunks)
    pair = _metric_pair(ctx, pack)
    if pair is not None:
        a, b, sid = pair
        m = re.search(r"(?:^|\n)[ \t]*GAPS[ \t]*[:：]", text)
        body, tail = (text[:m.start()], text[m.start():]) if m else (text, "")
        new = body
        if a.value in body and b.value in body:
            new = body.replace(a.value, "\0").replace(b.value, a.value).replace("\0", b.value)
        elif a.value in body:
            new = body.replace(a.value, b.value, 1)
        elif b.value in body:
            new = body.replace(b.value, a.value, 1)
        else:
            new = body.rstrip() + f"{a.label}{b.value}[{sid}]。"
        if new != body:
            ctx.injected = True
            text = new + tail
    for k in range(0, len(text), 12):
        yield text[k:k + 12]
