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
    parse_cites,
    retrieval_payload,
    note_retrieval,
    split_gaps_fallback,
    stream_text,
    trim_leak,
    verify_and_fix,
)
from ragwriter.contracts import ContextPack, Event, Plan, RetrievalResult, Sentence, SentenceVerdict, Snippet
from ragwriter.generate.topic import brief_text

GENRE_LABEL = {"blog": "官网 Blog", "faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}


def handle(ctx: TurnContext) -> Iterator[Event]:
    d = ctx.deps
    genre = ctx.intent.genre or "blog"
    topic = ctx.intent.topic or ""
    constraints = tuple(dict.fromkeys([*(ctx.slots.get("constraints") or []), *ctx.intent.constraints,
                                       *brief_text(ctx.options.get("confirmed_brief"))]))
    is_faq = genre == "faq"
    is_intro = genre in ("brand_intro", "product_intro")
    max_snippets = int(ctx.options.get("max_snippets", 14))  # 10 时第 9 页全球数据排第 11/12 被截，见 10-22 真跑
    split_gaps = getattr(d, "split_gaps", None) or split_gaps_fallback

    # ① 规划
    plan: Plan = d.plan(d.client, topic, genre, ctx.profile, constraints)
    yield ev("plan", title=plan.title, genre=genre, topic=topic,
             outline=[s.heading for s in plan.sections],
             sub_questions=[s.sub_question for s in plan.sections],
             expected_pages=[s.expected_pages_hint for s in plan.sections],
             purposes=[s.purpose for s in plan.sections], thesis=plan.thesis, products=list(plan.products),
             insufficient=list(plan.insufficient),
             lede_points=list(plan.lede_points), ending_points=list(plan.ending_points))

    boundary: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []  # 给 build_report：{topic, signal}
    brief = ctx.options.get("confirmed_brief") or {}
    if is_intro:
        plan.insufficient = []  # 介绍类按「有什么写什么」，资料里没有的节目静默跳过，不上边界卡
    elif brief:
        # 确认环节已经告诉用户「这部分资料里没有、文中不写」——边界卡就以它为准，规划不再另添（10-23）
        plan.insufficient = list(brief.get("premise_gaps") or [])
    for item in plan.insufficient:
        boundary.append({"topic": item, "source": "plan", "nearest_pages": []})
        insufficient.append({"topic": item, "signal": "planner"})
        yield ev("insufficient", section=None, topic=item, signal="planner", source="plan", nearest_pages=[])

    # ② 逐节检索 + 充分性判断 + 上下文（sid 跨节连续，全文共用一张映射表）
    packs: list[ContextPack | None] = []
    results: list[RetrievalResult] = []
    next_sid = 1
    scope = _product_facts(ctx, plan) if genre == "product_intro" else set()
    page_facts = ctx.page_facts if not scope else (lambda p: [f for f in ctx.page_facts(p) if f.fact_id in scope])
    for i, sec in enumerate(plan.sections, 1):
        result: RetrievalResult = note_retrieval(ctx, d.search(d.store, sec.sub_question, embedder=d.embedder))
        boosted = _boost_hint_pages(ctx, sec, result, scope)
        if is_intro and not boosted.sufficient and len(boosted.hits) > len(result.hits):
            # 介绍类：规划判断的页上取到了与本节字面相关的事实，就算资料充分（10-23）。
            # 「联系方式」检索问句「官网 邮箱」稠密余弦只有 0.52，可第 36 页的官网与邮箱原文就在那里；
            # τ 衡量的是语义相似，判不了「这一页就是封底联系方式」。Blog 仍按 τ 判——论证要实打实的支撑
            boosted = boosted.model_copy(update={"sufficient": True, "signal": "planner_hint"})
        result = boosted
        if scope:
            # 产品介绍只给选中产品的资料（10-23）：监控系统在品宣里只有第 27、28 页，结构却要写满六项，
            # 不圈页的话模型会把 OPC 模式的功能、项目交付的计费、别的案例效果说成监控系统的
            result = result.model_copy(update={"hits": [h for h in result.hits if h.fact.fact_id in scope]})
        results.append(result)
        if not result.sufficient or not result.hits:
            yield ev("retrieval", **retrieval_payload(result, section=i, kind="body"))
            near = nearest_pages([result])
            signal = result.signal or "below_threshold"
            if is_intro:
                # 介绍类按「有什么写什么」：资料里没有的节目直接不写，不上资料边界卡（10-23 子曰定）
                insufficient.append({"topic": sec.sub_question, "signal": signal})
                packs.append(None)
                continue
            boundary.append({"topic": sec.sub_question, "source": "retrieval", "nearest_pages": near,
                             "section": i, "heading": sec.heading})
            insufficient.append({"topic": sec.sub_question, "signal": signal})
            yield ev("insufficient", section=i, topic=sec.sub_question, signal=signal, source="retrieval",
                     nearest_pages=near)
            packs.append(None)
            continue
        pack: ContextPack = d.build_context(sec.sub_question, result, page_facts,
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
        # 同一事实只在一节里用（10-23）：前面小节**实际引用过**的事实，不再给后面的小节。
        # 🔴 不能按「检索时分给过」排除——第一节按整页回填会把第 2、13 页一次拿走，
        # 品牌介绍后面的「使命愿景 / 团队 / 联系方式」检索结果被排空，产品介绍各节拿不到本产品事实，
        # 模型就用别的产品的事实填空（实测：把「工具智能体」的 Token 计费写成了监控系统的功能）
        used = {c["fact_id"] for x in sections_out for c in x["citations"].values()}
        # 内容去重，不只按 fact_id：同一份「六个模块」在库里有清单、逐项、图示关系三种 fact，
        # 只按 id 排除，产品介绍五节会各列一遍（10-23 实测）
        from ragwriter.verify.checks import _bigrams

        used_grams = [_bigrams(c["text"]) for x in sections_out for c in x["citations"].values()]

        def repeats(sn: Snippet) -> bool:
            g = _bigrams(sn.text)
            return bool(g) and any(len(g & u) / len(g) >= 0.6 for u in used_grams if u)

        kept = [sn for sn in pack.snippets if sn.fact_id not in used and not repeats(sn)]
        if used and not kept:
            # 本节能用的资料都已在前面小节用过：这一节不写（介绍类「有什么写什么」，Blog 也不该复读）
            sections_out.append({"section": i, "heading": sec.heading, "text": "", "sentences": [],
                                 "citations": {}, "skipped": True, "pages": []})
            insufficient.append({"topic": sec.sub_question, "signal": "all_facts_used"})
            yield ev("section_final", section=i, kind="body", heading=sec.heading, text="",
                     sentences=[], citations={}, skipped=True)
            continue
        if used and len(kept) < len(pack.snippets):
            from ragwriter.generate.context import render as render_context

            pack = pack.model_copy(update={"snippets": kept, "rendered": render_context(
                pack.sub_question, kept, titles={c["page_no"]: c.get("title", "") for c in ctx.profile.get("catalog", [])})})
        for s in pack.snippets:
            all_snippets[s.sid] = s
        prev = [s["heading"] for s in sections_out if not s["skipped"]]
        chunks: Iterable[str] = d.stream_section(d.client, sec, pack, genre=genre, profile=ctx.profile,
                                                 constraints=constraints, thesis=plan.thesis,
                                                 outline=[x.heading for x in plan.sections], prev_headings=prev)
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
            # 写作节点报的缺口只进校验报告，不上资料边界卡（10-23）：写作前已和用户确认过「资料里没有什么」，
            # 边界卡以确认结果为准；写作节点每节都会凑缺口，还常把片段里有的内容也报成缺口
            if not brief and not is_intro and not is_faq:
                boundary.append({"topic": g, "source": "writer", "nearest_pages": pack_pages[:3],
                                 "section": i, "heading": sec.heading})
            insufficient.append({"topic": g, "signal": "writer"})
            yield ev("insufficient", section=i, topic=g, signal="writer", source="writer",
                     nearest_pages=pack_pages[:3])
        block: VerifiedBlock = yield from verify_and_fix(ctx, draft, section=i, kind="body",
                                                         snippets=pack.snippets)
        rewrites += block.rewrites
        deletions += block.deletions
        cites = citations_for(block.sentences, all_snippets)
        sec_out = {"section": i, "heading": sec.heading, "text": block.text,
                   "sentences": [s.model_dump() for s in block.sentences], "citations": cites,
                   "skipped": False, "pages": sorted({c["page_no"] for c in cites.values()}),
                   "verdicts": list(block.verdicts)}
        sections_out.append(sec_out)
        yield ev("section_final", section=i, kind="body", heading=sec.heading, text=block.text,
                 sentences=sec_out["sentences"], citations=cites, skipped=False)

    # ④ 导语与结尾：只能基于已过校验的小节，同样逐句校验
    snippet_list = list(all_snippets.values())
    verified_texts = [s["text"] for s in sections_out]  # 与 plan.sections 一一对齐，跳过的节是空串
    bookends: dict[str, VerifiedBlock] = {}
    for kind, sec_no in (() if is_faq else (("lede", 0), ("ending", len(plan.sections) + 1))):
        # FAQ 是问答对，没有导语和结尾（10-23 子曰定）
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
        rewrites += block.rewrites
        deletions += block.deletions
        bookends[kind] = block
        yield ev("section_final", section=sec_no, kind=kind, heading="", text=block.text,
                 sentences=[s.model_dump() for s in block.sentences],
                 citations=citations_for(block.sentences, all_snippets), skipped=False)

    # ⑤ 通篇润色：逐块重新校验，不过就回退（润色只能让文字更顺，不能让事实变差）
    polish_log = yield from _polish(ctx, plan, genre, sections_out, bookends, all_snippets)
    # 报告用每块的最终判定：润色被采纳的块是润色后的句子，否则是逐节定稿
    verdicts = list(bookends["lede"].verdicts) if "lede" in bookends else []
    for s in sections_out:
        verdicts += s.get("verdicts", [])
    if "ending" in bookends:
        verdicts += bookends["ending"].verdicts

    # ⑥ 汇总
    parts = [f"# {plan.title}"]
    if "lede" in bookends and bookends["lede"].text:
        parts.append(bookends["lede"].text)
    for s in sections_out:
        if not s["skipped"] and s["text"]:
            if is_faq:
                parts.append(f"**问：{s['heading']}**\n\n答：{s['text']}")  # 问答对（10-23 子曰定）
            else:
                parts.append(f"## {s['heading']}\n\n{s['text']}")
    if "ending" in bookends and bookends["ending"].text:
        parts.append(bookends["ending"].text)
    final_text = "\n\n".join(parts)
    jsonld = None
    if is_faq:
        from ragwriter.generate.jsonld import faq_jsonld

        jsonld = faq_jsonld([(s["heading"], s["text"]) for s in sections_out if not s["skipped"] and s["text"]])

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
                     title=plan.title, genre=genre, topic=topic, thesis=plan.thesis, jsonld=jsonld,
                     polish=polish_log,
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


# ── 通篇润色（10-23 子曰：模型参与写作过程，让语句通顺）──────────────────
def _polish(ctx: TurnContext, plan: Plan, genre: str, sections_out: list[dict], bookends: dict,
            all_snippets: dict[str, Snippet]) -> Iterator[Event]:
    """润色全文，逐块验收：句句过 V1–V5、数字与引用不越出原文，才替换；否则这一块保留润色前的定稿。

    生成器，返回润色日志（每块 accepted / reason），进 done 事件，答辩时能说清润色改了哪里、拦了哪里。
    """
    from ragwriter.generate.writer import polish_article
    from ragwriter.verify.checks import number_tokens

    d = ctx.deps
    if ctx.options.get("polish", True) is False:
        return []
    blocks: dict[str, str] = {}
    heads: dict[str, str] = {}
    if "lede" in bookends and bookends["lede"].text:
        blocks["LEDE"] = bookends["lede"].text
    for s in sections_out:
        if not s["skipped"] and s["text"]:
            blocks[f"SECTION {s['section']}"] = s["text"]
            heads[f"SECTION {s['section']}"] = s["heading"]
    if "ending" in bookends and bookends["ending"].text:
        blocks["ENDING"] = bookends["ending"].text
    if not blocks:
        return []
    try:
        polished = getattr(d, "polish_article", None) or polish_article
        out = polished(d.client, blocks, genre=genre, profile=ctx.profile, thesis=plan.thesis, headings=heads)
    except Exception as exc:  # noqa: BLE001 — 润色是锦上添花，失败就用逐节定稿
        yield ev("error", stage="polish", reason=str(exc), fallback=True)
        return [{"block": "ALL", "accepted": False, "reason": f"润色调用失败：{exc}"}]

    titles = {c["page_no"]: str(c.get("title", "")) for c in ctx.profile.get("catalog", [])}
    by_sid = {sid: sn.model_copy(update={"text": f"{titles[sn.page_no]}\n{sn.text}"})
              if titles.get(sn.page_no) else sn for sid, sn in all_snippets.items()}
    log = []
    for key, original in blocks.items():
        new = (out.get(key) or "").strip()
        if not new or new == original:
            log.append({"block": key, "accepted": False, "reason": "未改动"})
            continue
        sents = [x for x in d.split_sentences(new) if x.text.strip()]
        sents = [x if x.cites else x.model_copy(update={"cites": parse_cites(x.text)}) for x in sents]
        reason = ""
        if not set(number_tokens(new)) <= set(number_tokens(original)):
            reason = "润色引入了原文没有的数字"
        elif not {c for x in sents for c in x.cites} <= set(re.findall(r"S\d+", original)):
            reason = "润色引入了原文没有的引用"
        verdicts: list[SentenceVerdict] = []
        if not reason:
            for x in sents:
                v = d.verify_sentence(x, by_sid, ctx.page_metrics, strict_cite=(key == "LEDE"),
                                      case_pages=ctx.case_pages)
                if not v.ok:
                    reason = "润色后校验未过：" + "；".join(f"{r.check} {r.reason}" for r in v.results if not r.ok)
                    break
                verdicts.append(v)
        if reason:
            log.append({"block": key, "accepted": False, "reason": reason})
            continue
        text = "".join(x.text for x in sents)
        cites = citations_for(sents, all_snippets)
        if key == "LEDE" or key == "ENDING":
            kind = "lede" if key == "LEDE" else "ending"
            b = bookends[kind]
            b.sentences, b.verdicts = sents, verdicts
            sec_no = 0 if kind == "lede" else len(plan.sections) + 1
            yield ev("section_final", section=sec_no, kind=kind, heading="", text=b.text,
                     sentences=[x.model_dump() for x in sents], citations=cites, skipped=False, polished=True)
        else:
            n = int(key.split()[1])
            s = next(x for x in sections_out if x["section"] == n)
            s.update(text=text, sentences=[x.model_dump() for x in sents], citations=cites, verdicts=verdicts,
                     pages=sorted({c["page_no"] for c in cites.values()}))
            yield ev("section_final", section=n, kind="body", heading=s["heading"], text=text,
                     sentences=s["sentences"], citations=cites, skipped=False, polished=True)
        log.append({"block": key, "accepted": True, "reason": ""})
    return log


_BRAND_WORDS = re.compile(r"Trendee|万悉科技|万悉|LLM-?原生|GEO|[\s·（）()]", re.I)


def _product_facts(ctx: TurnContext, plan: Plan) -> set[str]:
    """选中产品的资料：页标题点名该产品的页取整页；其它页只取正文提到产品核心词的那几条。

    粒度是事实不是页：第 28 页有一句「全链路数据监控系统」，同页的「动态高潜提示词模型，预测精准度超 95%」
    讲的是公司整体技术体系——整页圈进来，它就被写成了监控系统的功能（10-23 实测）。
    找不到任何资料时返回空集，调用方不圈——宁可退回全量检索，也不让整篇产品介绍空掉。
    """
    terms = {_BRAND_WORDS.sub("", p) for p in plan.products}
    terms = {t for t in terms if len(t) >= 2}
    if not terms:
        return set()
    out: set[str] = set()
    for c in ctx.profile.get("catalog", []):
        facts = ctx.page_facts(c["page_no"])
        if any(t in str(c.get("title", "")) for t in terms):
            out.update(f.fact_id for f in facts)
        else:
            out.update(f.fact_id for f in facts if any(t in f.text for t in terms))
    return out


def _boost_hint_pages(ctx: TurnContext, sec, result: RetrievalResult, scope: set[str]) -> RetrievalResult:
    """规划给的预计页码（模型从目录判断的）上，与本节最相关的事实优先进上下文。

    10-23 实测品牌介绍每次结果都不一样：「联系方式」时而取到第 36 页的官网邮箱、时而写成「资料未说明」，
    「荣誉」「服务行业」时有时无——规划其实给了页码提示，检索却只靠一句泛化问句去碰运气。
    每个提示页按与本节标题 + 检索问句的字面重合取前 3 条，排到命中最前；已经命中的保持原样。
    """
    from ragwriter.contracts import Hit, USABLE_STATUSES
    from ragwriter.verify.checks import _bigrams

    hints = [p for p in (sec.expected_pages_hint or [])][:3]
    if not hints:
        return result
    q = _bigrams(f"{sec.heading}{sec.purpose}{sec.sub_question}")
    have = {h.fact.fact_id for h in result.hits}
    top = max((h.rrf_score for h in result.hits), default=0.0)
    extra = []
    for p in hints:
        facts = [f for f in ctx.page_facts(p) if f.status in USABLE_STATUSES and (not scope or f.fact_id in scope)]
        ranked = sorted(facts, key=lambda f: -len(q & _bigrams(f.text)))
        for f in ranked[:3]:
            if f.fact_id not in have and len(q & _bigrams(f.text)) > 0:
                extra.append(Hit(fact=f, rrf_score=top + 1e-4 * (len(hints) * 3 - len(extra))))
                have.add(f.fact_id)
    if not extra:
        return result
    return result.model_copy(update={"hits": extra + list(result.hits)})
