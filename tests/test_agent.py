"""会话 Agent 编排层：事件顺序、槽位继承、查询改写、资料不足、校验改写/删除、意图规则、工作窗口。

全部用 ragwriter.agent.testing 的假实现，不依赖 rag.db 与模型。
"""

from __future__ import annotations

import json

import pytest

from ragwriter.agent import Orchestrator, SessionStore
from ragwriter.agent.context import assemble, estimate_tokens, profile_slice, truncate_to_tokens
from ragwriter.agent.handlers import parse_cites, split_gaps_fallback
from ragwriter.agent.intent import SUGGESTED_TOPICS, classify, rule_classify
from ragwriter.agent.testing import DEMO_OUTPUTS, DEMO_PLAN, DEMO_PROFILE, FakeClient, FakeSearch, FakeStore, make_fake_deps

BLOG = "写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？"
DONE_KEYS = {"artifact_id", "final_text", "citations", "report", "stream_mode", "usage", "refused",
             "nearest_pages", "suggestions", "refusal_kind", "boundary", "injected", "dense_used"}


def run(orch: Orchestrator, message: str, session_id: str | None = None, **options):
    return list(orch.handle(message, session_id=session_id, options=options or None))


def types(events) -> list[str]:
    out: list[str] = []
    for e in events:
        if not out or out[-1] != e.type:
            out.append(e.type)
    return out


def first(events, type_):
    return next(e for e in events if e.type == type_)


def is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(x == y for y in it) for x in needle)


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def orch(tmp_path, client):
    return Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "sessions.db")


# ── A1 写作事件顺序 ───────────────────────────────────────
def test_a1_write_event_order(orch):
    evs = run(orch, BLOG)
    t = types(evs)
    assert evs[0].type == "session" and evs[1].type == "intent" and evs[-1].type == "done"
    assert is_subsequence(["session", "intent", "plan", "retrieval", "delta", "verify", "section_final", "done"], t)
    intent = evs[1].data
    assert intent["intent"] == "write" and intent["genre"] == "blog" and intent["source"] == "rule"
    assert "GEO" in intent["topic"]
    done = evs[-1].data
    assert DONE_KEYS <= set(done)
    assert done["refused"] is False and done["artifact_id"]
    assert "[S1]" in done["final_text"] and "61.66亿" in done["final_text"]
    assert done["citations"]["S1"] == {"fact_id": "p09-f01", "page_no": 9,
                                       "text": "全球市场 AI 网站月访问量：ChatGPT 61.66亿", "status": "verified"}
    assert done["stream_mode"] == "native"
    assert done["usage"]["calls"] >= 4 and "elapsed_ms" in done["usage"]
    assert done["dense_used"] is True and done["injected"] is False
    # sid 跨节连续：第 2 节从 S3 开始
    rets = [e.data for e in evs if e.type == "retrieval"]
    assert [s["sid"] for s in rets[1]["snippets"]] == ["S3", "S4"]
    # 每条 delta 都带节号
    assert all("section" in e.data for e in evs if e.type == "delta")


# ── A2 「再出一版 FAQ」继承主题 ────────────────────────────
def test_a2_faq_inherits_topic(orch):
    first_run = run(orch, BLOG)
    sid = first_run[0].data["session_id"]
    evs = run(orch, "再出一版 FAQ", sid)
    it = evs[1].data
    assert it["intent"] == "write" and it["genre"] == "faq"
    assert it["topic"] == first_run[1].data["topic"]
    assert it["inherited"] == {"topic": first_run[1].data["topic"]}
    assert first(evs, "plan").data["genre"] == "faq"
    assert evs[-1].data["refused"] is False


# ── A3 qa 多轮：指代句触发查询改写 ────────────────────────
def test_a3_qa_followup_rewrites_query(tmp_path):
    client = FakeClient({**DEMO_OUTPUTS,
                         "answer": ["据 Shopify 2025 年数据，AI驱动订单增长 15倍[S2]。",
                                    "据 Shopify 2025 年数据，AI驱动流量增长 8倍[S1]。"],
                         "query_rewrite": "第10页 AI驱动流量增长多少"})
    deps = make_fake_deps(client=client)
    orch = Orchestrator(deps, sessions_db=tmp_path / "s.db")
    e1 = run(orch, "第10页 AI驱动订单增长多少？")
    assert e1[1].data["intent"] == "qa"
    assert "query_rewrite" not in client.tags()  # 首轮没有上文，不改写
    assert "15倍" in e1[-1].data["final_text"]

    e2 = run(orch, "那流量呢？", e1[0].data["session_id"])
    assert e2[1].data["intent"] == "qa"
    assert client.tags().count("query_rewrite") == 1
    assert "第10页 AI驱动流量增长多少" in deps.fake_search.queries  # 改写句作第一路
    assert "那流量呢？" in deps.fake_search.queries  # 原话作第二路
    done = e2[-1].data
    assert "8倍" in done["final_text"] and done["refused"] is False
    assert {c["page_no"] for c in done["citations"].values()} == {10}
    assert done["query"] == "第10页 AI驱动流量增长多少"


# ── A4 全部小节资料不足 ⇒ 不生成文章 ──────────────────────
def test_a4_all_sections_insufficient(tmp_path, client):
    store = FakeStore()
    search = FakeSearch(store, rules={"流量规模": {"facts": ["p09-f01"], "sufficient": False},
                                      "电商增长": {"facts": ["p10-f01"], "sufficient": False}})
    orch = Orchestrator(make_fake_deps(store=store, client=client, search=search), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    assert "section_writer" not in client.tags()
    assert not any(e.type == "delta" and e.data["kind"] == "body" for e in evs)
    assert sum(e.type == "insufficient" for e in evs) == 2
    done = evs[-1].data
    assert done["refused"] is True and done["refusal_kind"] == "insufficient_all"
    assert done["nearest_pages"] == [9, 10]
    assert done["final_text"].startswith("品宣资料里没有关于『")
    assert "第 9、10 页" in done["final_text"]
    assert {b["source"] for b in done["boundary"]} == {"retrieval"}


def test_a4_business_question_goes_to_retrieval_not_out_of_scope(orch):
    """评审 M6：融资是正常的品牌问题，走检索 → 资料不足，而不是越界。"""
    evs = run(orch, "万悉科技的融资情况")
    assert evs[1].data["intent"] == "qa"
    done = evs[-1].data
    assert done["refused"] is True and done["refusal_kind"] == "insufficient_all"
    assert first(evs, "insufficient")


def test_partial_insufficient_goes_to_boundary(tmp_path):
    plan = {**DEMO_PLAN, "insufficient": ["中国出海品牌的整体规模数据"],
            "sections": [*DEMO_PLAN["sections"],
                         {"heading": "融资情况", "sub_question": "万悉融资", "expected_pages_hint": []}]}
    client = FakeClient({**DEMO_OUTPUTS, "planner": json.dumps(plan, ensure_ascii=False)})
    orch = Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    done = evs[-1].data
    assert done["refused"] is False and done["refusal_kind"] == "partial"
    assert {(b["source"], b["topic"]) for b in done["boundary"]} == {
        ("plan", "中国出海品牌的整体规模数据"), ("retrieval", "万悉融资")}
    assert "融资情况" not in done["final_text"]  # 不足的小节整节不写
    assert "品宣资料未涉及" not in done["final_text"]
    skipped = [e for e in evs if e.type == "section_final" and e.data.get("skipped")]
    assert [e.data["heading"] for e in skipped] == ["融资情况"]


def test_writer_gaps_line_stripped(tmp_path):
    client = FakeClient({**DEMO_OUTPUTS, "section_writer": [
        "AI搜索正成为新的超级流量入口[S2]。\nGAPS: [\"AI 搜索在东南亚的份额\"]",
        "据 Shopify 2025 年数据，AI驱动订单增长 15倍[S4]。"]}, chunk=3)
    orch = Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    streamed = "".join(e.data["text"] for e in evs if e.type == "delta")
    assert "GAPS" not in streamed and "GA" not in streamed.replace("GEO", "")
    done = evs[-1].data
    assert "GAPS" not in done["final_text"]
    assert {"topic": "AI 搜索在东南亚的份额", "source": "writer", "nearest_pages": [9],
            "section": 1, "heading": "AI 搜索正在成为新的流量入口"} in done["boundary"]
    assert done["refusal_kind"] == "partial"


# ── A5 校验失败 ⇒ patch rewrite / delete ──────────────────
WRONG = "据 Shopify 2025 年数据，AI驱动流量增长 15倍[S4]。"
RIGHT = "据 Shopify 2025 年数据，AI驱动订单增长 15倍[S4]。"


def test_a5_v3_fail_then_rewrite(tmp_path):
    client = FakeClient({**DEMO_OUTPUTS, "section_writer": [DEMO_OUTPUTS["section_writer"][0], WRONG],
                         "rewriter": RIGHT})
    orch = Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    fails = [e.data for e in evs if e.type == "verify" and e.data["result"] == "fail"]
    assert fails and fails[0]["failed"] == ["V3"] and fails[0]["section"] == 2
    patch = first(evs, "patch").data
    assert patch["action"] == "rewrite" and patch["text"] == RIGHT and patch["original"] == WRONG
    sec2 = [e.data for e in evs if e.type == "section_final" and e.data["section"] == 2][0]
    assert sec2["text"] == RIGHT
    done = evs[-1].data
    assert RIGHT in done["final_text"] and WRONG not in done["final_text"]
    assert [r["round"] for r in done["report"]["rewrites"]] == [1]
    assert done["report"]["rewrites"][0]["failed_checks"] == ["V3"]


def test_a5_two_rounds_then_delete(tmp_path):
    client = FakeClient({**DEMO_OUTPUTS, "section_writer": [DEMO_OUTPUTS["section_writer"][0], WRONG],
                         "rewriter": [WRONG, "AI驱动流量增长 15倍[S4]。"]})
    orch = Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    assert client.tags().count("rewriter") == 2
    sec2_verifies = [e.data for e in evs if e.type == "verify" and e.data["section"] == 2]
    assert [v["round"] for v in sec2_verifies] == [0, 1, 2]
    patches = [e.data for e in evs if e.type == "patch"]
    assert [p["action"] for p in patches] == ["delete"]
    done = evs[-1].data
    assert "15倍" not in done["final_text"]
    assert len(done["report"]["rewrites"]) == 2
    assert done["report"]["deletions"][0]["failed_checks"] == ["V3"]


def test_inject_fault_v3_swap(orch):
    evs = run(orch, BLOG, inject_fault="v3_swap")
    done = evs[-1].data
    assert done["injected"] is True
    assert any(e.type == "verify" and "V3" in e.data["failed"] for e in evs)


def test_rewrite_rounds_capped_by_default(orch, client):
    client.set("section_writer", [DEMO_OUTPUTS["section_writer"][0], WRONG]).set("rewriter", WRONG)
    run(orch, BLOG)
    assert client.tags().count("rewriter") == 2


# ── 意图规则快路 ──────────────────────────────────────────
@pytest.mark.parametrize("msg,slots,intent,genre", [
    (BLOG, {}, "write", "blog"),
    ("生成一篇 FAQ，主题是 LLM-原生 GEO", {}, "write", "faq"),
    ("帮我写一版品牌介绍：万悉科技", {}, "write", "brand_intro"),
    ("出一版产品介绍：Trendee 监控系统", {}, "write", "product_intro"),
    ("再出一版 FAQ", {"topic": "GEO"}, "write", "faq"),
    ("再出一版 FAQ", {}, "clarify", "faq"),
    ("写一篇博客", {}, "clarify", "blog"),
    ("第 10 页 AI 驱动订单增长多少", {}, "qa", None),
    ("那流量呢？", {}, "qa", None),
    ("万悉科技的融资情况", {}, "qa", None),
    ("万悉的营收是多少？", {}, "qa", None),
    ("帮我写个 Python 脚本", {}, "out_of_scope", None),
    ("今天天气怎么样", {}, "out_of_scope", None),
    ("你好", {}, "out_of_scope", None),
    ("第二节短一点", {"last_artifact_id": "a_1"}, "revise", None),
    ("第二节短一点", {}, "clarify", None),
    ("这句话依据哪页？", {"last_artifact_id": "a_1"}, "explain", None),
])
def test_rule_fast_path(msg, slots, intent, genre):
    res = rule_classify(msg, slots)
    assert res is not None and res.intent == intent
    if genre:
        assert res.genre == genre


def test_rule_miss_without_model_is_clarify():
    assert rule_classify("嗯", {}) is None
    assert classify("嗯", {}, client=None).intent == "clarify"


def test_model_classify_and_low_confidence():
    c = FakeClient({"intent": {"intent": "write", "genre": "none", "topic": "", "query": "", "constraints": [],
                               "confidence": 0.9}})
    res = classify("嗯，按刚才那个来", {"topic": "GEO 合规", "genre": "faq"}, client=c, profile=DEMO_PROFILE)
    assert res.source == "model" and res.intent == "write" and res.topic == "GEO 合规"
    assert res.inherited == {"genre": "faq", "topic": "GEO 合规"}
    low = FakeClient({"intent": {"intent": "qa", "genre": "none", "topic": "", "query": "x", "constraints": [],
                                 "confidence": 0.3}})
    assert classify("嗯", {}, client=low).intent == "clarify"
    # 模型挂了：不崩，退回追问
    bad = FakeClient({"intent": RuntimeError("boom")})
    assert classify("嗯", {}, client=bad).intent == "clarify"


def test_clarify_has_suggestions(orch):
    evs = run(orch, "写一篇")
    done = evs[-1].data
    assert done["refused"] is True and done["refusal_kind"] == "clarify"
    assert done["suggestions"] == list(SUGGESTED_TOPICS)


def test_out_of_scope_text_and_no_search(orch):
    evs = run(orch, "帮我写个 Python 脚本")
    done = evs[-1].data
    assert done["refusal_kind"] == "out_of_scope" and done["nearest_pages"] == []
    assert done["final_text"] == "这个问题超出了品宣资料的范围。我只能基于万悉科技品宣 PDF 写作或回答问题。"
    assert orch.deps.fake_search.queries == []


def test_revise_is_unsupported_not_faked(orch, client):
    sid = run(orch, BLOG)[0].data["session_id"]
    n_calls = len(client.calls)
    evs = run(orch, "第二节短一点", sid)
    assert evs[1].data["intent"] == "revise" and evs[1].data["supported"] is False
    done = evs[-1].data
    assert done["unsupported"] is True and "暂不支持" in done["final_text"]
    assert len(client.calls) == n_calls  # 没有调任何模型


def test_no_model_write_refuses(tmp_path):
    orch = Orchestrator(make_fake_deps(client=None), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    assert evs[1].data["intent"] == "write"
    assert evs[-1].data["refusal_kind"] == "no_model"


def test_handler_crash_still_ends_with_done(tmp_path):
    client = FakeClient({**DEMO_OUTPUTS, "planner": RuntimeError("planner down")})
    orch = Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")
    evs = run(orch, BLOG)
    assert [e.type for e in evs[-2:]] == ["error", "done"]
    assert "planner down" in evs[-2].data["reason"] and evs[-1].data["error"] is True


def test_force_intent(orch):
    evs = run(orch, "随便", force_intent={"intent": "write", "genre": "faq", "topic": "GEO 合规"})
    assert evs[1].data["source"] == "forced" and evs[1].data["topic"] == "GEO 合规"


# ── 会话记忆与工作窗口 ────────────────────────────────────
def test_window_excludes_assistant_body(tmp_path, orch):
    evs = run(orch, BLOG)
    sid = evs[0].data["session_id"]
    body_sentence = "ChatGPT 月访问量达 61.66亿"
    assert body_sentence in evs[-1].data["final_text"]

    store = SessionStore(tmp_path / "sessions.db")
    win = store.window(sid)
    rendered = win.render()
    assert body_sentence not in rendered and "61.66亿" not in rendered
    assert win.user_turns == [BLOG]
    assert win.assistant_turns[0]["pages"] == [9, 10]
    assert win.assistant_turns[0]["outline"][0].startswith("AI 搜索正在成为新的流量入口（第9页）")
    # L2 仍保留全文，只是不进窗口
    assert any(body_sentence in t["content"] for t in store.turns(sid) if t["role"] == "assistant")
    slots = store.load_slots(sid)
    assert slots["last_artifact_id"] == evs[-1].data["artifact_id"] and slots["referenced_pages"] == [9, 10]
    art = store.get_artifact(slots["last_artifact_id"])
    assert art["genre"] == "blog" and art["citations"]["S1"]["page_no"] == 9


def test_window_keeps_last_three_turns(tmp_path, orch):
    sid = None
    for i in range(5):
        evs = run(orch, f"第{i + 1}页讲了什么？", sid)
        sid = evs[0].data["session_id"]
    win = SessionStore(tmp_path / "sessions.db").window(sid)
    assert win.user_turns == ["第3页讲了什么？", "第4页讲了什么？", "第5页讲了什么？"]
    assert len(win.assistant_turns) == 3


def test_run_recorded(tmp_path, orch):
    evs = run(orch, BLOG)
    run_id = evs[-1].data["run_id"]
    rec = SessionStore(tmp_path / "sessions.db").get_run(run_id)
    lines = [json.loads(x) for x in rec["events_jsonl"].splitlines()]
    assert lines[0]["_meta"]["message"] == BLOG
    assert [x["type"] for x in lines[1:]] == [e.type for e in evs]
    assert all(isinstance(x["t"], int) for x in lines[1:])


# ── 上下文与小工具 ────────────────────────────────────────
def test_estimate_and_truncate():
    assert estimate_tokens("AI 搜索 GEO optimization") == 2 + 3
    text = "\n".join(f"第{i}行内容" for i in range(10))
    cut = truncate_to_tokens(text, 12)
    assert estimate_tokens(cut) <= 12 and cut.startswith("第0行内容")


def test_assemble_priority():
    ctx = assemble(l0="系统" * 10, task="任务", snippets="片段" * 20, l1="档案" * 20, l3="窗口" * 20, budget=70)
    assert "片段" * 20 in ctx.user  # 检索片段优先保留
    assert "窗口" not in ctx.user and "l3" in ctx.dropped
    assert ctx.system.startswith("系统" * 10) and "本轮任务\n任务" in ctx.user


def test_profile_slice_planner_compact():
    out = profile_slice(DEMO_PROFILE, "planner")
    assert out.splitlines()[0] == "p9 AI正在重塑搜索：AI搜索正成为新的超级流量入口"
    assert "思维连" in profile_slice(DEMO_PROFILE, "section_writer")
    assert "01 GEO背景介绍" in profile_slice(DEMO_PROFILE, "intent")


def test_parse_cites_and_gaps():
    assert parse_cites("A[S4][S5]，B[S6, S7]。") == ["S4", "S5", "S6", "S7"]
    assert split_gaps_fallback("正文[S1]。\nGAPS: [\"x\"]") == ("正文[S1]。", ["x"])


def test_drop_subsumed_duplicate_sentence():
    """真跑实例：改写句「…豆包 3.45 亿[S4]」被下一句「豆包 3.45 亿，DeepSeek 1.27 亿[S4][S5]」包含。"""
    from ragwriter.agent.handlers import VerifiedBlock, _drop_subsumed
    from ragwriter.contracts import CheckResult, Sentence, SentenceVerdict

    a = Sentence(idx=0, text="据数据，豆包月活 3.45 亿[S4]。", cites=["S4"])
    b = Sentence(idx=1, text="豆包月活 3.45 亿，DeepSeek 1.27 亿[S4][S5]。", cites=["S4", "S5"])
    c = Sentence(idx=2, text="品牌需要被 AI 记住[S9]。", cites=["S9"])
    block = VerifiedBlock()
    for s in (a, b, c):
        block.sentences.append(s)
        block.verdicts.append(SentenceVerdict(sentence=s, results=[CheckResult(check="V1", ok=True)]))
    evs = list(_drop_subsumed(block, section=1, kind="body"))
    assert [e.data["sentence"] for e in evs] == [0]
    assert [s.idx for s in block.sentences] == [1, 2]
