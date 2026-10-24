"""generate/ 的单元测试：规划、上下文构建、写作、改写、查询改写、提示词加载。

一律用假 client（记录收到的 system/user，返回预置文本），不联网、不调真模型。
上下文构建用 build/facts.jsonl 的真实事实，自己组装 Hit，不依赖检索层的实现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragwriter.contracts import USABLE_STATUSES, CheckResult, Hit, Plan, PlanSection, RetrievalResult
from ragwriter.generate import context, jsonld, planner, prompts, query, writer
from ragwriter.models import Fact

ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = ROOT / "build" / "facts.jsonl"
PAGES_PATH = ROOT / "build" / "pages.jsonl"


# ── 夹具 ──────────────────────────────────────────────────────

class FakeClient:
    """按 C 组约定的签名实现 complete / stream，只记录、不联网。"""

    def __init__(self, complete_out: str = "", stream_out: tuple[str, ...] = ()) -> None:
        self.complete_out = complete_out
        self.stream_out = stream_out
        self.calls: list[dict] = []

    def complete(self, system, user, *, json_schema=None, model=None, max_tokens=4096, tag=""):
        self.calls.append(dict(kind="complete", system=system, user=user, json_schema=json_schema, tag=tag))
        return self.complete_out

    def stream(self, system, user, *, model=None, max_tokens=1024, tag=""):
        self.calls.append(dict(kind="stream", system=system, user=user, tag=tag))
        return iter(self.stream_out)

    @property
    def last(self) -> dict:
        return self.calls[-1]


@pytest.fixture(scope="module")
def facts() -> dict[str, Fact]:
    with FACTS_PATH.open(encoding="utf-8") as f:
        return {fa.fact_id: fa for fa in (Fact.model_validate_json(line) for line in f if line.strip())}


@pytest.fixture(scope="module")
def page_facts(facts):
    """模拟检索层提供的回填函数：已按状态过滤。"""
    def _pf(page_no: int) -> list[Fact]:
        return [f for f in facts.values() if f.page_no == page_no and f.status in USABLE_STATUSES]
    return _pf


@pytest.fixture(scope="module")
def profile() -> dict:
    catalog = []
    with PAGES_PATH.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            catalog.append({"page_no": p["page_no"], "chapter": p["chapter"], "title": p["title"]})
    return {
        "catalog": catalog,
        "rules": {
            "corrections": [{"wrong": "思维连", "right": "思维链", "pages": [19, 25]}],
            "forbidden": ["第31页截图中的客户站点名与营收不得使用"],
        },
    }


def _hit(fact: Fact, score: float) -> Hit:
    return Hit(fact=fact, rrf_score=score)


def _result(hits: list[Hit]) -> RetrievalResult:
    return RetrievalResult(query="q", hits=hits, top_score=max((h.rrf_score for h in hits), default=0.0),
                           sufficient=bool(hits))


# ── 提示词加载器 ──────────────────────────────────────────────

def test_rules_injected_from_single_source():
    write, qa, rewrite = (prompts.rules_text(m) for m in ("write", "qa", "rewrite"))
    for r in (write, qa, rewrite):  # ①②③与附加规则各节点相同
        assert "[S编号]" in r and "数字对、对象错" in r and "据第 N 页图示" in r
    # 第 ④ 条按节点分开：写作列 GAPS，问答明说未涉及，改写放弃
    assert "GAPS: [" in write and "明说「品宣资料未涉及」" not in write
    assert "明说「品宣资料未涉及」" in qa and "GAPS" not in qa
    assert "无法改写" in rewrite and "GAPS" not in rewrite
    for name, heading, rules in [("section_writer", "System", write),
                                 ("section_writer", "System（导语与结尾）", write),
                                 ("answer", "System", qa), ("rewriter", "System", rewrite)]:
        assert "{{rules}}" in prompts.section(name, heading)
        rendered = prompts.render(name, heading, profile_rules="P", genre_spec="G", shape_rule="R",
                                  kind_name="导语", kind_spec="K")
        assert rules in rendered
        assert "{{" not in rendered
    # 一份原文：规则正文不得被复制进其它提示词文件
    marker = "数字对、对象错，是本系统最严重的错误"
    for path in prompts.PROMPT_DIR.glob("*.md"):
        if path.stem != "_rules":
            assert marker not in path.read_text(encoding="utf-8"), path.name


def test_render_is_single_pass_and_strict():
    # 变量值里的 {{rules}} 不会被再次展开（用户输入不能篡改提示词结构）
    out = prompts.render("answer", "User", context="{{rules}}", question="Q")
    assert "{{rules}}" in out and prompts.rules_text() not in out
    with pytest.raises(KeyError, match="question"):
        prompts.render("answer", "User", context="C")


def test_genre_specs_come_from_genre_files():
    genres = ("blog", "faq", "brand_intro", "product_intro")
    for part in ("规划", "写作", "导语", "结尾"):
        specs = {g: prompts.genre_spec(g, part) for g in genres}
        assert all(specs.values()) and all("{{" not in v for v in specs.values())
    plan = {g: prompts.genre_spec(g, "规划") for g in genres}
    assert len(set(plan.values())) == 4                          # 四种体裁结构互不相同
    assert "thesis" in plan["blog"] and "中心论点" in prompts.genre_spec("blog", "导语")
    assert "问答对" in plan["faq"] and "第一句直接回答问题" in prompts.genre_spec("faq", "写作")
    # 介绍类共用 intro.md，只换节目单（10-23：品牌介绍按市场通行结构，产品介绍先选品）
    assert "**使命与愿景**" in plan["brand_intro"] and "**联系方式**" in plan["brand_intro"]
    assert "**效果与案例**" not in plan["brand_intro"]
    assert "先选品" in plan["product_intro"] and "**效果与案例**" in plan["product_intro"]
    assert "**使命与愿景**" not in plan["product_intro"]
    # 体裁不再写在节点提示词里
    assert not any(k.startswith("体裁：") for k in prompts.load_sections("planner"))
    assert not any(k.startswith("体裁：") for k in prompts.load_sections("section_writer"))
    with pytest.raises(KeyError):
        prompts.genre_spec("poem", "规划")


def test_fenced_heading_is_not_a_section(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("说明\n## System\n```\n## 不是段\n```\n正文\n## User\nU", encoding="utf-8")
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    prompts.load_sections.cache_clear()
    try:
        secs = prompts.load_sections("x")
        assert list(secs) == ["System", "User"]
        assert "## 不是段" in secs["System"]
    finally:
        prompts.load_sections.cache_clear()


# ── 规划 ──────────────────────────────────────────────────────

PLAN_JSON = {
    "title": "为什么中国出海品牌需要进行 GEO 优化",
    "lede_points": ["AI 搜索改变了海外买家找品牌的方式"],
    "sections": [
        {"heading": "AI 搜索正在成为新的流量入口", "sub_question": "AI 搜索的流量规模与增长", "expected_pages_hint": [9, 10, 99]},
        {"heading": "AI 如何决定引用谁", "sub_question": "AI 问答引擎的工作链路与 GEO 注入点", "expected_pages_hint": [5]},
        {"heading": "", "sub_question": "", "expected_pages_hint": []},
    ],
    "insufficient": ["中国出海品牌的整体规模数据"],
    "ending_points": ["尽早布局 GEO"],
}


def test_plan_sends_full_catalog_and_parses(profile):
    client = FakeClient(complete_out=json.dumps(PLAN_JSON, ensure_ascii=False))
    topic = "为什么中国出海品牌需要进行 GEO 优化？"
    result = planner.plan(client, topic, "blog", profile, constraints=["不超过 800 字"])

    call = client.last
    assert call["json_schema"] is planner.PLAN_SCHEMA and call["tag"] == "planner"
    for entry in profile["catalog"]:  # 全部 36 页标题都进了 prompt
        assert f"第{entry['page_no']}页" in call["user"] and entry["title"] in call["user"]
    assert topic in call["user"] and "不超过 800 字" in call["user"]
    assert prompts.genre_spec("blog", "规划") in call["system"]  # blog 的体裁结构注入了
    assert "第 2 页（愿景页）" in call["system"]                   # 示例主题的连接点说明
    assert "{{" not in call["system"] + call["user"]

    assert isinstance(result, Plan)
    assert [s.sub_question for s in result.sections] == ["AI 搜索的流量规模与增长", "AI 问答引擎的工作链路与 GEO 注入点"]
    assert result.sections[0].expected_pages_hint == [9, 10]  # 目录外的 99 被丢掉
    assert result.insufficient == ["中国出海品牌的整体规模数据"]


def test_plan_schema_is_strict():
    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        elif node.get("type") == "array":
            walk(node["items"])
    walk(planner.PLAN_SCHEMA)


def test_plan_caps_sections_and_rejects_garbage(profile):
    many = dict(PLAN_JSON, sections=[{"heading": f"h{i}", "sub_question": f"q{i}", "expected_pages_hint": []}
                                     for i in range(9)])
    client = FakeClient(complete_out="```json\n" + json.dumps(many) + "\n```")
    assert len(planner.plan(client, "t", "blog", profile).sections) == 4  # 10-23：Blog 收到 4 节
    assert len(planner.plan(client, "t", "faq", profile).sections) == 8
    with pytest.raises(planner.PlanError):
        planner.plan(FakeClient(complete_out="不是 JSON"), "t", "blog", profile)


# ── 上下文构建 ────────────────────────────────────────────────

def _geo_hits(facts):
    return [
        _hit(facts["p10-f02"], 0.9),   # 15倍 · AI驱动订单增长
        _hit(facts["p05-v26"], 0.7),   # image_structural：GEO Phase 1 注入点
        _hit(facts["p10-v06"], 0.65),  # image_only：截图标题，必须被丢掉
        _hit(facts["p09-f01"], 0.5),   # ChatGPT 61.66亿
        _hit(facts["p10-f02"], 0.4),   # 重复命中
    ]


def test_build_context_backfill_dedupe_numbering(facts, page_facts):
    pack = context.build_context("AI 搜索的流量规模与增长", _result(_geo_hits(facts)), page_facts,
                                 sid_start=4, max_snippets=9)
    ids = [s.fact_id for s in pack.snippets]

    assert len(ids) == len(set(ids)) == 9                       # 预算内、无重复
    assert [s.sid for s in pack.snippets] == [f"S{i}" for i in range(4, 13)]  # 从 sid_start 连续编号
    assert "p10-f01" in ids                                     # 回填：同页兄弟数字（8倍 流量）
    assert all(facts[i].status in USABLE_STATUSES for i in ids)
    assert "p10-v06" not in ids and "p10-v07" not in ids         # image_only 不出现，回填也不出现

    # 按页去重：每页的片段连续成一组
    pages = [s.page_no for s in pack.snippets]
    blocks = [p for i, p in enumerate(pages) if i == 0 or pages[i - 1] != p]
    assert len(blocks) == len(set(blocks)) == 3

    # 首尾放置：最相关页（10）在最前，次相关页（5）在最后，第三（9）在中间
    assert blocks == [10, 9, 5]
    assert pack.snippets[0].fact_id == "p10-f02"                 # 组内命中在前
    # 轮流回填：三页都拿到了回填，不被第 5 页的 20 多条事实挤占
    for p in (5, 9, 10):
        assert sum(1 for s in pack.snippets if s.page_no == p) >= 2


def test_build_context_rendered_format(facts, page_facts):
    pack = context.build_context("AI 搜索的流量规模与增长", _result(_geo_hits(facts)), page_facts,
                                 sid_start=4, max_snippets=9)
    all_lines = pack.rendered.splitlines()
    assert all_lines[0] == "## 子问题：AI 搜索的流量规模与增长"
    # 每页一行标题头（模型要知道片段出自哪一页、那页讲什么），其余每行一个片段
    heads = [ln for ln in all_lines[1:] if ln.startswith("### 第")]
    assert "### 第10页《AI正在重塑购物：AI电商迎来爆发式增长》" in heads
    assert len(heads) == len({s.page_no for s in pack.snippets})
    lines = [all_lines[0]] + [ln for ln in all_lines[1:] if not ln.startswith("### ")]
    assert len(lines) == 1 + len(pack.snippets)
    by_id = {s.fact_id: (s, line) for s, line in zip(pack.snippets, lines[1:])}

    s, line = by_id["p10-f02"]
    assert line.startswith(f"[{s.sid} | 第10页] AI驱动订单增长：15倍")
    assert "〔出处口径：Shopify官方数据(2025)〕" in line           # metric 带出处口径

    s, line = by_id["p05-v26"]
    assert line.startswith(f"[{s.sid} | 第5页 · 图示] ")         # image_structural 标图示
    assert all("图示" not in ln for ln in lines[1:] if "| 第10页" in ln)


def test_build_context_empty_and_rerank_order(facts, page_facts):
    empty = context.build_context("中国出海品牌整体规模", _result([]), page_facts)
    assert empty.snippets == [] and context.EMPTY_NOTE in empty.rendered

    only_bad = context.build_context("x", _result([_hit(facts["p10-v06"], 1.0)]), page_facts)
    assert only_bad.snippets == []                              # 只命中 image_only 也不回填

    # 有重排分时按重排分排序，而不是 RRF
    hits = [Hit(fact=facts["p09-f01"], rrf_score=0.9, rerank_score=0.1),
            Hit(fact=facts["p10-f02"], rrf_score=0.1, rerank_score=0.8)]
    pack = context.build_context("x", _result(hits), page_facts, max_snippets=2)
    assert [s.fact_id for s in pack.snippets] == ["p10-f02", "p09-f01"]


# ── 写作 ──────────────────────────────────────────────────────

def _pack(facts, page_facts):
    return context.build_context("AI 搜索的流量规模与增长", _result(_geo_hits(facts)), page_facts, max_snippets=6)


def test_stream_section_sends_context_and_rules(facts, page_facts, profile):
    client = FakeClient(stream_out=("据 Shopify 2025 年数据，", "AI驱动订单增长15倍[S1]。"))
    pack = _pack(facts, page_facts)
    sec = PlanSection(heading="AI 搜索正在成为新的流量入口", sub_question=pack.sub_question)
    out = "".join(writer.stream_section(client, sec, pack, genre="faq", profile=profile,
                                        constraints=["语气偏正式"]))

    call = client.last
    assert out == "据 Shopify 2025 年数据，AI驱动订单增长15倍[S1]。"
    assert call["kind"] == "stream" and call["tag"] == "section_writer"
    assert pack.rendered in call["user"]                        # 上下文原样送进模型
    assert sec.heading in call["user"] and "语气偏正式" in call["user"]
    assert prompts.rules_text("write") in call["system"]        # 四条硬约束（写作版 ④）
    assert "「思维连」→「思维链」（第19、25页）" in call["system"]   # profile.rules 切片
    assert "第31页截图中的客户站点名与营收不得使用" in call["system"]
    assert prompts.genre_spec("faq", "写作") in call["system"]   # faq 体裁写法
    assert "第1页 · " not in call["system"] + call["user"]       # 写作节点不看目录


def test_stream_bookend_only_sees_verified_sentences(profile):
    client = FakeClient(stream_out=("导语。",))
    plan_ = Plan(title="为什么中国出海品牌需要进行 GEO 优化",
                 lede_points=["AI 搜索改变了买家找品牌的方式"],
                 sections=[PlanSection(heading="流量入口", sub_question="q1"),
                           PlanSection(heading="引用机制", sub_question="q2")])
    verified = ["据 Shopify 2025 年数据，AI驱动订单增长15倍[S5]。", "据第5页图示，GEO Phase 1 的注入点是实时外部索引[S9]。"]
    out = "".join(writer.stream_bookend(client, "lede", plan_, verified, genre="blog", profile=profile))

    call = client.last
    assert out == "导语。" and call["tag"] == "bookend_lede"
    assert "### 第1节 流量入口\n" + verified[0] in call["user"]
    assert verified[1] in call["user"] and "AI 搜索改变了买家找品牌的方式" in call["user"]
    assert "不得引入任何新事实" in call["system"] and prompts.rules_text("write") in call["system"]
    assert prompts.genre_spec("blog", "导语") in call["system"]  # Blog 导语结论先行


def test_stream_answer(facts, page_facts, profile):
    client = FakeClient(stream_out=("15倍[S1]。",))
    pack = _pack(facts, page_facts)
    assert "".join(writer.stream_answer(client, "AI 驱动订单增长了多少", pack, profile=profile)) == "15倍[S1]。"
    call = client.last
    assert pack.rendered in call["user"] and "AI 驱动订单增长了多少" in call["user"]
    assert writer.SHAPE_RULES["short"] in call["system"] and prompts.rules_text("qa") in call["system"]


def test_stream_answer_shape_drives_format(facts, page_facts, profile):
    """10-23：回答形状由意图识别给出，作答提示词按形状裁剪（问数字只给数字，核实先说对不对）。"""
    pack = _pack(facts, page_facts)
    for shape in ("number", "yes_no", "list", "explain"):
        client = FakeClient(stream_out=("x",))
        list(writer.stream_answer(client, "q", pack, profile=profile, shape=shape))
        assert writer.SHAPE_RULES[shape] in client.last["system"]


def test_rewrite_sentence_sends_failures_and_snippets(facts, page_facts, profile):
    pack = _pack(facts, page_facts)
    s = next(sn for sn in pack.snippets if sn.fact_id == "p10-f02")
    bad = f"据 Shopify 2025 年数据，AI驱动流量增长15倍[{s.sid}]。"
    failures = [CheckResult(check="V2", ok=True),
                CheckResult(check="V3", ok=False, reason="15倍的区分词「订单」缺失，兄弟区分词「流量」出现")]
    client = FakeClient(complete_out=f"「据 Shopify 2025 年数据，AI驱动订单增长15倍[{s.sid}]。」\n")
    out = writer.rewrite_sentence(client, bad, failures, [s], profile=profile)

    call = client.last
    assert out == f"据 Shopify 2025 年数据，AI驱动订单增长15倍[{s.sid}]。"
    assert call["kind"] == "complete" and call["tag"] == "rewriter"
    assert bad in call["user"]
    assert "- V3：15倍的区分词「订单」缺失，兄弟区分词「流量」出现" in call["user"]
    assert "V2：" not in call["user"]                            # 只送失败项
    assert context.render_snippet(s) in call["user"]             # 原片段
    assert prompts.rules_text("rewrite") in call["system"] and "禁用规则" in call["system"]

    assert writer.rewrite_sentence(FakeClient(complete_out="无法改写"), bad, failures, [s], profile=profile) == ""


# ── 查询改写 ──────────────────────────────────────────────────

def test_rewrite_query_completes_reference():
    slots = {"topic": "为什么中国出海品牌需要进行 GEO 优化", "referenced_pages": [10]}
    client = FakeClient(complete_out="改写后：AI 驱动订单增长了多少\n")
    out = query.rewrite_query(client, "那订单呢", slots, ["AI 驱动流量增长了多少"])
    assert out == "AI 驱动订单增长了多少"
    call = client.last
    assert "那订单呢" in call["user"] and "AI 驱动流量增长了多少" in call["user"]
    assert '"referenced_pages"' in call["user"] and call["tag"] == "query_rewrite"


def test_rewrite_query_rejects_leaked_answer():
    # 模型把答案「15」写进了问题 → 退回原消息
    client = FakeClient(complete_out="AI 驱动订单增长 15 倍的出处")
    assert query.rewrite_query(client, "那订单呢", {}, ["AI 驱动流量增长了多少"]) == "那订单呢"
    # 输入里已有的数字可以保留
    client = FakeClient(complete_out="第 10 页的订单增长数据")
    assert query.rewrite_query(client, "第10页那个订单呢", {}, []) == "第 10 页的订单增长数据"
    assert query.rewrite_query(FakeClient(complete_out="  "), "那订单呢", {}, []) == "那订单呢"


# ── GAPS 剥离（产品评审 M1）──────────────────────────────────

def test_split_gaps():
    text = "AI驱动订单增长15倍[S5]。\n\n据第5页图示，GEO Phase 1 的注入点是实时外部索引[S9]。\nGAPS: [\"中国出海品牌的整体规模数据\", \"GEO 投入成本\"]"
    body, gaps = writer.split_gaps(text)
    assert body == "AI驱动订单增长15倍[S5]。\n\n据第5页图示，GEO Phase 1 的注入点是实时外部索引[S9]。"
    assert gaps == ["中国出海品牌的整体规模数据", "GEO 投入成本"]
    assert writer.split_gaps("正文[S1]。\nGAPS: []") == ("正文[S1]。", [])
    assert writer.split_gaps("GAPS：[\"缺整体统计\"]") == ("", ["缺整体统计"])   # 正文留空、全角冒号
    assert writer.split_gaps("正文[S1]。\n`GAPS: [缺A、缺B]`") == ("正文[S1]。", ["缺A", "缺B"])  # 非法 JSON 兜底
    assert writer.split_gaps("没有 GAPS 行[S1]。") == ("没有 GAPS 行[S1]。", [])


# ── JSON-LD（产品评审 S1）────────────────────────────────────

def test_faq_jsonld():
    doc = jsonld.faq_jsonld([
        ("为什么出海品牌要关注 AI 搜索？", "据 Shopify 2025 年数据，AI驱动订单增长15倍[S5]。AI驱动流量增长8倍[S4][S6]。"),
        ("空答案会被跳过吗？", "[S3]"),
    ])
    assert doc["@context"] == "https://schema.org" and doc["@type"] == "FAQPage"
    assert doc["mainEntity"] == [{
        "@type": "Question",
        "name": "为什么出海品牌要关注 AI 搜索？",
        "acceptedAnswer": {"@type": "Answer", "text": "据 Shopify 2025 年数据，AI驱动订单增长15倍。AI驱动流量增长8倍。"},
    }]
    json.dumps(doc, ensure_ascii=False)  # 可直接序列化进 <script type="application/ld+json">


def test_plan_parses_thesis_purpose_products():
    """10-23：Blog 有中心论点、各节有论证作用；产品介绍先选品。"""
    raw = json.dumps({"title": "T", "thesis": "品牌要被 AI 记住", "products": ["GEO 监控系统"],
                      "lede_points": [], "ending_points": [], "insufficient": [],
                      "sections": [{"heading": "AI 正在成为入口", "purpose": "立住现象", "sub_question": "q",
                                    "expected_pages_hint": [9]}]})
    p = planner.parse_plan(raw)
    assert p.thesis == "品牌要被 AI 记住" and p.products == ["GEO 监控系统"]
    assert p.sections[0].purpose == "立住现象"


def test_section_writer_gets_full_text_context(facts, page_facts, profile):
    """写每节时带上全文论点、大纲与前文小节，节与节才接得上。"""
    client = FakeClient(stream_out=("x",))
    pack = _pack(facts, page_facts)
    sec = PlanSection(heading="品牌不被记住就会失去流量", sub_question="q", purpose="承接上一节")
    list(writer.stream_section(client, sec, pack, genre="blog", profile=profile, thesis="论点X",
                               outline=["A", "品牌不被记住就会失去流量"], prev_headings=["A"]))
    u = client.last["user"]
    assert "论点X" in u and "承接上一节" in u and "前面已经写完的小节：A" in u


def test_polish_blocks_roundtrip():
    out = writer.parse_blocks("<<<LEDE>>>\n导语[S1]。\n\n<<<SECTION 2>>>\n（小标题：H）\n正文[S3]。\n<<<ENDING>>>\n收尾。")
    assert out == {"LEDE": "导语[S1]。", "SECTION 2": "正文[S3]。", "ENDING": "收尾。"}


def test_check_topic_keeps_selected_genre_and_strips_digits(profile):
    from ragwriter.generate import topic as topic_mod

    class C:
        def complete_json(self, system, user, **kw):
            self.user = user
            return {"valid": True, "genre": "faq", "topic": "合规 GEO", "angle": "合规带来 15 倍增长",
                    "outline_points": ["要点 1"], "premise_gaps": [], "alternatives": [], "reason": "",
                    "confirm_text": "我理解的主题是合规 GEO，可以吗？"}
    c = C()
    r = topic_mod.check_topic(c, "写篇 Blog 讲合规", genre="blog", topic=None, profile=profile)
    assert r.genre == "blog"                      # 写作台指定的体裁不让模型改
    assert "15" not in r.angle                    # 这一步看不到正文，任何数字都没经过检索
    assert "品宣资料页标题目录" in c.user and "写篇 Blog 讲合规" in c.user
