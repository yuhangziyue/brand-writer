"""V1–V4 句级校验与校验报告的离线测试。

数据用真实的 build/facts.jsonl 第 9、10 页组装，不依赖索引层：
校验闸要拦的正是这两页上真实出现过的错法（62 亿、15 倍挂到流量上）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragwriter.contracts import Sentence, Snippet
from ragwriter.models import Fact, Metric
from ragwriter.verify.checks import is_factual, number_tokens, token_in_text, verify_sentence
from ragwriter.verify.report import build_report
from ragwriter.verify.sentences import parse_cites, split_sentences, strip_cites

FACTS = Path(__file__).resolve().parents[1] / "build" / "facts.jsonl"


@pytest.fixture(scope="module")
def facts() -> dict[str, Fact]:
    if not FACTS.exists():
        pytest.skip("build/facts.jsonl 未生成")
    out = {}
    for line in FACTS.read_text(encoding="utf-8").splitlines():
        f = Fact.model_validate_json(line)
        if f.page_no in (9, 10):
            out[f.fact_id] = f
    return out


def _snip(sid: str, f: Fact) -> Snippet:
    return Snippet(sid=sid, fact_id=f.fact_id, page_no=f.page_no, text=f.text, status=f.status, metric=f.metric)


@pytest.fixture(scope="module")
def cmap(facts) -> dict[str, Snippet]:
    # S1 ChatGPT 61.66亿(p9) / S2 Gemini(p9) / S3 DeepSeek 3.55亿(p9)
    # S4 流量8倍(p10) / S5 订单15倍(p10) / S6 64%(p10) / S7 $5万亿(p10) / S8 OpenAI 购物功能(p10)
    ids = ["p09-f01", "p09-f02", "p09-f03", "p10-f01", "p10-f02", "p10-f03", "p10-f04", "p10-f05"]
    return {f"S{i + 1}": _snip(f"S{i + 1}", facts[fid]) for i, fid in enumerate(ids)}


@pytest.fixture(scope="module")
def page_metrics(facts):
    by_page: dict[int, list[Metric]] = {}
    for f in facts.values():
        if f.kind == "metric" and f.metric:
            by_page.setdefault(f.page_no, []).append(f.metric)
    return lambda p: by_page.get(p, [])


def _run(text: str, cmap, page_metrics):
    [s] = split_sentences(text)
    v = verify_sentence(s, cmap, page_metrics)
    return v, {r.check: r for r in v.results}


# ── 断句 ─────────────────────────────────────────────────────
def test_cites_before_and_after_terminator():
    text = "AI 驱动流量增长 8 倍[S4]。订单增长 15 倍。[S5][S6]第三句 [S4, S5]；第四句[S7，S8]！"
    ss = split_sentences(text)
    assert [s.cites for s in ss] == [["S4"], ["S5", "S6"], ["S4", "S5"], ["S7", "S8"]]
    assert ss[1].text == "订单增长 15 倍。[S5][S6]"
    assert [s.idx for s in ss] == [0, 1, 2, 3]


def test_headings_blank_lines_and_list_items():
    text = "# 标题\n\n## 小标题\n- 第一项 8 倍[S4]\n1. 第二项。第三项[S5]\n3.55亿是 DeepSeek 的数字[S3]\n"
    ss = split_sentences(text)
    assert [s.text for s in ss] == ["第一项 8 倍[S4]", "第二项。", "第三项[S5]", "3.55亿是 DeepSeek 的数字[S3]"]


def test_orphan_cite_line_merges_back_and_decimal_not_split():
    ss = split_sentences("ChatGPT 月访问量 61.66 亿，统计口径 2026.4。\n[S1]")
    assert len(ss) == 1 and ss[0].cites == ["S1"]


def test_parse_and_strip_cites():
    assert parse_cites("a[S4][s5]b[S4, S6]") == ["S4", "S5", "S6"]
    assert strip_cites("增长 8 倍 [S4]。") == "增长 8 倍。"


# ── 数字 token ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,expect",
    [
        ("61.66亿", ["61.66亿"]), ("15 倍", ["15倍"]), ("64%", ["64%"]), ("$5万亿", ["5万亿美元"]),
        ("95%+", ["95%+"]), ("300+ 品牌", ["300+"]), ("2025 年", ["2025年"]), ("2 个月", ["2个月"]),
        ("3—5 倍", ["3-5倍"]), ("11,000次", ["11000次"]), ("2026年4月上线", ["2026年4月"]),
        ("GPT-4o 与 B2B 不算数字", []), ("[S4] 不算数字", []),
    ],
)
def test_number_tokens(text, expect):
    assert number_tokens(text) == expect


def test_token_boundaries():
    assert not token_in_text("5倍", "AI驱动订单增长：15倍")
    assert not token_in_text("1.66亿", "ChatGPT：61.66亿")
    assert not token_in_text("5万", "$5万亿")
    assert token_in_text("2025年", "Shopify官方数据(2025)")
    assert token_in_text("2026年4月", "统计, 2026.4；来源")
    assert token_in_text("61.66亿次", "ChatGPT：61.66亿")


# ── V1 ──────────────────────────────────────────────────────
def test_v1_unknown_cite_blocked(cmap, page_metrics):
    v, r = _run("ChatGPT 月访问量 61.66 亿[S99]。", cmap, page_metrics)
    assert not r["V1"].ok and "S99" in r["V1"].reason
    assert [x.check for x in v.results] == ["V1", "V2", "V3", "V4", "V5"]


# ── V2 ──────────────────────────────────────────────────────
def test_v2_wrong_number_blocked(cmap, page_metrics):
    _, r = _run("ChatGPT 月访问量 62 亿[S1]。", cmap, page_metrics)
    assert not r["V2"].ok
    assert "62亿" in r["V2"].reason and "第9页" in r["V2"].reason


def test_v2_unknown_latin_name_blocked(cmap, page_metrics):
    _, r = _run("据 Similarweb 统计，ChatGPT 月访问量 61.66 亿[S1]。", cmap, page_metrics)
    assert not r["V2"].ok and "Similarweb" in r["V2"].reason


def test_v2_source_year_and_spacing_pass(cmap, page_metrics):
    v, r = _run("据 Shopify 2025 年数据，AI 驱动订单增长 15 倍[S5]。", cmap, page_metrics)
    assert r["V2"].ok, r["V2"].reason
    assert v.ok


def test_v2_dollar_and_date(cmap, page_metrics):
    _, r = _run("麦肯锡预测，到 2030 年全球 AI 电商市场规模将达 5 万亿美元[S7]。", cmap, page_metrics)
    assert r["V2"].ok, r["V2"].reason
    _, r = _run("2025年4月，OpenAI 向 ChatGPT 添加了购物功能[S8]。", cmap, page_metrics)
    assert r["V2"].ok, r["V2"].reason


# ── V3 ──────────────────────────────────────────────────────
def test_v3_wrong_object_blocked(cmap, page_metrics):
    _, r = _run("AI 驱动流量增长 15 倍[S5]。", cmap, page_metrics)
    assert r["V2"].ok  # 数字本身是真的
    assert not r["V3"].ok
    assert "订单" in r["V3"].reason and "流量" in r["V3"].reason


def test_v3_both_pairs_in_one_sentence_pass(cmap, page_metrics):
    v, r = _run("AI 驱动流量增长 8 倍、订单增长 15 倍[S4][S5]。", cmap, page_metrics)
    assert r["V3"].ok, r["V3"].reason
    assert v.ok


def test_v3_latin_label_fallback(cmap, page_metrics):
    # ChatGPT 的区分词被 builder 算成残片「h」「T」，V3 改用整词「ChatGPT」
    v, r = _run("ChatGPT 月访问量 61.66 亿[S1]。", cmap, page_metrics)
    assert r["V3"].ok, r["V3"].reason
    assert v.ok
    _, r = _run("Gemini 月访问量 61.66 亿[S1]。", cmap, page_metrics)
    assert not r["V3"].ok


def test_v3_same_label_twins_fall_back_to_label(cmap, page_metrics):
    # 第 9 页两条 DeepSeek（两个统计口径）互相抵消，builder 给空区分词；
    # V3 改为只跟不同标签的兄弟比，「DeepSeek」本身就是区分词
    v, r = _run("DeepSeek 月访问量 3.55 亿[S3]。", cmap, page_metrics)
    assert r["V3"].ok, r["V3"].reason
    _, r = _run("ChatGPT 月访问量 3.55 亿[S3]。", cmap, page_metrics)
    assert not r["V3"].ok and "ChatGPT" in r["V3"].reason


def test_v3_empty_discriminator_skipped_with_reason():
    # 第 2 页的真实形态：两条同名「发表论文」，其余标签把「发」「论文」都占了，只剩单字「表」
    twins = [
        Metric(value="30+篇", label="发表论文"), Metric(value="40+篇", label="发表论文"),
        Metric(value="11000次", label="论文被引次数"), Metric(value="50大", label="全球50大发明之一"),
        Metric(value="80%以上", label="AI研发开发人员占比"),
    ]
    sn = Snippet(sid="S1", fact_id="p02-f02", page_no=2, text="发表论文：30+篇", status="verified", metric=twins[0])
    [s] = split_sentences("团队发表论文 30+ 篇[S1]。")
    r = verify_sentence(s, {"S1": sn}, lambda p: twins).results[2]
    assert r.ok and "跳过" in r.reason


# ── V4 ──────────────────────────────────────────────────────
def test_v4_uncited_number_blocked(cmap, page_metrics):
    _, r = _run("AI 驱动订单增长 15 倍。", cmap, page_metrics)
    assert not r["V4"].ok and "15倍" in r["V4"].reason


def test_v4_faq_question_and_insufficient_exempt(cmap, page_metrics):
    _, r = _run("Q：Shopify 2025 年的 AI 订单增长了多少？", cmap, page_metrics)
    assert all(x.ok for x in r.values())
    _, r = _run("万悉科技的融资情况，品宣资料未涉及。", cmap, page_metrics)
    assert r["V4"].ok


def test_is_factual():
    assert is_factual("64% 的消费者愿意用 AI 购物")
    assert is_factual("Trendee 推出了 GEO 解决方案")
    assert not is_factual("下面分三部分展开。")
    assert not is_factual("AI 搜索为什么重要？")


# ── 正常段落全过 + 报告 ──────────────────────────────────────
PARA = (
    "## AI 正在重塑购物\n"
    "据 Shopify 2025 年数据，AI 驱动流量增长 8 倍[S4]，订单增长 15 倍[S5]。"
    "64% 的消费者表示愿意使用 AI 购物[S6]。\n"
    "麦肯锡预测全球 AI 电商市场规模将达 $5 万亿[S7]。\n"
    "下面看具体案例。\n"
)


def test_normal_paragraph_all_pass(cmap, page_metrics):
    ss = split_sentences(PARA)
    assert len(ss) == 4
    vs = [verify_sentence(s, cmap, page_metrics) for s in ss]
    bad = [(v.sentence.text, r.check, r.reason) for v in vs for r in v.results if not r.ok]
    assert bad == []


def test_build_report_counts(cmap, page_metrics):
    ss = split_sentences(PARA + "AI 驱动流量增长 15 倍[S5]。ChatGPT 月访问量 62 亿。")
    vs = [verify_sentence(s, cmap, page_metrics) for s in ss]
    rep = build_report(
        vs,
        rewrites=[{"original": "x", "failed_checks": ["V3"], "rewritten": "y", "round": 1}],
        deletions=[{"original": "z", "failed_checks": "V2,V4", "reason": "两轮仍不过"}],
        insufficient=[{"topic": "融资情况", "signal": "top_score<τ"}],
        corrections=[{"original": "同义千问", "corrected": "通义千问"}],
        citation_map=cmap,
    )
    assert rep["sentence_total"] == 6
    assert rep["factual_sentences"] == 5
    assert rep["cited_sentences"] == 4
    assert rep["passed_sentences"] == 4
    # V3：终稿 1 + 改写 1；V2：删除 1（62 亿无引用，V2 不重复计）；V4：终稿 1 + 删除 1
    assert rep["gate_blocks"] == {"V1": 0, "V2": 1, "V3": 2, "V4": 2, "V5": 0}
    nums = {(r["number"], r["sentence"][:6]): r for r in rep["numbers"]}
    row = nums[("8倍", "据 Shop")]
    assert row["pages"] == [10] and row["consistent"] and "8倍" in row["snippet"]
    assert nums[("62亿", "ChatGP")]["consistent"] is False
    assert rep["numbers_inconsistent"] == 1
    assert len(rep["insufficient"]) == 1 and len(rep["corrections"]) == 1
    json.dumps(rep, ensure_ascii=False)  # 要能直接进 SSE / 落盘


def test_v2_page_reference_is_not_a_fact_number():
    """「据第 N 页图示」是规定的出处写法，页码不能被当成事实数字拦下；
    但指向未引用页的页码照常回查。集成时发现：不豁免的话所有图示句都会被删。"""
    from ragwriter.contracts import Snippet
    from ragwriter.verify.checks import check_v2
    from ragwriter.verify.sentences import split_sentences

    cm = {"S1": Snippet(sid="S1", fact_id="p05-v26", page_no=5,
                        text="GEO Phase 1 的注入点是实时外部索引环节", status="image_structural")}
    ok = split_sentences("据第 5 页图示，GEO Phase 1 的注入点是实时外部索引环节[S1]。")[0]
    assert check_v2(ok, cm).ok
    bad = split_sentences("据第 9 页图示，GEO Phase 1 的注入点是实时外部索引环节[S1]。")[0]
    assert not check_v2(bad, cm).ok


def test_v5_lexical_catches_unrelated_citation():
    """10-22 真跑实例：改写为过 V4 挂了第 32 页 OEM 案例结构，字面支持度 0.05。"""
    from ragwriter.contracts import Snippet
    from ragwriter.verify.checks import check_v5_lexical
    from ragwriter.verify.sentences import split_sentences

    cm = {"S18": Snippet(sid="S18", fact_id="p32-v16", page_no=32, status="image_structural",
                         text="案例的三段结构：痛点（原有传统外贸获客渠道客户流量持续减少、在AI世界不存在）到 解决方案 再到 效益转化"),
          "S14": Snippet(sid="S14", fact_id="p10-f02", page_no=10, status="verified",
                         text="AI驱动订单增长：15倍（Shopify官方数据(2025)）")}
    bad = split_sentences("当买家不再逐页翻找搜索结果而是向 AI 提问时，能否被看见就决定了订单是否流向商家[S18]。")[0]
    assert not check_v5_lexical(bad, cm).ok
    good = split_sentences("据 Shopify 官方数据（2025），AI 驱动订单增长 15 倍[S14]。")[0]
    assert check_v5_lexical(good, cm).ok


def test_v4_strict_for_lede():
    from ragwriter.verify.checks import check_v4
    from ragwriter.verify.sentences import split_sentences

    s = split_sentences("AI 搜索正在取代传统搜索成为用户获取品牌信息的第一入口。")[0]
    assert check_v4(s).ok                    # 正文里：无数字无断言词，按非事实句放过
    assert not check_v4(s, strict=True).ok   # 导语里：一律要引用


def test_v2_scope_and_case_attribution():
    """10-22 第二次真跑的两类漏网：全球数据写成「海外买家」；第 32 页案例痛点写成普遍现象。"""
    from ragwriter.contracts import Snippet
    from ragwriter.verify.checks import check_v2, check_v4
    from ragwriter.verify.sentences import split_sentences

    cm = {"S15": Snippet(sid="S15", fact_id="p10-f01", page_no=10, status="verified",
                         text="AI驱动流量增长：8倍（Shopify官方数据(2025)）"),
          "S23": Snippet(sid="S23", fact_id="p32-f02", page_no=32, status="verified",
                         text="原有传统外贸获客渠道客户流量持续减少、在AI世界不存在，无法被AI看到，被竞品反超")}
    one = lambda t: split_sentences(t)[0]
    assert not check_v2(one("AI 驱动流量增长 8 倍，正在改变海外买家的选择方式[S15]。"), cm).ok
    assert check_v2(one("AI 驱动流量增长 8 倍[S15]。"), cm).ok
    pain = "原有传统外贸获客渠道客户流量持续减少，无法被 AI 看到[S23]。"
    assert not check_v2(one(pain), cm, case_pages=frozenset({32})).ok
    assert check_v2(one("在一家牛仔服饰工厂的案例中，" + pain), cm, case_pages=frozenset({32})).ok
    assert not check_v4(one("海外用户获取信息的第一站正在转向 AI 对话。")).ok


def test_number_unit_kept_before_latin_word():
    from ragwriter.verify.checks import number_tokens
    assert number_tokens("AI 驱动订单增长 15 倍 Shopify 数据") == ["15倍"]
    assert number_tokens("61.66亿ChatGPT") == ["61.66亿"]
    assert number_tokens("GPT-4o 发布") == []
