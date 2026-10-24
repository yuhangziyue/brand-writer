"""评估 agent 的离线测试：自造 CaseRun 样本，逐项验证每个指标确实能区分好坏。

不碰网络、不需要 key；Judge 用假 client。语料用测试内自建的小 Corpus（取自第 9/10/30/31 页真实事实），
不依赖 build/ 产物是否重建过。
"""

from __future__ import annotations

import json

import pytest
import yaml

from ragwriter.eval import evaluate as E
from ragwriter.eval import metrics as M
from ragwriter.eval.judge import Judge, resolve_judge_model

SHOP = "Shopify官方数据(2025)"
FACTS = {
    "p09-f01": {"fact_id": "p09-f01", "page_no": 9, "kind": "point", "status": "verified", "metric": None,
                "text": "AI搜索正成为新的超级流量入口"},
    "p10-f01": {"fact_id": "p10-f01", "page_no": 10, "kind": "metric", "status": "verified",
                "text": f"AI驱动流量增长：8倍（{SHOP}）",
                "metric": {"value": "8倍", "label": "AI驱动流量增长", "source_note": SHOP, "method": "llm",
                           "discriminators": ["流量"]}},
    "p10-f02": {"fact_id": "p10-f02", "page_no": 10, "kind": "metric", "status": "verified",
                "text": f"AI驱动订单增长：15倍（{SHOP}）",
                "metric": {"value": "15倍", "label": "AI驱动订单增长", "source_note": SHOP, "method": "llm",
                           "discriminators": ["订单"]}},
    "p30-f01": {"fact_id": "p30-f01", "page_no": 30, "kind": "point", "status": "verified", "metric": None,
                "text": "外贸制造：跨境出海与智能制造"},
    "p31-v01": {"fact_id": "p31-v01", "page_no": 31, "kind": "point", "status": "image_only", "metric": None,
                "text": "Nihaojewelry 站点月营收截图"},
    "p20-f01": {"fact_id": "p20-f01", "page_no": 20, "kind": "point", "status": "verified", "metric": None,
                "text": "品牌价值蒸馏：提炼品牌核心价值，形成可被AI记住的差异化标签体系"},
}
PAGES = {
    9: {"page_no": 9, "chapter": "01 GEO背景介绍", "title": "AI搜索成为新的超级流量入口", "text": ""},
    10: {"page_no": 10, "chapter": "01 GEO背景介绍", "title": "AI正在重塑购物：AI电商迎来爆发式增长", "text": ""},
    20: {"page_no": 20, "chapter": "03 技术体系", "title": "品牌价值蒸馏", "text": ""},
    30: {"page_no": 30, "chapter": "04 服务行业", "title": "服务行业：外贸制造 跨境出海", "text": ""},
    31: {"page_no": 31, "chapter": "05 案例", "title": "客户案例", "text": ""},
}
CORPUS = M.Corpus(facts=FACTS, pages=PAGES)


def cit(sid: str, fid: str) -> dict:
    f = FACTS[fid]
    return {sid: {"fact_id": fid, "page_no": f["page_no"], "text": f["text"], "status": f["status"]}}


CITS = {**cit("S1", "p09-f01"), **cit("S2", "p10-f01"), **cit("S3", "p10-f02"), **cit("S4", "p30-f01")}

GOOD_BLOG = """# 为什么中国出海品牌需要 GEO

AI 搜索正成为新的超级流量入口[S1]。

## AI 搜索正在改变流量格局

AI搜索正成为新的超级流量入口[S1]。

## AI 电商的增长

据 Shopify 2025 年数据，AI 驱动流量增长 8 倍[S2]，AI 驱动订单增长 15 倍[S3]。

## 适合出海行业

外贸制造正走向跨境出海与智能制造[S4]。

## 结语

因此，出海品牌值得尽早布局 GEO。
"""

WRITE_EXPECT = {
    "intent": "write", "genre": "blog", "forbidden_strings": ["Nihaojewelry"],
    "numbers": [{"value": "15倍", "object": "订单"}, {"value": "8倍", "object": "流量"}],
    "key_points": [{"text": "AI搜索正成为新的超级流量入口", "page": 9},
                   {"text": "AI驱动订单增长 15倍 Shopify官方数据(2025)", "page": 10},
                   {"text": "外贸制造 跨境出海与智能制造", "page": 30}],
    "must_say_insufficient": ["中国出海品牌的总体统计"],
}
BOUNDARY = [{"topic": "中国出海品牌的总体统计数据", "source": "plan", "nearest_pages": [30]}]


def turn(final_text: str, *, intent: str = "write", citations: dict | None = None, events: list | None = None,
         **kw) -> dict:
    t = {"seq": 0, "user": "u", "intent": intent, "final_text": final_text,
         "citations": CITS if citations is None else citations, "events": events or []}
    t.update(kw)
    return t


def case_of(expect: dict, *, cid: str = "C1", options: dict | None = None) -> dict:
    return {"id": cid, "title": "t", "turns": [{"user": "u", "expect": expect, "options": options}]}


def run_of(*turns: dict, cid: str = "C1") -> dict:
    return {"case_id": cid, "turns": list(turns)}


def score(expect: dict, t: dict, *, judge=None, options=None) -> E.TurnScore:
    return E.evaluate_run(case_of(expect, options=options), run_of(t), judge=judge, corpus=CORPUS).turns[0]


# ── 假评审 ───────────────────────────────────────────────────
class FakeClient:
    """按任务返回预设 JSON，并记录收到的 schema，验证 strict 结构化输出确实传下去了。"""

    def __init__(self, support=None, covers=None, expansion=None):
        self.resp = {"support": support, "covers": covers, "expansion": expansion}
        self.calls: list[dict] = []

    def complete_json(self, system, user, *, json_schema=None, model=None, max_tokens=4096, tag=""):
        task = tag.split(".")[-1]
        self.calls.append({"task": task, "schema": json_schema, "model": model, "user": user})
        r = self.resp[task]
        return r(user) if callable(r) else r


def fake_judge(**kw) -> Judge:
    return Judge(FakeClient(**kw), model="judge-x", generator_model="gen-y")


# ── 原文匹配度 ───────────────────────────────────────────────
def test_good_blog_scores_full_on_deterministic_items():
    ts = score(WRITE_EXPECT, turn(GOOD_BLOG, boundary=BOUNDARY))
    f, c = ts.faithfulness, ts.completeness
    assert f["引用覆盖率"].value == 1.0
    assert f["数字一致率"].value == 1.0 and f["数字一致率"].den == 2
    assert f["数字对象一致率"].value == 1.0
    assert f["越界内容"].value == 1.0
    assert f["期望数字命中"].value == 1.0
    assert c["要点覆盖率"].value == 1.0
    assert c["体裁结构"].value == 1.0, c["体裁结构"].detail
    assert c["资料不足处理"].value == 1.0
    assert ts.gates_failed == []
    assert ts.faith_score == 1.0


def test_fabricated_number_breaks_number_consistency_and_gate():
    bad = GOOD_BLOG.replace("订单增长 15 倍", "订单增长 20 倍")
    ts = score(WRITE_EXPECT, turn(bad, boundary=BOUNDARY))
    assert ts.faithfulness["数字一致率"].value == 0.5
    assert ts.faithfulness["引用支持率·确定性"].value < 1.0
    assert "数字一致率 < 100%" in ts.gates_failed
    assert any("20倍" in x for x in ts.findings)


def test_15x_attached_to_wrong_object_is_caught_by_v3():
    bad = GOOD_BLOG.replace("AI 驱动流量增长 8 倍[S2]，AI 驱动订单增长 15 倍[S3]", "AI 驱动流量增长 15 倍[S2][S3]")
    ts = score(WRITE_EXPECT, turn(bad, boundary=BOUNDARY))
    # 数字本身在片段里（V2 过），但挂错了对象（V3 不过）
    assert ts.faithfulness["数字一致率"].value == 1.0
    assert ts.faithfulness["数字对象一致率"].value == 0.0
    assert "数字对象一致率 < 100%" in ts.gates_failed
    assert ts.faithfulness["期望数字命中"].value < 1.0


def test_citing_image_only_fact_is_out_of_bounds():
    cits = {**CITS, **cit("S9", "p31-v01")}
    bad = GOOD_BLOG.replace("外贸制造正走向跨境出海与智能制造[S4]。", "外贸制造正走向跨境出海与智能制造[S4][S9]。")
    ts = score(WRITE_EXPECT, turn(bad, citations=cits, boundary=BOUNDARY))
    assert ts.faithfulness["越界内容"].value == 0.0
    assert "越界内容" in ts.gates_failed
    assert any("image_only" in x for x in ts.findings)


def test_image_only_status_backfilled_from_corpus_when_missing():
    cits = {"S9": {"fact_id": "p31-v01", "page_no": 31, "text": "截图"}}  # 编排层漏填 status
    assert M.non_citable_cites("营收很高[S9]。", cits, CORPUS)[0]["status"] == "image_only"


def test_forbidden_string_hit():
    ts = score(WRITE_EXPECT, turn(GOOD_BLOG + "\n案例站点 Nihaojewelry 月营收可观。", boundary=BOUNDARY))
    assert ts.faithfulness["越界内容"].value == 0.0


def test_uncited_fact_sentence_lowers_coverage():
    bad = GOOD_BLOG.replace("外贸制造正走向跨境出海与智能制造[S4]。", "据 Shopify 数据，外贸制造正走向跨境出海与智能制造。")
    ts = score(WRITE_EXPECT, turn(bad, boundary=BOUNDARY))
    assert ts.faithfulness["引用覆盖率"].value < 1.0


def test_qa_reference_elements_not_bleu():
    exp = {"intent": "qa", "reference_answer": "AI 驱动订单增长 15 倍，来自 Shopify 2025 年数据。",
           "key_points": [{"text": "AI驱动订单增长 15倍", "page": 10}]}
    good = score(exp, turn("据 Shopify 2025 年数据，AI 驱动订单增长 15 倍[S3]。", intent="qa"))
    bad = score(exp, turn("据 Shopify 2025 年数据，AI 驱动订单增长 8 倍[S3]。", intent="qa"))
    assert good.faithfulness["参考要素匹配"].value == 1.0
    assert bad.faithfulness["参考要素匹配"].value < 1.0
    # 字符级 F1 只作参考：错答的 F1 仍然很高，正说明它不能当分数用
    assert bad.reference_only["字符级F1(仅供参考)"] > 0.8
    assert "字符级F1(仅供参考)" not in bad.faithfulness


# ── 扩写完整度 ───────────────────────────────────────────────
def test_missing_key_point_lowers_coverage():
    missing = GOOD_BLOG.replace("## 适合出海行业\n\n外贸制造正走向跨境出海与智能制造[S4]。\n\n", "")
    ts = score(WRITE_EXPECT, turn(missing, boundary=BOUNDARY))
    assert ts.completeness["要点覆盖率"].value == pytest.approx(2 / 3)
    assert any("漏要点（第30页）" in x for x in ts.findings)


def test_uncertain_key_point_is_unrated_without_judge_and_judged_with_one():
    # 只引了第 10 页、却没写 15 倍：页对上、关键词没过半 ⇒ 不确定
    text = "# T\n\n导语[S2]。\n\n## a\n\nShopify 数据显示 AI 驱动流量增长 8 倍[S2]。"
    exp = {"intent": "write", "key_points": [{"text": "AI驱动订单增长 15倍 Shopify官方数据(2025)", "page": 10}]}
    no_j = score(exp, turn(text))
    assert no_j.completeness["要点覆盖率"].status == "未评"
    j = fake_judge(covers={"verdict": "未覆盖", "evidence": [], "reason": "只写了流量"})
    with_j = score(exp, turn(text), judge=j)
    assert with_j.completeness["要点覆盖率"].value == 0.0


def test_faq_with_only_three_pairs_fails_structure():
    faq3 = "# FAQ\n\n" + "".join(
        f"## AI 驱动订单增长多少{i}？\n\nAI 驱动订单增长 15 倍[S3]。\n\n" for i in range(3))
    ts = score({"intent": "write", "genre": "faq"}, turn(faq3))
    st = ts.completeness["体裁结构"]
    assert st.value < 1.0 and "Q/A 组数" in st.detail


def _faq(n: int, *, preamble: bool = False) -> str:
    body = "# FAQ\n\n"
    for i in range(n):
        a = "随着 AI 的发展，情况很复杂。" if preamble and i == 0 else "AI 驱动订单增长 15 倍[S3]。"
        body += f"## AI 驱动订单增长多少（{i}）？\n\n{a}\n\n"
    return body


def _jsonld(n: int) -> dict:
    return {"@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": f"q{i}", "acceptedAnswer": {"@type": "Answer", "text": "a"}}
                           for i in range(n)]}


def test_faq_direct_answer_and_jsonld():
    good = M.check_structure(_faq(5), "faq", jsonld=_jsonld(5))
    assert good.ok, good.items
    pre = M.check_structure(_faq(5, preamble=True), "faq", jsonld=_jsonld(5))
    assert not pre.items["答案首句直接回答"]
    mismatch = M.check_structure(_faq(5), "faq", jsonld=_jsonld(4))
    assert not mismatch.items["JSON-LD 为 FAQPage 且组数一致"]
    assert not M.check_structure(_faq(5), "faq", jsonld=None).items["JSON-LD 为 FAQPage 且组数一致"]


def test_blog_lede_must_lead_with_conclusion():
    pre = GOOD_BLOG.replace("AI 搜索正成为新的超级流量入口[S1]。\n\n## AI 搜索正在", "随着 AI 时代到来，一切都在变化。\n\n## AI 搜索正在")
    assert not M.check_structure(pre, "blog").items["导语结论先行"]
    assert M.check_structure(GOOD_BLOG, "blog").items["导语结论先行"]


def test_blog_without_ending_and_too_few_sections():
    st = M.check_structure("# T\n\n导语句子在这里[S1]。\n\n## a\n\n内容[S1]。\n\n## b\n\n内容[S1]。", "blog")
    assert not st.items["正文小节 3–4"] and not st.items["有结尾"]


def test_brand_intro_sections_order():
    ok = ("# 万悉科技\n\n万悉科技是一家 GEO 公司[S1]。\n\n## 我们是谁\n\n团队介绍[S1]。\n\n## 我们做什么\n\n业务[S1]。\n\n"
          "## 凭什么做好\n\n技术[S1]。\n\n## 服务谁\n\n客户行业[S1]。")
    bad = ok.replace("## 我们是谁\n\n团队介绍[S1]。\n\n", "").replace("## 服务谁", "## 服务谁\n\n客户[S1]。\n\n## 我们是谁\n\n团队介绍[S1]。\n\n## 其他")
    assert M.check_structure(ok, "brand_intro").ok
    st = M.check_structure(bad, "brand_intro")
    assert not st.items["节目顺序"]
    assert "节目清单来源" in st.note


def test_genre_sections_loaded_from_prompts_genres(tmp_path):
    (tmp_path / "brand_intro.yaml").write_text(
        yaml.safe_dump({"sections": [{"name": "愿景", "keywords": ["愿景"]}, "荣誉"]}, allow_unicode=True), encoding="utf-8")
    spec, src = M.load_genre_sections("brand_intro", tmp_path)
    assert [s["name"] for s in spec] == ["愿景", "荣誉"] and "genres/" in src


def test_write_boundary_rules():
    subs = ["中国出海品牌的总体统计"]
    ok = M.check_boundary_write(GOOD_BLOG, subs, BOUNDARY)
    assert ok[0].status == "ok"
    assert M.check_boundary_write(GOOD_BLOG, subs, [])[0].status == "missing_boundary"
    phrase = GOOD_BLOG + "\n中国出海品牌的总体统计，品宣资料未涉及。"
    assert M.check_boundary_write(phrase, subs, BOUNDARY)[0].status == "body_phrase"
    fab = GOOD_BLOG + "\n中国出海品牌总体统计显示已超过 3 万家。"
    assert M.check_boundary_write(fab, subs, BOUNDARY)[0].status == "fabricated"


def test_qa_insufficient_must_be_explicit():
    subs = ["万悉的融资情况"]
    assert M.check_insufficient_qa("万悉的融资情况，品宣资料未涉及。", subs)[0].ok
    assert M.check_insufficient_qa("万悉已完成数千万元融资[S1]。", subs)[0].status == "fabricated"
    assert M.check_insufficient_qa("万悉专注 GEO 技术[S1]。", subs)[0].status == "skipped"


# ── 拒答 / 追问 / 解释 ───────────────────────────────────────
def test_refusal_without_nearest_pages_is_flagged():
    """资料不足类拒答必须给最近页；越界拒答（写代码、问天气）与品宣无关，不考核这一项。"""
    exp = {"intent": "qa", "refused": True, "refusal_kind": "insufficient_all", "forbidden_strings": ["Nihaojewelry"]}
    good = score(exp, turn("品宣资料未涉及该内容。", intent="qa", citations={}, refused=True,
                           refusal_kind="insufficient_all", nearest_pages=[9, 10]))
    bad = score(exp, turn("品宣资料未涉及该内容。", intent="qa", citations={}, refused=True,
                          refusal_kind="insufficient_all", nearest_pages=[]))
    assert good.behavior["给出 nearest_pages"].value == 1.0
    assert bad.behavior["给出 nearest_pages"].value == 0.0
    assert any("最近页" in x for x in bad.findings)
    oos = score({"intent": "out_of_scope", "refused": True},
                turn("这超出了品宣资料的范围。", intent="out_of_scope", citations={}, refused=True,
                     refusal_kind="out_of_scope", nearest_pages=[]))
    assert "给出 nearest_pages" not in oos.behavior


def test_refusal_kind_distinguishes_insufficient_all_from_out_of_scope():
    exp = {"intent": "qa", "refused": True, "refusal_kind": "insufficient_all", "must_say_insufficient": ["融资"]}
    wrong = score(exp, turn("万悉的融资情况，品宣资料未涉及。", intent="qa", citations={}, refused=True,
                            refusal_kind="out_of_scope", nearest_pages=[9]))
    right = score(exp, turn("万悉的融资情况，品宣资料未涉及。", intent="qa", citations={}, refused=True,
                            refusal_kind="insufficient_all", nearest_pages=[9]))
    assert wrong.behavior["拒答理由正确"].value == 0.0
    assert right.behavior["拒答理由正确"].value == 1.0
    assert right.completeness["资料不足处理"].value == 1.0


def test_trap_not_refused_fails_gate():
    exp = {"intent": "qa", "refused": True, "refusal_kind": "insufficient_all"}
    ts = score(exp, turn("万悉已完成 A 轮融资 5000 万元。", intent="qa", citations={}, refused=False))
    assert ts.behavior["拒答正确"].value == 0.0 and ts.behavior["未生成正文"].value == 0.0
    assert "陷阱题拒答" in ts.gates_failed


def test_clarify_suggestions_must_be_in_catalog():
    exp = {"intent": "clarify"}
    good = score(exp, turn("想写哪个主题？", intent="clarify", citations={}, suggestions=["品牌价值蒸馏", "AI电商爆发式增长"]))
    bad = score(exp, turn("想写哪个主题？", intent="clarify", citations={}, suggestions=["品牌价值蒸馏", "万悉的融资历程"]))
    assert good.behavior["是否追问"].value == 1.0
    assert good.behavior["建议在资料覆盖内"].value == 1.0
    assert bad.behavior["建议在资料覆盖内"].value == 0.5


def test_explain_pages_must_have_been_cited_before():
    case = {"id": "X", "title": "", "turns": [{"user": "写", "expect": {"intent": "write"}},
                                               {"user": "依据？", "expect": {"intent": "explain"}}]}
    t0 = turn(GOOD_BLOG)
    ok = E.evaluate_run(case, run_of(t0, turn("这句依据第 10 页。", intent="explain", citations={}), cid="X"),
                        judge=None, corpus=CORPUS).turns[1]
    bad = E.evaluate_run(case, run_of(t0, turn("这句依据第 12 页。", intent="explain", citations={}), cid="X"),
                         judge=None, corpus=CORPUS).turns[1]
    assert ok.behavior["解释依据一致"].value == 1.0
    assert bad.behavior["解释依据一致"].value == 0.0


def test_inherited_topic_and_intent_accuracy():
    ts = score({"intent": "write", "inherited_topic": True}, turn(GOOD_BLOG, intent="clarify", inherited={}))
    assert ts.behavior["意图正确"].value == 0.0
    assert ts.behavior["主题继承"].value == 0.0


# ── 拦截可见性 ───────────────────────────────────────────────
V3_FAIL = ("verify", {"section": 2, "sentence": 0, "result": "fail", "failed": ["V3"], "reason": "V3: 挂错对象"})
PATCH = ("patch", {"section": 2, "sentence": 0, "action": "rewrite", "text": "…"})


def _events(*evs):
    return [{"type": t, "data": d} for t, d in evs]


def test_interception_visible():
    opts = {"inject_fault": "v3_swap"}
    ok = score(WRITE_EXPECT, turn(GOOD_BLOG, boundary=BOUNDARY, events=_events(V3_FAIL, PATCH)), options=opts)
    assert ok.behavior["拦截可见性"].value == 1.0
    no_patch = score(WRITE_EXPECT, turn(GOOD_BLOG, boundary=BOUNDARY, events=_events(V3_FAIL)), options=opts)
    assert no_patch.behavior["拦截可见性"].value == 0.0 and "拦截可见性" in no_patch.gates_failed
    dirty = GOOD_BLOG.replace("AI 驱动流量增长 8 倍[S2]，AI 驱动订单增长 15 倍[S3]", "AI 驱动流量增长 15 倍[S2][S3]")
    leaked = score(WRITE_EXPECT, turn(dirty, boundary=BOUNDARY, events=_events(V3_FAIL, PATCH)), options=opts)
    assert leaked.behavior["拦截可见性"].value == 0.0


# ── 模型评审 ─────────────────────────────────────────────────
def test_judge_none_marks_model_items_unrated_not_zero():
    ts = score(WRITE_EXPECT, turn(GOOD_BLOG, boundary=BOUNDARY))
    assert ts.faithfulness["引用召回·模型"].status == "未评"
    assert ts.faithfulness["引用召回·模型"].value is None
    assert ts.completeness["扩写质量·模型"].status == "未评"


def test_judge_support_with_verified_evidence_counts():
    def support(user):
        sid = "S3" if "订单" in user.split("## 待判句")[1] else ("S2" if "流量" in user.split("## 待判句")[1] else
                                                             ("S1" if "超级流量" in user else "S4"))
        snip = {"S1": "AI搜索正成为新的超级流量入口", "S2": "AI驱动流量增长：8倍", "S3": "AI驱动订单增长：15倍",
                "S4": "外贸制造：跨境出海与智能制造"}[sid]
        return {"verdict": "支持", "evidence": [{"sid": sid, "quote": snip}], "reason": "原文一致"}

    j = fake_judge(support=support, covers={"verdict": "覆盖", "evidence": [], "reason": ""},
                   expansion={"score": 4, "links": [{"pages": [9, 10], "quote": "AI 驱动流量增长 8 倍"}],
                              "new_facts": [], "reason": "有递进"})
    ts = score(WRITE_EXPECT, turn(GOOD_BLOG, boundary=BOUNDARY), judge=j)
    assert ts.faithfulness["引用召回·模型"].status == "ok"
    assert ts.faithfulness["引用召回·模型"].value > 0.7
    assert ts.completeness["扩写质量·模型"].value == pytest.approx(0.8)
    # strict schema 确实传给了 client，且用的是评审模型
    call = j.client.calls[0]
    assert call["schema"]["additionalProperties"] is False and call["model"] == "judge-x"


def test_judge_fabricated_evidence_is_voided():
    j = fake_judge(support={"verdict": "支持", "evidence": [{"sid": "S3", "quote": "订单增长二十倍"}], "reason": "x"})
    v = j.support("AI 驱动订单增长 15 倍", [{"sid": "S3", "page_no": 10, "text": "AI驱动订单增长：15倍"}])
    assert v.evidence_ok is False and v.score is None and j.invalid == 1
    j2 = fake_judge(support={"verdict": "支持", "evidence": [], "reason": "x"})
    assert j2.support("句", [{"sid": "S1", "text": "片段"}]).score is None  # 判支持却不给依据 ⇒ 作废


def test_expansion_new_facts_cap_score():
    out = "AI 驱动订单增长 15 倍。万悉估值 10 亿。"
    j = fake_judge(expansion={"score": 5, "links": [], "new_facts": ["万悉估值 10 亿"], "reason": ""})
    v = j.expansion(out, [{"sid": "S3", "page_no": 10, "text": "AI驱动订单增长：15倍"}])
    assert v.score == pytest.approx(0.4)


def test_judge_model_resolution_and_same_model_flag():
    assert resolve_judge_model({"LLM_MODEL": "a", "EVAL_JUDGE_MODEL": "b", "LLM_JUDGE_MODEL": "c"}) == ("b", "a", False)
    assert resolve_judge_model({"LLM_MODEL": "a", "LLM_JUDGE_MODEL": "c"}) == ("c", "a", False)
    assert resolve_judge_model({"LLM_MODEL": "a"}) == ("a", "a", True)
    j = Judge(FakeClient(), model="a", generator_model="a")
    rep = E.evaluate_suite([case_of(WRITE_EXPECT)], [run_of(turn(GOOD_BLOG, boundary=BOUNDARY))], judge=None, corpus=CORPUS)
    assert any("未启用模型评审" in n for n in rep.notes)
    assert j.same_model


# ── 汇总、渲染、命令入口 ──────────────────────────────────────
def _suite():
    cases = [case_of(WRITE_EXPECT, cid="W", options={"inject_fault": "v3_swap"}),
             case_of({"intent": "out_of_scope", "refused": True}, cid="O"),
             case_of({"intent": "qa", "key_points": [{"text": "AI驱动订单增长 15倍", "page": 10}],
                      "reference_answer": "15 倍"}, cid="Q")]
    retr = {"hits": [{"fact_id": "p10-f02", "page_no": 10}]}
    runs = [run_of(turn(GOOD_BLOG, boundary=BOUNDARY, events=_events(V3_FAIL, PATCH,
                                                                   ("delta", {"section": 1, "text": GOOD_BLOG}))),
                   cid="W"),
            run_of(turn("超出范围。", intent="out_of_scope", citations={}, refused=True,
                        refusal_kind="out_of_scope", nearest_pages=[]), cid="O"),
            run_of(turn("AI 驱动订单增长 15 倍[S3]。", intent="qa", events=_events(("retrieval", retr))), cid="Q")]
    return cases, runs


def test_suite_headline_and_render():
    cases, runs = _suite()
    rep = E.evaluate_suite(cases, runs, judge=None, corpus=CORPUS)
    h = rep.headline
    assert h["数字一致率"]["pass"] is True
    assert h["拦截可见性"]["pass"] is True
    assert h["检索命中率"]["pass"] is True
    assert h["引用召回"]["value"] == "未评"
    assert h["引用格式遵从率"]["note"].startswith("改写前")
    assert set(rep.by_intent) == {"write", "out_of_scope", "qa"}
    md = E.render_report(rep)
    assert md.startswith(E.RESULT_START) and md.rstrip().endswith(E.RESULT_END)
    for sec in ("首页：校验闸是否在工作", "06 号指标总表", "意图 write", "意图 out_of_scope", "意图 qa", "人工抽检表"):
        assert sec in md
    assert "拒答没给最近页" in md


def test_real_caserun_model_is_accepted():
    from ragwriter.contracts import Event
    from ragwriter.eval.dialogue_runner import CaseRun, TurnRun, absorb

    t = TurnRun(seq=0, user="u")
    for ev in [Event(type="intent", data={"intent": "write"}),
               Event(type="done", data={"final_text": GOOD_BLOG, "citations": CITS, "boundary": BOUNDARY})]:
        t.events.append(ev)
        absorb(t, ev)
    cs = E.evaluate_run(case_of(WRITE_EXPECT), CaseRun(case_id="C1", turns=[t]), judge=None, corpus=CORPUS)
    assert cs.turns[0].faith_score == 1.0 and cs.turns[0].completeness["资料不足处理"].value == 1.0


def test_main_eval_writes_report_preserving_method_part(tmp_path):
    cases_file = tmp_path / "c.yaml"
    cases_file.write_text(yaml.safe_dump([{
        "id": "W", "title": "t", "turns": [{"user": "写一篇", "expect": {
            "intent": "write", "genre": "blog", "forbidden_strings": ["Nihaojewelry"],
            "key_points": [{"text": "AI搜索正成为新的超级流量入口", "page": 9}]}}]}], allow_unicode=True), encoding="utf-8")
    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text(json.dumps(run_of(turn(GOOD_BLOG), cid="W"), ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "eval-report.md"
    out.write_text(f"# 报告\n\n方法段保留\n\n{E.RESULT_START}\n旧结果\n{E.RESULT_END}\n\n尾注\n", encoding="utf-8")
    rc = E.main_eval(["--cases", str(cases_file), "--runs", str(runs_file), "--no-judge",
                      "--out", str(out), "--json", str(tmp_path / "r.json")])
    text = out.read_text(encoding="utf-8")
    assert rc == 0 and "方法段保留" in text and "尾注" in text and "旧结果" not in text
    assert "## 评估结果" in text
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["cases"][0]["case_id"] == "W"


def test_page_references_are_not_fact_numbers():
    # 「据第 10 页图示」是出处写法；无引用的「最接近的资料在第 9、10 页」是导航 —— 都不是事实数字
    s = M.split_sentences("据第10页图示，AI 驱动订单增长 15 倍[S3]。")[0]
    assert [n.raw for n in M.check_numbers(s, CITS, CORPUS)] == ["15倍"]
    nav = M.split_sentences("最接近的资料在第 9、10 页。")[0]
    assert M.check_numbers(nav, CITS, CORPUS) == [] and not M.is_fact_sentence(nav)
    # 指向未引用页的页码照常回查
    other = M.split_sentences("第 31 页显示 AI 驱动订单增长 15 倍[S3]。")[0]
    assert any(not n.ok for n in M.check_numbers(other, CITS, CORPUS))
