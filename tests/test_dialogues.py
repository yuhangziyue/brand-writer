"""多轮对话用例：数据自检 + runner/checks 的正负例 + （可选）真链路。

三种模式：
- 默认：脚本化假编排器（tests/dialogue/fakes.py + scripts/<id>.yaml），验证 runner 与 checks 本身
- E 组 ragwriter.agent.testing 就绪后：真 Orchestrator + 假依赖（test_orchestrator_with_fakes，未就绪则 skip）
- RAG_E2E=1：Deps.from_env() 跑真链路（pytest.mark.e2e，默认 skip）
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
from pathlib import Path

import pytest

from dialogue.fakes import ScriptedOrchestrator, load_script
from ragwriter.contracts import Event
from ragwriter.eval.dialogue_checks import (
    CheckOutcome, bigram_recall, check_case, check_numbers, failures, format_outcomes,
)
from ragwriter.eval.dialogue_runner import DIALOGUE_DIR, NumberExpect, TurnExpect, TurnRun, load_cases, run_case

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "docs" / "structured-content.md"
CASES = load_cases(DIALOGUE_DIR)
CASE_BY_ID = {c.id: c for c in CASES}


def _page_sections() -> dict[int, str]:
    text = CORPUS.read_text(encoding="utf-8")
    parts = re.split(r"^### p(\d+) ", text, flags=re.M)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


PAGES = _page_sections()


def _run(case_id: str, script: dict | None = None):
    case = CASE_BY_ID[case_id]
    orch = ScriptedOrchestrator(script or load_script(case_id))
    run = run_case(case, orch)
    return case, run, orch, check_case(case, run)


def _failed_names(outcomes: list[CheckOutcome]) -> set[str]:
    return {o.name for o in failures(outcomes)}


# ── 用例数据自检 ────────────────────────────────────────────────
def test_case_set_coverage():
    assert len(CASES) >= 16
    intents = {t.expect.intent for c in CASES for t in c.turns}
    assert intents == {"write", "qa", "revise", "explain", "out_of_scope", "clarify"}
    genres = {t.expect.genre for c in CASES for t in c.turns if t.expect.intent == "write"}
    assert {"blog", "faq", "brand_intro", "product_intro"} <= genres
    assert any(len(c.turns) > 1 for c in CASES)
    assert any(t.options for c in CASES for t in c.turns), "缺故障注入用例"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_case_grounded_in_corpus(case):
    """期望必须能指回 structured-content.md 的页：页号存在、要点文本出自该页、数字出自语料。"""
    corpus = re.sub(r"\s+", "", CORPUS.read_text(encoding="utf-8"))
    for t in case.turns:
        e = t.expect
        for p in [*e.cite_pages_any, *e.cite_pages_all, *(k.page for k in e.key_points)]:
            assert p in PAGES, f"{case.id}: 第 {p} 页不存在"
        for kp in e.key_points:
            r = bigram_recall(kp.text, PAGES[kp.page])
            assert r >= 0.75, f"{case.id}: 要点「{kp.text}」与第 {kp.page} 页原文重合仅 {r:.2f}"
        for n in e.numbers:
            assert re.sub(r"\s+", "", n.value) in corpus, f"{case.id}: 数字 {n.value} 不在语料里"
        if e.intent in ("write", "qa") and e.refused is False:
            assert e.cite_pages_any or e.cite_pages_all or e.no_clarify or e.key_points, f"{case.id}: 有答案的轮次没有可核对的期望"


# ── 假编排器：每条用例都应全部通过 ───────────────────────────────
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_dialogue_fake(case):
    _, run, orch, outcomes = _run(case.id)
    assert len(run.turns) == len(case.turns)
    assert run.session_id and all(t.session_id == run.session_id for t in run.turns)
    # 第一轮不带 session，后续轮次沿用第一轮的
    assert orch.received[0]["session_id"] is None
    assert all(r["session_id"] == run.session_id for r in orch.received[1:])
    # options 原样透传
    for t, r in zip(case.turns, orch.received):
        assert r["options"] == (t.options or None)
    assert all(t.elapsed_ms >= 0 and t.has_done for t in run.turns)
    bad = failures(outcomes)
    assert not bad, format_outcomes(bad)


# ── 负例：让假模型犯错，checks 必须抓住 ──────────────────────────
def _mut(case_id: str, turn: int, **patch) -> dict:
    s = copy.deepcopy(load_script(case_id))
    s["turns"][turn].update(patch)
    return s


def test_neg_p31_client_name_leaked():
    s = _mut("D16", 1, final_text="这家头部首饰供应商就是 Nihaojewelry[S1]。", cite={"S1": "p31-v15"}, hits=["p31-v15"])
    _, _, _, out = _run("D16", s)
    assert "forbidden_strings" in _failed_names(out)
    assert any(o.name.startswith("must_say_insufficient") for o in failures(out))


def test_neg_15x_attached_to_traffic():
    s = _mut("D05", 0, final_text="据 Shopify官方数据(2025)，AI 驱动流量增长 15 倍[S1]。")
    _, _, _, out = _run("D05", s)
    assert "numbers[15倍->订单]" in _failed_names(out)


def test_neg_refusal_without_nearest_pages():
    s = _mut("D11", 0, nearest_pages=[])
    _, _, _, out = _run("D11", s)
    names = _failed_names(out)
    assert "refusal_nearest_pages" in names


def test_neg_clarify_when_should_not():
    s = _mut("D01", 1, intent={"intent": "clarify", "source": "model"}, suggestions=["a", "b", "c"])
    _, _, _, out = _run("D01", s)
    assert {"no_clarify", "intent"} <= _failed_names(out)


def test_neg_cites_previous_turn():
    # 第二轮检索只命中第 5 页，却引用了上一轮的 p10-f02
    s = _mut("D06", 1, hits=["p05-v26"], cite={"S1": "p10-f02"},
             final_text="据 Shopify官方数据(2025)，AI驱动流量增长 8倍[S1]。")
    _, _, _, out = _run("D06", s)
    bad = [o for o in failures(out) if o.name == "citations_from_this_turn"]
    assert bad and "上一轮" in bad[0].detail


def test_neg_session_drift():
    s = _mut("D06", 2, session_id="s_other")
    _, _, _, out = _run("D06", s)
    assert "session_consistent" in _failed_names(out)


def test_neg_topic_not_inherited():
    s = copy.deepcopy(load_script("D01"))
    s["turns"][1]["intent"]["topic"] = "FAQ"
    s["turns"][1]["intent"]["inherited"] = {}
    _, _, _, out = _run("D01", s)
    assert {"topic_contains", "inherited_topic", "inherit.topic"} <= _failed_names(out)


def test_neg_constraints_dropped():
    s = copy.deepcopy(load_script("D03"))
    s["turns"][1]["intent"]["slots"]["constraints"] = ["语气正式"]
    _, _, _, out = _run("D03", s)
    bad = [o for o in failures(out) if o.name == "inherit.constraints"]
    assert bad and "800" in bad[0].detail


def test_neg_write_insufficient_in_body():
    s = _mut("D01", 0, boundary=[],
             final_text=load_script("D01")["turns"][0]["final_text"] + "关于中国出海品牌的总体统计，品宣资料未涉及。")
    _, _, _, out = _run("D01", s)
    bad = [o for o in failures(out) if o.name.startswith("must_say_insufficient")]
    assert bad and "品宣资料未涉及" in bad[0].detail


def test_neg_v3_fault_without_patch():
    s = copy.deepcopy(load_script("D19"))
    s["turns"][0]["events"] = [e for e in s["turns"][0]["events"] if e != "patch"]
    _, _, _, out = _run("D19", s)
    assert {"inject.v3_swap", "events_in_order"} <= _failed_names(out)


def test_neg_v3_fault_final_text_still_swapped():
    s = _mut("D19", 0, final_text="据 Shopify官方数据(2025)，AI驱动流量增长 15倍[S1]，AI驱动流量增长 8倍[S2]。")
    _, _, _, out = _run("D19", s)
    assert "numbers[15倍->订单]" in _failed_names(out)


def test_neg_faq_without_jsonld():
    s = copy.deepcopy(load_script("D18"))
    del s["turns"][0]["jsonld"]
    _, _, _, out = _run("D18", s)
    assert "faq_jsonld" in _failed_names(out)


def test_neg_clarify_suggestions_count():
    s = _mut("D17", 0, suggestions=["为什么品牌要被 AI 记住"])
    _, _, _, out = _run("D17", s)
    assert "clarify_suggestions" in _failed_names(out)


def test_neg_refusal_kind_and_out_of_scope_mislabel():
    s = copy.deepcopy(load_script("D12"))
    s["turns"][0]["intent"]["intent"] = "out_of_scope"
    s["turns"][0]["refusal_kind"] = "out_of_scope"
    _, _, _, out = _run("D12", s)
    assert {"intent", "refusal_kind"} <= _failed_names(out)


def test_neg_unresolvable_citation():
    s = _mut("D05", 0, final_text="据 Shopify官方数据(2025)，AI驱动订单增长 15倍[S1][S9]。")
    _, _, _, out = _run("D05", s)
    assert "citations_resolvable" in _failed_names(out)


def test_runner_records_exception_and_continues():
    class Boom:
        def __init__(self):
            self.inner = ScriptedOrchestrator(load_script("D06"))
            self.n = 0

        def handle(self, message, session_id=None, options=None):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("relay 502")
            return self.inner.handle(message, session_id=session_id, options=options)

    run = run_case(CASE_BY_ID["D06"], Boom())
    assert len(run.turns) == 3
    assert run.turns[1].errors and "relay 502" in run.turns[1].errors[0]["reason"]
    out = check_case(CASE_BY_ID["D06"], run)
    assert any(o.name == "done" and o.turn == 1 for o in failures(out))


# ── 检测细节的单元测试 ──────────────────────────────────────────
def _numbers(text: str, *pairs: tuple[str, str]) -> dict[str, bool]:
    t = TurnRun(seq=0, user="", final_text=text)
    e = TurnExpect(intent="qa", numbers=[NumberExpect(value=v, object=o) for v, o in pairs])
    return {o.name: o.ok for o in check_numbers(t, e)}


def test_numbers_clause_proximity():
    pairs = (("15倍", "订单"), ("8倍", "流量"))
    assert all(_numbers("AI驱动流量增长8倍，订单增长15倍。", *pairs).values())
    assert all(_numbers("订单和流量分别增长 15 倍和 8 倍。", *pairs).values())
    assert not _numbers("AI驱动流量增长15倍，订单增长8倍。", *pairs)["numbers[15倍->订单]"]
    assert all(_numbers("增幅达 115倍的指标与订单无关。", ("15倍", "流量")).values())  # 115倍 不是 15倍
    assert all(_numbers("公司获20+省央媒报道、10+科技门户报道。", ("20+", "省央媒"), ("10+", "科技门户")).values())


def test_events_parsing_tolerant_retrieval_shape():
    """retrieval 载荷形状未定，runner 递归找 fact_id。"""
    from ragwriter.eval.dialogue_runner import absorb
    t = TurnRun(seq=0, user="")
    absorb(t, Event(type="retrieval", data={"per_question": [{"q": "x", "hits": [{"fact": {"fact_id": "p10-f02", "page_no": 10}}]}]}))
    assert t.retrieved_fact_ids == ["p10-f02"] and t.retrieved_pages == [10]


# ── 真 Orchestrator ─────────────────────────────────────────────
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not (_has("ragwriter.agent.orchestrator") and _has("ragwriter.agent.testing")),
                    reason="E 组 ragwriter.agent.orchestrator / ragwriter.agent.testing 尚未落地")
def test_orchestrator_with_fakes_smoke(tmp_path):
    """E 组假依赖就绪后的冒烟：真编排器 + 假依赖跑通 D05，事件流形状满足 runner 的解析。
    假依赖用 E 组 testing.make_fake_deps()（演示语料含第 10 页的 8倍/15倍 配对）。"""
    from ragwriter.agent import testing
    from ragwriter.agent.orchestrator import Orchestrator
    orch = Orchestrator(testing.make_fake_deps(), sessions_db=tmp_path / "s.db")
    run = run_case(CASE_BY_ID["D05"], orch)
    assert run.turns[0].has_done and run.turns[0].session_id
    assert run.turns[0].event_types[:2] == ["session", "intent"]


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("RAG_E2E") != "1", reason="真链路需 RAG_E2E=1")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_dialogue_e2e(case, tmp_path):
    from ragwriter.agent.orchestrator import Deps, Orchestrator
    orch = Orchestrator(Deps.from_env(), sessions_db=tmp_path / "s.db")
    run = run_case(case, orch)
    bad = failures(check_case(case, run))
    assert not bad, format_outcomes(bad)
