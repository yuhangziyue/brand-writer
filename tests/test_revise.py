"""写前确认、指定体裁入口、改写上一篇（revise）。

全部用 ragwriter.agent.testing 的假实现：意图由 FakeClient 的 intent 预置按消息返回，
主题审核 / 改写 / 起标题按 tag 预置（topic_check / revise_body / revise_ending / revise_heading / revise_title）。
"""

from __future__ import annotations

import pytest

from ragwriter.agent import Orchestrator
from ragwriter.agent.intent import normalize_revise, rule_classify, rule_revise
from ragwriter.agent.testing import DEMO_OUTPUTS, FakeClient, make_fake_deps

BLOG = "写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？"
TOPIC = "为什么中国出海品牌需要进行 GEO 优化？"


def intent_out(intent: str, **kw) -> dict:
    base = {"intent": intent, "genre": "none", "topic": "", "query": "", "search_queries": [], "answer_shape": "short",
            "focus": "", "constraints": [], "revise_target": "", "revise_action": "none", "revise_instruction": "",
            "confirm_action": "none", "confidence": 0.9}
    base.update(kw)
    return base


INTENTS: dict[str, dict] = {
    BLOG: intent_out("write", genre="blog", topic=TOPIC),
    "确认": intent_out("confirm", confirm_action="accept"),
    "换个角度，讲合规": intent_out("confirm", confirm_action="modify", revise_instruction="换个角度，讲合规"),
    "算了不写了": intent_out("confirm", confirm_action="cancel"),
    "AI 驱动订单增长多少？": intent_out("qa", query="AI 驱动订单增长多少", answer_shape="number"),
    "第二节改短一点": intent_out("revise", revise_target="section:2", revise_action="shorten",
                          revise_instruction="短一点"),
    "第二节再口语一点": intent_out("revise", revise_target="section:2", revise_action="tone",
                           revise_instruction="更口语"),
    "结尾换成更口语的语气": intent_out("revise", revise_target="ending", revise_action="tone",
                              revise_instruction="更口语"),
    "删掉第一节": intent_out("revise", revise_target="section:1", revise_action="delete",
                        revise_instruction="删除第一节"),
    "第五节改短": intent_out("revise", revise_target="section:5", revise_action="shorten",
                        revise_instruction="短一点"),
    "加一节讲订单增长": intent_out("revise", revise_target="all", revise_action="add_section",
                           revise_instruction="订单增长"),
    "加一节讲牛仔服饰工厂的案例": intent_out("revise", revise_target="all", revise_action="add_section",
                                 revise_instruction="牛仔服饰工厂的案例"),
    "把标题改得更吸引人": intent_out("revise", revise_target="title", revise_action="retitle",
                             revise_instruction="更吸引人"),
}


def _intent(system: str, user: str, tag: str) -> dict:
    msg = user.rsplit("用户消息：", 1)[-1].strip()
    return INTENTS[msg]


@pytest.fixture
def client():
    return FakeClient({**DEMO_OUTPUTS, "intent": _intent})


@pytest.fixture
def orch(tmp_path, client):
    return Orchestrator(make_fake_deps(client=client), sessions_db=tmp_path / "s.db")


def run(orch, message, sid=None, **options):
    return list(orch.handle(message, session_id=sid, options=options or None))


def of(evs, t):
    return [e for e in evs if e.type == t]


def written(orch) -> tuple[str, list]:
    """写一篇并确认，返回 (session_id, 写作轮事件)。"""
    evs = run(orch, BLOG)
    sid = evs[0].data["session_id"]
    return sid, run(orch, "确认", sid)


def slots(orch, sid):
    return orch.memory.load_slots(sid)


# ── ① 指定体裁入口 ─────────────────────────────────────────
def test_selected_mode_skips_intent_model(orch, client):
    evs = run(orch, BLOG, genre="faq", topic="GEO 合规")
    it = evs[1].data
    assert it["source"] == "selected" and it["intent"] == "write" and it["genre"] == "faq" and it["topic"] == "GEO 合规"
    assert "intent" not in client.tags()
    # 指定模式同样先确认：不检索、不写作
    assert [e.type for e in evs[2:]] == ["confirm", "done"]
    assert orch.deps.fake_search.queries == []


def test_selected_mode_ignored_without_topic(orch, client):
    evs = run(orch, BLOG, genre="faq", topic="  ")
    assert evs[1].data["source"] == "model" and "intent" in client.tags()


# ── ② 写前确认 ────────────────────────────────────────────
def test_confirm_card_then_accept_writes(orch, client):
    client.set("topic_check", {"valid": True, "genre": "blog", "topic": "出海品牌为什么需要 GEO",
                               "angle": "AI 搜索改变流量入口", "outline_points": ["AI 搜索流量", "AI 电商增长"],
                               "premise_gaps": ["中国出海品牌总体数据"], "confirm_text": "这样写可以吗？"})
    evs = run(orch, BLOG)
    sid = evs[0].data["session_id"]
    card = of(evs, "confirm")[0].data
    assert card["awaiting"] is True and card["topic"] == "出海品牌为什么需要 GEO"
    done = evs[-1].data
    assert done["awaiting_confirmation"] is True and done["refused"] is False and done["final_text"] == "这样写可以吗？"
    assert of(evs, "plan") == [] and orch.deps.fake_search.queries == []
    pend = slots(orch, sid)["pending_write"]
    assert pend["topic"] == "出海品牌为什么需要 GEO" and pend["premise_gaps"] == ["中国出海品牌总体数据"]

    evs2 = run(orch, "确认", sid)
    assert evs2[1].data["intent"] == "write" and evs2[1].data["confirm_action"] == "accept"
    assert evs2[1].data["topic"] == "出海品牌为什么需要 GEO"  # 用审核后的精确主题
    assert of(evs2, "plan") and evs2[-1].data["artifact_id"]
    assert slots(orch, sid)["pending_write"] is None


def test_confirm_button_skips_intent_model(orch, client):
    sid = run(orch, BLOG)[0].data["session_id"]
    n = client.tags().count("intent")
    evs = run(orch, "确认", sid, confirm="accept")
    assert client.tags().count("intent") == n
    assert evs[1].data["intent"] == "write" and evs[1].data["source"] == "selected" and of(evs, "plan")


def test_confirm_modify_rechecks_with_feedback(orch, client):
    seen = []

    def check(system, user, tag):
        seen.append(user)
        return {"topic": "合规 GEO 为什么重要" if "合规" in user else TOPIC}

    client.set("topic_check", check)
    sid = run(orch, BLOG)[0].data["session_id"]
    evs = run(orch, "换个角度，讲合规", sid)
    assert evs[1].data["intent"] == "confirm" and evs[1].data["confirm_action"] == "modify"
    card = of(evs, "confirm")[0].data
    assert card["topic"] == "合规 GEO 为什么重要" and card["feedback"] == "换个角度，讲合规"
    assert "意见：换个角度，讲合规" in seen[-1]
    assert slots(orch, sid)["pending_write"]["topic"] == "合规 GEO 为什么重要"
    assert of(evs, "plan") == []


def test_invalid_topic_then_pick_alternative(orch, client):
    client.set("topic_check", lambda s, u, t: (
        {"valid": False, "topic": "万悉的融资历程", "reason": "品宣资料没有融资信息",
         "alternatives": ["为什么品牌要被 AI 记住", "为什么 GEO 必须合规"]}
        if "融资" in u else {"valid": True, "topic": "为什么 GEO 必须合规"}))
    INTENTS["写一篇 Blog：万悉的融资历程"] = intent_out("write", genre="blog", topic="万悉的融资历程")
    evs = run(orch, "写一篇 Blog：万悉的融资历程")
    sid = evs[0].data["session_id"]
    card = of(evs, "confirm")[0].data
    assert card["valid"] is False and card["alternatives"] and "融资" in evs[-1].data["final_text"]
    assert slots(orch, sid)["pending_write"]["alternatives"] == card["alternatives"]

    evs2 = run(orch, "就写：为什么 GEO 必须合规", sid, confirm="pick", topic="为什么 GEO 必须合规")
    card2 = of(evs2, "confirm")[0].data
    assert card2["valid"] is True and card2["topic"] == "为什么 GEO 必须合规"
    assert of(evs2, "plan") == []  # 选了备选也要再确认一次


def test_confirm_cancel_clears_pending(orch):
    sid = run(orch, BLOG)[0].data["session_id"]
    evs = run(orch, "算了不写了", sid)
    assert evs[-1].data["final_text"].startswith("好的，已取消") and evs[-1].data["refused"] is False
    assert slots(orch, sid)["pending_write"] is None


def test_unrelated_question_keeps_pending(orch, client):
    sid = run(orch, BLOG)[0].data["session_id"]
    evs = run(orch, "AI 驱动订单增长多少？", sid)
    assert evs[1].data["intent"] == "qa" and evs[1].data["pending"] is True
    assert slots(orch, sid)["pending_write"]
    evs2 = run(orch, "确认", sid)
    assert evs2[1].data["intent"] == "write" and of(evs2, "plan")


def test_confirm_without_pending_is_clarify(orch):
    evs = run(orch, "确认", confirm="accept")
    assert evs[1].data["intent"] == "clarify" and evs[-1].data["refusal_kind"] == "clarify"
    evs = run(orch, "确认")
    assert evs[1].data["intent"] == "clarify" and evs[1].data["reason"] == "no_pending"


def test_force_intent_skips_confirmation(orch):
    evs = run(orch, "随便", force_intent={"intent": "write", "genre": "blog", "topic": "GEO"})
    assert of(evs, "confirm") == [] and of(evs, "plan")
    evs = run(orch, "随便", force_intent={"intent": "write", "genre": "blog", "topic": "GEO", "confirmed": False})
    assert of(evs, "confirm") and of(evs, "plan") == []


# ── ③ revise ──────────────────────────────────────────────
def test_revise_shorten_keeps_citations_and_new_artifact(orch, client):
    sid, w = written(orch)
    base = w[-1].data["artifact_id"]
    client.set("revise_body", "AI驱动订单增长 15倍[S4]。")
    evs = run(orch, "第二节改短一点", sid)
    it = evs[1].data
    assert (it["intent"], it["revise_target"], it["revise_action"]) == ("revise", "section:2", "shorten")
    plan = of(evs, "plan")[0].data
    assert plan["revise"] is True and plan["targets"] == ["AI 电商带来的增长"]
    # 只有第 2 节被改写、被校验；第 1 节原样定稿
    assert {e.data["section"] for e in of(evs, "verify")} == {2}
    assert {e.data["section"] for e in of(evs, "delta") if e.data.get("kind") == "body"} == {2}
    finals = {(e.data["kind"], e.data["section"]): e.data for e in of(evs, "section_final")}
    assert finals[("body", 2)]["text"] == "AI驱动订单增长 15倍[S4]。" and finals[("body", 2)]["changed"] is True
    assert finals[("body", 1)]["changed"] is False and "[S1]" in finals[("body", 1)]["text"]
    done = evs[-1].data
    assert done["citations"]["S4"]["fact_id"] == "p10-f02" and done["citations"]["S1"]["fact_id"] == "p09-f01"
    assert done["base_artifact_id"] == base and done["artifact_id"] != base
    assert done["revised"]["action"] == "shorten" and done["revised"]["changed"] is True
    assert "## AI 电商带来的增长\n\nAI驱动订单增长 15倍[S4]。" in done["final_text"]
    assert done["report"]["sentence_total"] >= 3  # 报告覆盖整篇，不只改过的那句
    assert slots(orch, sid)["last_artifact_id"] == done["artifact_id"]


def test_revise_twice_builds_on_latest(orch, client):
    sid, _ = written(orch)
    client.set("revise_body", ["AI驱动订单增长 15倍[S4]。", "订单涨了 15倍，AI驱动订单增长[S4]。"])
    first = run(orch, "第二节改短一点", sid)[-1].data
    second = run(orch, "第二节再口语一点", sid)[-1].data
    assert second["base_artifact_id"] == first["artifact_id"]
    assert "订单涨了 15倍" in second["final_text"]
    assert slots(orch, sid)["last_artifact_id"] == second["artifact_id"]
    revise_users = [c["user"] for c in client.calls if c["tag"] == "revise_body"]
    assert "AI驱动订单增长 15倍[S4]。" in revise_users[-1]  # 第二次改的是第一次的结果


def test_revise_tone_on_ending(orch, client):
    sid, _ = written(orch)
    client.set("revise_ending", "GEO 这件事，现在就该提上日程啦。")
    evs = run(orch, "结尾换成更口语的语气", sid)
    finals = {e.data["kind"]: e.data for e in of(evs, "section_final")}
    assert finals["ending"]["text"] == "GEO 这件事，现在就该提上日程啦。" and finals["ending"]["changed"]
    assert evs[-1].data["final_text"].endswith("GEO 这件事，现在就该提上日程啦。")
    assert "revise_body" not in client.tags()


def test_revise_rewrite_failing_verification_keeps_original(orch, client):
    sid, _ = written(orch)
    client.set("revise_body", "AI驱动订单增长 99倍[S4]。").set("rewriter", "")
    evs = run(orch, "第二节改短一点", sid)
    assert any(e.data.get("action") == "delete" for e in of(evs, "patch"))
    done = evs[-1].data
    assert "15倍[S4]" in done["final_text"] and "99倍" not in done["final_text"]
    assert any("保留原文" in n for n in done["revised"]["notes"])


def test_revise_empty_output_keeps_original(orch, client):
    sid, w = written(orch)
    client.set("revise_body", "")
    done = run(orch, "第二节改短一点", sid)[-1].data
    assert done["revised"]["changed"] is False and done["artifact_id"] == w[-1].data["artifact_id"]
    assert any("没有产出" in n for n in done["revised"]["notes"])


def test_revise_delete_section_no_model(orch, client):
    sid, _ = written(orch)
    n = len(client.calls)
    evs = run(orch, "删掉第一节", sid)
    assert [c["tag"] for c in client.calls[n:]] == ["intent"]
    done = evs[-1].data
    assert "AI 搜索正在成为新的流量入口" not in done["final_text"] and "S1" not in done["citations"]
    assert of(evs, "plan")[0].data["outline"] == ["AI 电商带来的增长"]


def test_revise_add_section(orch, client):
    sid, _ = written(orch)
    client.set("revise_heading", "AI 带来的订单增长").set("section_writer", "AI驱动流量增长 8倍[S5]。")
    evs = run(orch, "加一节讲订单增长", sid)
    assert orch.deps.fake_search.queries[-1] == "订单增长"
    sids = [s["sid"] for s in of(evs, "retrieval")[0].data["snippets"]]
    assert sids[0] == "S5"  # 续接现有最大编号
    done = evs[-1].data
    assert done["final_text"].index("## AI 电商带来的增长") < done["final_text"].index("## AI 带来的订单增长")
    assert done["citations"]["S5"]["fact_id"] == "p10-f01" and done["citations"]["S4"]["fact_id"] == "p10-f02"
    assert of(evs, "plan")[0].data["outline"][-1] == "AI 带来的订单增长"


def test_revise_add_section_insufficient(orch, client):
    sid, w = written(orch)
    evs = run(orch, "加一节讲牛仔服饰工厂的案例", sid)
    assert of(evs, "insufficient")
    done = evs[-1].data
    assert done["refused"] is True and "未新增" in done["final_text"]
    assert slots(orch, sid)["last_artifact_id"] == w[-1].data["artifact_id"]


def test_revise_retitle(orch, client):
    sid, _ = written(orch)
    client.set("revise_title", ["标题：《3 个理由：出海品牌必须做 GEO》", "AI 搜索时代，出海品牌为何要做 GEO"])
    bad = run(orch, "把标题改得更吸引人", sid)[-1].data
    assert bad["revised"]["changed"] is False and bad["title"] == "为什么中国出海品牌需要 GEO"
    good = run(orch, "把标题改得更吸引人", sid)[-1].data
    assert good["title"] == "AI 搜索时代，出海品牌为何要做 GEO"
    assert good["final_text"].startswith("# AI 搜索时代，出海品牌为何要做 GEO")


def test_revise_target_not_found(orch):
    sid, w = written(orch)
    evs = run(orch, "第五节改短", sid)
    done = evs[-1].data
    assert done["refused"] is True and "只有 2 节" in done["final_text"] and "AI 电商带来的增长" in done["final_text"]
    assert slots(orch, sid)["last_artifact_id"] == w[-1].data["artifact_id"]


def test_revise_without_artifact_is_clarify(orch):
    evs = run(orch, "第二节改短一点")
    assert evs[1].data["intent"] == "clarify" and evs[1].data["reason"] == "no_artifact"


# ── 规则兜底 ──────────────────────────────────────────────
@pytest.mark.parametrize("msg,target,action", [
    ("第二节改短一点", "section:2", "shorten"),
    ("第3节展开讲讲", "section:3", "expand"),
    ("结尾换成更口语的语气", "ending", "tone"),
    ("开头控制在 50 字内", "lede", "shorten"),
    ("删掉第一节", "section:1", "delete"),
    ("加一节讲案例", "all", "add_section"),
    ("把标题改得更吸引人", "title", "retitle"),
    ("重写一下", "all", "rewrite"),
])
def test_rule_revise(msg, target, action):
    assert rule_revise(msg) == (target, action)


def test_normalize_revise():
    assert normalize_revise("title", "tone") == ("title", "retitle")
    assert normalize_revise("section:x", "shorten") == ("all", "shorten")
    assert normalize_revise("", "") == ("all", "rewrite")


def test_rule_genre_switch_is_write_not_revise():
    res = rule_classify("换成 FAQ", {"topic": "GEO", "last_artifact_id": "a_1"})
    assert res.intent == "write" and res.genre == "faq" and res.topic == "GEO"


def test_rule_confirm_when_pending():
    assert rule_classify("确认", {"pending_write": {"topic": "x"}}).confirm_action == "accept"
    assert rule_classify("取消", {"pending_write": {"topic": "x"}}).confirm_action == "cancel"
    assert rule_classify("主题改为：合规", {"pending_write": {"topic": "x"}}).confirm_action == "modify"


def test_shorten_that_barely_shortens_gets_second_pass(orch, client):
    sid, _ = written(orch)
    client.set("revise_body", ["据 Shopify 2025 年数据，AI驱动订单增长 15倍[S4]。", "AI驱动订单增长 15倍[S4]。"])
    evs = run(orch, "第二节改短一点", sid)
    trims = [e.data for e in of(evs, "patch") if e.data["action"] == "trim"]
    assert len(trims) == 1 and trims[0]["text"] == "AI驱动订单增长 15倍[S4]。"
    assert [c["tag"] for c in client.calls].count("revise_body") == 2
    assert "## AI 电商带来的增长\n\nAI驱动订单增长 15倍[S4]。" in evs[-1].data["final_text"]
