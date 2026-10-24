"""写作前的主题审核与确认（10-23 子曰定：每次写作都先确认，写作台与对话入口没有例外）。

流程：写作请求 → 主题审核（模型，`generate.topic.check_topic`）→ 发 confirm 卡片并**停在这里** →
用户回复「确认 / 修改 / 选备选 / 取消」→ 确认后才检索与写作。

为什么停下来等，而不是边确认边写：确认卡的价值在于让用户在花掉一整篇的模型调用之前纠正理解偏差
（主题太宽、前提资料里没有、侧重不对）。先写再问，用户要么忍着读完、要么整篇重来。

本处理器只管「问」与「取消」；accept 由编排器直接转成 write，并把确认过的要点经
`ctx.options["confirmed_brief"]` 交给 write 处理器——handlers 之间不互相调用（10 号文档第 7 节）。
"""

from __future__ import annotations

from typing import Any, Iterator

from ragwriter.agent.handlers import TurnContext, done_event, ev
from ragwriter.contracts import Event, TopicCheck

GENRE_LABEL = {"blog": "官网 Blog", "faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}
CANCEL_TEXT = "好的，已取消。想写别的随时告诉我。"


def handle(ctx: TurnContext) -> Iterator[Event]:
    it = ctx.intent
    pending: dict[str, Any] = dict(ctx.slots.get("pending_write") or {})
    if it.intent == "confirm" and it.confirm_action == "cancel":
        ctx.slots["pending_write"] = None
        yield ev("delta", section=0, kind="notice", text=CANCEL_TEXT)
        ctx.outcome.update(kind="confirm", outline=["已取消待确认的写作"], pages=[], final_text=CANCEL_TEXT)
        yield done_event(ctx, final_text=CANCEL_TEXT, cancelled=True)
        return

    if it.intent == "confirm" and it.confirm_action == "pick":
        request, genre, topic, feedback = it.topic or "", pending.get("genre"), it.topic, ""
    elif it.intent == "confirm":  # modify
        request = pending.get("request") or ctx.message
        genre, topic, feedback = pending.get("genre"), pending.get("topic"), it.revise_instruction or ctx.message
    else:  # write，尚未确认
        request, genre, topic, feedback = ctx.message, it.genre, it.topic, ""
    constraints = list(dict.fromkeys([*(pending.get("constraints") or []), *it.constraints]))

    tc = yield from _check(ctx, request, genre=genre, topic=topic, feedback=feedback)
    text = tc.confirm_text.strip() or fallback_text(tc)
    ctx.slots["pending_write"] = {
        "genre": tc.genre, "topic": tc.topic, "angle": tc.angle, "outline_points": list(tc.outline_points),
        "premise_gaps": list(tc.premise_gaps), "alternatives": list(tc.alternatives), "valid": tc.valid,
        "request": request, "constraints": constraints,
    }
    data = tc.model_dump()
    data.update(confirm_text=text, awaiting=True, request=request, genre_label=GENRE_LABEL.get(tc.genre, tc.genre),
                feedback=feedback)
    yield ev("confirm", **data)
    ctx.outcome.update(kind="confirm", outline=[f"待确认：{tc.topic}"], pages=[], final_text=text)
    yield done_event(ctx, final_text=text, awaiting_confirmation=True, topic_check=data, genre=tc.genre,
                     topic=tc.topic)


def _check(ctx: TurnContext, request: str, *, genre: str | None, topic: str | None, feedback: str):
    """调主题审核；函数缺席或调用失败时退回「原样复述」的确认卡。

    退回而不是报错：审核是给用户的一次纠偏机会，它挂了不该让写作整个不可用；
    但退化要显式——发一条 error（soft），前端能看到这次确认没经过模型审核。
    """
    d = ctx.deps
    fn = getattr(d, "check_topic", None)
    if fn is not None:
        try:
            tc = fn(d.client, request, genre=genre, topic=topic, profile=ctx.profile, slots=ctx.slots,
                    feedback=feedback)
            if isinstance(tc, dict):
                tc = TopicCheck(**tc)
            return tc
        except Exception as exc:  # noqa: BLE001
            yield ev("error", stage="topic_check", reason=f"{type(exc).__name__}: {exc}", fallback=True)
    else:
        yield ev("error", stage="topic_check", reason="主题审核未接入（generate.topic.check_topic 缺席），按原主题确认",
                 fallback=True)
    return TopicCheck(valid=True, genre=genre or "blog", topic=(topic or request).strip())


def fallback_text(tc: TopicCheck) -> str:
    """模型没给确认话术时由程序拼一句。只复述已知字段，不编内容。"""
    label = GENRE_LABEL.get(tc.genre, tc.genre)
    if not tc.valid:
        alts = "、".join(f"「{a}」" for a in tc.alternatives)
        return (f"「{tc.topic}」{('：' + tc.reason) if tc.reason else '品宣资料撑不起这个主题'}。"
                + (f"可以换成 {alts}，选一个或告诉我新的主题。" if alts else "可以换个主题再试。"))
    parts = [f"我打算写一篇{label}：「{tc.topic}」。"]
    if tc.angle:
        parts.append(f"侧重：{tc.angle}。")
    if tc.outline_points:
        parts.append("要点：" + "；".join(tc.outline_points) + "。")
    if tc.premise_gaps:
        parts.append("品宣资料里没有：" + "、".join(tc.premise_gaps) + "，文中不会写。")
    parts.append("确认后开始写作，也可以告诉我怎么改。")
    return "".join(parts)
