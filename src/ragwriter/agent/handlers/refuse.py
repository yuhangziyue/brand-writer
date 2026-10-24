"""chitchat / out_of_scope / clarify / 暂不支持（explain）/ 无模型 的统一出口。

revise 已有自己的处理器（handlers/revise.py），这里只剩它的两个兜底：没有上一篇（no_artifact）与无模型。

不调写作模型：拒答与追问的文案由程序给定，避免模型在拒答里顺手「补充一点背景知识」。

越界 ≠ 资料不足（评审 M6）：越界请求与品牌资料无关，检索出来的「最接近的页」没有意义，
所以 out_of_scope 不检索；「资料不足」只在 write / qa 链路里由检索充分性判断产生。
10-23 起追问、闲聊、暂不支持也不再检索附「参考第 N 页」——用户没问资料，给页码是答非所问。
"""

from __future__ import annotations

from typing import Iterator

from ragwriter.agent.handlers import OUT_OF_SCOPE_TEXT, TurnContext, done_event, ev, nearest_pages, note_retrieval
from ragwriter.agent.intent import suggest_topics
from ragwriter.contracts import Event

_UNSUPPORTED = {
    "explain": "每句话的依据就在句末角标里，点开能看到原文和页码；被拦下的句子与原因在文末校验报告里。",
}
CHITCHAT_TEXT = "我可以基于万悉科技品宣资料写 Blog、FAQ、品牌或产品介绍，也可以回答资料里的具体问题。"


def _chitchat(message: str) -> str:
    """问候、道谢、问能力各回各的。10-23 实测：「谢谢」被回了一句「你好！我可以……」。"""
    if any(w in message for w in ("谢", "thank", "辛苦")):
        return "不客气，有需要随时说。"
    if any(w in message.lower() for w in ("你好", "您好", "hi", "hello", "早上好", "晚上好", "在吗")):
        return "你好！" + CHITCHAT_TEXT
    return CHITCHAT_TEXT


def _message(ctx: TurnContext) -> tuple[str, list[str], str]:
    """返回 (文案, 主题建议, refusal_kind)。每种情况只说该说的那一句，不附带没问的内容。"""
    it = ctx.intent
    if it.reason == "no_model":
        return ("当前没有可用的模型，无法实时生成。配置 LLM_API_KEY 后重试，或使用回放模式。", [], "no_model")
    if it.intent == "chitchat":
        return _chitchat(ctx.message), [], "chitchat"
    if it.intent in _UNSUPPORTED:
        if it.reason == "no_artifact":
            return "还没有生成过内容，先告诉我想写什么吧。", [], "clarify"
        return _UNSUPPORTED[it.intent], [], "unsupported"
    if it.intent == "out_of_scope":
        return OUT_OF_SCOPE_TEXT, [], "out_of_scope"
    # clarify：只问缺的那一项。缺主题才给主题建议——那时建议是答案的一部分，别的时候是噪声
    if it.reason == "missing_topic":
        suggestions = suggest_topics(ctx.profile)
        tips = "、".join(f"「{x}」" for x in suggestions)
        kind = {"faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}.get(it.genre or "", "文章")
        return f"想写哪个主题的{kind}？比如 {tips}。", suggestions, "clarify"
    if it.reason == "no_artifact":
        return "还没有生成过内容，先告诉我想写什么吧。", [], "clarify"
    if it.reason == "no_pending":
        return "现在没有等待确认的写作。想写什么主题？", [], "clarify"
    return "你是想让我写一篇内容，还是想问品宣资料里的某个问题？", [], "clarify"


def handle(ctx: TurnContext) -> Iterator[Event]:
    text, suggestions, kind = _message(ctx)
    near: list[int] = []
    d = ctx.deps
    # 追问、闲聊、越界都不检索：用户没问资料，给「参考第 N 页」就是回答了没问的问题（10-23 子曰定）
    want_search = ctx.options.get("nearest_pages", False) and kind == "no_model" and d.store is not None
    if want_search:
        try:
            near = nearest_pages([note_retrieval(ctx, d.search(d.store, ctx.message, embedder=d.embedder))])
        except Exception:  # noqa: BLE001 — 只为给参考页，失败就不给
            near = []
    yield ev("delta", section=0, kind="notice", text=text)
    ctx.outcome.update(kind="refuse", outline=[f"{ctx.intent.intent}：{ctx.intent.reason}"], pages=near,
                       final_text=text)
    yield done_event(ctx, final_text=text, refused=True, nearest=near, suggestions=suggestions,
                     refusal_kind=kind, reason=ctx.intent.reason, unsupported=kind == "unsupported")
