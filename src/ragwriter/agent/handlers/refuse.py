"""out_of_scope / clarify / 暂不支持（revise、explain）/ 无模型 的统一出口。

不调写作模型：拒答与追问的文案由程序给定，避免模型在拒答里顺手「补充一点背景知识」。

越界 ≠ 资料不足（评审 M6）：越界请求与品牌资料无关，检索出来的「最接近的页」没有意义，
所以 out_of_scope 不检索；「资料不足」只在 write / qa 链路里由检索充分性判断产生。
暂不支持与追问可以用原话检索一次给出参考页（10 号文档 4.1 节），检索失败不影响拒答本身。
"""

from __future__ import annotations

from typing import Iterator

from ragwriter.agent.handlers import OUT_OF_SCOPE_TEXT, TurnContext, done_event, ev, nearest_pages, note_retrieval
from ragwriter.agent.intent import suggest_topics
from ragwriter.contracts import Event

_UNSUPPORTED_LABEL = {"revise": "修改上一篇内容", "explain": "解释上一篇的依据与校验过程"}


def _message(ctx: TurnContext) -> tuple[str, list[str], str]:
    """返回 (文案, 主题建议, refusal_kind)。"""
    it = ctx.intent
    if it.reason == "no_model":
        return ("当前没有可用的模型，本条请求无法实时生成。配置 LLM_API_KEY 后重试，或使用回放模式。",
                [], "no_model")
    if it.intent in _UNSUPPORTED_LABEL:
        return (f"「{_UNSUPPORTED_LABEL[it.intent]}」本版本暂不支持，已记录这条需求。"
                "可以重新发起一篇写作，或直接就品宣资料提问。", [], "unsupported")
    if it.intent == "out_of_scope":
        return OUT_OF_SCOPE_TEXT, [], "out_of_scope"
    suggestions = suggest_topics(ctx.profile)
    if it.reason == "no_artifact":
        lead = "还没有可以修改或解释的上一篇内容。"
    elif it.reason == "missing_topic":
        lead = "请告诉我写作主题。"
    else:
        lead = "没能确定你想做什么，可以说得具体一点吗？"
    tips = "；".join(f"「{s}」" for s in suggestions)
    return f"{lead}资料能支撑的主题，例如：{tips}。", suggestions, "clarify"


def handle(ctx: TurnContext) -> Iterator[Event]:
    text, suggestions, kind = _message(ctx)
    near: list[int] = []
    d = ctx.deps
    want_search = (ctx.options.get("nearest_pages", True) and kind in ("unsupported", "clarify", "no_model")
                   and ctx.intent.reason not in ("empty", "missing_topic") and d.store is not None)
    if want_search:
        try:
            near = nearest_pages([note_retrieval(ctx, d.search(d.store, ctx.message, embedder=d.embedder))])
        except Exception:  # noqa: BLE001 — 只为给参考页，失败就不给
            near = []
    if near:
        text += "可以参考第 " + "、".join(str(p) for p in near) + " 页。"
    yield ev("delta", section=0, kind="notice", text=text)
    ctx.outcome.update(kind="refuse", outline=[f"{ctx.intent.intent}：{ctx.intent.reason}"], pages=near,
                       final_text=text)
    yield done_event(ctx, final_text=text, refused=True, nearest=near, suggestions=suggestions,
                     refusal_kind=kind, reason=ctx.intent.reason, unsupported=kind == "unsupported")
