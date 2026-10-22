"""在线链路各段之间的数据契约。

检索、上下文、写作、校验、编排分头实现，彼此只通过这里的类型交接。
改这里的任何字段 = 改接口，要同时通知所有下游，不要在某一段里私自加字段。
架构见需求文档 10-agent-architecture.md。
"""

from __future__ import annotations

import json

from typing import Any, Literal

from pydantic import BaseModel, Field

from ragwriter.models import Fact, FactStatus, Metric

# 默认可以进生成上下文的状态。image_only / unverified 只入库，不进上下文
USABLE_STATUSES: tuple[FactStatus, ...] = ("verified", "image_structural")

Genre = Literal["blog", "faq", "brand_intro", "product_intro"]
Intent = Literal["write", "qa", "revise", "explain", "out_of_scope", "clarify"]


# ── 检索（retrieve/）──────────────────────────────────────────
class Hit(BaseModel):
    """一条命中。两路排名都留着，前端和答辩要能讲清「为什么命中」。"""

    fact: Fact
    rrf_score: float
    bm25_rank: int | None = None  # 1 起；该路未命中为 None
    dense_rank: int | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    query: str
    hits: list[Hit]
    top_score: float  # 有重排用重排分，否则用 RRF 分
    sufficient: bool  # 充分性判断（05 第 3.2 节的阈值 τ）
    signal: str = ""  # 判为不足时的原因，进校验报告
    dense_used: bool = False  # 稠密路实际有没有参与，退化必须显式


# ── 上下文（generate/context.py）──────────────────────────────
class Snippet(BaseModel):
    """上下文里一个可被引用的片段。sid 是本次请求内的编号，句末写 [S4]。"""

    sid: str  # "S4"
    fact_id: str
    page_no: int
    text: str
    status: FactStatus
    metric: Metric | None = None


class ContextPack(BaseModel):
    sub_question: str
    snippets: list[Snippet]
    rendered: str  # 原样送进模型的文本：「## 子问题 … [S4 | 第9页] …」


# ── 规划（generate/planner.py）────────────────────────────────
class PlanSection(BaseModel):
    heading: str
    sub_question: str
    expected_pages_hint: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    title: str
    lede_points: list[str] = Field(default_factory=list)
    sections: list[PlanSection]
    insufficient: list[str] = Field(default_factory=list)
    ending_points: list[str] = Field(default_factory=list)


# ── 校验（verify/）────────────────────────────────────────────
CheckName = Literal["V1", "V2", "V3", "V4", "V5"]


class Sentence(BaseModel):
    idx: int  # 节内序号，0 起
    text: str  # 含 [S编号]
    cites: list[str] = Field(default_factory=list)  # ["S4", "S5"]


class CheckResult(BaseModel):
    check: CheckName
    ok: bool
    reason: str = ""


class SentenceVerdict(BaseModel):
    sentence: Sentence
    results: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


# ── 编排与流式（agent/）────────────────────────────────────────
EventType = Literal[
    "session", "intent", "plan", "retrieval", "delta", "verify", "patch",
    "section_final", "insufficient", "done", "error",
]


class Event(BaseModel):
    """一条 SSE 事件。字段约定见 10-agent-architecture.md 第 6.2 节。"""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False, default=str)}\n\n"
