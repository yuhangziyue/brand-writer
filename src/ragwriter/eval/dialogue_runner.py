"""多轮对话用例的加载与逐轮执行。

一条用例 = 同一会话里的若干轮用户消息，每轮带期望（expect）。本模块只负责
「把消息逐轮喂给编排器、把事件流收集成结构化的逐轮记录」，不做任何判定——
判定在 dialogue_checks.py（确定性）与 G 组的评估器（模型判别）里。

用例文件：eval/dialogues/*.yaml，格式与 G 组共用，字段改动须双方同意。
编排器只需满足鸭子类型 `handle(message, session_id=..., options=...) -> Iterator[Event]`，
所以假实现与真 Orchestrator 跑的是同一套代码路径。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragwriter.contracts import Event, Genre, Intent

ROOT = Path(__file__).resolve().parents[3]
DIALOGUE_DIR = ROOT / "eval" / "dialogues"


# ── 用例格式（与 G 组共用，extra=forbid：写错字段名直接报错，不静默忽略）──────
class NumberExpect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str  # 照抄原文的数字，如 "15倍"
    object: str  # 该数字描述的对象关键词，如 "订单"


class KeyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    page: int


class TurnExpect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent
    genre: Genre | None = None
    topic_contains: list[str] = Field(default_factory=list)
    refused: bool | None = None
    events_in_order: list[str] = Field(default_factory=list)
    cite_pages_any: list[int] = Field(default_factory=list)
    cite_pages_all: list[int] = Field(default_factory=list)
    forbidden_strings: list[str] = Field(default_factory=list)
    numbers: list[NumberExpect] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    must_say_insufficient: list[str] = Field(default_factory=list)
    no_clarify: bool | None = None
    reference_answer: str = ""
    # 2026-10-22 产品评审后新增（可选）
    refusal_kind: str | None = None  # 如 "insufficient_all"，对 done.data.refusal_kind
    inherited_topic: bool | None = None  # true ⇒ intent 事件 inherited.topic 须非空


class DialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str
    expect: TurnExpect
    options: dict[str, Any] | None = None  # 原样透传给 handle(options=...)，如 {inject_fault: v3_swap}


class DialogueCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    tags: list[str] = Field(default_factory=list)
    turns: list[DialogueTurn]


def load_cases(path: str | Path | None = None) -> list[DialogueCase]:
    """读一个 yaml 文件或一个目录下全部 *.yaml，按文件名、文件内顺序返回。id 不得重复。"""
    p = Path(path) if path else DIALOGUE_DIR
    files = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    cases: list[DialogueCase] = []
    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        cases.extend(DialogueCase.model_validate(c) for c in raw)
    ids = [c.id for c in cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"用例 id 重复：{sorted(dup)}")
    return cases


# ── 逐轮记录 ────────────────────────────────────────────────────
class TurnRun(BaseModel):
    seq: int  # 0 起
    user: str
    events: list[Event] = Field(default_factory=list)
    session_id: str | None = None
    turn_id: str | None = None
    # intent 事件
    intent: str | None = None
    genre: str | None = None
    topic: str | None = None
    query: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    intent_source: str | None = None
    inherited: dict[str, Any] = Field(default_factory=dict)  # intent.inherited：{topic?, genre?}
    # done 事件
    artifact_id: str | None = None
    final_text: str = ""
    citations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    refused: bool = False
    nearest_pages: list[int] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    stream_mode: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    refusal_kind: str | None = None
    boundary: list[dict[str, Any]] = Field(default_factory=list)  # 写作的资料不足：{topic, source, nearest_pages}
    jsonld: Any = None  # FAQ 的 schema.org FAQPage
    options: dict[str, Any] | None = None
    # 过程
    retrieved_fact_ids: list[str] = Field(default_factory=list)  # 本轮全部 retrieval 事件命中
    retrieved_pages: list[int] = Field(default_factory=list)
    insufficient: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)  # error 事件 + 运行异常
    has_done: bool = False
    elapsed_ms: float = 0.0
    first_event_ms: float | None = None

    @property
    def event_types(self) -> list[str]:
        return [e.type for e in self.events]


class CaseRun(BaseModel):
    case_id: str
    title: str = ""
    session_id: str | None = None  # 第一轮 session 事件给出的会话
    turns: list[TurnRun] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class SupportsHandle(Protocol):
    def handle(self, message: str, session_id: str | None = None,
               options: dict | None = None) -> Iterable[Event]: ...


# ── 事件解析 ────────────────────────────────────────────────────
def _walk_facts(obj: Any) -> Iterator[dict]:
    """retrieval 载荷的精确形状由 E 组定；这里递归找所有带 fact_id 的 dict，对形状不敏感。"""
    if isinstance(obj, dict):
        if "fact_id" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_facts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_facts(v)


def _page_of(fact_id: str) -> int | None:
    # fact_id 约定 "p10-f03" / "p05-v26"
    if fact_id.startswith("p"):
        head = fact_id[1:].split("-", 1)[0]
        if head.isdigit():
            return int(head)
    return None


def _as_int_list(v: Any) -> list[int]:
    out: list[int] = []
    for x in v or []:
        if isinstance(x, dict):
            x = x.get("page_no", x.get("page"))
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def absorb(turn: TurnRun, ev: Event) -> None:
    """把一条事件并入逐轮记录。"""
    d = ev.data or {}
    if ev.type == "session":
        turn.session_id = d.get("session_id")
        turn.turn_id = d.get("turn_id")
    elif ev.type == "intent":
        turn.intent = d.get("intent")
        slots = d.get("slots") or {}
        turn.slots = slots if isinstance(slots, dict) else {}
        turn.genre = d.get("genre") or turn.slots.get("genre")
        turn.topic = d.get("topic") or turn.slots.get("topic")
        turn.query = d.get("query")
        turn.intent_source = d.get("source")
        inh = d.get("inherited") or {}
        turn.inherited = inh if isinstance(inh, dict) else {}
    elif ev.type == "retrieval":
        for f in _walk_facts(d):
            fid = str(f["fact_id"])
            if fid not in turn.retrieved_fact_ids:
                turn.retrieved_fact_ids.append(fid)
            pg = f.get("page_no", f.get("page", _page_of(fid)))
            if pg is not None and int(pg) not in turn.retrieved_pages:
                turn.retrieved_pages.append(int(pg))
    elif ev.type == "insufficient":
        turn.insufficient.append(dict(d))
    elif ev.type == "error":
        turn.errors.append(dict(d))
    elif ev.type == "done":
        turn.has_done = True
        turn.artifact_id = d.get("artifact_id")
        turn.final_text = d.get("final_text") or ""
        cits = d.get("citations") or {}
        turn.citations = {str(k): dict(v) for k, v in cits.items()} if isinstance(cits, dict) else {}
        turn.refused = bool(d.get("refused", False))
        turn.nearest_pages = _as_int_list(d.get("nearest_pages"))
        turn.suggestions = [str(s) for s in (d.get("suggestions") or [])]
        turn.stream_mode = d.get("stream_mode")
        turn.usage = d.get("usage") or {}
        turn.refusal_kind = d.get("refusal_kind")
        b = d.get("boundary") or []
        turn.boundary = [dict(x) if isinstance(x, dict) else {"topic": str(x)} for x in b] if isinstance(b, list) else []
        turn.jsonld = d.get("jsonld")


# ── 执行 ────────────────────────────────────────────────────────
def run_case(case: DialogueCase | dict, orch: SupportsHandle, *,
             session_id: str | None = None, options: dict | None = None) -> CaseRun:
    """逐轮喂消息。第一轮 session_id 传入值（默认 None，由编排器新建），
    之后每轮沿用第一轮 session 事件给出的 session_id——跨轮一致性由 checks 判定，
    这里不偷偷修正。单轮抛异常记入 errors 并继续下一轮，不让一条坏轮次吞掉整条用例。"""
    if isinstance(case, dict):
        case = DialogueCase.model_validate(case)
    run = CaseRun(case_id=case.id, title=case.title)
    sid = session_id
    t_case = time.perf_counter()
    for seq, t in enumerate(case.turns):
        # 评估测的是写作质量：默认跳过写作前的主题确认（10-23 新增的确认轮由 tests/test_revise.py 覆盖）；
        # 用例要测确认流程时在该轮 options 里写 confirmed: false
        turn_opts = {"confirmed": True, **(options or {}), **(t.options or {})}
        turn = TurnRun(seq=seq, user=t.user, options=turn_opts)
        t0 = time.perf_counter()
        try:
            for ev in orch.handle(t.user, session_id=sid, options=turn_opts):
                if not isinstance(ev, Event):
                    ev = Event.model_validate(ev)
                if turn.first_event_ms is None:
                    turn.first_event_ms = (time.perf_counter() - t0) * 1000
                turn.events.append(ev)
                absorb(turn, ev)
        except Exception as exc:  # noqa: BLE001 — 记录而不是中断，评估要看到全部轮次
            turn.errors.append({"stage": "runner", "reason": f"{type(exc).__name__}: {exc}"})
        turn.elapsed_ms = (time.perf_counter() - t0) * 1000
        if sid is None:
            sid = turn.session_id
            run.session_id = sid
        run.turns.append(turn)
    if run.session_id is None and run.turns:
        run.session_id = run.turns[0].session_id
    run.elapsed_ms = (time.perf_counter() - t_case) * 1000
    return run
