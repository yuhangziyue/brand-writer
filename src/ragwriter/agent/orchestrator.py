"""会话 Agent 编排器：载入会话 → 意图识别 → 路由处理器 → 回写会话与运行日志。

这是**受控工作流**，不是自主 Agent（10 号文档第 0 节）：路由由程序按意图决定，
检索是必经步骤而不是模型可选的工具，每次请求的步骤序列确定、可回放、可审计。

所有外部能力（检索、写作、校验、模型）都经 `Deps` 注入：生产环境 `Deps.from_env()` 组装，
测试与评估逐项替换成假实现（见 `ragwriter.agent.testing`）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from ragwriter.agent import intent as intent_mod
from ragwriter.agent.handlers import TurnContext, done_event, ev
from ragwriter.agent.handlers import confirm as confirm_handler
from ragwriter.agent.handlers import qa as qa_handler
from ragwriter.agent.handlers import refuse as refuse_handler
from ragwriter.agent.handlers import revise as revise_handler
from ragwriter.agent.handlers import write as write_handler
from ragwriter.agent.intent import IntentResult
from ragwriter.agent.memory import SessionStore, new_id
from ragwriter.contracts import Event

log = logging.getLogger("ragwriter.agent")

DEFAULT_SESSIONS_DB = "build/runtime/sessions.db"
DEFAULT_RAG_DB = "build/index/rag.db"


def _missing(name: str, exc: BaseException) -> Callable[..., Any]:
    """他组模块缺失时的占位：导入期不炸（回放模式仍要能起服务），调用时才明确报错。"""

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError(f"{name} 不可用：{exc}")

    _raise.__name__ = f"missing_{name}"
    return _raise


@dataclass
class Deps:
    """依赖容器。字段逐项可替换：`replace(deps, verify_sentence=fake)`。

    client 为 None 表示无模型（无 key）：意图只走规则，write/qa 给出明确的「无模型」拒答，
    API 层据此切到离线回放。
    """

    store: Any = None
    client: Any = None
    embedder: Any = None
    # 检索（A 组）
    search: Callable[..., Any] | None = None
    search_many: Callable[..., Any] | None = None
    # 生成（D 组）
    plan: Callable[..., Any] | None = None
    build_context: Callable[..., Any] | None = None
    stream_section: Callable[..., Any] | None = None
    stream_bookend: Callable[..., Any] | None = None
    stream_answer: Callable[..., Any] | None = None
    rewrite_sentence: Callable[..., Any] | None = None
    rewrite_query: Callable[..., Any] | None = None
    check_topic: Callable[..., Any] | None = None  # 写作前主题审核；缺席时确认卡退回原样复述
    split_gaps: Callable[..., Any] | None = None  # 缺席时处理器用兜底实现
    # 校验（B 组）
    split_sentences: Callable[..., Any] | None = None
    verify_sentence: Callable[..., Any] | None = None
    build_report: Callable[..., Any] | None = None
    # 诊断：哪些部件没装上、为什么。healthz 直接展示，退化必须显式
    notes: dict[str, str] = field(default_factory=dict)
    _profile: dict | None = field(default=None, repr=False)

    @property
    def profile(self) -> dict:
        if self._profile is None:
            try:
                self._profile = dict(self.store.profile()) if self.store is not None else {}
            except Exception as exc:  # noqa: BLE001
                self.notes["profile"] = str(exc)
                self._profile = {}
        return self._profile

    @property
    def live(self) -> bool:
        return self.client is not None

    def with_(self, **kw: Any) -> "Deps":
        return replace(self, **kw)

    @classmethod
    def from_env(cls, *, rag_db: Path | str | None = None, load_dense: bool | None = None) -> "Deps":
        notes: dict[str, str] = {}
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        # 模型
        client = None
        try:
            from ragwriter.llm.client import LLMClient, LLMConfig, LLMNotConfigured, RunLog

            try:
                client = LLMClient(LLMConfig.from_env(), RunLog(Path("build/runtime/llm-runlog.jsonl")))
            except LLMNotConfigured as exc:
                notes["client"] = f"replay: {exc}"
        except ImportError as exc:
            notes["client"] = str(exc)

        # 存储与检索
        store = None
        db = Path(rag_db or os.getenv("RAG_DB", DEFAULT_RAG_DB))
        try:
            from ragwriter.index.store import RagStore

            if db.exists():
                store = RagStore(db)
            else:
                notes["store"] = f"{db} 不存在，先跑 make index"
        except Exception as exc:  # noqa: BLE001
            notes["store"] = str(exc)

        embedder = None
        if store is not None and client is not None:
            # 回放模式下不加载 2GB 的嵌入模型：用不上，还拖慢启动
            try:
                from ragwriter.index.embed import load_embedder

                embedder = load_embedder(load_dense)
                if embedder is None:
                    notes["embedder"] = "dense disabled: BM25 only"
            except Exception as exc:  # noqa: BLE001
                notes["embedder"] = str(exc)

        if client is not None and store is not None:
            _warn_dense(store, embedder, notes)

        fns: dict[str, Callable[..., Any] | None] = {}
        for mod, names in _WIRING:
            try:
                m = __import__(mod, fromlist=list(names))
            except Exception as exc:  # noqa: BLE001
                for n in names:
                    fns[n] = None if n in _OPTIONAL else _missing(n, exc)
                notes[mod] = str(exc)
                continue
            for n in names:
                fn = getattr(m, n, None)
                if fn is None:
                    fns[n] = None if n in _OPTIONAL else _missing(n, AttributeError(f"{mod}.{n}"))
                    notes[f"{mod}.{n}"] = "missing"
                else:
                    fns[n] = fn
        return cls(store=store, client=client, embedder=embedder, notes=notes, **fns)


# 他组模块的落点（按实际包结构，generate/ 与 verify/ 的 __init__ 不做再导出）
_WIRING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ragwriter.retrieve.hybrid", ("search", "search_many")),
    ("ragwriter.generate.planner", ("plan",)),
    ("ragwriter.generate.context", ("build_context",)),
    ("ragwriter.generate.writer", ("stream_section", "stream_bookend", "stream_answer", "rewrite_sentence",
                                   "split_gaps")),
    ("ragwriter.generate.query", ("rewrite_query",)),
    ("ragwriter.generate.topic", ("check_topic",)),
    ("ragwriter.verify.sentences", ("split_sentences",)),
    ("ragwriter.verify.checks", ("verify_sentence",)),
    ("ragwriter.verify.report", ("build_report",)),
)
_OPTIONAL = frozenset({"split_gaps", "check_topic"})


def _warn_dense(store: Any, embedder: Any, notes: dict[str, str]) -> None:
    """稠密路不可用时打一行显眼警告。不私自覆盖用户的 USE_DENSE：退化可以，但必须让人看见。

    A 组实测：BM25 单路的充分性判断错 3/18，拒答会变得不可靠。
    """
    try:
        meta = store.meta() if hasattr(store, "meta") else {}
    except Exception:  # noqa: BLE001
        meta = {}
    reason = ""
    if embedder is None:
        reason = "embedder 未加载（USE_DENSE=0 或未装 .[dense]）"
    elif not meta.get("dense"):
        reason = "rag.db 建库时没有稠密向量（meta.dense=0），先 make index"
    elif meta.get("embed_model") and meta.get("embed_model") != getattr(embedder, "model_name", None):
        reason = f"嵌入模型与库不一致：库 {meta.get('embed_model')} / 运行 {getattr(embedder, 'model_name', '?')}"
    if reason:
        notes["dense"] = f"BM25 only: {reason}"
        log.warning("⚠️⚠️⚠️ 稠密检索未启用，退化为 BM25 单路，充分性判断不可靠：%s", reason)


_ROUTES: dict[str, Any] = {
    "write": write_handler.handle,
    "qa": qa_handler.handle,
    "revise": revise_handler.handle,
    "confirm": confirm_handler.handle,
}
# 不调模型也能完成的改写动作：无 key 时照样可用
_REVISE_NO_MODEL = frozenset({"delete"})


class Orchestrator:
    def __init__(self, deps: Deps, *, sessions_db: Path | str = DEFAULT_SESSIONS_DB) -> None:
        self.deps = deps
        self.memory = SessionStore(sessions_db)

    # ── 入口 ────────────────────────────────────────────────
    def handle(self, message: str, *, session_id: str | None = None,
               options: dict | None = None) -> Iterator[Event]:
        """同步生成器。首个事件 session，第二个 intent，最后一个 done（出错时 error 之后仍有 done）。"""
        options = dict(options or {})
        t0 = time.perf_counter()
        sid = self.memory.ensure_session(session_id)
        turn_id = self.memory.add_turn(sid, "user", message)
        run_id = new_id("r")
        records: list[dict[str, Any]] = []

        def emit(e: Event) -> Event:
            records.append({"t": round((time.perf_counter() - t0) * 1000), "type": e.type, "data": e.data})
            return e

        slots = self.memory.load_slots(sid)
        window = self.memory.window(sid, exclude_turn=turn_id)
        runlog = getattr(self.deps.client, "runlog", None)
        ctx = TurnContext(
            deps=self.deps, memory=self.memory, session_id=sid, turn_id=turn_id, run_id=run_id,
            message=message, intent=IntentResult(intent="clarify"), slots=slots, window=window,
            options=options, profile=self.deps.profile,
            usage_start=len(getattr(runlog, "records", []) or []),
        )
        finished = False
        try:
            yield emit(ev("session", session_id=sid, turn_id=turn_id, run_id=run_id,
                          mode="live" if self.deps.live else "no_model"))

            res = self._classify(message, slots, window.render(), options)
            new_slots = intent_mod.merge_slots(slots, res)
            if res.intent == "confirm" and res.confirm_action == "accept":
                # 确认通过 ⇒ 本轮就是 write：用待确认的体裁与精确主题，审核给的要点交给写作链路
                res, new_slots = _accept(res, new_slots, options)
            ctx.intent = res
            ctx.slots = new_slots
            self.memory.update_turn_intent(turn_id, res.to_dict())
            yield emit(ev("intent", intent=res.intent, genre=res.genre, topic=res.topic, query=res.query,
                          constraints=res.constraints, confidence=res.confidence, source=res.source,
                          reason=res.reason, inherited=res.inherited, slots=new_slots,
                          answer_shape=res.answer_shape, focus=res.focus,
                          revise_target=res.revise_target, revise_action=res.revise_action,
                          revise_instruction=res.revise_instruction, confirm_action=res.confirm_action,
                          pending=bool(new_slots.get("pending_write")),
                          supported=res.intent in intent_mod.SUPPORTED))

            needs_model = res.intent in ("write", "qa") or (
                res.intent == "revise" and res.revise_action not in _REVISE_NO_MODEL)
            if needs_model and not self.deps.live:
                res.reason = "no_model"
                handler = refuse_handler.handle
            elif res.intent == "write" and not _confirmed(res, options):
                # 每次写作先审主题、与用户确认（10-23 子曰定）：本轮只发确认卡，不检索、不写作
                handler = confirm_handler.handle
            else:
                handler = _ROUTES.get(res.intent, refuse_handler.handle)

            for e in handler(ctx):
                if e.type == "done":
                    e.data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
                    e.data.setdefault("usage", {})["elapsed_ms"] = e.data["elapsed_ms"]
                    e.data["run_id"] = run_id
                    e.data["session_id"] = sid
                    finished = True
                yield emit(e)
        except Exception as exc:  # noqa: BLE001 — 任何一步炸了都要给前端一个收口，不留半截流
            if not finished:
                yield emit(ev("error", stage=_stage_of(records), reason=f"{type(exc).__name__}: {exc}",
                              fallback=False))
                d = done_event(ctx, final_text="", refused=True, error=True)
                d.data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
                d.data["run_id"] = run_id
                d.data["session_id"] = sid
                finished = True
                yield emit(d)
        finally:
            self._persist(ctx, records)

    # ── 内部 ────────────────────────────────────────────────
    def _classify(self, message: str, slots: dict, window_text: str, options: dict) -> IntentResult:
        forced = options.get("force_intent")
        if forced:
            # /v1/generate 与评估脚本用：跳过识别，直接指定意图与槽位，保证可复现
            res = IntentResult(intent=forced.get("intent", "write"), genre=forced.get("genre"),
                               topic=forced.get("topic"), query=forced.get("query"),
                               constraints=list(forced.get("constraints") or []), source="forced")
            if res.intent == "write" and not res.topic and slots.get("topic"):
                res.topic, res.inherited = slots["topic"], {"topic": slots["topic"]}
            return res
        action = options.get("confirm")
        if action:
            # 确认卡上的按钮：动作已经由用户点明，不必再问意图模型
            if not slots.get("pending_write"):
                return IntentResult(intent="clarify", source="selected", reason="no_pending")
            if action not in intent_mod.CONFIRM_ACTIONS:
                action = "accept"
            topic = str(options.get("topic") or "").strip() or None
            if action == "pick" and not topic:
                action = "modify"
            return IntentResult(intent="confirm", source="selected", confirm_action=action,
                                topic=topic if action == "pick" else None,
                                revise_instruction=str(options.get("feedback") or message) if action == "modify" else "")
        genre, topic = options.get("genre"), str(options.get("topic") or "").strip()
        if genre in intent_mod.GENRES and topic:
            # 写作台（前端选了体裁、填了主题、点「生成」）= 指定模式：用户已经说清楚要什么，不做意图识别
            return IntentResult(intent="write", genre=genre, topic=topic, source="selected",
                                constraints=[str(c) for c in options.get("constraints") or [] if str(c).strip()])
        return intent_mod.classify(message, slots, client=self.deps.client, profile=self.deps.profile,
                                   window_text=window_text)

    def _persist(self, ctx: TurnContext, records: list[dict[str, Any]]) -> None:
        """回写放在 finally：客户端中途断开（GeneratorExit）也要留下已发出的事件，供排障与回放。"""
        try:
            out = ctx.outcome
            slots = dict(ctx.slots)
            if out.get("kind") in ("write",):
                slots["last_artifact_id"] = out.get("artifact_id")
                slots["last_artifact_outline"] = out.get("outline", [])
            if out.get("pages"):
                slots["referenced_pages"] = sorted({*(slots.get("referenced_pages") or []), *out["pages"]})
            self.memory.save_slots(ctx.session_id, slots)
            if out:
                # 助手正文存 content（L2 全量），大纲与页码存 intent_json；工作窗口只读后者
                self.memory.add_turn(ctx.session_id, "assistant", out.get("final_text", ""), {
                    "intent": ctx.intent.intent, "outline": out.get("outline", []),
                    "pages": out.get("pages", []), "artifact_id": out.get("artifact_id"),
                    "run_id": ctx.run_id,
                })
            cfg = getattr(self.deps.client, "cfg", None)
            meta = {"_meta": {"message": ctx.message, "intent": ctx.intent.intent, "run_id": ctx.run_id,
                              "session_id": ctx.session_id, "topic": ctx.intent.topic,
                              "genre": ctx.intent.genre, "model": getattr(cfg, "model", None)}}
            lines = [json.dumps(meta, ensure_ascii=False)]
            lines += [json.dumps(r, ensure_ascii=False, default=str) for r in records]
            self.memory.save_run(ctx.run_id, ctx.session_id, ctx.turn_id, "\n".join(lines) + "\n")
        except Exception:  # noqa: BLE001 — 回写失败不能反过来让已完成的回答报错
            pass


def _confirmed(res: IntentResult, options: dict) -> bool:
    """这一轮 write 是否已经确认过。

    force_intent（/v1/generate、评估、回放录制）默认视为已确认，评估要一步出稿才可复现；
    显式传 force_intent.confirmed=False 才走确认。
    """
    if options.get("confirmed"):
        return True
    forced = options.get("force_intent")
    if res.source == "forced" and isinstance(forced, dict):
        return forced.get("confirmed", True) is not False
    return False


def _accept(res: IntentResult, slots: dict, options: dict) -> tuple[IntentResult, dict]:
    """confirm/accept → write。pending 里存的是审核后的精确主题，写作按它来，而不是用户最初的原话。"""
    pending = dict(slots.get("pending_write") or {})
    write = IntentResult(intent="write", genre=pending.get("genre") or slots.get("genre") or "blog",
                         topic=pending.get("topic") or slots.get("topic"),
                         constraints=list(pending.get("constraints") or []), source=res.source,
                         confidence=res.confidence, confirm_action="accept")
    options["confirmed"] = True
    options["confirmed_brief"] = {"angle": pending.get("angle", ""),
                                  "outline_points": list(pending.get("outline_points") or []),
                                  "premise_gaps": list(pending.get("premise_gaps") or [])}
    new_slots = intent_mod.merge_slots(slots, write)
    new_slots["pending_write"] = None
    return write, new_slots


def _stage_of(records: list[dict[str, Any]]) -> str:
    """按最后一个已发事件推断出错阶段，给前端一个能看懂的位置。"""
    last = records[-1]["type"] if records else "session"
    return {"session": "intent", "intent": "route", "plan": "retrieval", "retrieval": "generate",
            "delta": "generate", "verify": "verify", "patch": "verify", "section_final": "generate",
            "insufficient": "retrieval"}.get(last, last)
