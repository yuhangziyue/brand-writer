"""HTTP 接口：SSE 会话流、会话查询、页面信息、回放模式与前端静态页（10 号文档 6.2 节、05 第 6 节）。

编排器是同步生成器；StreamingResponse 收到同步迭代器时会放进线程池逐个取（iterate_in_threadpool），
不阻塞事件循环。依赖在第一次请求时才组装：导入本模块（测试、uvicorn 热重载）不该触发加载 2GB 的嵌入模型。

无 API key ⇒ 回放模式：/v1/chat 只重放录过的主题，没录过的明确返回 replay_unmatched（评审 M4）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragwriter.agent.intent import SUGGESTED_TOPICS
from ragwriter.agent.orchestrator import DEFAULT_SESSIONS_DB, Deps, Orchestrator
from ragwriter.agent.replay import DEFAULT_FIXTURES, DEFAULT_SPEED, FixtureLibrary, replay_stream
from ragwriter.contracts import Event

log = logging.getLogger("ragwriter.api")

GENRE_LABEL = {"blog": "官网 Blog", "faq": "FAQ", "brand_intro": "品牌介绍", "product_intro": "产品介绍"}
REPO_ROOT = Path(__file__).resolve().parents[3]


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str
    options: dict[str, Any] = Field(default_factory=dict)


class GenerateIn(BaseModel):
    topic: str
    genre: str = "blog"
    session_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class _State:
    """懒加载的依赖与编排器。加锁：并发首请求只组装一次。"""

    def __init__(self, deps: Deps | None, factory: Callable[[], Deps], sessions_db: Path | str,
                 fixtures_dir: Path | str) -> None:
        self._deps = deps
        self._factory = factory
        self._orch: Orchestrator | None = None
        self._lock = threading.Lock()
        self.sessions_db = sessions_db
        self.library = FixtureLibrary(fixtures_dir)

    @property
    def deps(self) -> Deps:
        if self._deps is None:
            with self._lock:
                if self._deps is None:
                    self._deps = self._factory()
                    if not self._deps.live:
                        log.warning("未配置 LLM_API_KEY：进入回放模式，只重放 %s 里录过的主题", self.library.dir)
        return self._deps

    @property
    def orch(self) -> Orchestrator:
        if self._orch is None:
            with self._lock:
                if self._orch is None:
                    self._orch = Orchestrator(self.deps, sessions_db=self.sessions_db)
        return self._orch

    @property
    def mode(self) -> str:
        return "live" if self.deps.live else "replay"


def _sse(events: Iterator[Event]) -> Iterator[str]:
    for e in events:
        yield e.to_sse()


def _stream(events: Iterator[Event]) -> StreamingResponse:
    # X-Accel-Buffering：经 nginx 反代时关掉缓冲，否则「流式」会被攒成一次性返回
    return StreamingResponse(_sse(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def create_app(*, deps: Deps | None = None, deps_factory: Callable[[], Deps] | None = None,
               sessions_db: Path | str | None = None, fixtures_dir: Path | str | None = None,
               web_dir: Path | str | None = None, pages_jsonl: Path | str | None = None) -> FastAPI:
    state = _State(
        deps, deps_factory or Deps.from_env,
        sessions_db or os.getenv("SESSIONS_DB", DEFAULT_SESSIONS_DB),
        fixtures_dir or os.getenv("FIXTURES_DIR", str(DEFAULT_FIXTURES)),
    )
    pages_path = Path(pages_jsonl or os.getenv("PAGES_JSONL", "build/pages.jsonl"))
    app = FastAPI(title="ragwriter", version="0.1.0")
    app.state.rw = state

    def _events(message: str, session_id: str | None, options: dict[str, Any]) -> Iterator[Event]:
        if state.mode == "replay":
            speed = float(options.get("replay_speed", os.getenv("REPLAY_SPEED", DEFAULT_SPEED)))
            return replay_stream(state.library, message, speed=speed, session_id=session_id)
        return state.orch.handle(message, session_id=session_id, options=options)

    # ── 会话 ───────────────────────────────────────────────
    @app.post("/v1/chat")
    def chat(body: ChatIn) -> StreamingResponse:
        return _stream(_events(body.message, body.session_id, body.options))

    @app.post("/v1/generate")
    def generate(body: GenerateIn) -> StreamingResponse:
        """薄封装：新会话 + 一条 write 消息（10 号文档第 7 节）。跳过意图识别，保证评估脚本可复现。"""
        message = f"写一篇{GENRE_LABEL.get(body.genre, body.genre)}：{body.topic}"
        options = {**body.options, "force_intent": {"intent": "write", "genre": body.genre, "topic": body.topic}}
        return _stream(_events(message, body.session_id, options))

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        out = state.orch.memory.get_session(session_id)
        if out is None:
            raise HTTPException(404, f"会话 {session_id} 不存在")
        return out

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = state.orch.memory.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"运行 {run_id} 不存在")
        lines = [json.loads(x) for x in run["events_jsonl"].splitlines() if x.strip()]
        return {"run_id": run_id, "session_id": run["session_id"], "turn_id": run["turn_id"],
                "meta": next((x["_meta"] for x in lines if "_meta" in x), {}),
                "events": [x for x in lines if "_meta" not in x]}

    # ── 页面 ───────────────────────────────────────────────
    def _page(n: int) -> dict[str, Any] | None:
        store = state.deps.store
        if store is not None and hasattr(store, "page"):
            try:
                page = store.page(n)
            except (KeyError, IndexError, LookupError):
                return None
            data = page.model_dump() if hasattr(page, "model_dump") else dict(page)
            try:
                data["facts"] = [{"fact_id": f.fact_id, "kind": f.kind, "text": f.text, "status": f.status}
                                 for f in store.page_facts(n, statuses=("verified", "image_structural",
                                                                         "image_only", "unverified"))]
            except Exception:  # noqa: BLE001
                data["facts"] = []
            return data
        # 没有 rag.db（例如纯回放演示）时退到建库中间产物，前端缩略图与高亮照样能用
        if pages_path.exists():
            for line in pages_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("page_no") == n:
                        return row
        return None

    @app.get("/v1/pages/{n}")
    def get_page(n: int) -> dict[str, Any]:
        data = _page(n)
        if data is None:
            raise HTTPException(404, f"第 {n} 页不存在")
        data["image_url"] = f"/v1/pages/{n}/image"
        data.pop("image_path", None)  # 不向前端暴露本机绝对路径
        return data

    @app.get("/v1/pages/{n}/image")
    def get_page_image(n: int) -> FileResponse:
        data = _page(n)
        path = Path((data or {}).get("image_path") or REPO_ROOT / "build" / "pages" / f"p{n:02d}.png")
        if not path.exists():
            raise HTTPException(404, f"第 {n} 页没有渲染图")
        return FileResponse(path, media_type="image/png")

    # ── 前端辅助 ───────────────────────────────────────────
    @app.get("/v1/suggestions")
    def suggestions() -> dict[str, Any]:
        return {"suggestions": list(SUGGESTED_TOPICS)}

    @app.get("/v1/mode")
    def mode() -> dict[str, Any]:
        d = state.deps
        if d.live:
            model = getattr(getattr(d.client, "cfg", None), "model", None)
            return {"mode": "live", "model": model, "recorded_topics": state.library.recorded_topics()}
        models = state.library.models()
        return {"mode": "replay", "model": models[0] if models else None,
                "recorded_topics": state.library.recorded_topics()}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        d = state.deps
        return {"ok": True, "mode": state.mode, "store": d.store is not None, "dense": d.embedder is not None,
                "notes": d.notes}

    # 静态前端最后挂：路由按注册顺序匹配，/v1/* 与 /healthz 优先
    web = Path(web_dir) if web_dir else REPO_ROOT / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
