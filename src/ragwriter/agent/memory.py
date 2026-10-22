"""会话记忆：L2 会话上下文（sessions.db）与 L3 工作窗口（10 号文档 2.3、2.4 节）。

sessions.db 与只读的 rag.db 分开放：rag.db 是建库产物，随时可以整个删掉重建；
会话是运行数据，不能跟着建库一起丢。

每个操作单独开关连接：API 用线程池迭代同步生成器，同一个流的前后几步可能落在不同线程，
sqlite3 连接默认不允许跨线程共享，「用完即关」最省心，这点开销在单用户演示里可以忽略。
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

WINDOW_TURNS = 3  # 工作窗口保留最近几轮，见 2.4 节

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, slots_json TEXT);
CREATE TABLE IF NOT EXISTS turns(
  turn_id TEXT PRIMARY KEY, session_id TEXT, seq INTEGER, role TEXT,
  content TEXT, intent_json TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, seq);
CREATE TABLE IF NOT EXISTS artifacts(
  artifact_id TEXT PRIMARY KEY, session_id TEXT, turn_id TEXT, genre TEXT, topic TEXT,
  body_json TEXT, citations_json TEXT, verify_json TEXT);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, session_id TEXT, turn_id TEXT, events_jsonl TEXT);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def empty_slots() -> dict[str, Any]:
    return {
        "genre": None,
        "topic": None,
        "constraints": [],
        "last_query": None,
        "last_artifact_id": None,
        "last_artifact_outline": [],
        "referenced_pages": [],
    }


@dataclass
class WorkingWindow:
    """L3 工作窗口：结构化槽位 + 最近 3 轮用户原文 + 最近 3 轮助手「大纲 + 引用页码」。

    🔴 助手正文**刻意不进窗口**。窗口会被送进意图识别、查询改写等模型节点，
    正文一旦进去，模型就可能把自己上一轮写的句子当成事实来源（自我污染），
    而且这条路径上没有校验去拦它。正文按 last_artifact_id 从 artifacts 表取，
    只在改写链路里载入、且只载入已通过校验的句子。见 2.4 节。
    """

    slots: dict[str, Any]
    user_turns: list[str] = field(default_factory=list)
    assistant_turns: list[dict[str, Any]] = field(default_factory=list)  # {intent, outline, pages}

    def render(self) -> str:
        lines = ["槽位：" + json.dumps(self.slots, ensure_ascii=False)]
        for i, u in enumerate(self.user_turns, 1):
            lines.append(f"用户[{i}]：{u}")
        for i, a in enumerate(self.assistant_turns, 1):
            pages = "、".join(str(p) for p in a.get("pages", []))
            outline = "；".join(a.get("outline", []))
            lines.append(f"助手[{i}]（{a.get('intent', '')}）大纲：{outline}" + (f"（第{pages}页）" if pages else ""))
        return "\n".join(lines)


class SessionStore:
    def __init__(self, db_path: Path | str = "build/runtime/sessions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── sessions ────────────────────────────────────────────
    def ensure_session(self, session_id: str | None) -> str:
        """给了 id 且存在就沿用；给了 id 但不存在就按这个 id 新建（前端可自带 id）。"""
        sid = session_id or new_id("s")
        now = time.time()
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM sessions WHERE session_id=?", (sid,)).fetchone()
            if row is None:
                c.execute("INSERT INTO sessions VALUES(?,?,?,?)",
                          (sid, now, now, json.dumps(empty_slots(), ensure_ascii=False)))
        return sid

    def load_slots(self, session_id: str) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT slots_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        slots = empty_slots()
        if row and row["slots_json"]:
            slots.update(json.loads(row["slots_json"]))
        return slots

    def save_slots(self, session_id: str, slots: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET slots_json=?, updated_at=? WHERE session_id=?",
                      (json.dumps(slots, ensure_ascii=False), time.time(), session_id))

    # ── turns ───────────────────────────────────────────────
    def add_turn(self, session_id: str, role: str, content: str,
                 intent: dict[str, Any] | None = None) -> str:
        turn_id = new_id("t")
        with self._conn() as c:
            seq = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM turns WHERE session_id=?",
                            (session_id,)).fetchone()[0]
            c.execute("INSERT INTO turns VALUES(?,?,?,?,?,?,?)",
                      (turn_id, session_id, seq, role, content,
                       json.dumps(intent or {}, ensure_ascii=False), time.time()))
        return turn_id

    def update_turn_intent(self, turn_id: str, intent: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("UPDATE turns SET intent_json=? WHERE turn_id=?",
                      (json.dumps(intent, ensure_ascii=False, default=str), turn_id))

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM turns WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [{**dict(r), "intent": json.loads(r["intent_json"] or "{}")} for r in rows]

    def window(self, session_id: str, *, exclude_turn: str | None = None,
               n: int = WINDOW_TURNS) -> WorkingWindow:
        turns = [t for t in self.turns(session_id) if t["turn_id"] != exclude_turn]
        users = [t["content"] for t in turns if t["role"] == "user"][-n:]
        assistants = []
        for t in [t for t in turns if t["role"] == "assistant"][-n:]:
            meta = t["intent"]
            # 只取大纲与页码；t["content"] 里的正文不读，理由见 WorkingWindow 的说明
            assistants.append({"intent": meta.get("intent", ""),
                               "outline": meta.get("outline", []),
                               "pages": meta.get("pages", [])})
        return WorkingWindow(slots=self.load_slots(session_id), user_turns=users, assistant_turns=assistants)

    # ── artifacts / runs ────────────────────────────────────
    def save_artifact(self, *, session_id: str, turn_id: str, genre: str, topic: str,
                      body: dict, citations: dict, verify: dict, artifact_id: str | None = None) -> str:
        aid = artifact_id or new_id("a")
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
                      (aid, session_id, turn_id, genre, topic,
                       json.dumps(body, ensure_ascii=False, default=str),
                       json.dumps(citations, ensure_ascii=False, default=str),
                       json.dumps(verify, ensure_ascii=False, default=str)))
        return aid

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if r is None:
            return None
        return {"artifact_id": r["artifact_id"], "session_id": r["session_id"], "turn_id": r["turn_id"],
                "genre": r["genre"], "topic": r["topic"], "body": json.loads(r["body_json"]),
                "citations": json.loads(r["citations_json"]), "verify": json.loads(r["verify_json"])}

    def save_run(self, run_id: str, session_id: str, turn_id: str, events_jsonl: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?)", (run_id, session_id, turn_id, events_jsonl))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(r) if r else None

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            s = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if s is None:
                return None
            arts = c.execute("SELECT artifact_id, turn_id, genre, topic FROM artifacts WHERE session_id=?",
                             (session_id,)).fetchall()
            runs = c.execute("SELECT run_id, turn_id FROM runs WHERE session_id=?", (session_id,)).fetchall()
        return {
            "session_id": session_id,
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "slots": json.loads(s["slots_json"] or "{}"),
            "turns": [{k: t[k] for k in ("turn_id", "seq", "role", "content", "intent", "created_at")}
                      for t in self.turns(session_id)],
            "artifacts": [dict(a) for a in arts],
            "runs": [dict(r) for r in runs],
        }
