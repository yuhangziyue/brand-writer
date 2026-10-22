"""离线回放：把真实运行的事件流存成 fixture，无 API key 时按原节奏重放（10 号文档 A6）。

每次运行编排器都已把完整事件流写进 sessions.db 的 runs 表（带毫秒时间戳）；
这里负责导出成 `build/fixtures/<name>.jsonl`，以及在回放模式下按消息挑 fixture 重放。

🔴 只放录过的主题（评审 M4）：收到没录过的消息时返回 `replay_unmatched` 错误并列出录制主题，
**不拿「最接近的 fixture」冒充**——评审输入 A 却看到 B 的答案，会以为系统在造假。

fixture 格式：首行 `{"_meta": {message, topic, genre, intent, model, run_id}}`，
其后每行 `{"t": 毫秒偏移, "type": 事件类型, "data": {...}}`。

用法：
  python -m ragwriter.agent.replay list
  python -m ragwriter.agent.replay export --run r_xxx --name blog_geo
  python -m ragwriter.agent.replay record --message "写一篇官网 Blog：……" --name blog_geo
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ragwriter.agent.memory import SessionStore
from ragwriter.contracts import Event

DEFAULT_FIXTURES = Path("build/fixtures")
DEFAULT_SPEED = 4.0  # 默认四倍速：原速一篇要一两分钟，演示时太拖
MAX_GAP_MS = 1500  # 单个间隔上限（加速前）：模型偶发的长停顿在回放里没有信息量
MATCH_THRESHOLD = 0.75

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _norm(text: str | None) -> str:
    return _PUNCT.sub("", (text or "").lower())


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def similarity(a: str, b: str) -> float:
    """字符二元组 Jaccard。只用来容忍标点、空格与个别字的差异，不做语义匹配。"""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    x, y = _bigrams(na), _bigrams(nb)
    return len(x & y) / len(x | y)


@dataclass
class Fixture:
    name: str
    path: Path
    meta: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def message(self) -> str:
        return str(self.meta.get("message") or "")


def read_fixture(path: Path) -> Fixture:
    meta: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "_meta" in rec:
            meta = rec["_meta"]
        else:
            events.append(rec)
    return Fixture(name=path.stem, path=path, meta=meta, events=events)


class FixtureLibrary:
    def __init__(self, fixtures_dir: Path | str = DEFAULT_FIXTURES) -> None:
        self.dir = Path(fixtures_dir)

    def all(self) -> list[Fixture]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("*.jsonl")):
            try:
                out.append(read_fixture(p))
            except (OSError, ValueError):
                continue
        return out

    def recorded_topics(self) -> list[str]:
        """返回可以原样输入的录制消息（前端提示「回放模式下可以试这些」）。"""
        return list(dict.fromkeys(f.message for f in self.all() if f.message))

    def models(self) -> list[str]:
        return list(dict.fromkeys(str(f.meta["model"]) for f in self.all() if f.meta.get("model")))

    def match(self, message: str, *, threshold: float = MATCH_THRESHOLD) -> Fixture | None:
        """严格匹配：消息与录制消息几乎一致，或录制的写作主题完整出现在消息里。"""
        best, best_score = None, 0.0
        nm = _norm(message)
        for f in self.all():
            score = similarity(message, f.message)
            topic = _norm(f.meta.get("topic"))
            if topic and len(topic) >= 4 and topic in nm and f.meta.get("intent") == "write":
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = f, score
        return best if best_score >= threshold else None


def play(fixture: Fixture, *, speed: float = DEFAULT_SPEED, session_id: str | None = None,
         sleep: Callable[[float], None] = time.sleep) -> Iterator[Event]:
    """按原时间间隔重放；speed<=0 表示不等待（测试用）。session_id 给定时替换录制时的会话号。"""
    prev = 0
    for rec in fixture.events:
        t = int(rec.get("t", prev))
        gap = min(max(t - prev, 0), MAX_GAP_MS)
        prev = t
        if speed > 0 and gap:
            sleep(gap / 1000 / speed)
        data = dict(rec.get("data") or {})
        if rec.get("type") in ("session", "done"):
            data["replay"] = True
            data["fixture"] = fixture.name
            if session_id:
                data["session_id"] = session_id
        yield Event(type=rec["type"], data=data)


def replay_stream(library: FixtureLibrary, message: str, *, speed: float = DEFAULT_SPEED,
                  session_id: str | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> Iterator[Event]:
    fx = library.match(message)
    if fx is not None:
        yield from play(fx, speed=speed, session_id=session_id, sleep=sleep)
        return
    topics = library.recorded_topics()
    yield Event(type="session", data={"session_id": session_id, "turn_id": None, "replay": True, "mode": "replay"})
    yield Event(type="error", data={"stage": "replay", "reason": "replay_unmatched", "recorded_topics": topics,
                                    "fallback": False})
    # 仍发一个 done 收口：前端据此结束加载态；refused + error 明示这不是一次真实回答
    yield Event(type="done", data={
        "replay": True, "error": True, "refused": True, "refusal_kind": "replay_unmatched",
        "artifact_id": None, "final_text": "", "citations": {}, "report": {}, "stream_mode": "replay",
        "usage": {}, "nearest_pages": [], "suggestions": topics, "boundary": [], "recorded_topics": topics,
    })


def export_run(sessions_db: Path | str, run_id: str, name: str,
               fixtures_dir: Path | str = DEFAULT_FIXTURES) -> Path:
    run = SessionStore(sessions_db).get_run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} 不存在")
    out = Path(fixtures_dir) / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run["events_jsonl"], encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragwriter.agent.replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    ex = sub.add_parser("export")
    ex.add_argument("--run", required=True)
    ex.add_argument("--name", required=True)
    rc = sub.add_parser("record", help="实跑一条消息并导出为 fixture（需要 LLM_API_KEY）")
    rc.add_argument("--message", required=True)
    rc.add_argument("--name", required=True)
    rc.add_argument("--session")
    for p in (ex, rc):
        p.add_argument("--db", default="build/runtime/sessions.db")
        p.add_argument("--dir", default=str(DEFAULT_FIXTURES))
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for f in FixtureLibrary().all():
            print(f"{f.name}\t{f.meta.get('intent')}\t{f.message}")
        return 0
    if args.cmd == "export":
        print(export_run(args.db, args.run, args.name, args.dir))
        return 0
    from ragwriter.agent.orchestrator import Deps, Orchestrator

    deps = Deps.from_env()
    if not deps.live:
        print("没有 LLM_API_KEY，无法录制")
        return 2
    orch = Orchestrator(deps, sessions_db=args.db)
    run_id = None
    for e in orch.handle(args.message, session_id=args.session):
        if e.type == "done":
            run_id = e.data.get("run_id")
        print(e.type, json.dumps(e.data, ensure_ascii=False)[:160])
    if run_id:
        print(export_run(args.db, run_id, args.name, args.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
