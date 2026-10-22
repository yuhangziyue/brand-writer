"""脚本化的假编排器：按 tests/dialogue/scripts/<用例id>.yaml 逐轮吐出事件流。

用途：在 E 组的 Orchestrator 落地前，验证 dialogue_runner 与 dialogue_checks 本身是对的。
它不做意图识别、不检索、不调模型——每轮该吐什么由脚本写死，脚本里的引用一律用 build/facts.jsonl 的真实 fact_id。
E 组的 ragwriter.agent.testing（FakeStore / FakeClient / FakeSearch）可用后，
test_dialogues.py 会改为「真 Orchestrator + 假依赖」跑同一套用例，这里只保留负例需要的部分。

脚本每轮字段（都可省）：
  intent: {intent, genre, topic, query, slots, inherited, source}
  hits: [fact_id, ...]                 # 本轮 retrieval 事件命中
  final_text: str
  cite: {S1: fact_id, ...}             # done.citations，page_no/text/status 从 facts.jsonl 补
  refused / refusal_kind / nearest_pages / suggestions / boundary / jsonld / insufficient
  verify: [{check, result, reason, sentence}]   # 额外的 verify 事件（如 V3 fail）
  patch:  [{sentence, action, text}]
  events: [type, ...]                  # 显式指定事件顺序；省略则按意图给默认顺序
  session_id: str                      # 强行指定本轮 session（负例用）
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from ragwriter.contracts import Event

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent / "scripts"
FACTS = ROOT / "build" / "facts.jsonl"

_FACTS: dict[str, dict] | None = None


def facts() -> dict[str, dict]:
    global _FACTS
    if _FACTS is None:
        _FACTS = {}
        if FACTS.exists():
            for line in FACTS.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    f = json.loads(line)
                    _FACTS[f["fact_id"]] = f
    return _FACTS


def page_of(fid: str) -> int:
    return int(fid[1:].split("-", 1)[0])


def load_script(case_id: str) -> dict:
    return yaml.safe_load((SCRIPTS / f"{case_id}.yaml").read_text(encoding="utf-8"))


DEFAULT_EVENTS = {
    "write": ["session", "intent", "plan", "retrieval", "delta", "verify", "section_final", "done"],
    "qa": ["session", "intent", "retrieval", "delta", "verify", "section_final", "done"],
    "qa_refused": ["session", "intent", "retrieval", "insufficient", "done"],
    "out_of_scope": ["session", "intent", "retrieval", "insufficient", "done"],
    "clarify": ["session", "intent", "done"],
    "revise": ["session", "intent", "done"],
    "explain": ["session", "intent", "done"],
}


class ScriptedOrchestrator:
    """鸭子类型同 Orchestrator.handle。一个实例 = 一份脚本；每次 handle 消费一轮。"""

    _sid_counter = itertools.count(1)

    def __init__(self, script: dict):
        self.turns: list[dict] = copy.deepcopy(script.get("turns", []))
        self.i = 0
        self.sessions: set[str] = set()
        self.received: list[dict] = []  # 记录收到的 (message, session_id, options)，供断言透传

    def handle(self, message: str, session_id: str | None = None,
               options: dict | None = None) -> Iterator[Event]:
        self.received.append({"message": message, "session_id": session_id, "options": options})
        if self.i >= len(self.turns):
            raise RuntimeError(f"脚本只有 {len(self.turns)} 轮，收到第 {self.i + 1} 轮：{message}")
        t = self.turns[self.i]
        self.i += 1
        sid = t.get("session_id") or session_id or f"s_fake{next(self._sid_counter):03d}"
        self.sessions.add(sid)
        intent = dict(t.get("intent") or {})
        kind = intent.get("intent", "qa")
        refused = bool(t.get("refused", kind in ("out_of_scope", "revise", "explain")))
        order = t.get("events") or DEFAULT_EVENTS["qa_refused" if (kind == "qa" and refused) else kind]
        fdb = facts()
        hits = t.get("hits") or [*(t.get("cite") or {}).values()]

        def citations() -> dict[str, dict]:
            out = {}
            for s, fid in (t.get("cite") or {}).items():
                f = fdb.get(fid, {})
                out[s] = {"fact_id": fid, "page_no": f.get("page_no", page_of(fid)),
                          "text": f.get("text", ""), "status": f.get("status", "verified")}
            return out

        for typ in order:
            if typ == "session":
                yield Event(type="session", data={"session_id": sid, "turn_id": f"{sid}-t{self.i}"})
            elif typ == "intent":
                yield Event(type="intent", data={
                    "intent": kind, "genre": intent.get("genre"), "topic": intent.get("topic"),
                    "query": intent.get("query"), "slots": intent.get("slots") or {},
                    "inherited": intent.get("inherited") or {}, "source": intent.get("source", "rule")})
            elif typ == "plan":
                yield Event(type="plan", data={"title": intent.get("topic"), "sections": [], "insufficient": []})
            elif typ == "retrieval":
                yield Event(type="retrieval", data={
                    "query": intent.get("query") or message,
                    "hits": [{"fact_id": h, "page_no": fdb.get(h, {}).get("page_no", page_of(h)),
                              "bm25_rank": k + 1, "dense_rank": None} for k, h in enumerate(hits)]})
            elif typ == "delta":
                yield Event(type="delta", data={"section": 0, "text": t.get("draft_text", t.get("final_text", ""))})
            elif typ == "verify":
                extra = t.get("verify") or []
                for v in extra:
                    yield Event(type="verify", data={"section": 0, **v})
                if not extra:
                    yield Event(type="verify", data={"section": 0, "sentence": 0, "check": "V3", "result": "pass"})
            elif typ == "patch":
                for p in t.get("patch") or []:
                    yield Event(type="patch", data={"section": 0, **p})
            elif typ == "section_final":
                yield Event(type="section_final", data={"section": 0, "citations": citations()})
            elif typ == "insufficient":
                for ins in t.get("insufficient") or [{"topic": message, "signal": "top_score<tau"}]:
                    yield Event(type="insufficient", data=ins)
            elif typ == "error":
                yield Event(type="error", data=t.get("error") or {"stage": "fake", "reason": "scripted"})
            elif typ == "done":
                data: dict[str, Any] = {
                    "artifact_id": f"a_{sid}_{self.i}" if kind == "write" else None,
                    "final_text": t.get("final_text", ""),
                    "citations": citations(),
                    "report": {}, "stream_mode": "real", "usage": {},
                    "refused": refused,
                }
                for k in ("refusal_kind", "nearest_pages", "suggestions", "boundary", "jsonld"):
                    if k in t:
                        data[k] = t[k]
                yield Event(type="done", data=data)
