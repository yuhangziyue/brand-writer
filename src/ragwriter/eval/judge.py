"""评估用的模型评审（辅）。

为什么只做两件事（支持度、要点覆盖）外加一个扩写质量打分：
凡是规则写得出来的判定都在 metrics.py 里程序判定了。模型评审有 generator-verifier gap ——
评审模型和生成模型能力相当时，生成者犯的错评审者多半也看不出来。所以这里：
  1. 评审模型尽量和生成模型不同：EVAL_JUDGE_MODEL > LLM_JUDGE_MODEL > LLM_MODEL；
     与生成模型相同时 `same_model=True`，报告里标「同模型评审，可信度降级」。
  2. 每条判定必须给依据原文，并且**程序回查依据是否真的逐字出现在原文里**。
     查不到 = 评审自己在编，判定作废（evidence_ok=False），不计分、单独计数。
  3. 结构化输出走 json_schema strict，判定值只能是枚举里的几档，不给自由打分的空间。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ragwriter.eval.metrics import norm, strip_cites

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "eval_judge.md"

SUPPORT_LEVELS = ("支持", "部分支持", "不支持")
COVER_LEVELS = ("覆盖", "部分覆盖", "未覆盖")
_LEVEL_SCORE = {"支持": 1.0, "部分支持": 0.5, "不支持": 0.0, "覆盖": 1.0, "部分覆盖": 0.5, "未覆盖": 0.0}

_EVIDENCE_ITEM = {
    "type": "object",
    "properties": {"sid": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["sid", "quote"],
    "additionalProperties": False,
}
SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(SUPPORT_LEVELS)},
        "evidence": {"type": "array", "items": _EVIDENCE_ITEM},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "evidence", "reason"],
    "additionalProperties": False,
}
COVERS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(COVER_LEVELS)},
        "evidence": {"type": "array", "items": _EVIDENCE_ITEM},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "evidence", "reason"],
    "additionalProperties": False,
}
EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"pages": {"type": "array", "items": {"type": "integer"}},
                               "quote": {"type": "string"}},
                "required": ["pages", "quote"],
                "additionalProperties": False,
            },
        },
        "new_facts": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["score", "links", "new_facts", "reason"],
    "additionalProperties": False,
}


class ChatClient(Protocol):
    """LLMClient 的最小子集。测试里用假 client 实现它。"""

    def complete_json(self, system: str, user: str, *, json_schema: dict | None = None,
                      model: str | None = None, max_tokens: int = 4096, tag: str = "") -> dict: ...


class Evidence(BaseModel):
    sid: str
    quote: str
    found: bool = False  # 程序回查：quote 是否逐字出现在对应原文


class JudgeVerdict(BaseModel):
    task: str  # support / covers / expansion
    verdict: str  # 枚举档位；expansion 为 "1".."5"
    score: float | None  # 支持 1 / 部分 0.5 / 不支持 0；expansion 为 score/5；依据不实为 None
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str = ""
    evidence_ok: bool = True  # False = 依据回查失败，本判定作废
    extra: dict[str, Any] = Field(default_factory=dict)
    model: str = ""


def _load_prompts(path: Path = PROMPT_PATH) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"<!--\s*task:\s*(\w+)\s*-->", text)
    out: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        out[parts[i]] = parts[i + 1].strip()
    return out


def _quote_in(quote: str, source: str) -> bool:
    q = norm(strip_cites(quote)).strip("…。.，,\"“”'")
    return bool(q) and q in norm(strip_cites(source))


def resolve_judge_model(env: dict[str, str] | None = None) -> tuple[str, str, bool]:
    """返回 (评审模型, 生成模型, 是否同模型)。

    优先 EVAL_JUDGE_MODEL：离线评估专用，和线上 V5 的 LLM_JUDGE_MODEL 分开配，
    才能做到「生成用 A、线上自检用 A'、离线打分用 B」。
    """
    env = dict(os.environ) if env is None else env
    gen = (env.get("LLM_MODEL") or "deepseek-chat").strip()
    judge = (env.get("EVAL_JUDGE_MODEL") or env.get("LLM_JUDGE_MODEL") or gen).strip()
    return judge, gen, judge == gen


class Judge:
    def __init__(self, client: ChatClient, *, model: str, generator_model: str = "",
                 prompt_path: Path = PROMPT_PATH) -> None:
        self.client = client
        self.model = model
        self.generator_model = generator_model
        self.same_model = bool(generator_model) and model == generator_model
        self._prompts = _load_prompts(prompt_path)
        self.prompt_hash = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()[:12]
        self._cache: dict[str, dict] = {}
        self.calls = 0
        self.invalid = 0  # 依据回查失败的次数，进报告

    @classmethod
    def from_env(cls, client: ChatClient | None = None) -> "Judge":
        """用 LLMClient 建评审。没 key 时抛 LLMNotConfigured，调用方据此退到 judge=None。"""
        model, gen, _ = resolve_judge_model()
        if client is None:
            from ragwriter.llm.client import LLMClient, LLMConfig

            client = LLMClient(LLMConfig.from_env())
        return cls(client, model=model, generator_model=gen)

    # ── 对外 ────────────────────────────────────────────────
    def support(self, sentence: str, snippets: list[dict]) -> JudgeVerdict:
        """snippets: [{sid, page_no, text}]，只放本句所引的片段。"""
        body = "## 片段\n" + "\n".join(
            f"[{s.get('sid')} | 第{s.get('page_no')}页] {s.get('text', '')}" for s in snippets
        ) + f"\n\n## 待判句\n{sentence}"
        raw = self._call("support", body, SUPPORT_SCHEMA)
        by_sid = {str(s.get("sid")): str(s.get("text", "")) for s in snippets}
        ev = [Evidence(sid=str(e.get("sid", "")), quote=str(e.get("quote", ""))) for e in raw.get("evidence") or []]
        for e in ev:
            e.found = e.sid in by_sid and _quote_in(e.quote, by_sid[e.sid])
        verdict = raw.get("verdict", "")
        # 判「支持 / 部分支持」却拿不出能回查的依据 ⇒ 判定作废。判「不支持」可以没有依据
        ok = verdict in SUPPORT_LEVELS and (verdict == "不支持" or (bool(ev) and all(e.found for e in ev)))
        return self._finish("support", verdict, ev, raw, ok)

    def covers(self, output: str, key_point: dict | str) -> JudgeVerdict:
        kp = key_point if isinstance(key_point, dict) else {"text": key_point}
        body = (f"## 要点\n{kp.get('text', '')}" + (f"（品宣第 {kp['page']} 页）" if kp.get("page") else "")
                + f"\n\n## 输出\n{output}")
        raw = self._call("covers", body, COVERS_SCHEMA)
        ev = [Evidence(sid="OUT", quote=str(e.get("quote", ""))) for e in raw.get("evidence") or []]
        for e in ev:
            e.found = _quote_in(e.quote, output)
        verdict = raw.get("verdict", "")
        ok = verdict in COVER_LEVELS and (verdict == "未覆盖" or (bool(ev) and all(e.found for e in ev)))
        return self._finish("covers", verdict, ev, raw, ok)

    def expansion(self, output: str, snippets: list[dict]) -> JudgeVerdict:
        """扩写质量（辅）。links 的衔接句与 new_facts 都必须能在输出里逐字找到。"""
        body = "## 片段\n" + "\n".join(
            f"[{s.get('sid')} | 第{s.get('page_no')}页] {s.get('text', '')}" for s in snippets
        ) + f"\n\n## 输出\n{output}"
        raw = self._call("expansion", body, EXPANSION_SCHEMA)
        links = raw.get("links") or []
        ev = [Evidence(sid="OUT", quote=str(l.get("quote", "")), found=_quote_in(str(l.get("quote", "")), output))
              for l in links]
        new_facts = [str(x) for x in raw.get("new_facts") or []]
        nf_ok = all(_quote_in(x, output) for x in new_facts)
        score = raw.get("score")
        ok = isinstance(score, int) and 1 <= score <= 5 and all(e.found for e in ev) and nf_ok
        if ok and new_facts and score > 2:
            score = 2  # 提示词约定：出现新事实不得高于 2，评审没守住就程序兜底
        v = self._finish("expansion", str(score), ev, raw, ok)
        v.extra = {"pages_linked": [l.get("pages") for l in links], "new_facts": new_facts}
        if v.evidence_ok:
            v.score = score / 5
        return v

    # ── 内部 ────────────────────────────────────────────────
    def _call(self, task: str, body: str, schema: dict) -> dict:
        system = self._prompts.get("common", "") + "\n\n" + self._prompts[task]
        key = hashlib.sha256(f"{self.model}|{task}|{body}".encode()).hexdigest()
        if key in self._cache:  # 同一句在多轮里重复出现（revise 保留的句子）不重复花钱
            return self._cache[key]
        self.calls += 1
        raw = self.client.complete_json(system, body, json_schema=schema, model=self.model,
                                        max_tokens=800, tag=f"eval_judge.{task}")
        # 评审返回空或非 JSON：当作无效判定（后续 ok=False 计入 invalid），不让一次坏响应中断整轮评估
        try:
            raw = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            raw = {}
        self._cache[key] = raw
        return raw

    def _finish(self, task: str, verdict: str, ev: list[Evidence], raw: dict, ok: bool) -> JudgeVerdict:
        if not ok:
            self.invalid += 1
        return JudgeVerdict(
            task=task, verdict=verdict, score=_LEVEL_SCORE.get(verdict) if ok else None,
            evidence=ev, reason=str(raw.get("reason", "")), evidence_ok=ok,
            extra={"_raw": raw}, model=self.model,
        )
