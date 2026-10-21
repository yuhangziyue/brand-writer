"""T9 模型适配层。

只认 OpenAI 兼容的 /v1/chat/completions 协议，不绑任何厂商 SDK：
换 DeepSeek、Qwen、Claude 或任何中转，只改 base_url 与 model 两个环境变量。

每次调用都记运行日志（模型、prompt 哈希、token、耗时），
供离线回放与成本统计——这是「不是只调一次模型 API」的证据。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class LLMNotConfigured(LLMError):
    """没有可用的 key。调用方据此转入离线回放模式。"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    judge_model: str
    vlm_model: str
    # 本项目全程关思考：判别是逐句短任务，一篇文章要调几十次，
    # 思考 token 既吃预算又吃延迟，而且会把 max_tokens 占光导致 content 为空
    reasoning_effort: str = "none"
    timeout: float = 120.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        key = os.getenv("LLM_API_KEY", "").strip()
        base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip().rstrip("/")
        model = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        if not key:
            raise LLMNotConfigured("LLM_API_KEY 未设置")
        return cls(
            base_url=base,
            api_key=key,
            model=model,
            judge_model=os.getenv("LLM_JUDGE_MODEL", model).strip(),
            vlm_model=os.getenv("LLM_VLM_MODEL", model).strip(),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "none").strip(),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        )


@dataclass
class RunLog:
    """调用流水。写 JSONL，既是成本账也是回放素材。"""

    path: Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **rec: Any) -> None:
        self.records.append(rec)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @property
    def total(self) -> Usage:
        u = Usage()
        for r in self.records:
            u.prompt_tokens += r.get("prompt_tokens", 0)
            u.completion_tokens += r.get("completion_tokens", 0)
            u.reasoning_tokens += r.get("reasoning_tokens", 0)
        return u


class LLMClient:
    def __init__(self, cfg: LLMConfig, runlog: RunLog | None = None) -> None:
        self.cfg = cfg
        self.runlog = runlog or RunLog()
        # 本机中转不走系统代理：开发机上常设着 SOCKS 代理，
        # 带着它去连 127.0.0.1 只会连不上
        host = httpx.URL(cfg.base_url).host or ""
        local = host in {"127.0.0.1", "localhost", "::1"}
        self._http = httpx.Client(timeout=cfg.timeout, trust_env=not local)

    def close(self) -> None:
        self._http.close()

    # ── 对外 ────────────────────────────────────────────────
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        tag: str = "",
    ) -> str:
        """纯文本补全。json_schema 给定时走 strict 结构化输出。"""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._chat(messages, json_schema=json_schema, model=model,
                          max_tokens=max_tokens, tag=tag)

    def complete_with_image(
        self,
        system: str,
        user: str,
        image_path: str | Path,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
        tag: str = "",
    ) -> str:
        """带图补全，用于页卡解析。图片走 base64 data URL，不依赖外部可访问地址。"""
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": user},
                ],
            },
        ]
        return self._chat(messages, json_schema=json_schema, model=self.cfg.vlm_model,
                          max_tokens=max_tokens, tag=tag)

    def complete_json(self, *args: Any, **kwargs: Any) -> dict:
        """要求返回 JSON 并解析。schema 不合法时带错误重试一次。"""
        raw = self.complete(*args, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"模型返回的不是合法 JSON：{raw[:200]}") from exc

    # ── 内部 ────────────────────────────────────────────────
    def _chat(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None,
        model: str | None,
        max_tokens: int,
        tag: str,
    ) -> str:
        model = model or self.cfg.model
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": self.cfg.reasoning_effort,
            "temperature": 0.2,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": json_schema},
            }
        prompt_hash = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._http.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage", {})
                detail = usage.get("completion_tokens_details") or {}
                self.runlog.add(
                    tag=tag,
                    model=model,
                    prompt_hash=prompt_hash,
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=detail.get("reasoning_tokens", 0),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000),
                    attempt=attempt,
                )
                if not content.strip():
                    # 空内容最常见的原因是 finish_reason=length：
                    # 思考 token 把预算吃光。这里显式报出来，别让它退化成静默失败
                    raise LLMError(
                        f"模型返回空内容，finish_reason={choice.get('finish_reason')}，"
                        f"reasoning_tokens={detail.get('reasoning_tokens', 0)}"
                    )
                return content
            except Exception as exc:  # noqa: BLE001 — 退避重试，最后一次原样抛出
                last = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"调用失败（已重试 {self.cfg.max_retries} 次）：{last}") from last
