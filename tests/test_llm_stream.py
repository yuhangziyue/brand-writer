"""LLMClient.stream() 的离线测试：全部走 httpx.MockTransport，不碰网络、不需要 key。"""

from __future__ import annotations

import json

import httpx
import pytest

from ragwriter.llm import client as client_mod
from ragwriter.llm.client import LLMClient, LLMConfig, LLMError, RunLog


def _cfg() -> LLMConfig:
    return LLMConfig(base_url="http://relay.test/v1", api_key="test-key", model="m",
                     judge_model="m", vlm_model="m", max_retries=2)


def _client(handler) -> LLMClient:
    c = LLMClient(_cfg(), RunLog())
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def _sse(*objs: dict | str) -> bytes:
    lines = []
    for o in objs:
        lines.append("data: " + (o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)))
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def _delta(text: str, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                                  "completion_tokens_details": {"reasoning_tokens": 0}}}
SSE_HEADERS = {"content-type": "text/event-stream; charset=utf-8"}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # 重试退避会真睡 1.5s/3s，测试里跳过
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    monkeypatch.delenv("LLM_STREAM", raising=False)


def test_native_sse_multi_chunk_with_usage():
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        body = _sse({"choices": [{"delta": {"role": "assistant"}}]},
                    _delta("RAG "), _delta("是检索"), _delta("增强生成。", "stop"),
                    USAGE, "[DONE]")
        return httpx.Response(200, headers=SSE_HEADERS, content=body)

    c = _client(handler)
    parts = list(c.stream("sys", "user", tag="t1"))
    assert parts == ["RAG ", "是检索", "增强生成。"]
    assert c.last_stream_mode == "native"
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert seen[0]["temperature"] == 0.2 and "reasoning_effort" in seen[0]
    rec = c.runlog.records[-1]
    assert rec["stream_mode"] == "native" and rec["chunks"] == 3
    assert rec["prompt_tokens"] == 11 and rec["completion_tokens"] == 7
    assert rec["finish_reason"] == "stop" and rec["tag"] == "t1"
    assert isinstance(rec["ttft_ms"], int)


def test_retry_on_500_before_first_token():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, headers=SSE_HEADERS,
                              content=_sse(_delta("好。"), USAGE, "[DONE]"))

    c = _client(handler)
    assert "".join(c.stream("s", "u")) == "好。"
    assert calls["n"] == 2
    assert c.runlog.records[-1]["attempt"] == 1


def test_error_after_first_token_raises_without_retry():
    calls = {"n": 0}

    def broken():
        yield _sse(_delta("已经吐出的半句"))
        raise httpx.ReadError("connection reset")

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, headers=SSE_HEADERS, content=broken())

    c = _client(handler)
    got: list[str] = []
    with pytest.raises(LLMError, match="不重试"):
        for p in c.stream("s", "u"):
            got.append(p)
    assert got == ["已经吐出的半句"]
    assert calls["n"] == 1  # 不能重来，否则前端会收到重复文本


def test_non_event_stream_falls_back_to_simulated():
    calls = {"n": 0}
    full = "第一句。第二句！第三句？"

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={
            "choices": [{"message": {"content": full}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9},
        })

    c = _client(handler)
    parts = list(c.stream("s", "u"))
    assert parts == ["第一句。", "第二句！", "第三句？"]
    assert c.last_stream_mode == "simulated"
    assert calls["n"] == 1  # 就地解析整段 JSON，不重复请求
    assert c.runlog.records[-1]["stream_mode"] == "simulated"


def test_env_stream_off_uses_chat(monkeypatch):
    monkeypatch.setenv("LLM_STREAM", "0")
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "甲。乙。\n丙"}, "finish_reason": "stop"}],
            "usage": {},
        })

    c = _client(handler)
    parts = list(c.stream("s", "u"))
    assert parts == ["甲。", "乙。", "\n", "丙"]
    assert "".join(parts) == "甲。乙。\n丙"
    assert c.last_stream_mode == "simulated"
    assert "stream" not in seen[0]


def test_empty_stream_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=SSE_HEADERS,
                              content=_sse(_delta("", "length"), USAGE, "[DONE]"))

    c = _client(handler)
    with pytest.raises(LLMError, match="空内容"):
        list(c.stream("s", "u"))
