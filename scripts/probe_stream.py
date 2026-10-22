"""流式探针：实测中转是「真流式」还是「攒完一次性发出」。

10 号文档 6.3「先探针，再依赖」：上次中转声明支持图片却静默丢图，
流式能力同样要实测，判定标准是首 token 明显早于整体完成。

用法：.venv/bin/python scripts/probe_stream.py
只打印 base_url 的 host 与 model 名，不打印 key。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragwriter.llm.client import LLMClient, LLMConfig, LLMError, RunLog  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = LLMConfig.from_env()
    client = LLMClient(cfg, RunLog())
    host = httpx.URL(cfg.base_url).host

    t0 = time.perf_counter()
    ttft: float | None = None
    chunks = 0
    text = ""
    err: str | None = None
    try:
        for piece in client.stream(
            "你是技术写作助手。",
            "用 6 句话介绍 RAG（检索增强生成），每句一行，总共约 300 字。",
            max_tokens=1024,
            tag="probe_stream",
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
            text += piece
    except LLMError as exc:
        err = str(exc)
    total = time.perf_counter() - t0
    client.close()

    mode = client.last_stream_mode
    rec = client.runlog.records[-1] if client.runlog.records else {}
    # simulated 模式下 yield 的是切片，块数和首 token 时间都没有意义，直接判攒包
    is_sse = mode == "native" and rec.get("stream_mode") == "native"
    real = is_sse and chunks >= 5 and ttft is not None and ttft < total * 0.5
    if err:
        verdict = f"结论：探针失败 —— {err}"
    elif real:
        verdict = "结论：真流式（首 token 明显早于整体完成，stream() 可直接依赖）"
    else:
        verdict = "结论：中转在攒包（stream() 应退化为 simulated，done 事件标注 stream=simulated）"

    print(verdict)
    print(f"host={host}  model={cfg.model}")
    print(f"event-stream={is_sse}  last_stream_mode={mode}")
    ttft_s = f"{ttft * 1000:.0f}ms" if ttft is not None else "n/a"
    print(f"首token={ttft_s}  总耗时={total * 1000:.0f}ms  块数={chunks}  "
          f"首token/总耗时={(ttft / total if ttft else float('nan')):.2f}")
    print(f"usage: prompt={rec.get('prompt_tokens')} completion={rec.get('completion_tokens')} "
          f"finish={rec.get('finish_reason')}  输出字数={len(text)}")
    print("── 输出 ──")
    print(text)
    return 0 if not err else 1


if __name__ == "__main__":
    raise SystemExit(main())
