"""生成示例输出（交付物 D7）：四种体裁各走一遍完整流程「请求 → 主题确认 → 确认 → 写作」。

每篇输出到 samples/<name>.md：确认卡、正文、引用依据表（编号 → 页码 → 原文片段）、校验摘要、资料边界、润色日志。
用法（仓库根目录）：USE_DENSE=1 .venv/bin/python scripts/make_samples.py [out_dir] [名称 ...]
"""

from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
from ragwriter.agent import Orchestrator  # noqa: E402
from ragwriter.agent.orchestrator import Deps  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
ONLY = set(sys.argv[2:])
CASES: dict[str, tuple[str, dict | None]] = {
    "blog_geo": ("写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？",
                 {"genre": "blog", "topic": "为什么中国出海品牌需要进行 GEO 优化？"}),
    "faq_compliance": ("出一版 FAQ，讲合规 GEO", None),
    "brand_intro_a": ("写一篇万悉科技的品牌介绍", None),
    "brand_intro_b": ("写一篇万悉科技的品牌介绍", None),
    "product_intro_monitor": ("写个产品介绍，讲讲怎么监控品牌在 AI 里的表现", None),
    "product_intro_distill": ("写一篇产品介绍，介绍品牌价值蒸馏模型", None),
}


def _table(citations: dict) -> list[str]:
    rows = ["| 编号 | 页码 | 状态 | 原文片段 |", "|---|---|---|---|"]
    for sid in sorted(citations, key=lambda x: int(x[1:])):
        c = citations[sid]
        text = str(c.get("text", "")).replace("|", "｜").replace("\n", " ")
        rows.append(f"| {sid} | 第 {c.get('page_no')} 页 | {c.get('status', '')} | {text[:120]} |")
    return rows


def run(name: str) -> str:
    deps = Deps.from_env()
    msg, opts = CASES[name]
    orch = Orchestrator(deps, sessions_db=Path(tempfile.mkdtemp()) / "s.db")
    first = list(orch.handle(msg, options=opts))
    sid = next(e.data["session_id"] for e in first if e.type == "session")
    conf = next((e.data for e in first if e.type == "confirm"), {})
    second = list(orch.handle("确认", session_id=sid, options={"confirm": "accept"}))
    done = next(e.data for e in second if e.type == "done")
    plan = next((e.data for e in second if e.type == "plan"), {})
    rep = done.get("report") or {}
    lines = [
        f"# 示例：{name}", "",
        f"> 请求：「{msg}」" + ("（写作台指定体裁）" if opts else "（对话识别体裁）"),
        f"> 模型：{deps.client.cfg.model if deps.client else '-'} · 耗时 {done.get('elapsed_ms', 0) / 1000:.0f}s"
        f" · 调用 {(done.get('usage') or {}).get('calls', '-')} 次 · 流式 {done.get('stream_mode')}", "",
        "## ① 主题审核与确认", "",
        f"- 有效：{conf.get('valid')} · 体裁：{conf.get('genre')} · 精确主题：{conf.get('topic')}",
        f"- 切入点 / 中心观点 / 选品：{conf.get('angle')}",
        "- 要点：" + "；".join(conf.get("outline_points") or []),
        "- 资料里没有、文中不写：" + ("；".join(conf.get("premise_gaps") or []) or "无"),
        f"- 确认话术：{conf.get('confirm_text') or done.get('final_text', '')[:0]}", "",
        "## ② 规划", "",
        f"- 中心论点：{plan.get('thesis') or '（本体裁无）'} · 选品：{plan.get('products') or '（无）'}",
        "- 大纲：" + " / ".join(plan.get("outline") or []), "",
        "## ③ 正文", "", done.get("final_text") or "", "",
        "## ④ 引用依据", "", *_table(done.get("citations") or {}), "",
        "## ⑤ 校验摘要", "",
        f"- 句子 {rep.get('sentence_total')} · 事实句 {rep.get('factual_sentences')} · 带引用 {rep.get('cited_sentences')}"
        f" · 数字不一致 {rep.get('numbers_inconsistent')}",
        f"- 各闸拦截：{json.dumps(rep.get('gate_blocks'), ensure_ascii=False)}"
        f" · 改写 {len(rep.get('rewrites') or [])} · 删除 {len(rep.get('deletions') or [])}",
        "- 资料边界：" + ("；".join(b.get("topic", "") for b in done.get("boundary") or []) or "无"),
        "- 润色：" + "；".join(f"{x['block']} {'采纳' if x['accepted'] else '保留原文（' + x['reason'] + '）'}"
                             for x in done.get("polish") or []),
    ]
    if done.get("jsonld"):
        lines += ["", "## ⑥ FAQPage JSON-LD", "", "```json", json.dumps(done["jsonld"], ensure_ascii=False, indent=2), "```"]
    if rep.get("deletions"):
        lines += ["", "### 被删除的句子（校验未过）", ""]
        lines += [f"- {d.get('original')}　—— {d.get('reason')}" for d in rep["deletions"]]
    errs = [e.data for e in second if e.type == "error"]
    if errs:
        lines += ["", f"> ⚠️ 运行中的非致命错误：{errs}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        for n in pool.map(run, [c for c in CASES if not ONLY or c in ONLY]):
            print("done", n, flush=True)
