"""可复用的假实现：编排器、多轮对话测试（F 组）、评估冒烟测试共用这一份，别各自再造。

设计要点：
- **FakeClient 是唯一的「模型旋钮」**：按 tag 前缀预置输出。假的生成函数（fake_plan、fake_stream_section…）
  也都经 FakeClient 取输出，tag 与 D 组真实实现一致（planner / section_writer / bookend_lede /
  bookend_ending / answer / rewriter / query_rewrite / intent），所以把真生成函数换进来时预置照样可用。
- **FakeSearch 按查询子串返回预置命中**，未命中默认「资料不足」——拒答路径不用额外配置。
- 假校验是 V1–V4 的简化版，够驱动编排器的改写/删除分支；真正的校验规则测试在 B 组。

最小用法：

    from ragwriter.agent.testing import make_fake_deps, FakeClient
    deps = make_fake_deps()                                   # 演示语料 + 演示输出，A1 能直接跑通
    deps = make_fake_deps(client=FakeClient({"section_writer": ["第一节……[S1]。", "第二节……[S3]。"]}))
    orch = Orchestrator(deps, sessions_db=tmp_path / "s.db")
    events = list(orch.handle("写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？"))
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Iterator

from ragwriter.contracts import (
    USABLE_STATUSES,
    CheckResult,
    ContextPack,
    Hit,
    Plan,
    PlanSection,
    RetrievalResult,
    Sentence,
    SentenceVerdict,
    Snippet,
    TopicCheck,
)
from ragwriter.models import Fact, Metric, Page

# ── 演示语料：第 9、10、11、19 页的几条事实，第 10 页带一对会被挂错的数字 ────────────
DEMO_FACTS: list[Fact] = [
    Fact(fact_id="p09-f01", page_no=9, kind="metric",
         text="全球市场 AI 网站月访问量：ChatGPT 61.66亿", prefix="第9页 · 01 GEO背景介绍 · AI正在重塑搜索",
         metric=Metric(value="61.66亿", label="ChatGPT月访问量", method="llm")),
    Fact(fact_id="p09-f02", page_no=9, kind="point", text="AI搜索正成为新的超级流量入口",
         prefix="第9页 · 01 GEO背景介绍 · AI正在重塑搜索"),
    Fact(fact_id="p10-f01", page_no=10, kind="metric", text="AI驱动流量增长 8倍（Shopify官方数据(2025)）",
         prefix="第10页 · 01 GEO背景介绍 · AI正在重塑购物",
         metric=Metric(value="8倍", label="AI驱动流量增长", source_note="Shopify官方数据(2025)", method="llm",
                       discriminators=["流量"])),
    Fact(fact_id="p10-f02", page_no=10, kind="metric", text="AI驱动订单增长 15倍（Shopify官方数据(2025)）",
         prefix="第10页 · 01 GEO背景介绍 · AI正在重塑购物",
         metric=Metric(value="15倍", label="AI驱动订单增长", source_note="Shopify官方数据(2025)", method="llm",
                       discriminators=["订单"])),
    Fact(fact_id="p11-f01", page_no=11, kind="point", text="平台破 1 亿用户时间：ChatGPT 2 个月",
         prefix="第11页 · 01 GEO背景介绍 · 紧抓AI问答时代流量红利"),
    Fact(fact_id="p19-f01", page_no=19, kind="point", text="LLM-原生GEO 技术体系覆盖记忆、推荐与监控三个环节",
         prefix="第19页 · 02 解决方案 · LLM-原生GEO 技术体系"),
]

DEMO_PROFILE: dict[str, Any] = {
    "entities": {"brand": "万悉科技", "aliases": ["Trendee"]},
    "catalog": [
        {"page_no": 9, "chapter": "01 GEO背景介绍", "title": "AI正在重塑搜索：AI搜索正成为新的超级流量入口"},
        {"page_no": 10, "chapter": "01 GEO背景介绍", "title": "AI正在重塑购物：AI电商迎来爆发式增长"},
        {"page_no": 11, "chapter": "01 GEO背景介绍", "title": "紧抓AI问答时代流量红利"},
        {"page_no": 19, "chapter": "02 解决方案: LLM-原生GEO", "title": "LLM-原生GEO “技术体系”"},
    ],
    "rules": {"corrections": [{"pages": [19], "wrong": "思维连", "right": "思维链"}],
              "forbidden": [], "status_policy": {"usable": list(USABLE_STATUSES)}},
}

DEMO_PLAN: dict[str, Any] = {
    "title": "为什么中国出海品牌需要 GEO",
    "lede_points": ["AI 搜索改变流量入口"],
    "sections": [
        {"heading": "AI 搜索正在成为新的流量入口", "sub_question": "AI 搜索的流量规模", "expected_pages_hint": [9]},
        {"heading": "AI 电商带来的增长", "sub_question": "AI 驱动的电商增长数据", "expected_pages_hint": [10]},
    ],
    "insufficient": [],
    "ending_points": ["尽早布局 GEO"],
}

DEMO_OUTPUTS: dict[str, Any] = {
    "planner": json.dumps(DEMO_PLAN, ensure_ascii=False),
    "section_writer": [
        "据第三方平台统计，ChatGPT 月访问量达 61.66亿[S1]。AI搜索正成为新的超级流量入口[S2]。",
        "据 Shopify 2025 年数据，AI驱动订单增长 15倍[S4]。",
    ],
    "bookend_lede": "AI 搜索正在改变海外买家找品牌的方式。",
    "bookend_ending": "现在就是布局 GEO 的时机。",
    "answer": "据 Shopify 2025 年数据，AI驱动订单增长 15倍[S2]。",
    "rewriter": "",
    "query_rewrite": "AI 驱动流量增长多少",
}

DEMO_RULES: dict[str, Any] = {
    "流量规模": ["p09-f01", "p09-f02"],
    "电商增长": ["p10-f01", "p10-f02"],
    "订单": ["p10-f01", "p10-f02"],
    "流量": ["p10-f01", "p10-f02", "p09-f01"],
    "第10页": ["p10-f01", "p10-f02"],
    "第 10 页": ["p10-f01", "p10-f02"],
}


# ── 存储 ──────────────────────────────────────────────────
class FakeStore:
    """RagStore 的最小替身：page_facts / page_metrics / profile / page / pages / meta。"""

    def __init__(self, facts: Iterable[Fact] | None = None, profile: dict | None = None,
                 pages: Iterable[Page] | None = None, *, dense: bool = True) -> None:
        self._facts = {f.fact_id: f for f in (facts if facts is not None else DEMO_FACTS)}
        self._profile = profile if profile is not None else DEMO_PROFILE
        if pages is None:
            pages = [Page(page_no=r["page_no"], chapter=r["chapter"], title=r["title"],
                          text="\n".join(f.text for f in self._facts.values() if f.page_no == r["page_no"]))
                     for r in self._profile.get("catalog", [])]
        self._pages = {p.page_no: p for p in pages}
        self._dense = dense

    def fact(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def page_facts(self, page_no: int, statuses=USABLE_STATUSES) -> list[Fact]:
        return [f for f in self._facts.values() if f.page_no == page_no and f.status in statuses]

    def page_metrics(self, page_no: int) -> list[Metric]:
        return [f.metric for f in self._facts.values() if f.page_no == page_no and f.metric is not None]

    def profile(self) -> dict:
        return self._profile

    def page(self, page_no: int) -> Page:
        if page_no not in self._pages:
            raise KeyError(page_no)
        return self._pages[page_no]

    def pages(self) -> list[Page]:
        return list(self._pages.values())

    def meta(self) -> dict:
        return {"dense": self._dense, "embed_model": "fake"}


# ── 模型 ──────────────────────────────────────────────────
class FakeRunLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, **rec: Any) -> None:
        self.records.append(rec)


class FakeClient:
    """按「tag 前缀 → 预置输出」应答。

    预置值可以是：
      str        每次都返回它
      list[str]  依次返回，用完后重复最后一个（多节写作、多轮改写常用）
      dict       complete_json 直接返回；complete 返回其 JSON
      callable   fn(system, user, tag) -> str | dict，需要按输入分支时用
      Exception  调用时抛出（测失败路径）
    多个前缀都匹配时取最长的。没有预置的 tag 抛 KeyError，免得测试悄悄走了默认值。
    """

    def __init__(self, outputs: dict[str, Any] | None = None, *, stream_mode: str = "native",
                 chunk: int = 8, model: str = "fake-model") -> None:
        self.outputs: dict[str, Any] = dict(DEMO_OUTPUTS if outputs is None else outputs)
        self._cursor: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.runlog = FakeRunLog()
        self.stream_mode = stream_mode
        self.last_stream_mode = stream_mode
        self.chunk = chunk
        self.cfg = type("Cfg", (), {"model": model})()

    def set(self, tag_prefix: str, output: Any) -> "FakeClient":
        self.outputs[tag_prefix] = output
        self._cursor.pop(tag_prefix, None)
        return self

    def tags(self) -> list[str]:
        return [c["tag"] for c in self.calls]

    def _lookup(self, system: str, user: str, tag: str) -> Any:
        keys = [k for k in self.outputs if tag.startswith(k)]
        if not keys:
            raise KeyError(f"FakeClient 没有为 tag={tag!r} 预置输出")
        key = max(keys, key=len)
        out = self.outputs[key]
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, list):
            i = self._cursor.get(key, 0)
            self._cursor[key] = i + 1
            out = out[min(i, len(out) - 1)] if out else ""
        if callable(out):
            out = out(system, user, tag)
        return out

    def _log(self, kind: str, system: str, user: str, tag: str, out: str) -> None:
        self.calls.append({"kind": kind, "tag": tag, "system": system, "user": user, "output": out})
        self.runlog.add(tag=tag, model="fake", prompt_tokens=len(system) + len(user), completion_tokens=len(out))

    def complete(self, system: str, user: str, *, json_schema: dict | None = None, model: str | None = None,
                 max_tokens: int = 4096, tag: str = "") -> str:
        out = self._lookup(system, user, tag)
        text = json.dumps(out, ensure_ascii=False) if isinstance(out, dict) else str(out)
        self._log("complete", system, user, tag, text)
        return text

    def complete_json(self, system: str, user: str, **kw: Any) -> dict:
        return json.loads(self.complete(system, user, **kw))

    def stream(self, system: str, user: str, *, model: str | None = None, max_tokens: int = 2048,
               tag: str = "") -> Iterator[str]:
        out = self._lookup(system, user, tag)
        text = json.dumps(out, ensure_ascii=False) if isinstance(out, dict) else str(out)
        self._log("stream", system, user, tag, text)
        self.last_stream_mode = self.stream_mode
        for i in range(0, len(text), self.chunk):
            yield text[i:i + self.chunk]


# ── 检索 ──────────────────────────────────────────────────
class FakeSearch:
    """按查询子串返回预置命中。

    rules: {查询子串: [fact_id, ...] | {"facts": [...], "sufficient": bool, "signal": str}}
    多条规则都匹配时取第一条（字典顺序）。没匹配上 ⇒ 无命中、资料不足。
    `insufficient` 里的子串强制判为不足（命中照给，用于测「有最接近的页但不足」）。
    """

    def __init__(self, store: FakeStore | None = None, rules: dict[str, Any] | None = None, *,
                 insufficient: Iterable[str] = (), dense_used: bool = True) -> None:
        self.store = store
        self.rules = dict(DEMO_RULES if rules is None else rules)
        self.insufficient = list(insufficient)
        self.dense_used = dense_used
        self.queries: list[str] = []

    def _resolve(self, store: Any, query: str) -> RetrievalResult:
        store = store if store is not None else self.store
        facts: list[str] = []
        sufficient = False
        signal = "no_hit"
        for key, val in self.rules.items():
            if key in query:
                if isinstance(val, dict):
                    facts = list(val.get("facts", []))
                    sufficient = bool(val.get("sufficient", True))
                    signal = val.get("signal", "" if sufficient else "below_threshold")
                else:
                    facts, sufficient, signal = list(val), True, ""
                break
        if any(k in query for k in self.insufficient):
            sufficient, signal = False, "below_threshold"
        hits = [Hit(fact=store.fact(fid), rrf_score=1.0 / (60 + i + 1), bm25_rank=i + 1, dense_rank=i + 1)
                for i, fid in enumerate(facts)]
        return RetrievalResult(query=query, hits=hits, top_score=hits[0].rrf_score if hits else 0.0,
                               sufficient=sufficient and bool(hits), signal=signal, dense_used=self.dense_used)

    def search(self, store: Any, query: str, *, embedder: Any = None, **_: Any) -> RetrievalResult:
        self.queries.append(query)
        return self._resolve(store, query)

    def search_many(self, store: Any, queries: list[str], *, embedder: Any = None, **_: Any) -> RetrievalResult:
        queries = [q for q in queries if q and q.strip()]
        self.queries.extend(queries)
        results = [self._resolve(store, q) for q in queries]
        best = next((r for r in results if r.sufficient), results[0] if results else None)
        if best is None:
            return RetrievalResult(query="", hits=[], top_score=0.0, sufficient=False, signal="空查询")
        seen: set[str] = set()
        hits = []
        for r in [best, *results]:
            for h in r.hits:
                if h.fact.fact_id not in seen:
                    seen.add(h.fact.fact_id)
                    hits.append(h)
        return best.model_copy(update={"query": " || ".join(queries), "hits": hits})


# ── 生成（签名与 D 组真实实现一致）─────────────────────────
def fake_plan(client: Any, topic: str, genre: str, profile: dict, constraints: Iterable[str] = ()) -> Plan:
    data = client.complete_json("planner", f"主题：{topic}\n体裁：{genre}", tag="planner")
    data["sections"] = [PlanSection(**s) for s in data.get("sections", [])]
    return Plan(**data)


def fake_build_context(sub_question: str, result: RetrievalResult, page_facts: Callable[[int], list[Fact]], *,
                       sid_start: int = 1, max_snippets: int = 10) -> ContextPack:
    snippets = [Snippet(sid=f"S{sid_start + i}", fact_id=h.fact.fact_id, page_no=h.fact.page_no,
                        text=h.fact.text, status=h.fact.status, metric=h.fact.metric)
                for i, h in enumerate(result.hits[:max_snippets])]
    rendered = "\n".join([f"## 子问题：{sub_question}"] +
                         [f"[{s.sid} | 第{s.page_no}页] {s.text}" for s in snippets])
    return ContextPack(sub_question=sub_question, snippets=snippets, rendered=rendered)


def fake_stream_section(client: Any, section: PlanSection, pack: ContextPack, *, genre: str, profile: dict,
                        constraints: Iterable[str] = (), thesis: str = "", outline: Iterable[str] = (),
                        prev_headings: Iterable[str] = ()) -> Iterator[str]:
    return client.stream("section_writer", f"小节：{section.heading}\n{pack.rendered}", tag="section_writer")


def fake_stream_bookend(client: Any, kind: str, plan: Plan, verified_sections: list[str], *, genre: str,
                        profile: dict) -> Iterator[str]:
    return client.stream(f"bookend_{kind}", "\n".join(verified_sections), tag=f"bookend_{kind}")


def fake_stream_answer(client: Any, question: str, pack: ContextPack, *, profile: dict,
                       shape: str = "short") -> Iterator[str]:
    return client.stream("answer", f"问题：{question}\n{pack.rendered}", tag="answer")


def fake_rewrite_sentence(client: Any, sentence: str, failures: list[CheckResult], snippets: list[Snippet], *,
                          profile: dict) -> str:
    reasons = "；".join(f.reason for f in failures)
    return client.complete("rewriter", f"原句：{sentence}\n失败：{reasons}", tag="rewriter").strip()


def fake_check_topic(client: Any, request: str, *, genre: str | None, topic: str | None, profile: dict,
                     slots: dict, feedback: str = "") -> TopicCheck:
    """主题审核假实现。预置了 tag=topic_check 就用预置（dict 或 callable），否则原样确认请求里的体裁与主题。

    默认不调 client：现有用例不为它配预置也能跑通确认流程；要测「无效主题 / 带修改意见」时再预置。
    """
    try:
        out = client.complete_json("topic_check", f"请求：{request}\n体裁：{genre}\n主题：{topic}\n意见：{feedback}",
                                   tag="topic_check")
    except KeyError:
        out = {}
    data = {"valid": True, "genre": genre or "blog", "topic": topic or request, **(out or {})}
    return TopicCheck(**data)


def fake_rewrite_query(client: Any, message: str, slots: dict, recent_user_turns: list[str]) -> str:
    out = client.complete("query_rewrite", f"上文：{recent_user_turns}\n本句：{message}", tag="query_rewrite")
    return out.strip() or message


# ── 校验（V1–V4 简化版）──────────────────────────────────
_SENT = re.compile(r".+?(?:[。！？!?](?:\s*\[[^\]]*S\d+[^\]]*\])*|$)", re.S)
_CITE = re.compile(r"\[([^\]]*S\d+[^\]]*)\]")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def fake_split_sentences(text: str) -> list[Sentence]:
    out = []
    for m in _SENT.finditer(text.strip()):
        t = m.group(0).strip()
        if not t:
            continue
        cites = [s for g in _CITE.findall(t) for s in re.findall(r"S\d+", g)]
        out.append(Sentence(idx=len(out), text=t, cites=list(dict.fromkeys(cites))))
    return out


def _disc(label: str, sibling: str) -> str:
    """去掉兄弟标签的公共前后缀，剩下的就是区分词（「AI驱动流量增长」vs「…订单…」⇒ 流量）。"""
    i = 0
    while i < min(len(label), len(sibling)) and label[i] == sibling[i]:
        i += 1
    j = 0
    while j < min(len(label), len(sibling)) - i and label[-1 - j] == sibling[-1 - j]:
        j += 1
    return label[i:len(label) - j]


def fake_verify_sentence(sentence: Sentence, citation_map: dict[str, Snippet],
                         page_metrics: Callable[[int], list[Metric]], **_: object) -> SentenceVerdict:
    body = _CITE.sub("", sentence.text)
    cited = [citation_map[c] for c in sentence.cites if c in citation_map]
    v1 = all(c in citation_map for c in sentence.cites)
    material = " ".join(s.text for s in cited)
    nums = [n.strip() for n in _NUM.findall(body)]
    v2 = all(n in material.replace(" ", "") or n in material for n in nums)
    v3_ok, v3_reason = True, ""
    for page in sorted({s.page_no for s in cited}):
        metrics = page_metrics(page)
        for m in metrics:
            if m.value not in body:
                continue
            for sib in metrics:
                if sib is m or sib.value == m.value:
                    continue
                mine, theirs = _disc(m.label, sib.label), _disc(sib.label, m.label)
                if theirs and theirs in body and mine not in body:
                    v3_ok, v3_reason = False, f"{m.value} 属于「{m.label}」，句中写成了「{theirs}」"
    v4 = bool(sentence.cites) or not nums
    return SentenceVerdict(sentence=sentence, results=[
        CheckResult(check="V1", ok=v1, reason="" if v1 else "引用编号不在映射表"),
        CheckResult(check="V2", ok=v2, reason="" if v2 else "数字不在所引片段"),
        CheckResult(check="V3", ok=v3_ok, reason=v3_reason),
        CheckResult(check="V4", ok=v4, reason="" if v4 else "事实句未带引用"),
    ])


def fake_build_report(verdicts: list[SentenceVerdict], *, rewrites: list[dict], deletions: list[dict],
                      insufficient: list[dict], corrections: list[dict],
                      citation_map: dict[str, Snippet] | None = None) -> dict[str, Any]:
    return {
        "sentence_total": len(verdicts),
        "cited_sentences": sum(1 for v in verdicts if v.sentence.cites),
        "passed_sentences": sum(1 for v in verdicts if v.ok),
        "rewrites": rewrites,
        "deletions": deletions,
        "insufficient": insufficient,
        "corrections": corrections,
        "citation_count": len(citation_map or {}),
    }


def make_fake_deps(*, store: FakeStore | None = None, client: Any = "default", search: FakeSearch | None = None,
                   **overrides: Any) -> Any:
    """组装一套全假依赖。client=None 模拟无 key；overrides 可替换任意 Deps 字段（含真实函数）。"""
    from ragwriter.agent.orchestrator import Deps

    store = store or FakeStore()
    search = search or FakeSearch(store)
    deps = Deps(
        store=store,
        client=FakeClient() if client == "default" else client,
        embedder=None,
        search=search.search,
        search_many=search.search_many,
        plan=fake_plan,
        build_context=fake_build_context,
        stream_section=fake_stream_section,
        stream_bookend=fake_stream_bookend,
        stream_answer=fake_stream_answer,
        rewrite_sentence=fake_rewrite_sentence,
        rewrite_query=fake_rewrite_query,
        check_topic=fake_check_topic,
        split_sentences=fake_split_sentences,
        verify_sentence=fake_verify_sentence,
        build_report=fake_build_report,
    )
    if overrides:
        deps = deps.with_(**overrides)
    deps.fake_search = search  # type: ignore[attr-defined]  方便测试断言检索过哪些查询
    return deps
