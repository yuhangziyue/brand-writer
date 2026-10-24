"""处理器共用的部件：单轮上下文、流式吐字、逐句校验与改写循环、done 载荷。

放在包的 __init__ 而不是某个处理器里：10 号文档第 7 节要求 handlers 之间不互相调用，
共用逻辑只能下沉到这里。各处理器只 import 本模块，不 import 兄弟模块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Iterable, Iterator

from ragwriter.contracts import CheckResult, Event, RetrievalResult, Sentence, SentenceVerdict, Snippet

if TYPE_CHECKING:
    from ragwriter.agent.intent import IntentResult
    from ragwriter.agent.memory import SessionStore, WorkingWindow
    from ragwriter.agent.orchestrator import Deps

MAX_REWRITE_ROUNDS = 2  # 05 第 4.2 节：单句改写最多两轮，仍不过则删除
NOT_COVERED = "品宣资料未涉及"

# 三种拒答三套文案（评审 M6）：越界、资料完全不足、部分不足（后者不发拒答，走 done.boundary）
OUT_OF_SCOPE_TEXT = "这个我帮不上忙，我只负责基于万悉科技品宣资料写作和回答问题。"


def insufficient_all_text(topic: str, pages: list[int], *, verb: str) -> str:
    """资料不足的一句话。topic 用意图识别给的焦点词（「万悉科技的融资情况」），不复述整句原话。

    最接近的页不写进正文：前端用 done.nearest_pages 渲染成按钮，正文只回答「有没有」。
    """
    if verb == "写作":
        return f"品宣资料未涉及「{topic}」，写不成一篇有依据的内容。"
    return f"品宣资料未涉及{topic}。"


_GAPS_MARK = "GAPS"
_CITE_GROUP = re.compile(r"\[([^\[\]]*?S\d+[^\[\]]*?)\]")
_SID = re.compile(r"S\d+")
_GAPS_LINE = re.compile(r"(?:^|\n)[ \t]*GAPS[ \t]*[:：][ \t]*(?P<tail>.*?)\s*$", re.S)


def split_gaps_fallback(text: str) -> tuple[str, list[str]]:
    """D 组 `generate.writer.split_gaps` 缺席时的兜底：只认末尾 `GAPS: [...]` 一行。

    兜底存在的理由：GAPS 行一旦漏进正文，会被当成句子去校验，还会出现在定稿里。
    """
    import json as _json

    m = _GAPS_LINE.search(text)
    if not m:
        return text, []
    tail = m.group("tail").strip()
    try:
        raw = _json.loads(tail)
        gaps = [str(g) for g in raw] if isinstance(raw, list) else [str(raw)]
    except ValueError:
        gaps = [g.strip(" \"'「」") for g in tail.strip("[]").split(",")]
    return text[:m.start()].rstrip(), [g for g in gaps if g]


def parse_cites(text: str) -> list[str]:
    """从句子里抽 [S4] / [S4][S5] / [S4, S5] 形式的引用编号，保序去重。"""
    out: list[str] = []
    for grp in _CITE_GROUP.findall(text):
        for sid in _SID.findall(grp):
            if sid not in out:
                out.append(sid)
    return out


@dataclass
class TurnContext:
    """一轮请求里各步共享的状态。处理器把产出写进 outcome，编排器据此回写会话。"""

    deps: "Deps"
    memory: "SessionStore"
    session_id: str
    turn_id: str
    run_id: str
    message: str
    intent: "IntentResult"
    slots: dict[str, Any]
    window: "WorkingWindow"
    options: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    stream_modes: list[str] = field(default_factory=list)
    usage_start: int = 0
    outcome: dict[str, Any] = field(default_factory=dict)
    injected: bool = False
    dense_flags: list[bool] = field(default_factory=list)  # 每次检索稠密路是否真参与了
    _last_emitted: str = ""
    _metrics_cache: dict[int, list] = field(default_factory=dict)

    _seen_body: list = field(default_factory=list)  # 前面小节已定稿的 (引用集合, 二元组)，跨节去重用

    @property
    def case_pages(self) -> frozenset[int]:
        """标题以「案例」开头的页。第 30/35/36 页虽在案例章节，但不是案例，不能按章节划。"""
        return frozenset(c["page_no"] for c in self.profile.get("catalog", [])
                         if str(c.get("title", "")).startswith("案例"))

    @property
    def max_rounds(self) -> int:
        return int(self.options.get("max_rewrite_rounds", MAX_REWRITE_ROUNDS))

    _facts_cache: dict[int, list] = field(default_factory=dict)

    def page_metrics(self, page_no: int) -> list:
        """按页取数字配对，带缓存。以回调形式交给 verify_sentence（V3 按所引页取兄弟标签）。"""
        if page_no not in self._metrics_cache:
            store = self.deps.store
            try:
                self._metrics_cache[page_no] = list(store.page_metrics(page_no)) if store is not None else []
            except Exception:  # noqa: BLE001 — 拿不到数字配对时 V3 只是少一道依据，不该让整轮失败
                self._metrics_cache[page_no] = []
        return self._metrics_cache[page_no]

    def page_facts(self, page_no: int) -> list:
        """按页取可用事实，带缓存。以回调形式交给 build_context 做整页回填。"""
        if page_no not in self._facts_cache:
            store = self.deps.store
            try:
                self._facts_cache[page_no] = list(store.page_facts(page_no)) if store is not None else []
            except Exception:  # noqa: BLE001
                self._facts_cache[page_no] = []
        return self._facts_cache[page_no]

    def note_stream_mode(self) -> None:
        mode = getattr(self.deps.client, "last_stream_mode", None)
        if mode:
            self.stream_modes.append(str(mode))

    def stream_mode(self) -> str:
        """有一次退化成模拟流式，整轮就标 simulated（A7：前端据此提示）。"""
        if not self.stream_modes:
            return "none"
        if "simulated" in self.stream_modes:
            return "simulated"
        return self.stream_modes[-1]

    def usage(self) -> dict[str, int]:
        runlog = getattr(self.deps.client, "runlog", None)
        records = list(getattr(runlog, "records", []) or [])[self.usage_start:]
        u = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": len(records)}
        for r in records:
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                u[k] += int(r.get(k, 0) or 0)
        return u


# ── 事件构造 ──────────────────────────────────────────────
def ev(type_: str, **data: Any) -> Event:
    return Event(type=type_, data=data)  # type: ignore[arg-type]


def note_retrieval(ctx: TurnContext, result: RetrievalResult) -> RetrievalResult:
    ctx.dense_flags.append(bool(result.dense_used))
    return result


def retrieval_payload(result: RetrievalResult, *, section: int | None, kind: str,
                      sids: dict[str, str] | None = None) -> dict[str, Any]:
    """两路排名都给：评审要能看到「这条为什么命中」（10 号文档 6.2 节）。"""
    sid_of = {v: k for k, v in (sids or {}).items()}  # fact_id → sid
    return {
        "section": section,
        "kind": kind,
        "query": result.query,
        "sufficient": result.sufficient,
        "signal": result.signal,
        "top_score": result.top_score,
        "dense_used": result.dense_used,
        "hits": [
            {
                "fact_id": h.fact.fact_id,
                "page_no": h.fact.page_no,
                "status": h.fact.status,
                "rrf_score": h.rrf_score,
                "bm25_rank": h.bm25_rank,
                "dense_rank": h.dense_rank,
                "rerank_score": h.rerank_score,
                "sid": sid_of.get(h.fact.fact_id),
            }
            for h in result.hits
        ],
    }


def nearest_pages(results: Iterable[RetrievalResult | None], n: int = 3) -> list[int]:
    """拒答时附「最接近的页」：按命中顺序取页码去重。"""
    pages: list[int] = []
    for r in results:
        if r is None:
            continue
        for h in r.hits:
            if h.fact.page_no not in pages:
                pages.append(h.fact.page_no)
    return pages[:n]


def snippet_dict(s: Snippet) -> dict[str, Any]:
    return {"fact_id": s.fact_id, "page_no": s.page_no, "text": s.text, "status": s.status}


def citations_for(sentences: Iterable[Sentence], by_sid: dict[str, Snippet]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in sentences:
        for sid in s.cites or parse_cites(s.text):
            if sid in by_sid and sid not in out:
                out[sid] = snippet_dict(by_sid[sid])
    return out


# ── 流式吐字 ──────────────────────────────────────────────
def stream_text(ctx: TurnContext, chunks: Iterable[str], *, section: int, kind: str
                ) -> Generator[Event, None, str]:
    """转发模型增量为 delta，返回完整原文（含 GAPS 行，由调用方剥离）。

    写作节点会在末尾输出 `GAPS: [...]`（评审 M1）。为了不让这行闪现在前端，
    这里压住尾部可能构成 GAPS 标记的几个字符，一旦确认出现 GAPS 就停止转发；
    万一仍有漏出，调用方用 patch(action=trim) 兜底清掉。
    """
    buf = ""
    emitted = 0
    stopped = False
    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        if stopped:
            continue
        mark = buf.find(_GAPS_MARK, emitted)
        if mark >= 0:
            end, stopped = mark, True
        else:
            end = len(buf)
            # 尾部是 "G" / "GA" / "GAP" 时先压住，等下一块再判断
            for k in range(len(_GAPS_MARK) - 1, 0, -1):
                if buf.endswith(_GAPS_MARK[:k]):
                    end = len(buf) - k
                    break
        if end > emitted:
            yield ev("delta", section=section, kind=kind, text=buf[emitted:end])
            emitted = end
    if not stopped and emitted < len(buf):
        yield ev("delta", section=section, kind=kind, text=buf[emitted:])
        emitted = len(buf)
    ctx.note_stream_mode()
    ctx._last_emitted = buf[:emitted]
    return buf


# ── 逐句校验与改写循环 ────────────────────────────────────
@dataclass
class VerifiedBlock:
    sentences: list[Sentence] = field(default_factory=list)  # 定稿句（已过校验）
    verdicts: list[SentenceVerdict] = field(default_factory=list)  # 定稿句的最终判定
    rewrites: list[dict[str, Any]] = field(default_factory=list)
    deletions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.sentences)


def _verify_payload(section: int, kind: str, v: SentenceVerdict, round_: int) -> dict[str, Any]:
    failed = [r for r in v.results if not r.ok]
    return {
        "section": section,
        "kind": kind,
        "sentence": v.sentence.idx,
        "text": v.sentence.text,
        "round": round_,
        "result": "pass" if v.ok else "fail",
        "checks": [r.model_dump() for r in v.results],
        "failed": [r.check for r in failed],
        "reason": "；".join(f"{r.check}: {r.reason}" for r in failed),
    }


def verify_and_fix(ctx: TurnContext, text: str, *, section: int, kind: str, snippets: list[Snippet]
                   ) -> Generator[Event, None, VerifiedBlock]:
    """草稿 → 逐句校验 → 不过的改写 ≤2 轮（每轮重验）→ 仍不过删除。

    两阶段事件（6.1 节）：草稿已经以 delta 发出去了，这里发 verify / patch 让前端替换，
    评审人能亲眼看到闸门在工作。
    """
    d = ctx.deps
    # 校验用的原文 = 页标题 + 片段正文。页标题也是文字层原文，模型点明案例时用的正是它
    # （「头部牛仔服饰OEM工厂」只在第 32 页标题里）；只给正文，V2 会把「OEM」当编造的专名拦下
    titles = {c["page_no"]: str(c.get("title", "")) for c in ctx.profile.get("catalog", [])}
    by_sid = {s.sid: s.model_copy(update={"text": f"{titles[s.page_no]}\n{s.text}"})
              if titles.get(s.page_no) else s for s in snippets}
    block = VerifiedBlock()

    def check(sent: Sentence) -> SentenceVerdict:
        return d.verify_sentence(sent, by_sid, ctx.page_metrics, strict_cite=(kind == "lede"),
                                 case_pages=ctx.case_pages)

    text = strip_meta(text)
    for sent in d.split_sentences(text):
        if not sent.text.strip():
            continue
        if not sent.cites:
            sent = sent.model_copy(update={"cites": parse_cites(sent.text)})
        verdict = check(sent)
        yield ev("verify", **_verify_payload(section, kind, verdict, 0))
        cur, cur_v = sent, verdict
        rounds = 0
        while not cur_v.ok and rounds < ctx.max_rounds:
            rounds += 1
            failures: list[CheckResult] = [r for r in cur_v.results if not r.ok]
            try:
                new_text = d.rewrite_sentence(d.client, cur.text, failures, snippets, profile=ctx.profile) or ""
            except Exception as exc:  # noqa: BLE001 — 改写失败按「改不好」处理，落到删除，不中断全文
                yield ev("error", stage="rewrite", reason=str(exc), section=section,
                         sentence=sent.idx, fallback=False)
                break
            new_text = new_text.strip()
            if not new_text:
                break
            cand = Sentence(idx=sent.idx, text=new_text, cites=parse_cites(new_text))
            cand_v = check(cand)
            # 每轮一条记录（B 组 build_report 约定：一句改两轮就是两条）
            block.rewrites.append({
                "section": section, "kind": kind, "sentence": sent.idx, "round": rounds,
                "original": cur.text, "rewritten": new_text,
                "failed_checks": [r.check for r in failures], "reason": "；".join(r.reason for r in failures),
                "ok": cand_v.ok,
            })
            cur, cur_v = cand, cand_v
            yield ev("verify", **_verify_payload(section, kind, cur_v, rounds))

        if cur_v.ok:
            if cur is not sent:
                yield ev("patch", section=section, kind=kind, sentence=sent.idx, action="rewrite",
                         text=cur.text, original=sent.text, rounds=rounds)
            block.sentences.append(cur)
            block.verdicts.append(cur_v)
        else:
            failed = [r for r in cur_v.results if not r.ok]
            reason = "；".join(f"{r.check}: {r.reason}" for r in failed)
            yield ev("patch", section=section, kind=kind, sentence=sent.idx, action="delete",
                     text="", original=sent.text, rounds=rounds, reason=reason)
            block.deletions.append({"section": section, "kind": kind, "sentence": sent.idx,
                                    "original": sent.text, "failed_checks": [r.check for r in failed],
                                    "reason": reason, "rounds": rounds})
    yield from _judge_and_fix(ctx, block, by_sid, snippets, section=section, kind=kind, check=check)
    yield from _drop_subsumed(block, section=section, kind=kind,
                              seen=ctx._seen_body if kind == "body" else None)
    return block


# 上下文里给模型看的内部标注（render_snippet 加的出处口径提示），模型偶尔会照抄进正文（10-23 实测 FAQ）
_META = re.compile(r"〔出处口径：[^〕]*〕")


def strip_meta(text: str) -> str:
    return _META.sub("", text)


# ── V5 模型判别（10-23 补上，原为 P2）──────────────────────────
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["idx", "supported", "reason"],
        "properties": {"idx": {"type": "integer"}, "supported": {"type": "boolean"}, "reason": {"type": "string"}}}}},
}


def judge_block(ctx: TurnContext, sentences: list[Sentence], by_sid: dict[str, Snippet]) -> dict[int, str]:
    """批量判一节：返回 {句号: 不支持的原因}。只判带引用的句子；失败抛异常由调用方降级。"""
    from ragwriter.generate import prompts
    from ragwriter.generate.context import render_snippet

    items = []
    for s in sentences:
        cited = [by_sid[c] for c in s.cites if c in by_sid]
        if not cited:
            continue
        items.append(f"### 句 {s.idx}\n{s.text}\n所引片段：\n" + "\n".join(render_snippet(c) for c in cited))
    if not items:
        return {}
    system = prompts.render("judge", "System")
    user = prompts.render("judge", "User", items="\n\n".join(items))
    # 判别用 LLM_JUDGE_MODEL（与生成不同源时盲区更小）；没配就和生成同一个模型
    judge_model = getattr(getattr(ctx.deps.client, "cfg", None), "judge_model", None)
    kw = {"model": judge_model} if judge_model else {}
    raw = ctx.deps.client.complete(system, user, json_schema=JUDGE_SCHEMA, max_tokens=1500, tag="judge", **kw)
    return {int(x["idx"]): str(x.get("reason") or "片段不支持本句") for x in _judge_items(raw)
            if isinstance(x, dict) and "idx" in x and not x.get("supported", True)}


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _judge_items(raw: str) -> list:
    """判别输出的宽容解析。10-24 实测：中转下模型不遵守 strict schema，返回带 ```json 围栏的**数组**
    而不是 {"items": [...]}，原来的 complete_json 直接失败、按降级跳过——V5 等于没跑。
    数组与对象两种形状都收；仍解析不了就抛，由调用方发 error 事件降级。"""
    import json

    data = json.loads(_FENCE.sub("", raw.strip()))
    if isinstance(data, dict):
        data = data.get("items") or []
    return data if isinstance(data, list) else []


def _judge_and_fix(ctx: TurnContext, block: "VerifiedBlock", by_sid: dict[str, Snippet], snippets: list[Snippet],
                   *, section: int, kind: str, check) -> Iterator[Event]:
    """V5 模型判别：判为不支持的句子带着原因改写一次，程序校验 + 再判一次都过才采纳，否则删除。

    放在确定性校验之后：程序能拦的先拦掉，模型只负责程序拦不住的「主体错位 / 以个体代整体」。
    """
    d = ctx.deps
    if ctx.options.get("judge", True) is False or getattr(d, "client", None) is None or not block.sentences:
        return
    try:
        bad = judge_block(ctx, block.sentences, by_sid)
    except Exception as exc:  # noqa: BLE001 — 判别失败不拖垮写作，确定性校验已经兜住数字与字面
        yield ev("error", stage="judge", reason=str(exc), section=section, fallback=True)
        return
    if not bad:
        return
    keep_s, keep_v = [], []
    for sent, verdict in zip(block.sentences, block.verdicts):
        reason = bad.get(sent.idx)
        if reason is None:
            keep_s.append(sent)
            keep_v.append(verdict)
            continue
        fail = CheckResult(check="V5", ok=False, reason=f"判别模型：{reason}")
        yield ev("verify", section=section, kind=kind, sentence=sent.idx, text=sent.text, round=0, result="fail",
                 checks=[fail.model_dump()], failed=["V5"], reason=f"V5: {fail.reason}")
        new_text = ""
        try:
            new_text = (d.rewrite_sentence(d.client, sent.text, [fail], snippets, profile=ctx.profile) or "").strip()
        except Exception:  # noqa: BLE001 — 改写失败按删除处理
            new_text = ""
        accepted = False
        if new_text:
            cand = Sentence(idx=sent.idx, text=new_text, cites=parse_cites(new_text))
            cand_v = check(cand)
            if cand_v.ok:
                try:
                    accepted = cand.idx not in judge_block(ctx, [cand], by_sid)
                except Exception:  # noqa: BLE001
                    accepted = False
            block.rewrites.append({"section": section, "kind": kind, "sentence": sent.idx, "round": 1,
                                   "original": sent.text, "rewritten": new_text, "failed_checks": ["V5"],
                                   "reason": fail.reason, "ok": accepted})
            if accepted:
                yield ev("patch", section=section, kind=kind, sentence=sent.idx, action="rewrite",
                         text=new_text, original=sent.text, rounds=1)
                keep_s.append(cand)
                keep_v.append(cand_v)
                continue
        yield ev("patch", section=section, kind=kind, sentence=sent.idx, action="delete", text="",
                 original=sent.text, rounds=1, reason=f"V5: {fail.reason}")
        block.deletions.append({"section": section, "kind": kind, "sentence": sent.idx, "original": sent.text,
                                "failed_checks": ["V5"], "reason": fail.reason, "rounds": 1})
    block.sentences, block.verdicts = keep_s, keep_v


def _drop_subsumed(block: "VerifiedBlock", *, section: int, kind: str,
                   seen: list | None = None) -> Iterator[Event]:
    """删掉被本节后文完全包含的句子。

    10-22 真跑实例：V4 改写把「这一变化在国内市场已有数据佐证」补成「……豆包 3.45 亿[S4]」，
    紧接着下一句是「豆包 3.45 亿，DeepSeek 1.27 亿……[S4][S5][S6]」——两句都合规，读起来是复读。
    判据纯确定性：数字非空，且数字集合与引用集合都是另一句的子集。
    """
    from ragwriter.verify.checks import number_tokens

    def subsumed(i: int, j: int) -> bool:
        """第 i 句是否被第 j 句包含。完全相同的两句只删后一句。

        例外：被删句点明了另一句没有的**主体**（案例主体词或专名），不删。10-23 实测，
        「一家头部牛仔服饰 OEM 工厂的 GEO 实践……询盘量增加 20%[S62]」被后一句「这套动作……20%、17.6%[S62][S64]」
        按数字与引用判为重复删掉，后句开头的「该案例」就没了指代。
        不用字面重合度当判据：带了句开场白的真复读（「这一变化已有数据佐证，据……豆包 3.45 亿」）重合度也不高，会漏删。
        """
        (n1, c1), (n2, c2) = feats[i], feats[j]
        if i == j:
            return False
        # 不含数字的复读句（10-24 实测：「GEO……旨在提升品牌与产品在 AI 生成答案中的可见性[S15]」连说两遍）：
        # 引用相同、文字高度重合（字二元组 Jaccard ≥ 0.7）就删后一句
        if j < i and c1 and c1 <= c2 and grams[i] and grams[j]:
            if len(grams[i] & grams[j]) / len(grams[i] | grams[j]) >= 0.7:
                return True
        if not n1 or not (n1 <= n2 and c1 <= c2):
            return False
        if subjects[i] - subjects[j]:
            return False
        return (n1, c1) != (n2, c2) or j < i

    from ragwriter.verify.checks import _bigrams
    from ragwriter.verify.sentences import strip_cites

    def repeats_earlier_section(s: Sentence) -> bool:
        """跨节复读：引用集合相同、字面重合 ≥ 0.6。真跑里「GEO 作为统筹大脑……」在三节各出现一次。"""
        if seen is None or not s.cites:
            return False
        bg = _bigrams(strip_cites(s.text))
        return any(set(s.cites) == c and bg and len(bg & b) / len(bg | b) >= 0.6 for c, b in seen)

    keep: list[int] = []
    feats = [(set(number_tokens(s.text)), set(s.cites)) for s in block.sentences]
    from ragwriter.verify.checks import CASE_MARKERS, latin_names

    def subject_terms(text: str) -> set[str]:
        body = strip_cites(text)
        return {m for m in CASE_MARKERS if m in body} | {n.lower() for n in latin_names(body)}

    subjects = [subject_terms(s.text) for s in block.sentences]
    grams = [_bigrams(strip_cites(s.text)) for s in block.sentences]
    for i in range(len(feats)):
        s = block.sentences[i]
        cross = repeats_earlier_section(s)
        if not cross and not any(subsumed(i, j) for j in range(len(feats))):
            keep.append(i)
            continue
        reason = "与前文小节重复" if cross else "与本节另一句重复（数字与引用已被其包含）"
        yield ev("patch", section=section, kind=kind, sentence=s.idx, action="delete",
                 text="", original=s.text, rounds=0, reason=reason)
        block.deletions.append({"section": section, "kind": kind, "sentence": s.idx,
                                "original": s.text, "failed_checks": [], "reason": reason, "rounds": 0})
    block.sentences = [block.sentences[i] for i in keep]
    block.verdicts = [block.verdicts[i] for i in keep]
    if seen is not None:
        seen.extend((set(s.cites), _bigrams(strip_cites(s.text))) for s in block.sentences if s.cites)


# ── done 载荷 ─────────────────────────────────────────────
def done_event(ctx: TurnContext, *, final_text: str, artifact_id: str | None = None,
               citations: dict | None = None, report: dict | None = None, refused: bool = False,
               nearest: list[int] | None = None, suggestions: list[str] | None = None,
               refusal_kind: str | None = None, boundary: list[dict] | None = None,
               **extra: Any) -> Event:
    """done 的最小字段集是 F/G 组的接口，见编排器入口约定；这里统一出口，免得各处理器漏字段。"""
    data: dict[str, Any] = {
        "artifact_id": artifact_id,
        "intent": ctx.intent.intent,
        "final_text": final_text,
        "citations": citations or {},
        "report": report or {},
        "stream_mode": ctx.stream_mode(),
        "usage": ctx.usage(),
        "refused": refused,
        "nearest_pages": nearest or [],
        "suggestions": suggestions or [],
        # out_of_scope | insufficient_all | partial（评审 M6）；另有 clarify / unsupported / no_model
        "refusal_kind": refusal_kind,
        # 资料边界卡（评审 M1）：[{topic, source: plan|retrieval|writer, nearest_pages}]
        "boundary": boundary or [],
        "injected": ctx.injected,
        # 稠密路是否参与（A 组：BM25 单路的充分性判断不可靠，退化必须显式）；没检索过为 None
        "dense_used": (all(ctx.dense_flags) if ctx.dense_flags else None),
        "elapsed_ms": None,
    }
    data.update(extra)
    return Event(type="done", data=data)



def applied_corrections(profile: dict, final_text: str, snippets: Iterable[Snippet]) -> list[dict[str, str]]:
    """错字修正记录 {original, corrected}：片段里有错字原文、定稿里出现了修正后的写法，才算一次修正。

    品牌档案的修正表是「可能的修正」，报告要的是「本篇实际发生的修正」，两者不能混。
    """
    texts = " ".join(s.text for s in snippets)
    out: list[dict[str, str]] = []
    for c in ((profile or {}).get("rules") or {}).get("corrections") or []:
        if not isinstance(c, dict):
            continue
        wrong = c.get("wrong") or c.get("original")
        right = c.get("right") or c.get("corrected")
        if wrong and right and wrong in texts and right in final_text:
            out.append({"original": wrong, "corrected": right})
    return out


def trim_leak(ctx: TurnContext, body: str, *, section: int, kind: str) -> Iterator[Event]:
    """GAPS 行若已有部分以 delta 流出（标记被拆在奇怪的位置），发 patch(trim) 让前端用正文整体替换。"""
    if ctx._last_emitted.strip() != body.strip() and len(ctx._last_emitted.strip()) > len(body.strip()):
        yield ev("patch", section=section, kind=kind, sentence=None, action="trim", text=body,
                 original=ctx._last_emitted)
