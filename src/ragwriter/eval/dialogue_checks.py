"""对话质量的确定性检测：不调模型，只看事件流与 done 载荷。

两类检测：
- 逐轮：把用例里每轮的 expect 字段逐项判定（字段语义见本模块各函数 docstring）
- 隐含规则（不占 expect 字段）：clarify 的 suggestions 恰为 3 条；写 FAQ 时 done.jsonld 为 FAQPage；
  turn.options.inject_fault=v3_swap 时须出现 V3 失败的 verify 事件且其后有 patch
- 跨轮：会话一致、槽位继承、引用只来自本轮检索、拒答仍给最接近的页、[S编号] 可解析

需要模型判断的（参考答案匹配度、要点语义覆盖）交给 G 组评估器；
这里的 key_points 只做「页被引用 + 字面重合」的确定性下限。
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel

from ragwriter.eval.dialogue_runner import CaseRun, DialogueCase, TurnExpect, TurnRun

# 「资料不足」的明确说法。05 / 10 号文档的硬约束原话是「品宣资料未涉及」
INSUFFICIENT_MARKERS = ("未涉及", "资料不足", "没有涉及", "未提及", "没有提及", "没有相关", "无相关", "并未披露", "未披露", "不包含")
KEY_POINT_MIN_OVERLAP = 0.4  # 要点与正文的字二元组召回下限
KEY_POINT_MIN_COVERAGE = 0.6  # 用例要点覆盖率下限
INSUFFICIENT_MIN_OVERLAP = 0.3
# 写作类正文不得再出现这句（2026-10-22 产品评审：资料不足移到 done.boundary）
WRITE_BODY_BANNED = "品宣资料未涉及"
CLARIFY_SUGGESTIONS = (
    "LLM-原生 GEO 与传统 GEO 有什么不同",
    "为什么品牌要被 AI 记住",
    "为什么 GEO 必须合规",
)

_SID = re.compile(r"\[(S\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


class CheckOutcome(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    turn: int | None = None  # 0 起；跨轮检测为 None 或所涉轮次


# ── 文本工具 ────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def strip_cites(s: str) -> str:
    return _SID.sub("", s or "")


def sentences(text: str) -> list[str]:
    return [x for x in (p.strip() for p in _SENT_SPLIT.split(strip_cites(text))) if x]


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[\s\W_]+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def bigram_recall(needle: str, hay: str) -> float:
    a = _bigrams(needle)
    return len(a & _bigrams(hay)) / len(a) if a else 0.0


def cited_sids(turn: TurnRun) -> list[str]:
    out: list[str] = []
    for s in _SID.findall(turn.final_text):
        if s not in out:
            out.append(s)
    return out


def cited_pages(turn: TurnRun) -> set[int]:
    """正文里实际出现的 [S编号] 对应的页。citations 里有但正文没用到的不算引用。"""
    pages: set[int] = set()
    for s in cited_sids(turn):
        c = turn.citations.get(s)
        if c and c.get("page_no") is not None:
            pages.add(int(c["page_no"]))
    return pages


def _value_positions(text: str, value: str) -> list[int]:
    v = re.escape(_norm(value))
    # 前面不能紧挨数字或小数点：「15倍」不匹配「115倍」
    return [m.start() for m in re.finditer(rf"(?<![0-9.]){v}", text)]


# ── 逐轮 expect ────────────────────────────────────────────────
def check_intent(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    return CheckOutcome(name="intent", ok=t.intent == e.intent, detail=f"期望 {e.intent}，实际 {t.intent}")


def check_genre(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    return CheckOutcome(name="genre", ok=t.genre == e.genre, detail=f"期望 {e.genre}，实际 {t.genre}")


def check_topic_contains(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    topic = _norm(t.topic or "")
    miss = [w for w in e.topic_contains if _norm(w) not in topic]
    return CheckOutcome(name="topic_contains", ok=not miss, detail=f"topic={t.topic!r} 缺 {miss}" if miss else f"topic={t.topic!r}")


def check_refused(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    return CheckOutcome(name="refused", ok=t.refused == e.refused, detail=f"期望 {e.refused}，实际 {t.refused}")


def check_events_in_order(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    """子序列匹配：期望的事件类型按序出现即可，中间可以夹别的事件。"""
    it = iter(t.event_types)
    missing_from = None
    for i, want in enumerate(e.events_in_order):
        if not any(got == want for got in it):
            missing_from = i
            break
    ok = missing_from is None
    detail = "" if ok else f"从第 {missing_from} 项 {e.events_in_order[missing_from]!r} 起未按序出现；实际 {t.event_types}"
    return CheckOutcome(name="events_in_order", ok=ok, detail=detail)


def check_cite_pages_any(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    got = cited_pages(t)
    ok = bool(got & set(e.cite_pages_any))
    return CheckOutcome(name="cite_pages_any", ok=ok, detail=f"引用页 {sorted(got)}，期望至少其一 {e.cite_pages_any}")


def check_cite_pages_all(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    got = cited_pages(t)
    miss = sorted(set(e.cite_pages_all) - got)
    return CheckOutcome(name="cite_pages_all", ok=not miss, detail=f"引用页 {sorted(got)}，缺 {miss}")


def check_forbidden_strings(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    text = _norm(t.final_text)
    hit = [s for s in e.forbidden_strings if _norm(s) in text]
    return CheckOutcome(name="forbidden_strings", ok=not hit, detail=f"出现禁用串 {hit}" if hit else "")


_CLAUSE_SPLIT = re.compile(r"[，,、：:]")


def _nearest_object(text: str, pos: int, vlen: int, objects: list[str]) -> str | None:
    """text 内离数字最近的对象词（按与数字两端的字符距离）。"""
    best: tuple[int, str] | None = None
    for o in objects:
        i = text.rfind(o, 0, pos)
        if i >= 0:
            d = pos - (i + len(o))
            if best is None or d < best[0]:
                best = (d, o)
        j = text.find(o, pos + vlen)
        if j >= 0:
            d = j - (pos + vlen)
            if best is None or d < best[0]:
                best = (d, o)
    return best[1] if best else None


def check_numbers(t: TurnRun, e: TurnExpect) -> list[CheckOutcome]:
    """数字对象一致：正文每次出现 value，须挂在 object 上。

    判定：先看数字所在分句（按 ，、：切），取本轮 numbers 声明过的对象词里离数字最近的一个；
    分句里一个对象词都没有时，退到整句取最近的。它必须是期望的 object。
    句中含「分别」时退化为「同句含 object 即可」，避免「订单和流量分别增长 15 倍和 8 倍」误报。
    value 没出现不算错（是否必须出现由 key_points / 参考答案管）。
    """
    objects = [_norm(n.object) for n in e.numbers]
    out: list[CheckOutcome] = []
    for n in e.numbers:
        want, val = _norm(n.object), _norm(n.value)
        bad: list[str] = []
        seen = 0
        for sent in sentences(t.final_text):
            s = _norm(sent)
            if "分别" in s:
                if _value_positions(s, n.value):
                    seen += 1
                    if want not in s:
                        bad.append(sent)
                continue
            for clause in (c for c in _CLAUSE_SPLIT.split(s) if c):
                for pos in _value_positions(clause, n.value):
                    seen += 1
                    near = _nearest_object(clause, pos, len(val), objects)
                    if near is None:
                        cpos = s.find(clause)
                        near = _nearest_object(s, cpos + pos, len(val), objects)
                    if near != want:
                        bad.append(sent)
        name = f"numbers[{n.value}->{n.object}]"
        if bad:
            out.append(CheckOutcome(name=name, ok=False, detail=f"数字挂错对象：{bad[:2]}"))
        else:
            out.append(CheckOutcome(name=name, ok=True, detail=f"出现 {seen} 次" if seen else "未出现"))
    return out


def check_key_points(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    """确定性下限：要点所在页被引用，且要点与正文字二元组召回 ≥ KEY_POINT_MIN_OVERLAP。"""
    pages = cited_pages(t)
    body = strip_cites(t.final_text)
    covered, missed = [], []
    for kp in e.key_points:
        if kp.page in pages and bigram_recall(kp.text, body) >= KEY_POINT_MIN_OVERLAP:
            covered.append(kp.text)
        else:
            missed.append(f"{kp.text}(p{kp.page})")
    ratio = len(covered) / len(e.key_points) if e.key_points else 1.0
    return CheckOutcome(name="key_points", ok=ratio >= KEY_POINT_MIN_COVERAGE,
                        detail=f"覆盖 {len(covered)}/{len(e.key_points)}；未覆盖 {missed}")


def check_must_say_insufficient(t: TurnRun, e: TurnExpect) -> list[CheckOutcome]:
    """资料不足的子话题须被明确交代，口径按意图分（2026-10-22 产品评审）：

    - write：子话题出现在 done.boundary[].topic 里（字二元组召回 ≥ 阈值），且正文不含「品宣资料未涉及」
    - 其他（qa 等）：正文某句同时含「资料不足」说法与该话题的字面重合；或有 insufficient 事件点名它
    """
    out = []
    if e.intent == "write":
        body_clean = WRITE_BODY_BANNED not in t.final_text
        for item in e.must_say_insufficient:
            hit = [b for b in t.boundary if bigram_recall(item, str(b.get("topic", ""))) >= INSUFFICIENT_MIN_OVERLAP]
            ok = bool(hit) and body_clean
            detail = ("boundary 命中 " + str([b.get("topic") for b in hit])) if hit else f"boundary 里没有：{[b.get('topic') for b in t.boundary]}"
            if not body_clean:
                detail += f"；正文仍含「{WRITE_BODY_BANNED}」"
            out.append(CheckOutcome(name=f"must_say_insufficient[{item}]", ok=ok, detail=detail))
        return out
    sents = sentences(t.final_text)
    for item in e.must_say_insufficient:
        in_text = any(any(m in s for m in INSUFFICIENT_MARKERS) and bigram_recall(item, s) >= INSUFFICIENT_MIN_OVERLAP
                      for s in sents)
        in_event = any(bigram_recall(item, " ".join(str(v) for v in ev.values())) >= INSUFFICIENT_MIN_OVERLAP
                       for ev in t.insufficient)
        out.append(CheckOutcome(name=f"must_say_insufficient[{item}]", ok=in_text or in_event,
                                detail="正文" if in_text else ("insufficient 事件" if in_event else "正文与事件都没有明确说资料未涉及")))
    return out


def check_refusal_kind(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    return CheckOutcome(name="refusal_kind", ok=t.refusal_kind == e.refusal_kind,
                        detail=f"期望 {e.refusal_kind}，实际 {t.refusal_kind}")


def check_inherited_topic(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    got = bool(t.inherited.get("topic"))
    return CheckOutcome(name="inherited_topic", ok=got == bool(e.inherited_topic),
                        detail=f"intent.inherited={t.inherited}")


def check_clarify_suggestions(t: TurnRun) -> CheckOutcome:
    ok = len(t.suggestions) == len(CLARIFY_SUGGESTIONS)
    extra = [s for s in t.suggestions if s not in CLARIFY_SUGGESTIONS]
    detail = f"suggestions={t.suggestions}" + (f"（非约定常量：{extra}，仅提示）" if extra else "")
    return CheckOutcome(name="clarify_suggestions", ok=ok, detail=detail)


def _jsonld_types(obj) -> set[str]:
    types: set[str] = set()
    if isinstance(obj, dict):
        tp = obj.get("@type")
        if isinstance(tp, str):
            types.add(tp)
        elif isinstance(tp, list):
            types.update(str(x) for x in tp)
        for v in obj.values():
            types |= _jsonld_types(v)
    elif isinstance(obj, list):
        for v in obj:
            types |= _jsonld_types(v)
    return types


def check_faq_jsonld(t: TurnRun) -> CheckOutcome:
    obj = t.jsonld
    if isinstance(obj, str):
        try:
            import json
            obj = json.loads(obj)
        except ValueError:
            obj = None
    types = _jsonld_types(obj)
    ok = "FAQPage" in types
    return CheckOutcome(name="faq_jsonld", ok=ok, detail=f"@type={sorted(types)}" if obj is not None else "done 里没有 jsonld")


def _is_v3_fail(ev) -> bool:
    d = ev.data or {}
    check = d.get("check") or d.get("name")
    res = d.get("result", d.get("ok"))
    return check == "V3" and (res in ("fail", "failed", False))


def check_injected_v3(t: TurnRun) -> CheckOutcome:
    """故障注入 v3_swap：须先出现 V3 失败的 verify，之后有 patch（A5：闸门可见）。正文错配由 numbers 判。"""
    types = t.event_types
    fail_at = next((i for i, ev in enumerate(t.events) if ev.type == "verify" and _is_v3_fail(ev)), None)
    patch_after = fail_at is not None and "patch" in types[fail_at + 1:]
    return CheckOutcome(name="inject.v3_swap", ok=bool(patch_after),
                        detail="V3 fail → patch" if patch_after else ("没有 V3 失败的 verify 事件" if fail_at is None else "V3 失败后没有 patch"))


def check_no_clarify(t: TurnRun, e: TurnExpect) -> CheckOutcome:
    ok = t.intent != "clarify"
    return CheckOutcome(name="no_clarify", ok=ok, detail="" if ok else f"本轮不该追问，却进入 clarify：{t.suggestions}")


def check_turn(t: TurnRun, e: TurnExpect) -> list[CheckOutcome]:
    out: list[CheckOutcome] = []
    out.append(CheckOutcome(name="done", ok=t.has_done and not t.errors,
                            detail=f"errors={t.errors}" if t.errors else ("" if t.has_done else "没有 done 事件")))
    out.append(check_intent(t, e))
    if e.genre is not None:
        out.append(check_genre(t, e))
    if e.topic_contains:
        out.append(check_topic_contains(t, e))
    if e.refused is not None:
        out.append(check_refused(t, e))
    if e.events_in_order:
        out.append(check_events_in_order(t, e))
    if e.cite_pages_any:
        out.append(check_cite_pages_any(t, e))
    if e.cite_pages_all:
        out.append(check_cite_pages_all(t, e))
    if e.forbidden_strings:
        out.append(check_forbidden_strings(t, e))
    if e.numbers:
        out.extend(check_numbers(t, e))
    if e.key_points:
        out.append(check_key_points(t, e))
    if e.must_say_insufficient:
        out.extend(check_must_say_insufficient(t, e))
    if e.no_clarify:
        out.append(check_no_clarify(t, e))
    if e.refusal_kind is not None:
        out.append(check_refusal_kind(t, e))
    if e.inherited_topic is not None:
        out.append(check_inherited_topic(t, e))
    # 隐含规则
    if e.intent == "clarify":
        out.append(check_clarify_suggestions(t))
    if e.intent == "write" and e.genre == "faq":
        out.append(check_faq_jsonld(t))
    if (t.options or {}).get("inject_fault") == "v3_swap":
        out.append(check_injected_v3(t))
    for o in out:
        o.turn = t.seq
    return out


# ── 跨轮 / 结构性检测 ────────────────────────────────────────────
def check_session_consistent(run: CaseRun) -> CheckOutcome:
    ids = [t.session_id for t in run.turns]
    ok = bool(ids) and all(ids) and len(set(ids)) == 1
    return CheckOutcome(name="session_consistent", ok=ok, detail=f"session_ids={ids}")


def check_citations_resolvable(t: TurnRun) -> CheckOutcome:
    miss = [s for s in cited_sids(t) if s not in t.citations]
    return CheckOutcome(name="citations_resolvable", ok=not miss, turn=t.seq,
                        detail=f"正文里的 {miss} 在 citations 里找不到" if miss else "")


def check_citations_from_this_turn(t: TurnRun, prev: list[TurnRun]) -> CheckOutcome:
    """引用只能来自本轮检索（10 号文档第 2 节核心原则）。

    fact_id 在本轮 retrieval 事件里 → 通过；不在但所在页本轮被检索到 → 视为整页回填，通过并注明；
    都不是 → 失败，并指出它是否出现在前几轮的引用里（典型的跨轮自我污染）。
    """
    if t.intent == "revise":
        # 改写只动目标块，引用编号沿用上一篇（handlers/revise.py），本轮不检索
        return CheckOutcome(name="citations_from_this_turn", ok=True, turn=t.seq, detail="改写沿用上一篇引用")
    this_ids = set(t.retrieved_fact_ids)
    this_pages = set(t.retrieved_pages)
    prev_ids = {c.get("fact_id") for p in prev for c in p.citations.values()}
    bad, backfill = [], []
    for s in cited_sids(t):
        c = t.citations.get(s)
        if not c:
            continue
        fid = c.get("fact_id")
        if fid in this_ids:
            continue
        if c.get("page_no") is not None and int(c["page_no"]) in this_pages:
            backfill.append(s)
            continue
        bad.append(f"{s}={fid}" + ("（上一轮的引用）" if fid in prev_ids else ""))
    detail = f"不来自本轮检索：{bad}" if bad else (f"页回填：{backfill}" if backfill else "")
    return CheckOutcome(name="citations_from_this_turn", ok=not bad, turn=t.seq, detail=detail)


def check_refusal_has_nearest_pages(t: TurnRun) -> CheckOutcome:
    return CheckOutcome(name="refusal_nearest_pages", ok=bool(t.nearest_pages), turn=t.seq,
                        detail=f"nearest_pages={t.nearest_pages}")


def check_slot_inheritance(case: DialogueCase, run: CaseRun) -> list[CheckOutcome]:
    """槽位继承：
    - topic：本轮期望的 topic 关键词若不在本轮用户原话里，它只能来自前几轮 → 必须出现在 topic 中
    - constraints：前一轮槽位里的约束，后续 write 轮若用户没有改口（不限 / 取消 / 不用），须仍在槽位里
    """
    out: list[CheckOutcome] = []
    prev_constraints: list[str] = []
    for i, (turn_def, t) in enumerate(zip(case.turns, run.turns)):
        e = turn_def.expect
        if i > 0 and e.topic_contains:
            inherited = [w for w in e.topic_contains if _norm(w) not in _norm(turn_def.user)]
            if inherited:
                miss = [w for w in inherited if _norm(w) not in _norm(t.topic or "")]
                out.append(CheckOutcome(name="inherit.topic", ok=not miss, turn=i,
                                        detail=f"应继承 {inherited}，topic={t.topic!r}" + (f" 缺 {miss}" if miss else "")))
        cons = t.slots.get("constraints")
        if isinstance(cons, list):
            if i > 0 and prev_constraints and t.intent == "write" and not re.search(r"不限|取消|不用|去掉", turn_def.user):
                miss = [c for c in prev_constraints if c not in cons]
                out.append(CheckOutcome(name="inherit.constraints", ok=not miss, turn=i,
                                        detail=f"上轮约束 {prev_constraints}，本轮 {cons}" + (f" 丢了 {miss}" if miss else "")))
            if cons:
                prev_constraints = [str(c) for c in cons]
    return out


def check_case(case: DialogueCase, run: CaseRun) -> list[CheckOutcome]:
    """一条用例的全部检测结果（逐轮 + 跨轮），顺序稳定，便于报告与比对。"""
    out: list[CheckOutcome] = []
    if len(run.turns) != len(case.turns):
        out.append(CheckOutcome(name="turn_count", ok=False, detail=f"用例 {len(case.turns)} 轮，实际 {len(run.turns)} 轮"))
    for i, (turn_def, t) in enumerate(zip(case.turns, run.turns)):
        out.extend(check_turn(t, turn_def.expect))
        out.append(check_citations_resolvable(t))
        out.append(check_citations_from_this_turn(t, run.turns[:i]))
        # 越界请求与品宣无关，没有「最接近的页」，只对资料不足类拒答要求 nearest_pages
        if t.refused and t.intent in ("qa", "write") and t.refusal_kind != "out_of_scope":
            out.append(check_refusal_has_nearest_pages(t))
    out.append(check_session_consistent(run))
    out.extend(check_slot_inheritance(case, run))
    return out


def failures(outcomes: Iterable[CheckOutcome]) -> list[CheckOutcome]:
    return [o for o in outcomes if not o.ok]


def format_outcomes(outcomes: Iterable[CheckOutcome]) -> str:
    lines = []
    for o in outcomes:
        mark = "ok " if o.ok else "FAIL"
        where = f"T{o.turn}" if o.turn is not None else "--"
        lines.append(f"[{mark}] {where} {o.name}: {o.detail}")
    return "\n".join(lines)
