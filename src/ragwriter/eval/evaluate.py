"""按意图逐轮评估「输入 → 实际输出 → 期望」，汇总成评估报告。

两项核心分数：
  原文匹配度（faithfulness）  输出是否忠于品宣原文：引用覆盖、引用支持、数字、数字对象、越界、qa 要素
  扩写完整度（completeness）  输出是否覆盖该主题应有的要点、体裁结构是否完整、资料不足是否处理对

为什么核心分数是「子项均值」而子项才是验收口径：06 号文档的三条 100% 硬指标（数字一致、数字对象一致、
陷阱题拒答）不能被其他子项的高分「平均掉」，所以它们另列为硬闸（gates），任何一条不过就在报告首页标红；
核心分数只用于同一套用例下的横向比较（如 E1 RAG vs 全文）。

输入是 F 组 dialogue_runner 的 CaseRun（pydantic 或其 model_dump 的 dict 都行）与用例（DialogueCase 或 dict）。
judge=None 时只算确定性指标，模型项一律标「未评」，不记 0 分 —— 0 分会把「没评」伪装成「评了很差」。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragwriter.eval import metrics as M
from ragwriter.eval.judge import Judge, resolve_judge_model

ROOT = M.ROOT
REPORT_MD = ROOT / "docs" / "eval-report.md"
REPORT_JSON = ROOT / "build" / "eval" / "eval-report.json"
RESULT_START = "<!-- eval:results:start -->"
RESULT_END = "<!-- eval:results:end -->"
INTENTS = ("write", "qa", "out_of_scope", "clarify", "revise", "explain")
REFUSAL_KINDS = ("out_of_scope", "insufficient_all")


# ── 结果结构 ─────────────────────────────────────────────────
class MetricValue(BaseModel):
    value: float | None = None
    num: float = 0
    den: float = 0
    kind: str = "det"  # det 确定性 / model 模型评审 / auto 自动统计（耗时、成本）
    status: str = "ok"  # ok / 未评 / n/a
    detail: str = ""

    def fmt(self) -> str:
        if self.status == "未评":
            return "未评"
        if self.status == "n/a" or self.value is None:
            return "—"
        if self.den and self.kind != "auto":
            return f"{self.value:.2f} ({_n(self.num)}/{_n(self.den)})"
        return f"{self.value:.2f}"


def _n(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def ratio(num: float, den: float, kind: str = "det", detail: str = "") -> MetricValue:
    if not den:
        return MetricValue(num=num, den=den, kind=kind, status="n/a", detail=detail)
    return MetricValue(value=num / den, num=num, den=den, kind=kind, detail=detail)


def unrated(detail: str = "judge=None") -> MetricValue:
    return MetricValue(kind="model", status="未评", detail=detail)


def flag(ok: bool, detail: str = "") -> MetricValue:
    return ratio(1 if ok else 0, 1, detail=detail)


class TurnScore(BaseModel):
    case_id: str
    seq: int
    user: str
    intent_expected: str
    intent_actual: str | None
    genre: str | None = None
    faithfulness: dict[str, MetricValue] = Field(default_factory=dict)
    completeness: dict[str, MetricValue] = Field(default_factory=dict)
    behavior: dict[str, MetricValue] = Field(default_factory=dict)  # 拒答 / 追问 / 解释 / 继承 / 拦截等意图特有项
    reference_only: dict[str, float] = Field(default_factory=dict)  # 仅供参考，不进任何分数
    faith_score: float | None = None
    comp_score: float | None = None
    gates_failed: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    spot_rows: list[dict[str, Any]] = Field(default_factory=list)
    # 汇总 06 号指标用的原始计数
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def intent_ok(self) -> bool:
        return self.intent_actual == self.intent_expected


class CaseScore(BaseModel):
    case_id: str
    title: str = ""
    turns: list[TurnScore] = Field(default_factory=list)


class SuiteReport(BaseModel):
    generated_at: str
    judge_model: str | None = None
    generator_model: str | None = None
    same_model: bool = False
    judge_prompt_hash: str | None = None
    judge_calls: int = 0
    judge_invalid: int = 0
    cases: list[CaseScore] = Field(default_factory=list)
    headline: dict[str, dict[str, Any]] = Field(default_factory=dict)  # 06 号指标总表
    by_intent: dict[str, dict[str, Any]] = Field(default_factory=dict)
    interception: list[dict[str, Any]] = Field(default_factory=list)
    gates_failed: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ── 输入规整 ─────────────────────────────────────────────────
def _dump(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return o


def _turn_view(t: Any) -> dict[str, Any]:
    """TurnRun（或其 dict）→ 评估要的视图；字段缺失按空处理，并从 done 事件回补。"""
    d = dict(_dump(t))
    events = [(_dump(e)["type"], _dump(e).get("data") or {}) for e in d.get("events") or []]
    done = next((data for typ, data in reversed(events) if typ == "done"), {})
    for k in ("final_text", "citations", "refused", "refusal_kind", "boundary", "jsonld",
              "nearest_pages", "suggestions", "usage"):
        if d.get(k) in (None, "", [], {}) and done.get(k) not in (None, ""):
            d[k] = done[k]
    if not d.get("intent"):
        d["intent"] = next((data.get("intent") for typ, data in events if typ == "intent"), None) or done.get("intent")
    d["citations"] = {str(k): dict(_dump(v)) for k, v in (d.get("citations") or {}).items()}
    d["final_text"] = d.get("final_text") or ""
    d["events"] = events
    d["done"] = done
    return d


def _case_turns(case: Any) -> tuple[str, str, list[dict]]:
    c = _dump(case)
    turns = [dict(_dump(t)) for t in c.get("turns") or []]
    if not turns and c.get("expect"):
        turns = [{"user": c.get("user", ""), "expect": c["expect"], "options": c.get("options")}]
    for t in turns:
        t["expect"] = dict(_dump(t.get("expect") or {}))
    return str(c.get("id") or c.get("case_id") or ""), str(c.get("title") or ""), turns


def _draft_text(events: list[tuple[str, dict]]) -> str:
    """改写前的草稿：把 delta 按节拼回去。06 号「引用格式遵从率在改写前统计」要用它。"""
    by_sec: dict[Any, list[str]] = {}
    for typ, d in events:
        if typ == "delta":
            by_sec.setdefault(d.get("section"), []).append(str(d.get("text", "")))
    return "\n\n".join("".join(v) for v in by_sec.values())


def _mean(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


# ── 原文匹配度 ───────────────────────────────────────────────
def _faithfulness(ts: TurnScore, tv: dict, exp: dict, judge: Judge | None, corpus: M.Corpus) -> None:
    text, cits = tv["final_text"], tv["citations"]
    sents = M.split_sentences(text)
    facts = [s for s in sents if M.is_fact_sentence(s)]
    legal = [s for s in facts if s.cites and all(c in cits for c in s.cites)]
    cited = [s for s in facts if any(c in cits for c in s.cites)]
    f = ts.faithfulness
    f["引用覆盖率"] = ratio(len(legal), len(facts), detail="事实句中带合法引用编号的比例（成稿）")

    draft = _draft_text(tv["events"])
    if draft.strip():
        dfacts = [s for s in M.split_sentences(draft) if M.is_fact_sentence(s)]
        dlegal = [s for s in dfacts if s.cites and all(c in cits for c in s.cites)]
        ts.raw["format_pre"] = (len(dlegal), len(dfacts))
    else:
        ts.raw["format_pre"] = None
    ts.raw["format_final"] = (len(legal), len(facts))

    # 引用支持（确定性）
    det_ok = 0
    sup_by_sent: dict[int, M.SupportCheck] = {}
    for i, s in enumerate(cited):
        sc = M.deterministic_support(s, cits, corpus)
        sup_by_sent[i] = sc
        det_ok += sc.ok
        if not sc.ok:
            ts.findings.append(f"引用不支持（确定性）：{s.text}｜{sc.reason}")
    f["引用支持率·确定性"] = ratio(det_ok, len(cited), detail="数字与拉丁专名都在所引片段（= V2）")

    # 数字 / 数字对象
    nchecks = [nc for s in sents if not M.says_insufficient(s.plain) for nc in M.check_numbers(s, cits, corpus)]
    n_ok = sum(nc.ok for nc in nchecks)
    obj = [nc for nc in nchecks if nc.object_ok is not None]
    obj_ok = sum(bool(nc.object_ok) for nc in obj)
    f["数字一致率"] = ratio(n_ok, len(nchecks), detail="= V2，逐数字")
    f["数字对象一致率"] = ratio(obj_ok, len(obj), detail="= V3，只计所引页的配对数值")
    for nc in nchecks:
        if not nc.ok:
            ts.findings.append(f"数字不一致「{nc.raw}」：{nc.sentence}｜{nc.reason}")
        elif nc.object_ok is False:
            ts.findings.append(f"数字挂错对象「{nc.raw}」：{nc.sentence}｜{nc.object_reason}")
    if nchecks and n_ok < len(nchecks):
        ts.gates_failed.append("数字一致率 < 100%")
    if obj and obj_ok < len(obj):
        ts.gates_failed.append("数字对象一致率 < 100%")

    # 越界
    hits = M.forbidden_hits(text, exp.get("forbidden_strings") or [])
    bad_cites = M.non_citable_cites(text, cits, corpus)
    f["越界内容"] = flag(not hits and not bad_cites,
                       detail="；".join([f"禁用串 {hits}"] * bool(hits) + [f"引用了不可引用事实 {bad_cites}"] * bool(bad_cites)))
    if hits:
        ts.findings.append(f"命中 forbidden_strings：{hits}")
    if bad_cites:
        ts.findings.append("引用了 image_only / unverified 事实：" + "、".join(
            f"{b['sid']}={b['fact_id']}({b['status']})" for b in bad_cites))
    if hits or bad_cites:
        ts.gates_failed.append("越界内容")

    # 期望数字（含对象）
    nums = exp.get("numbers") or []
    if nums:
        ok = [n for n in nums if M.number_object_present(text, str(n.get("value")), str(n.get("object", "")))]
        f["期望数字命中"] = ratio(len(ok), len(nums), detail="数字出现且同句含其对象词")

    # qa：参考答案要素
    ref = exp.get("reference_answer") or ""
    if ref:
        els = M.answer_elements(ref, [n.get("value") for n in nums])
        hit, miss = M.element_match(text, els)
        f["参考要素匹配"] = ratio(len(hit), len(els), detail=f"缺 {miss}" if miss else "")
        ts.reference_only["字符级F1(仅供参考)"] = round(M.char_f1(text, ref), 3)

    # 模型：引用召回 / 引用精确（ALCE）
    if judge is None:
        f["引用召回·模型"] = unrated()
        f["引用精确·模型"] = unrated()
    elif facts:
        supported, invalid, prec_num, prec_den = 0.0, 0, 0, 0
        verdicts: dict[int, Any] = {}
        for i, s in enumerate(cited):
            snips = [{"sid": c, **cits[c]} for c in s.cites if c in cits]
            v = judge.support(s.plain, snips)
            verdicts[id(s)] = v
            if not v.evidence_ok:
                invalid += 1
                ts.findings.append(f"评审依据不实，判定作废：{s.text}")
                continue
            supported += 1.0 if v.verdict == "支持" else 0.0
            if v.verdict in ("支持", "部分支持"):
                used = {e.sid for e in v.evidence if e.found}
                cs = [c for c in s.cites if c in cits]
                prec_den += len(cs)
                prec_num += sum(1 for c in cs if c in used)
            if v.verdict != "支持":
                ts.findings.append(f"模型判{v.verdict}：{s.text}｜{v.reason}")
        den = len(facts) - invalid
        f["引用召回·模型"] = ratio(supported, den, kind="model",
                                 detail="事实句中被所引片段整体支持的比例；无引用的事实句计为不支持")
        f["引用精确·模型"] = ratio(prec_num, prec_den, kind="model",
                                 detail="被评审列为依据的引用 / 全部引用（ALCE 近似）")
        ts.raw["support_verdicts"] = {s.text: verdicts[id(s)].verdict for s in cited if id(s) in verdicts}
    else:
        f["引用召回·模型"] = ratio(0, 0, kind="model")
        f["引用精确·模型"] = ratio(0, 0, kind="model")

    # 人工抽检表的行（06 号第 5 节）
    vmap = ts.raw.get("support_verdicts", {})
    for s in facts:
        snip = "；".join(f"[{c}|第{cits[c].get('page_no')}页] {cits[c].get('text', '')}" for c in s.cites if c in cits)
        sc = next((sup_by_sent[i] for i, x in enumerate(cited) if x is s), None)
        ts.spot_rows.append({"sentence": s.text, "pages": sorted({cits[c].get("page_no") for c in s.cites if c in cits}),
                             "snippet": snip, "det": ("通过" if sc and sc.ok else ("无引用" if not sc else "不通过")),
                             "model": vmap.get(s.text, "未评")})
    # 编造候选 = 引用回查不过的句 + 无引用的事实句。只是候选，终判在人工抽检
    ts.raw["fabrication_candidates"] = (len(cited) - det_ok) + (len(facts) - len(cited))

    core = ["引用覆盖率", "引用支持率·确定性", "数字一致率", "数字对象一致率", "越界内容",
            "期望数字命中", "参考要素匹配", "引用召回·模型"]
    ts.faith_score = _mean([f[k].value for k in core if k in f and f[k].status == "ok"])


# ── 扩写完整度 ───────────────────────────────────────────────
def _completeness(ts: TurnScore, tv: dict, exp: dict, judge: Judge | None, corpus: M.Corpus,
                  *, intent: str) -> None:
    text, cits = tv["final_text"], tv["citations"]
    used = {sid: cits[sid] for sid in M.parse_cites(text) if sid in cits}
    c = ts.completeness

    kps = exp.get("key_points") or []
    if kps:
        covered, den, pending = 0.0, 0, 0
        for kp in kps:
            m = M.match_key_point(text, used, kp)
            if m.status == "covered":
                covered += 1; den += 1
            elif m.status == "missed":
                den += 1
                ts.findings.append(f"漏要点（第{m.page}页）：{m.text}")
            elif judge is None:
                pending += 1
            else:
                v = judge.covers(text, kp)
                if v.evidence_ok and v.score is not None:
                    covered += v.score; den += 1
                    if v.score < 1:
                        ts.findings.append(f"要点{v.verdict}（模型，第{m.page}页）：{m.text}｜{v.reason}")
                else:
                    pending += 1
                    ts.findings.append(f"要点覆盖评审依据不实，作废：{m.text}")
        c["要点覆盖率"] = ratio(covered, den, kind="det" if judge is None else "model",
                             detail=f"{pending} 条确定性判不了，未评" if pending else "")
        if pending and not den:
            c["要点覆盖率"] = unrated(f"{pending} 条均需模型判定")

    genre = exp.get("genre") or tv.get("genre")
    if intent in ("write", "revise") and genre:
        st = M.check_structure(text, genre, jsonld=tv.get("jsonld"))
        bad = [k for k, ok in st.items.items() if not ok]
        c["体裁结构"] = ratio(sum(st.items.values()), len(st.items),
                           detail=("缺：" + "、".join(bad) if bad else "") + (f"（{st.note}）" if st.note else ""))
        for k in bad:
            ts.findings.append(f"体裁结构不合格（{genre}）：{k} {st.detail.get('not_direct') or st.detail.get('jsonld') or ''}".rstrip())

    subs = exp.get("must_say_insufficient") or []
    if subs:
        if intent == "write":
            res = M.check_boundary_write(text, subs, tv.get("boundary") or [])
            label = "boundary 有该主题、正文未编数字、正文无「未涉及」字样"
        else:
            res = M.check_insufficient_qa(text, subs)
            label = "正文明确写出资料未涉及"
        c["资料不足处理"] = ratio(sum(r.ok for r in res), len(res), detail=label)
        for r in res:
            if not r.ok:
                ts.findings.append(f"资料不足处理失败「{r.subtopic}」：{r.status} {r.evidence}".rstrip())
    elif intent == "write":
        phrase = next((s.plain for s in M.split_sentences(text) if M.says_insufficient(s.plain)), None)
        if phrase:
            ts.findings.append(f"写作正文出现资料不足字样（应进 boundary）：{phrase}")

    if intent in ("write", "revise") and genre in ("blog", "brand_intro", "product_intro"):
        secs = M._sections(text)
        multi = sum(1 for _, b in secs if len({cits[x].get("page_no") for x in M.parse_cites(b) if x in cits}) >= 2)
        if secs:
            ts.reference_only["跨页引用节占比(仅供参考)"] = round(multi / len(secs), 3)
        if judge is None:
            c["扩写质量·模型"] = unrated()
        elif used:
            v = judge.expansion(text, [{"sid": k, **v} for k, v in used.items()])
            if v.evidence_ok:
                c["扩写质量·模型"] = MetricValue(value=v.score, num=float(v.verdict), den=5, kind="model", detail=v.reason)
                if v.extra.get("new_facts"):
                    ts.findings.append(f"模型指出片段外新事实：{v.extra['new_facts']}")
            else:
                c["扩写质量·模型"] = MetricValue(kind="model", status="n/a", detail="评审依据不实，作废")

    ts.comp_score = _mean([m.value for m in c.values() if m.status == "ok"])


# ── 意图特有行为 ─────────────────────────────────────────────
def _expects_refusal(exp: dict) -> bool:
    return bool(exp.get("refused")) or exp.get("intent") == "out_of_scope" or exp.get("refusal_kind") in REFUSAL_KINDS


def _refusal(ts: TurnScore, tv: dict, exp: dict, corpus: M.Corpus) -> None:
    b = ts.behavior
    text = tv["final_text"]
    refused = bool(tv.get("refused")) or tv.get("refusal_kind") in REFUSAL_KINDS
    b["拒答正确"] = flag(refused)
    want = exp.get("refusal_kind") or ("out_of_scope" if exp.get("intent") == "out_of_scope" else None)
    if want:
        b["拒答理由正确"] = flag(tv.get("refusal_kind") == want, detail=f"期望 {want}，实际 {tv.get('refusal_kind')}")
        if tv.get("refusal_kind") != want:
            ts.findings.append(f"拒答理由错：期望 {want}，实际 {tv.get('refusal_kind')}")
    near = list(tv.get("nearest_pages") or []) or [p for x in tv.get("boundary") or [] for p in x.get("nearest_pages") or []]
    valid = set(corpus.all_page_nos())
    need_near = tv.get("refusal_kind") not in ("out_of_scope", "unsupported")
    if need_near:
        # 越界请求（写代码、问天气）与暂不支持的功能，没有「最接近的页」可给，不考核这一项
        b["给出 nearest_pages"] = flag(bool(near) and all(int(p) in valid for p in near), detail=f"{near}")
    hits = M.forbidden_hits(text, exp.get("forbidden_strings") or [])
    b["未泄露 forbidden"] = flag(not hits, detail=f"{hits}" if hits else "")
    body = [s for s in M.split_sentences(text) if M.is_fact_sentence(s) and M.numbers_of(s.plain)]
    b["未生成正文"] = flag(not body, detail=body[0].text if body else "")
    if not refused:
        ts.findings.append("应拒答而未拒答")
    if need_near and not near:
        ts.findings.append("拒答没给最近页")
    if hits:
        ts.findings.append(f"拒答泄露 forbidden：{hits}")
    ts.raw["trap"] = {"ok": refused}
    if not refused:
        ts.gates_failed.append("陷阱题拒答")


def _clarify(ts: TurnScore, tv: dict, corpus: M.Corpus) -> None:
    b = ts.behavior
    text = tv["final_text"]
    sugg = tv.get("suggestions") or []
    asked = tv.get("intent") == "clarify" and (("？" in text or "?" in text) or bool(sugg))
    b["是否追问"] = flag(asked)
    b["给出建议"] = flag(bool(sugg))
    if sugg:
        cov = [(s, M.suggestion_pages(s, corpus)) for s in sugg]
        b["建议在资料覆盖内"] = ratio(sum(1 for _, p in cov if p), len(cov),
                                detail="；".join(f"{s}→{p[:4]}" for s, p in cov))
        for s, p in cov:
            if not p:
                ts.findings.append(f"追问建议在资料里找不到对应页：{s}")
    body = [s for s in M.split_sentences(text) if M.is_fact_sentence(s) and s.cites]
    b["未生成正文"] = flag(not body)


def _explain(ts: TurnScore, tv: dict, prev: list[dict]) -> None:
    """explain 只读引用映射与校验报告：它提到的页必须是上一篇真引用过的页，否则就是在编依据。"""
    text = tv["final_text"]
    pages = {int(x) for x in re.findall(r"第\s*(\d+)\s*页", text)}
    known = {int(c.get("page_no")) for t in prev for c in t["citations"].values() if c.get("page_no") is not None}
    known |= {int(c.get("page_no")) for c in tv["citations"].values() if c.get("page_no") is not None}
    ok = [p for p in pages if p in known]
    ts.behavior["解释依据一致"] = ratio(len(ok), len(pages), detail=f"提到 {sorted(pages)}，上文引用过 {sorted(known)}")
    for p in sorted(pages - known):
        ts.findings.append(f"解释提到第 {p} 页，但上文从未引用该页")


def _common_behavior(ts: TurnScore, tv: dict, exp: dict, turn_case: dict, case_opts: dict, corpus: M.Corpus) -> None:
    b = ts.behavior
    b["意图正确"] = flag(ts.intent_ok, detail=f"期望 {ts.intent_expected}，实际 {ts.intent_actual}")
    if not ts.intent_ok:
        ts.findings.append(f"意图识别错：期望 {ts.intent_expected}，实际 {ts.intent_actual}")
    if exp.get("inherited_topic"):
        inh = (tv.get("inherited") or {}).get("topic")
        b["主题继承"] = flag(bool(inh), detail=f"inherited.topic={inh!r}")
        if not inh:
            ts.findings.append("应继承上一轮主题而未继承")
    opts = {**(case_opts or {}), **(turn_case.get("options") or {}), **(tv.get("options") or {})}
    if opts.get("inject_fault"):
        ic = M.check_interception(tv["events"], tv["final_text"], tv["citations"], corpus)
        b["拦截可见性"] = flag(ic.ok, detail=f"V3失败事件={ic.v3_fail_seen} 其后patch={ic.patch_after} 成稿干净={ic.final_clean} {ic.detail}")
        ts.raw["interception"] = {"fault": opts["inject_fault"], "v3_fail_seen": ic.v3_fail_seen,
                                  "patch_after": ic.patch_after, "final_clean": ic.final_clean, "detail": ic.detail}
        done = tv.get("done") or {}
        if not ic.ok and done.get("injected") is False:
            # 故障根本没注入（常见原因：该轮被判资料不足直接拒答，没走到写作与校验），要和「注入了却没拦住」分开报
            b["拦截可见性"].detail += f"；故障未注入（done.injected=False，refusal_kind={done.get('refusal_kind')}）"
        if not ic.ok:
            ts.gates_failed.append("拦截可见性")
            ts.findings.append(f"注入故障 {opts['inject_fault']} 未被可见地拦截：{b['拦截可见性'].detail}")


# ── 对外接口 ─────────────────────────────────────────────────
def evaluate_run(case: Any, run: Any, *, judge: Judge | None, corpus: M.Corpus | None = None) -> CaseScore:
    """逐轮打分。按**期望意图**选指标：意图识别错了也按期望的口径评，错误本身记在「意图正确」。"""
    corpus = corpus or M.default_corpus()
    case_id, title, cturns = _case_turns(case)
    r = _dump(run)
    rturns = [_turn_view(t) for t in (r.get("turns") or [r])]
    case_opts = (_dump(case).get("options") or {}) if isinstance(_dump(case), dict) else {}
    cs = CaseScore(case_id=case_id or str(r.get("case_id", "")), title=title or str(r.get("title", "")))
    prev: list[dict] = []
    for i, ct in enumerate(cturns):
        exp = ct["expect"]
        if i >= len(rturns):
            ts = TurnScore(case_id=cs.case_id, seq=i, user=ct.get("user", ""), intent_expected=exp.get("intent", ""),
                           intent_actual=None, findings=["该轮没有运行记录"], gates_failed=["缺运行记录"])
            cs.turns.append(ts)
            continue
        tv = rturns[i]
        intent = exp.get("intent") or ""
        ts = TurnScore(case_id=cs.case_id, seq=i, user=ct.get("user", tv.get("user", "")),
                       intent_expected=intent, intent_actual=tv.get("intent"),
                       genre=exp.get("genre") or tv.get("genre"))
        for e in tv.get("errors") or []:
            ts.findings.append(f"运行错误：{e}")
        _common_behavior(ts, tv, exp, ct, case_opts, corpus)
        if _expects_refusal(exp):
            _refusal(ts, tv, exp, corpus)
            if intent == "qa":
                _completeness(ts, tv, exp, judge, corpus, intent=intent)
        elif intent in ("write", "revise", "qa"):
            _faithfulness(ts, tv, exp, judge, corpus)
            _completeness(ts, tv, exp, judge, corpus, intent=intent)
            if intent == "qa":
                _qa_extra(ts, tv, exp)
        elif intent == "clarify":
            _clarify(ts, tv, corpus)
        elif intent == "explain":
            _explain(ts, tv, prev)
            hits = M.forbidden_hits(tv["final_text"], exp.get("forbidden_strings") or [])
            ts.faithfulness["越界内容"] = flag(not hits, detail=f"{hits}" if hits else "")
        ts.raw["elapsed_ms"] = tv.get("elapsed_ms")
        ts.raw["usage"] = tv.get("usage") or {}
        ts.raw["retrieved_pages"] = tv.get("retrieved_pages") or []
        ts.raw["retrieval_topk"] = M.retrieval_topk_pages([d for t, d in tv["events"] if t == "retrieval"])
        cs.turns.append(ts)
        prev.append(tv)
    return cs


def _qa_extra(ts: TurnScore, tv: dict, exp: dict) -> None:
    """有答案的 qa：误拒与检索命中（06 号「误拒率」「检索命中率」）。"""
    refused = bool(tv.get("refused")) or tv.get("refusal_kind") == "insufficient_all"
    # 期望本身就是「资料不足」的轮次（参考答案以「品宣资料未涉及」开头，或带 must_say_insufficient）
    # 不是「有答案的问题」，拒答是正确行为，不能算误拒（10-22 真跑：D10 / D16#1 / D16#2 被误计）
    ref = str(exp.get("reference_answer") or "")
    expects_gap = bool(exp.get("must_say_insufficient")) or ref.startswith("品宣资料未涉及")
    has_answer = bool(exp.get("key_points") or ref) and not expects_gap
    if has_answer:
        ts.behavior["未误拒"] = flag(not refused)
        ts.raw["false_refusal"] = refused
        if refused:
            ts.findings.append("有答案的问题被判为资料不足（误拒）")
        gold = {int(k["page"]) for k in exp.get("key_points") or [] if k.get("page") is not None}
        if gold:
            topk = M.retrieval_topk_pages([d for t, d in tv["events"] if t == "retrieval"])
            if topk is not None:
                hit = any(gold & set(p) for p in topk)
                caliber = "top5"
            else:
                hit = bool(gold & set(tv.get("retrieved_pages") or []))
                caliber = "命中并集（事件无有序列表）"
            ts.behavior["检索命中"] = flag(hit, detail=f"黄金页 {sorted(gold)}，口径 {caliber}")
            ts.raw["retrieval_hit"] = hit


# ── 汇总 ─────────────────────────────────────────────────────
TARGETS = {
    "检索命中率": (">= 0.80", lambda v: v >= 0.80, "自动"),
    "引用召回": (">= 0.90", lambda v: v >= 0.90, "模型判别"),
    "引用精确": (">= 0.85", lambda v: v >= 0.85, "模型判别"),
    "数字一致率": ("= 100%", lambda v: v >= 1.0, "确定性"),
    "数字对象一致率": ("= 100%", lambda v: v >= 1.0, "确定性"),
    "引用格式遵从率": (">= 95%", lambda v: v >= 0.95, "确定性"),
    "陷阱题拒答率": ("= 100%", lambda v: v >= 1.0, "自动"),
    "误拒率": ("<= 10%", lambda v: v <= 0.10, "自动"),
    "编造数": ("= 0", lambda v: v == 0, "人工"),
    "单篇耗时(s)": ("<= 90", lambda v: v <= 90, "自动"),
    "单篇成本(元)": ("<= 1", lambda v: v <= 1, "自动"),
    "拦截可见性": ("= 100%", lambda v: v >= 1.0, "确定性"),
}


def _micro(turns: list[TurnScore], group: str, name: str) -> MetricValue:
    vals = [getattr(t, group).get(name) for t in turns]
    vals = [v for v in vals if v is not None and v.status == "ok"]
    if not vals:
        anyv = [getattr(t, group).get(name) for t in turns if getattr(t, group).get(name) is not None]
        if anyv and all(v.status == "未评" for v in anyv):
            return unrated()
        return MetricValue(status="n/a")
    if all(v.den for v in vals):
        return ratio(sum(v.num for v in vals), sum(v.den for v in vals), kind=vals[0].kind)
    return MetricValue(value=statistics.mean(v.value for v in vals), kind=vals[0].kind)


def _headline(turns: list[TurnScore], judge: Judge | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def put(name: str, mv: MetricValue, note: str = "") -> None:
        tgt, ok_fn, nature = TARGETS[name]
        passed = None if mv.status != "ok" or mv.value is None else bool(ok_fn(mv.value))
        out[name] = {"value": mv.fmt() if name not in ("编造数", "单篇耗时(s)", "单篇成本(元)") or mv.status != "ok"
                     else (_n(mv.value) if name == "编造数" else f"{mv.value:.1f}" if "耗时" in name else f"{mv.value:.3f}"),
                     "target": tgt, "pass": passed, "nature": nature, "note": note}

    hits = [t.raw["retrieval_hit"] for t in turns if "retrieval_hit" in t.raw]
    put("检索命中率", ratio(sum(hits), len(hits), kind="auto"), "有答案 qa；前 5 命中含黄金页")
    gen = [t for t in turns if t.intent_expected in ("write", "qa", "revise")]
    put("引用召回", _micro(gen, "faithfulness", "引用召回·模型"),
        "" if judge else "judge=None")
    put("引用精确", _micro(gen, "faithfulness", "引用精确·模型"), "" if judge else "judge=None")
    put("数字一致率", _micro(gen, "faithfulness", "数字一致率"))
    put("数字对象一致率", _micro(gen, "faithfulness", "数字对象一致率"))
    pre = [t.raw["format_pre"] for t in gen if t.raw.get("format_pre")]
    if pre:
        put("引用格式遵从率", ratio(sum(a for a, _ in pre), sum(b for _, b in pre)), "改写前（delta 草稿）")
    else:
        fin = [t.raw["format_final"] for t in gen if t.raw.get("format_final")]
        put("引用格式遵从率", ratio(sum(a for a, _ in fin), sum(b for _, b in fin)), "无 delta 草稿，退用成稿口径")
    traps = [t.raw["trap"]["ok"] for t in turns if "trap" in t.raw]
    put("陷阱题拒答率", ratio(sum(traps), len(traps), kind="auto"), "含 out_of_scope 与 insufficient_all")
    fr = [t.raw["false_refusal"] for t in turns if "false_refusal" in t.raw]
    put("误拒率", ratio(sum(fr), len(fr), kind="auto"))
    cand = sum(int(t.raw.get("fabrication_candidates") or 0) for t in gen)
    out["编造数"] = {"value": "人工", "target": "= 0", "pass": None, "nature": "人工",
                     "note": f"确定性候选 {cand} 句（见问题清单与抽检表），终判以人工为准"}
    blog = [t.raw["elapsed_ms"] / 1000 for t in turns
            if t.intent_expected == "write" and t.genre == "blog" and t.raw.get("elapsed_ms")]
    put("单篇耗时(s)", MetricValue(value=max(blog), kind="auto") if blog else MetricValue(status="n/a"),
        f"Blog 最慢一篇，均值 {statistics.mean(blog):.1f}s" if blog else "")
    costs = [float(t.raw["usage"].get(k)) for t in turns if t.intent_expected == "write"
             for k in ("cost_cny", "cost") if isinstance(t.raw.get("usage"), dict) and t.raw["usage"].get(k) is not None][:]
    put("单篇成本(元)", MetricValue(value=max(costs), kind="auto") if costs else MetricValue(status="n/a"),
        "done.usage 无成本字段" if not costs else "")
    ic = [t.raw["interception"] for t in turns if "interception" in t.raw]
    ok = [x["v3_fail_seen"] and x["patch_after"] and x["final_clean"] for x in ic]
    put("拦截可见性", ratio(sum(ok), len(ok)), "inject_fault 用例：V3 失败事件 → patch → 成稿无错配")
    return out


def evaluate_suite(cases: list[Any], runs: list[Any], *, judge: Judge | None,
                   corpus: M.Corpus | None = None) -> SuiteReport:
    corpus = corpus or M.default_corpus()
    by_id = {str(_dump(r).get("case_id")): r for r in runs}
    jm, gm, same = resolve_judge_model()
    rep = SuiteReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        judge_model=judge.model if judge else None,
        generator_model=(judge.generator_model if judge else gm) or gm,
        same_model=bool(judge and judge.same_model),
        judge_prompt_hash=judge.prompt_hash if judge else None,
    )
    for c in cases:
        cid = _case_turns(c)[0]
        r = by_id.get(cid)
        if r is None:
            rep.notes.append(f"用例 {cid} 没有运行记录，跳过")
            continue
        rep.cases.append(evaluate_run(c, r, judge=judge, corpus=corpus))
    turns = [t for c in rep.cases for t in c.turns]
    rep.headline = _headline(turns, judge)
    for intent in INTENTS:
        ts = [t for t in turns if t.intent_expected == intent]
        if not ts:
            continue
        agg: dict[str, Any] = {"turns": len(ts),
                               "原文匹配度": _mean([t.faith_score for t in ts]),
                               "扩写完整度": _mean([t.comp_score for t in ts]),
                               "metrics": {}}
        for group in ("faithfulness", "completeness", "behavior"):
            names = list(dict.fromkeys(n for t in ts for n in getattr(t, group)))
            agg["metrics"][group] = {n: _micro(ts, group, n).model_dump() for n in names}
        rep.by_intent[intent] = agg
    rep.interception = [{"case_id": t.case_id, "seq": t.seq, **t.raw["interception"]}
                        for t in turns if "interception" in t.raw]
    rep.gates_failed = sorted({f"{t.case_id}#{t.seq} {g}" for t in turns for g in t.gates_failed})
    if judge is None:
        rep.notes.append("本次未启用模型评审（judge=None）：引用召回、引用精确、扩写质量与不确定要点均为「未评」，不是 0 分。")
    elif judge.same_model:
        rep.notes.append(f"⚠️ 同模型评审，可信度降级：评审模型 {judge.model} 与生成模型相同（未配置 EVAL_JUDGE_MODEL）。")
    if judge:
        rep.judge_calls, rep.judge_invalid = judge.calls, judge.invalid
    return rep


# ── 渲染 ─────────────────────────────────────────────────────
def _mv(d: dict | MetricValue) -> str:
    return (d if isinstance(d, MetricValue) else MetricValue(**d)).fmt()


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_report(report: SuiteReport) -> str:
    """Markdown 结果部分，包在 eval:results 标记里，写进 docs/eval-report.md 时只替换这一段。"""
    L: list[str] = [RESULT_START, "", "## 评估结果", ""]
    L.append(f"- 运行时间：{report.generated_at}")
    L.append(f"- 生成模型：`{report.generator_model}` · 评审模型：`{report.judge_model or '未启用'}`"
             + (f" · 评审 prompt 哈希 `{report.judge_prompt_hash}` · 评审调用 {report.judge_calls} 次，依据不实作废 {report.judge_invalid} 次"
                if report.judge_model else ""))
    n_turns = sum(len(c.turns) for c in report.cases)
    L.append(f"- 用例 {len(report.cases)} 条 · {n_turns} 轮")
    for n in report.notes:
        L.append(f"- {n}")
    L.append("")

    L += ["### 首页：校验闸是否在工作", ""]
    hl = report.headline.get("拦截可见性", {})
    L.append(f"**拦截可见性 {hl.get('value', '—')}**（目标 {hl.get('target', '= 100%')}）——注入「数字挂错对象」故障后，"
             "必须看到 V3 失败的 verify 事件、其后的 patch，且成稿不含错配。")
    L.append("")
    if report.interception:
        L += ["| 用例 | 故障 | V3 失败事件 | 其后 patch | 成稿无错配 |", "|---|---|---|---|---|"]
        for x in report.interception:
            yn = lambda b: "✅" if b else "❌"  # noqa: E731
            L.append(f"| {x['case_id']}#{x['seq']} | {x['fault']} | {yn(x['v3_fail_seen'])} | {yn(x['patch_after'])} | {yn(x['final_clean'])} |")
    else:
        L.append("_本批用例没有 inject_fault 轮次。_")
    L.append("")
    if report.gates_failed:
        L.append("**硬闸未过（06 号要求 100% 的项）：**")
        L += [f"- ❌ {g}" for g in report.gates_failed]
    else:
        L.append("**硬闸全部通过**：数字一致、数字对象一致、越界、陷阱题拒答、拦截可见性。")
    L.append("")

    L += ["### 06 号指标总表", "", "| 指标 | 值 | 目标 | 达标 | 性质 | 备注 |", "|---|---|---|---|---|---|"]
    for name, h in report.headline.items():
        p = {True: "✅", False: "❌", None: "—"}[h["pass"]]
        L.append(f"| {name} | {h['value']} | {h['target']} | {p} | {h['nature']} | {_esc(h['note'])} |")
    L.append("")

    L += ["### 两项核心分数（按意图）", "",
          "> 核心分数 = 子项均值，只用于横向比较；验收看上表与各意图子项。模型项未评时不计入均值。", "",
          "| 意图 | 轮数 | 原文匹配度 | 扩写完整度 |", "|---|---|---|---|"]
    for intent, a in report.by_intent.items():
        L.append(f"| {intent} | {a['turns']} | {_pct(a['原文匹配度'])} | {_pct(a['扩写完整度'])} |")
    L.append("")

    turns_by_intent: dict[str, list[TurnScore]] = {}
    for c in report.cases:
        for t in c.turns:
            turns_by_intent.setdefault(t.intent_expected, []).append(t)
    for intent in INTENTS:
        ts = turns_by_intent.get(intent)
        if not ts:
            continue
        cols = list(dict.fromkeys(f"{g}:{n}" for t in ts for g in ("faithfulness", "completeness", "behavior")
                                  for n in getattr(t, g)))
        L += [f"### 意图 {intent}", ""]
        L.append("| 轮次 | " + " | ".join(c.split(":", 1)[1] for c in cols) + " | 匹配度 | 完整度 |")
        L.append("|---" * (len(cols) + 3) + "|")
        for t in ts:
            cells = []
            for col in cols:
                g, n = col.split(":", 1)
                mv = getattr(t, g).get(n)
                cells.append(mv.fmt() if mv else "—")
            L.append(f"| {t.case_id}#{t.seq} | " + " | ".join(cells) + f" | {_pct(t.faith_score)} | {_pct(t.comp_score)} |")
        agg = report.by_intent[intent]["metrics"]
        L.append("| **合计** | " + " | ".join(_mv(agg[c.split(':', 1)[0]][c.split(':', 1)[1]]) for c in cols)
                 + f" | {_pct(report.by_intent[intent]['原文匹配度'])} | {_pct(report.by_intent[intent]['扩写完整度'])} |")
        ref = {k for t in ts for k in t.reference_only}
        for k in sorted(ref):
            vals = [t.reference_only[k] for t in ts if k in t.reference_only]
            L.append(f"\n_{k}：均值 {statistics.mean(vals):.3f}（不进分数）_")
        L.append("")

    L += ["### 问题清单", ""]
    any_f = False
    for c in report.cases:
        for t in c.turns:
            if t.findings:
                any_f = True
                L.append(f"- **{t.case_id}#{t.seq}**（{t.intent_expected}）{_esc(t.user)[:40]}")
                L += [f"  - {_esc(f)}" for f in t.findings[:20]]
    if not any_f:
        L.append("_无。_")
    L.append("")

    L += ["### 人工抽检表（写作类前 3 篇，06 号第 5 节）", "",
          "| 轮次 | 句子 | 引用页 | 原文片段 | 确定性 | 模型 | 人工判定 | 备注 |", "|---|---|---|---|---|---|---|---|"]
    picked = [t for c in report.cases for t in c.turns if t.intent_expected == "write" and t.spot_rows][:3]
    for t in picked:
        for r in t.spot_rows:
            L.append(f"| {t.case_id}#{t.seq} | {_esc(r['sentence'])} | {r['pages']} | {_esc(r['snippet'])[:120]} | "
                     f"{r['det']} | {r['model']} |  |  |")
    L += ["", RESULT_END, ""]
    return "\n".join(L)


def compare_reports(reports: dict[str, SuiteReport]) -> str:
    """对照实验（06 号第 4 节 E1–E3）：同一套用例、不同配置的 06 号指标并排。"""
    names = list(next(iter(reports.values())).headline) if reports else []
    L = ["| 指标 | " + " | ".join(reports) + " |", "|---" * (len(reports) + 1) + "|"]
    for n in names:
        L.append(f"| {n} | " + " | ".join(r.headline.get(n, {}).get("value", "—") for r in reports.values()) + " |")
    return "\n".join(L)


def write_report(md: str, path: Path = REPORT_MD) -> None:
    """只替换结果段；方法段与人写的部分原样保留。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if RESULT_START in old and RESULT_END in old:
            head, rest = old.split(RESULT_START, 1)
            tail = rest.split(RESULT_END, 1)[1]
            path.write_text(head + md.strip() + tail, encoding="utf-8")
            return
        path.write_text(old.rstrip() + "\n\n" + md, encoding="utf-8")
        return
    path.write_text("# 评估报告\n\n> 评估方法见 [eval-method.md](eval-method.md)。\n\n" + md, encoding="utf-8")


# ── 命令入口 ─────────────────────────────────────────────────
def _load_runs(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("runs", [data])


def _live_runs(cases: list[Any]) -> list[Any]:
    from ragwriter.agent.orchestrator import Deps, Orchestrator
    from ragwriter.eval.dialogue_runner import run_case

    orch = Orchestrator(Deps.from_env())
    return [run_case(c, orch) for c in cases]


def main_eval(argv: list[str] | None = None) -> int:
    """`ragwriter eval` 的实现。返回码：0 正常；--strict 且有硬闸未过时 1。"""
    ap = argparse.ArgumentParser(prog="ragwriter eval", description="按意图评估多轮对话用例，输出 docs/eval-report.md")
    ap.add_argument("--cases", default=str(ROOT / "eval" / "dialogues"), help="用例 yaml 文件或目录")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--runs", help="已录制的 CaseRun（.json / .jsonl，model_dump 格式）")
    src.add_argument("--live", action="store_true", help="现场跑编排器（需要 rag.db 与模型或回放 fixtures）")
    ap.add_argument("--save-runs", help="--live 时把 CaseRun 另存为 jsonl，便于之后离线重评")
    ap.add_argument("--no-judge", action="store_true", help="只算确定性指标，模型项标「未评」")
    ap.add_argument("--out", default=str(REPORT_MD))
    ap.add_argument("--json", default=str(REPORT_JSON))
    ap.add_argument("--strict", action="store_true", help="硬闸未过时返回 1（给 make eval / CI 用）")
    args = ap.parse_args(argv)

    from ragwriter.eval.dialogue_runner import load_cases

    cases = load_cases(args.cases)
    if args.runs:
        runs: list[Any] = _load_runs(Path(args.runs))
    elif args.live:
        runs = _live_runs(cases)
        if args.save_runs:
            p = Path(args.save_runs)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(_dump(r), ensure_ascii=False, default=str) for r in runs) + "\n",
                         encoding="utf-8")
    else:
        ap.error("需要 --runs 或 --live")
        return 2

    judge: Judge | None = None
    if not args.no_judge:
        try:
            judge = Judge.from_env()
        except Exception as exc:  # noqa: BLE001 — 没 key 就退到确定性评估，并在报告里写明
            print(f"[eval] 模型评审不可用，退到 judge=None：{exc}", file=sys.stderr)

    report = evaluate_suite(cases, runs, judge=judge)
    md = render_report(report)
    write_report(md, Path(args.out))
    jp = Path(args.json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"[eval] {len(report.cases)} 条用例 → {args.out}；硬闸未过 {len(report.gates_failed)} 项")
    return 1 if (args.strict and report.gates_failed) else 0


if __name__ == "__main__":
    sys.exit(main_eval())
