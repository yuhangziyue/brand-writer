"""revise 链路：取上一篇 → 定位目标块 → 按动作改 → 改动块逐句再校验 → 存为新版本。

为什么按「块」改而不是整篇重写（10 号文档 4.1、5.1 节）：用户说「第二节短一点」，别的块已经过了校验、
用户也没意见；整篇重写会让它们重新冒出问题，还要多花一整篇的模型调用。未改动的块原样保留，S 编号不变，
所以旧引用照样能点开、校验报告能对得上。

事实来源只有片段：上一篇的引用表（S编号 → fact_id）按 fact_id 回库取原文重建 Snippet，
**不拿上一篇的正文当事实**（2.4 节的自我污染）。改写提示词里原文与片段一起给，改完照常跑 V1–V5。

动作分流：
- shorten / expand / tone / rewrite：调 prompts/revise.md 流式改写目标块；expand 允许按该节小标题补检索一次
- delete：去掉目标块，不调模型
- add_section：用户要求作子问题检索 → 写作节点写一节 → 校验 → 插入
- retitle：一次非流式调用改标题（不含数字，不承诺正文没有的东西）

事件不新增类型：plan（data.revise=true）→ 逐块 delta / verify / patch / section_final → done，
前端按普通写作轮渲染新版全文。未改动的块只发 section_final。
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from ragwriter.agent.handlers import (
    TurnContext,
    VerifiedBlock,
    applied_corrections,
    citations_for,
    done_event,
    ev,
    nearest_pages,
    note_retrieval,
    parse_cites,
    retrieval_payload,
    split_gaps_fallback,
    stream_text,
    trim_leak,
    verify_and_fix,
)
from ragwriter.contracts import Event, PlanSection, Sentence, Snippet

PROMPT = "revise"
ACTION_NAME = {"shorten": "缩短", "expand": "扩写", "tone": "换语气", "rewrite": "重写", "delete": "删除",
               "add_section": "新增一节", "retitle": "改标题"}
_BLOCK_SPEC = {
    "lede": "这一块是导语：一到两句，结论先行，每句都要带引用。",
    "ending": "这一块是结尾：一到三句，收束全文观点，不引入正文没写过的事实。",
    "body": "这一块是正文的一节：写成一个段落（FAQ 时是对这个问题的回答），每个事实句带引用。",
}
_REWRITE_ACTIONS = frozenset({"shorten", "expand", "tone", "rewrite"})
_CITE_RE = re.compile(r"\[[^\[\]]*?S\d+[^\[\]]*?\]")
_DIGIT = re.compile(r"\d")
EXPAND_SNIPPETS = 8


@dataclass
class Block:
    kind: str  # lede | body | ending
    heading: str = ""
    sentences: list[Sentence] = field(default_factory=list)
    changed: bool = False

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.sentences)


@dataclass
class Article:
    title: str
    lede: Block | None
    sections: list[Block]
    ending: Block | None
    boundary: list[dict[str, Any]]

    def blocks(self) -> list[tuple[int, Block]]:
        """按显示顺序给出 (事件里的 section 编号, 块)：导语 0，正文 1..n，结尾 n+1——与 write 一致。"""
        out: list[tuple[int, Block]] = []
        if self.lede is not None:
            out.append((0, self.lede))
        out += [(i, b) for i, b in enumerate(self.sections, 1)]
        if self.ending is not None:
            out.append((len(self.sections) + 1, self.ending))
        return out


# ── 入口 ──────────────────────────────────────────────────
def handle(ctx: TurnContext) -> Iterator[Event]:
    it = ctx.intent
    target, action = it.revise_target or "all", it.revise_action or "rewrite"
    instruction = it.revise_instruction or ctx.message
    base_id = ctx.slots.get("last_artifact_id")
    art = ctx.memory.get_artifact(base_id) if base_id else None
    if art is None:
        yield from _notice_done(ctx, "找不到上一篇内容，先告诉我想写什么吧。", kind="clarify", reason="no_artifact")
        return

    article = _load_article(art["body"])
    snippets = _rebuild_snippets(ctx, art.get("citations") or {})
    genre = art.get("genre") or "blog"
    revised = {"target": target, "action": action, "instruction": instruction,
               "label": target_label(target, action)}

    # ① 定位目标。找不到就一句话说明并列出现有小节，不动原文
    problem = _check_target(article, target, action)
    if problem:
        yield from _notice_done(ctx, problem, kind="clarify", reason="target_not_found", revised=revised,
                                base_artifact_id=base_id)
        return

    state = _State(ctx=ctx, article=article, snippets=snippets, genre=genre, instruction=instruction,
                   action=action, target=target)

    # ② 需要在 plan 之前定下来的东西：新标题、新增小节（标题与检索），这样 plan 里的大纲就是新版大纲
    new_section: tuple[Block, Any] | None = None
    if action == "retitle":
        title = _retitle(state)
        if title and title != article.title:
            article.title = title
            state.title_changed = True
        else:
            state.notes.append("新标题没有通过检查（为空、含数字或过长），标题保持不变。")
    elif action == "add_section":
        prepared = yield from _prepare_add_section(state)
        if prepared is None:
            return
        new_section = prepared
        pos = int(target.split(":")[1]) if target.startswith("section:") else len(article.sections)
        article.sections.insert(pos, new_section[0])
    elif action == "delete":
        _apply_delete(article, target)
        state.notes.append(f"已删除{target_label(target)}。")

    yield ev("plan", revise=True, title=article.title, genre=genre, topic=art.get("topic"),
             outline=[b.heading for b in article.sections], sub_questions=[], expected_pages=[],
             insufficient=[], lede_points=[], ending_points=[], revise_target=target, revise_action=action,
             revise_instruction=instruction, revise_label=revised["label"], base_artifact_id=base_id,
             targets=_target_names(article, target, action, new_section))

    # ③ 逐块：目标块改写并校验，其余原样定稿
    targets = _target_blocks(article, target, action, new_section)
    for sec_no, block in article.blocks():
        if new_section is not None and block is new_section[0]:
            yield from _write_new_section(state, sec_no, block, new_section[1])
        elif any(block is t for t in targets):
            yield from _rewrite_block(state, sec_no, block)
        if block.sentences:
            yield _section_final(state, sec_no, block)

    # 改写后某块空了（全被删）就已保留原文；导语 / 结尾被删则不再出现
    yield from _finish(state, art, base_id, revised)


# ── 状态 ──────────────────────────────────────────────────
@dataclass
class _State:
    ctx: TurnContext
    article: Article
    snippets: dict[str, Snippet]
    genre: str
    instruction: str
    action: str
    target: str
    verdicts: list = field(default_factory=list)  # 本轮改动块的定稿判定
    rewrites: list = field(default_factory=list)
    deletions: list = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    title_changed: bool = False

    @property
    def next_sid(self) -> int:
        nums = [int(s[1:]) for s in self.snippets if s[1:].isdigit()]
        return (max(nums) if nums else 0) + 1


# ── 载入 ──────────────────────────────────────────────────
def _sentences(raw: Iterable[dict]) -> list[Sentence]:
    out = []
    for x in raw or []:
        s = Sentence(**x)
        if not s.cites:
            s = s.model_copy(update={"cites": parse_cites(s.text)})
        out.append(s)
    return out


def _load_article(body: dict) -> Article:
    """资料不足而跳过的小节不进新版：用户看不到它们，「第 N 节」按看得到的小节数（与 intent.md 约定一致）。"""
    lede = _sentences(body.get("lede") or [])
    ending = _sentences(body.get("ending") or [])
    sections = [Block(kind="body", heading=s.get("heading", ""), sentences=_sentences(s.get("sentences") or []))
                for s in body.get("sections") or [] if not s.get("skipped") and s.get("sentences")]
    return Article(title=body.get("title", ""), lede=Block("lede", sentences=lede) if lede else None,
                   sections=sections, ending=Block("ending", sentences=ending) if ending else None,
                   boundary=list(body.get("boundary") or []))


def _rebuild_snippets(ctx: TurnContext, citations: dict[str, dict]) -> dict[str, Snippet]:
    """按 fact_id 回库取原文重建片段，保持原 S 编号。库里取不到（重建过库）时退回引用表里存的原文。"""
    store = ctx.deps.store
    out: dict[str, Snippet] = {}
    for sid, c in citations.items():
        fact = None
        try:
            fact = store.fact(c["fact_id"]) if store is not None else None
        except Exception:  # noqa: BLE001
            fact = None
        if fact is not None:
            out[sid] = Snippet(sid=sid, fact_id=fact.fact_id, page_no=fact.page_no, text=fact.text,
                               status=fact.status, metric=fact.metric)
        else:
            out[sid] = Snippet(sid=sid, fact_id=str(c.get("fact_id", "")), page_no=int(c.get("page_no") or 0),
                               text=str(c.get("text", "")), status=c.get("status") or "verified")
    return out


# ── 目标 ──────────────────────────────────────────────────
def target_label(target: str, action: str | None = None) -> str:
    if target.startswith("section:"):
        name = f"第 {target.split(':')[1]} 节"
    else:
        name = {"lede": "导语", "ending": "结尾", "title": "标题", "all": "全文"}.get(target, target)
    if action == "add_section":
        name = f"第 {target.split(':')[1]} 节之后" if target.startswith("section:") else "文末"
    return f"{name} · {ACTION_NAME.get(action, action)}" if action else name


def _check_target(article: Article, target: str, action: str) -> str:
    headings = "、".join(f"「{b.heading}」" for b in article.sections) or "（没有正文小节）"
    if target.startswith("section:"):
        n = int(target.split(":")[1])
        if n > len(article.sections):
            return f"上一篇只有 {len(article.sections)} 节：{headings}，没有第 {n} 节。想改哪一节？"
        if action == "delete" and len(article.sections) == 1:
            return "上一篇只有这一节正文，删掉就没有内容了。可以改短，或者重新写一篇。"
    if target == "lede" and article.lede is None and action != "add_section":
        return "上一篇没有导语。可以改正文小节或结尾。"
    if target == "ending" and article.ending is None and action != "add_section":
        return "上一篇没有结尾。可以改正文小节或导语。"
    if action == "delete" and target in ("all", "title"):
        return "标题和整篇不能删除。想删哪一节？现有小节：" + headings + "。"
    return ""


def _target_blocks(article: Article, target: str, action: str, new_section) -> list[Block]:
    if action not in _REWRITE_ACTIONS:
        return []
    if target.startswith("section:"):
        return [article.sections[int(target.split(":")[1]) - 1]]
    if target == "lede":
        return [article.lede] if article.lede else []
    if target == "ending":
        return [article.ending] if article.ending else []
    return [b for _, b in article.blocks()]


def _target_names(article: Article, target: str, action: str, new_section) -> list[str]:
    if action == "retitle":
        return ["标题"]
    if action == "add_section" and new_section is not None:
        return [f"新增：{new_section[0].heading}"]
    if action == "delete":
        return [target_label(target)]
    names = {"lede": "导语", "ending": "结尾"}
    return [names.get(b.kind) or b.heading for b in _target_blocks(article, target, action, new_section)]


def _apply_delete(article: Article, target: str) -> None:
    if target.startswith("section:"):
        del article.sections[int(target.split(":")[1]) - 1]
    elif target == "lede":
        article.lede = None
    elif target == "ending":
        article.ending = None


# ── 提示词 ────────────────────────────────────────────────
def _prompts():
    from ragwriter.generate import prompts

    return prompts


def _render_snippets(ctx: TurnContext, snippets: Iterable[Snippet]) -> str:
    """按页分组、每组前带页标题，与写作节点的上下文版式一致：模型要知道每条片段在讲什么。"""
    from ragwriter.generate.context import render_snippet

    titles = {c["page_no"]: str(c.get("title", "")) for c in ctx.profile.get("catalog", [])}
    lines: list[str] = []
    last = None
    for s in sorted(snippets, key=lambda s: (s.page_no, int(s.sid[1:]) if s.sid[1:].isdigit() else 0)):
        if s.page_no != last and titles.get(s.page_no):
            lines.append(f"### 第{s.page_no}页《{titles[s.page_no]}》")
        last = s.page_no
        lines.append(render_snippet(s))
    return "\n".join(lines) or "（这一块没有可引用的片段：只能调整措辞与语气，不得出现任何事实、数字或专名）"


def _outline_text(article: Article) -> str:
    return "\n".join(f"{i}. {b.heading}" for i, b in enumerate(article.sections, 1)) or "（无）"


def _strip_cites(text: str) -> str:
    return _CITE_RE.sub("", text)


def _clean_line(raw: str) -> str:
    text = re.sub(r"```\w*", "", raw or "").strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(#+\s*|新?标题[:：]\s*|小标题[:：]\s*)", "", line)
    return line.strip("「」“”\"'《》 ")


# ── retitle ───────────────────────────────────────────────
def _retitle(st: _State) -> str:
    """标题不进逐句校验（它没有引用），所以由程序拦两条：不含原标题没有的数字、不超过 40 字。"""
    ctx, a = st.ctx, st.article
    p = _prompts()
    lede = _strip_cites(a.lede.text) if a.lede else "（无）"
    system = p.render(PROMPT, "System（标题）", genre=st.genre)
    user = p.render(PROMPT, "User（标题）", title=a.title, outline=_outline_text(a), lede=lede,
                    instruction=st.instruction)
    try:
        raw = ctx.deps.client.complete(system, user, max_tokens=80, tag="revise_title")
    except Exception as exc:  # noqa: BLE001
        st.notes.append(f"改标题调用失败：{exc}")
        return ""
    title = _clean_line(raw)
    new_digits = set(_DIGIT.findall(title)) - set(_DIGIT.findall(a.title))
    if not title or new_digits or len(title) > 40:
        return ""
    st.changes.append({"block": "标题", "before": a.title, "after": title})
    return title


# ── add_section ───────────────────────────────────────────
def _prepare_add_section(st: _State):
    """检索 → 资料不足就停（说明「未新增」）→ 足够则起小标题。返回 (新块, 片段包) 或 None。"""
    ctx, d = st.ctx, st.ctx.deps
    query = st.instruction
    result = note_retrieval(ctx, d.search(d.store, query, embedder=d.embedder))
    if not result.sufficient:
        yield ev("retrieval", **retrieval_payload(result, section=None, kind="body"))
        near = nearest_pages([result])
        yield ev("insufficient", section=None, topic=query, signal=result.signal or "below_threshold",
                 source="retrieval", nearest_pages=near)
        text = f"品宣资料不足以支撑「{query}」，未新增小节，原文保持不变。"
        yield from _notice_done(ctx, text, kind="insufficient_all", reason="add_section_insufficient",
                                nearest=near, revised={"target": st.target, "action": st.action,
                                                       "instruction": st.instruction,
                                                       "label": target_label(st.target, st.action)},
                                base_artifact_id=ctx.slots.get("last_artifact_id"))
        return None
    pack = d.build_context(query, result, ctx.page_facts, sid_start=st.next_sid,
                           max_snippets=int(ctx.options.get("max_snippets", 14)))
    yield ev("retrieval", **retrieval_payload(result, section=None, kind="body",
                                              sids={s.sid: s.fact_id for s in pack.snippets}),
             snippets=[{"sid": s.sid, "fact_id": s.fact_id, "page_no": s.page_no} for s in pack.snippets])
    heading = _new_heading(st, pack.snippets)
    return Block(kind="body", heading=heading, changed=True), pack


def _new_heading(st: _State, snippets: list[Snippet]) -> str:
    ctx, a = st.ctx, st.article
    fallback = re.sub(r"^(讲|写|介绍|关于|说说)", "", st.instruction).strip("，。 ")[:16] or "补充"
    p = _prompts()
    try:
        raw = ctx.deps.client.complete(
            p.render(PROMPT, "System（新增小节标题）"),
            p.render(PROMPT, "User（新增小节标题）", title=a.title, outline=_outline_text(a),
                     instruction=st.instruction, snippets=_render_snippets(ctx, snippets[:8])),
            max_tokens=60, tag="revise_heading")
    except Exception:  # noqa: BLE001 — 起标题失败不该拦住新增；用指令要点当标题
        return fallback
    heading = _clean_line(raw)
    if not heading or _DIGIT.search(heading) or len(heading) > 24:
        return fallback
    return heading


def _kw_supported(fn: Any, **kw: Any) -> dict[str, Any]:
    """只传对方签名里有的关键字参数：写作节点的新参数（outline、prev_headings）由另一人在加，旧签名也要能跑。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def _write_new_section(st: _State, sec_no: int, block: Block, pack: Any) -> Iterator[Event]:
    ctx, d = st.ctx, st.ctx.deps
    split_gaps = getattr(d, "split_gaps", None) or split_gaps_fallback
    _prime_seen(ctx, [b for b in st.article.sections if b is not block])
    headings = [b.heading for b in st.article.sections]
    sec = PlanSection(heading=block.heading, sub_question=st.instruction, purpose=f"用户要求新增：{st.instruction}")
    constraints = tuple(ctx.slots.get("constraints") or [])
    extra = _kw_supported(d.stream_section, outline=tuple(headings), prev_headings=tuple(headings[:sec_no - 1]))
    chunks = d.stream_section(d.client, sec, pack, genre=st.genre, profile=ctx.profile, constraints=constraints,
                              **extra)
    raw = yield from stream_text(ctx, chunks, section=sec_no, kind="body")
    draft, gaps = split_gaps(raw)
    yield from trim_leak(ctx, draft, section=sec_no, kind="body")
    _note_gaps(st, gaps, sec_no, block.heading, pack.snippets)
    vb: VerifiedBlock = yield from verify_and_fix(ctx, draft, section=sec_no, kind="body", snippets=pack.snippets)
    _collect(st, vb)
    for s in pack.snippets:
        st.snippets.setdefault(s.sid, s)
    block.sentences = vb.sentences
    if not vb.sentences:
        st.notes.append(f"新增的「{block.heading}」没有句子通过校验，未新增。")
        st.article.sections.remove(block)
        return
    st.changes.append({"block": f"新增：{block.heading}", "before_chars": 0,
                       "after_chars": len(_strip_cites(block.text))})


def _prime_seen(ctx: TurnContext, blocks: list[Block]) -> None:
    """跨节复读检测要知道已有各节说过什么：新增一节最容易把前文的数字再说一遍。"""
    try:
        from ragwriter.verify.checks import _bigrams
        from ragwriter.verify.sentences import strip_cites
    except Exception:  # noqa: BLE001
        return
    for b in blocks:
        for s in b.sentences:
            if s.cites:
                ctx._seen_body.append((set(s.cites), _bigrams(strip_cites(s.text))))


# ── shorten / expand / tone / rewrite ─────────────────────
def _block_name(block: Block, sec_no: int) -> str:
    if block.kind == "lede":
        return "导语"
    if block.kind == "ending":
        return "结尾"
    return f"第 {sec_no} 节《{block.heading}》"


def _rewrite_block(st: _State, sec_no: int, block: Block) -> Iterator[Event]:
    ctx, d = st.ctx, st.ctx.deps
    split_gaps = getattr(d, "split_gaps", None) or split_gaps_fallback
    cited = [st.snippets[c] for s in block.sentences for c in s.cites if c in st.snippets]
    allowed: dict[str, Snippet] = {s.sid: s for s in cited}
    if block.kind != "body" and not allowed:
        # 导语 / 结尾原本没引用：给全文已引片段，让它能把论断落到出处上（写作链路的导语同样看全文片段）
        allowed = dict(list(st.snippets.items())[:20])

    if st.action == "expand":
        yield from _expand_retrieval(st, sec_no, block, allowed)

    p = _prompts()
    name = _block_name(block, sec_no)
    before = len(_strip_cites(block.text))
    system = p.render(PROMPT, "System", block_name=name, genre=st.genre, block_spec=_BLOCK_SPEC[block.kind],
                      profile_rules=p.profile_rules_text(ctx.profile))
    user = p.render(PROMPT, "User", title=st.article.title, outline=_outline_text(st.article), block_name=name,
                    original=block.text or "（空）", snippets=_render_snippets(ctx, allowed.values()),
                    action_name=ACTION_NAME[st.action], instruction=_with_length(st.instruction, st.action, before))
    kind = block.kind
    try:
        chunks = d.client.stream(system, user, max_tokens=1200, tag=f"revise_{kind}")
        raw = yield from stream_text(ctx, chunks, section=sec_no, kind=kind)
    except Exception as exc:  # noqa: BLE001 — 一块改失败不拖垮整轮，保留原文
        yield ev("error", stage="revise", reason=str(exc), section=sec_no, fallback=True)
        st.notes.append(f"{name}改写失败，保留原文。")
        return
    draft, gaps = split_gaps(raw)
    yield from trim_leak(ctx, draft, section=sec_no, kind=kind)
    if not draft.strip():
        st.notes.append(f"{name}改写没有产出内容，保留原文。")
        return
    if st.action == "shorten" and not _DIGIT.search(st.instruction):
        draft = yield from _shorten_again(st, sec_no, block, draft, before, system, allowed)
    if st.action == "expand":
        _note_gaps(st, gaps, sec_no, block.heading or name, list(allowed.values()))
    verify_snips = list(allowed.values()) if kind == "body" else list({**st.snippets, **allowed}.values())
    vb: VerifiedBlock = yield from verify_and_fix(ctx, draft, section=sec_no, kind=kind, snippets=verify_snips)
    _collect(st, vb)
    if not vb.sentences:
        st.notes.append(f"{name}改写后没有句子通过校验，保留原文。")
        return
    for s in allowed.values():
        st.snippets.setdefault(s.sid, s)
    block.sentences = vb.sentences
    block.changed = True
    st.changes.append({"block": name, "before_chars": before, "after_chars": len(_strip_cites(block.text))})


SHORTEN_MAX_RATIO = 0.8  # 改短后仍超过原文 80%，视为「没怎么短」，再压一次


def _shorten_again(st: _State, sec_no: int, block: Block, draft: str, before: int, system: str,
                   allowed: dict[str, Snippet]) -> Iterator[Event]:
    """「短一点」只压了一成时再压一次。10-23 实跑两次：给了字数目标，模型仍只压 10%–16%。

    第二遍不流式（读者不必看两遍草稿），压完用 patch(action=trim) 整块替换前端已显示的草稿，再进逐句校验。
    只在用户没给字数时做：用户给了字数，以用户为准。
    """
    ctx, d = st.ctx, st.ctx.deps
    now = len(_strip_cites(draft))
    if now <= before * SHORTEN_MAX_RATIO:
        return draft
    goal = max(40, before * 55 // 100)
    p = _prompts()
    user = p.render(PROMPT, "User", title=st.article.title, outline=_outline_text(st.article),
                    block_name=_block_name(block, sec_no), original=draft,
                    snippets=_render_snippets(ctx, allowed.values()), action_name=ACTION_NAME["shorten"],
                    instruction=(f"上一版仍有约 {now} 字，用户要的是明显变短：必须压到 {goal} 字以内。"
                                 "只保留这一节最核心的论点和一到两个最关键的事实，其余句子整句删掉"))
    try:
        raw = "".join(d.client.stream(system, user, max_tokens=800, tag=f"revise_{block.kind}"))
    except Exception:  # noqa: BLE001 — 二次压缩失败就用第一版
        return draft
    split_gaps = getattr(d, "split_gaps", None) or split_gaps_fallback
    second, _ = split_gaps(raw)
    if not second.strip() or len(_strip_cites(second)) >= now:
        return draft
    yield ev("patch", section=sec_no, kind=block.kind, sentence=None, action="trim", text=second, original=draft,
             reason=f"篇幅 {now} 字仍超出目标，二次压缩到 {len(_strip_cites(second))} 字")
    return second


def _with_length(instruction: str, action: str, before: int) -> str:
    """用户没给字数时补一个篇幅目标。10-23 实跑：只说「短一点」，模型只压了 16%（522→437 字），读者几乎看不出变化。"""
    if _DIGIT.search(instruction):
        return instruction
    if action == "shorten":
        return f"{instruction}（原文约 {before} 字，改后控制在 {max(40, before * 55 // 100)} 字左右）"
    if action == "expand":
        return f"{instruction}（原文约 {before} 字，改后约 {before * 3 // 2} 字，新增内容只能来自片段）"
    return instruction


def _expand_retrieval(st: _State, sec_no: int, block: Block, allowed: dict[str, Snippet]) -> Iterator[Event]:
    """扩写允许按该节小标题补检索一次。同一 fact 已有编号就沿用旧编号，新 fact 从现有最大编号之后续接。"""
    ctx, d = st.ctx, st.ctx.deps
    query = block.heading or st.article.title
    try:
        result = note_retrieval(ctx, d.search(d.store, query, embedder=d.embedder))
    except Exception as exc:  # noqa: BLE001
        yield ev("error", stage="retrieval", reason=str(exc), section=sec_no, fallback=True)
        return
    if not result.sufficient:
        yield ev("retrieval", **retrieval_payload(result, section=sec_no, kind=block.kind))
        return
    pack = d.build_context(query, result, ctx.page_facts, sid_start=st.next_sid, max_snippets=EXPAND_SNIPPETS)
    by_fact = {s.fact_id: s for s in st.snippets.values()}
    added: list[Snippet] = []
    for s in pack.snippets:
        old = by_fact.get(s.fact_id)
        if old is not None:
            allowed.setdefault(old.sid, old)
            continue
        allowed[s.sid] = s
        added.append(s)
    sids = {s.sid: s.fact_id for s in allowed.values()}
    yield ev("retrieval", **retrieval_payload(result, section=sec_no, kind=block.kind, sids=sids),
             snippets=[{"sid": s.sid, "fact_id": s.fact_id, "page_no": s.page_no} for s in allowed.values()])
    for s in added:  # 占住编号，后面的块续接时不撞号
        st.snippets.setdefault(s.sid, s)


# ── 汇总 ──────────────────────────────────────────────────
def _collect(st: _State, vb: VerifiedBlock) -> None:
    st.verdicts += vb.verdicts
    st.rewrites += vb.rewrites
    st.deletions += vb.deletions


def _note_gaps(st: _State, gaps: list[str], sec_no: int, heading: str, snippets: list[Snippet]) -> None:
    pages = sorted({s.page_no for s in snippets})[:3]
    for g in gaps:
        st.gaps.append({"topic": g, "source": "writer", "nearest_pages": pages, "section": sec_no,
                        "heading": heading})


def _section_final(st: _State, sec_no: int, block: Block) -> Event:
    return ev("section_final", section=sec_no, kind=block.kind, heading=block.heading, text=block.text,
              sentences=[s.model_dump() for s in block.sentences],
              citations=citations_for(block.sentences, st.snippets), skipped=False, changed=block.changed)


def _silent_verdicts(st: _State) -> list:
    """未改动的句子也重跑一遍校验（不发事件）：新版的校验报告要覆盖整篇，而不只是改过的块。

    确定性规则，不调模型，成本可以忽略；万一某句在新版里不再成立（库重建过），报告会如实反映。
    """
    ctx, d = st.ctx, st.ctx.deps
    titles = {c["page_no"]: str(c.get("title", "")) for c in ctx.profile.get("catalog", [])}
    by_sid = {s.sid: s.model_copy(update={"text": f"{titles[s.page_no]}\n{s.text}"})
              if titles.get(s.page_no) else s for s in st.snippets.values()}
    out = []
    for _, b in st.article.blocks():
        if b.changed:
            continue
        for s in b.sentences:
            try:
                out.append(d.verify_sentence(s, by_sid, ctx.page_metrics, strict_cite=(b.kind == "lede"),
                                             case_pages=ctx.case_pages))
            except Exception:  # noqa: BLE001
                continue
    return out


def _finish(st: _State, art: dict, base_id: str, revised: dict) -> Iterator[Event]:
    ctx, d, a = st.ctx, st.ctx.deps, st.article
    for note in st.notes:
        yield ev("delta", section=0, kind="notice", text=note + "\n")
    changed = st.title_changed or st.action == "delete" or any(b.changed for _, b in a.blocks())

    parts = [f"# {a.title}"]
    if a.lede and a.lede.text:
        parts.append(a.lede.text)
    for b in a.sections:
        if b.text:
            # 版式与 write 一致：FAQ 是问答对，其余是二级小标题
            parts.append(f"**问：{b.heading}**\n\n答：{b.text}" if st.genre == "faq" else f"## {b.heading}\n\n{b.text}")
    if a.ending and a.ending.text:
        parts.append(a.ending.text)
    final_text = "\n\n".join(parts)

    all_sentences = [s for _, b in a.blocks() for s in b.sentences]
    citations = citations_for(all_sentences, st.snippets)
    cited = {sid: st.snippets[sid] for sid in citations}
    report = d.build_report([*_silent_verdicts(st), *st.verdicts], rewrites=st.rewrites, deletions=st.deletions,
                            insufficient=[{"topic": g["topic"], "signal": "writer"} for g in st.gaps],
                            corrections=applied_corrections(ctx.profile, final_text, cited.values()),
                            citation_map=cited)
    boundary = [*a.boundary, *st.gaps]
    revised = {**revised, "changed": changed, "changes": st.changes, "notes": st.notes}

    outline = []
    for b in a.sections:
        pages = sorted({st.snippets[c].page_no for s in b.sentences for c in s.cites if c in st.snippets})
        outline.append(b.heading + (f"（第{'、'.join(map(str, pages))}页）" if pages else ""))
    artifact_id = base_id
    if changed:
        body = {"title": a.title,
                "lede": [s.model_dump() for s in a.lede.sentences] if a.lede else [],
                "sections": [{"heading": b.heading, "sentences": [s.model_dump() for s in b.sentences],
                              "skipped": False} for b in a.sections],
                "ending": [s.model_dump() for s in a.ending.sentences] if a.ending else [],
                "boundary": boundary, "base_artifact_id": base_id, "revised": revised}
        artifact_id = ctx.memory.save_artifact(session_id=ctx.session_id, turn_id=ctx.turn_id,
                                               genre=art.get("genre") or st.genre, topic=art.get("topic") or "",
                                               body=body, citations=citations, verify=report)
    ref_pages = sorted({c["page_no"] for c in citations.values()})
    # kind=write：槽位 last_artifact_id / outline 指向新版本，下一次「再改短一点」改的是这一版
    ctx.outcome.update(kind="write", artifact_id=artifact_id, outline=outline, pages=ref_pages,
                       final_text=final_text)
    yield done_event(ctx, final_text=final_text, artifact_id=artifact_id, citations=citations, report=report,
                     refusal_kind="partial" if boundary else None, boundary=boundary, title=a.title,
                     genre=art.get("genre") or st.genre, topic=art.get("topic"), revised=revised,
                     base_artifact_id=base_id,
                     summary={"sentences": len(st.verdicts), "rewrites": len(st.rewrites),
                              "deletions": len(st.deletions), "boundary": len(boundary)})


def _notice_done(ctx: TurnContext, text: str, *, kind: str, reason: str, nearest: list[int] | None = None,
                 **extra: Any) -> Iterator[Event]:
    """不改动原文的一句话收口。槽位不动：last_artifact_id 仍指向原文，用户可以接着改。"""
    yield ev("delta", section=0, kind="notice", text=text)
    ctx.outcome.update(kind="refuse", outline=[f"revise：{reason}"], pages=nearest or [], final_text=text)
    yield done_event(ctx, final_text=text, refused=True, refusal_kind=kind, reason=reason, nearest=nearest,
                     **extra)
