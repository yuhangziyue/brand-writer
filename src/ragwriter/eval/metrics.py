"""评估用的确定性指标。

为什么单列一个纯函数模块、并且「能程序判定的都程序判定」：
用模型给模型打分有 generator-verifier gap —— 评审和生成能力相当时，评审往往发现不了生成者的错
（尤其是数字挂错对象这类「看着很顺」的错）。所以数字、引用编号、专名、体裁结构、禁用词这些
**规则写得出来的**判定都在这里做，模型评审（judge.py）只补程序判不了的语义支持度与要点覆盖。

口径对齐 06-quality-metrics.md：
  事实句 = 含数字、专名、或对万悉及行业做出断言的句子；纯过渡句、设问句不算。
  数字一致率 / 数字对象一致率 = 线上 V2 / V3。

数字抽取、数字回查、专名回查、V3 区分词**直接调用 B 组的 `ragwriter.verify.checks`**，
不另写一套：线上校验和离线评估如果口径不同，「评估 100%」就证明不了「线上闸在工作」。
本模块只在它上面加评估特有的东西：按句汇总成比例、要点覆盖、体裁结构、拒答与边界。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from ragwriter.contracts import Sentence as CSentence
from ragwriter.contracts import Snippet as CSnippet
from ragwriter.models import Metric
from ragwriter.verify import checks as vchecks

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FACTS = ROOT / "build" / "facts.jsonl"
DEFAULT_PAGES = ROOT / "build" / "pages.jsonl"
GENRES_DIR = ROOT / "prompts" / "genres"

# 不可引用的状态：image_only 文字层无法核验，unverified 两路解析冲突。被引用就是越界
NON_CITABLE_STATUSES = frozenset({"image_only", "unverified"})
_VALID_STATUSES = {"verified", "image_structural", "image_only", "unverified"}


# ── 语料视图 ──────────────────────────────────────────────────
@dataclass
class Corpus:
    """评估需要的那一小部分建库产物：fact 状态、同页数字配对（V3）、页目录。

    为什么不直接用 citations 里的片段：CaseRun 的 citations 只带 text 与 status，
    没有 metric 配对，而 V3 必须知道「同页的兄弟数字挂的是什么标签」。
    """

    facts: dict[str, dict] = field(default_factory=dict)
    pages: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, facts_path: Path = DEFAULT_FACTS, pages_path: Path = DEFAULT_PAGES) -> "Corpus":
        facts: dict[str, dict] = {}
        pages: dict[int, dict] = {}
        if Path(facts_path).exists():
            for line in Path(facts_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    facts[d["fact_id"]] = d
        if Path(pages_path).exists():
            for line in Path(pages_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    pages[int(d["page_no"])] = {k: d.get(k) for k in ("page_no", "chapter", "title", "text")}
        return cls(facts=facts, pages=pages)

    def page_metrics(self, page_no: int) -> list[Metric]:
        return [Metric(**f["metric"]) for f in self.facts.values()
                if f.get("page_no") == page_no and f.get("metric")]

    def status_of(self, fact_id: str) -> str | None:
        f = self.facts.get(fact_id)
        return f.get("status") if f else None

    def page_text(self, page_no: int) -> str:
        """页目录文本：章节 + 标题 + 页文字层 + 该页全部事实，用于判断「这个建议主题资料里有没有」。"""
        p = self.pages.get(page_no) or {}
        facts = " ".join(f["text"] for f in self.facts.values() if f.get("page_no") == page_no)
        return f"{p.get('chapter', '')} {p.get('title', '')} {p.get('text', '')} {facts}"

    def all_page_nos(self) -> list[int]:
        return sorted(set(self.pages) | {int(f["page_no"]) for f in self.facts.values()})

    def to_snippets(self, citations: dict[str, dict]) -> dict[str, CSnippet]:
        """CaseRun.citations → B 组校验要的 Snippet（补回 metric 配对，V3 要用）。"""
        out: dict[str, CSnippet] = {}
        for sid, c in citations.items():
            fid = str(c.get("fact_id", ""))
            f = self.facts.get(fid) or {}
            status = c.get("status") or f.get("status") or "verified"
            out[sid] = CSnippet(
                sid=sid, fact_id=fid, page_no=int(c.get("page_no") or f.get("page_no") or 0),
                text=str(c.get("text") or f.get("text") or ""),
                status=status if status in _VALID_STATUSES else "verified",
                metric=Metric(**f["metric"]) if f.get("metric") else None,
            )
        return out


@lru_cache(maxsize=1)
def default_corpus() -> Corpus:
    return Corpus.load()


# ── 文本规整与切句 ────────────────────────────────────────────
CITE_RE = re.compile(r"\[\s*([Ss]\d+(?:\s*[,，、;；]\s*[Ss]?\d+)*)\s*\]")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.、)）]\s*|>\s*)")
_SENT_END = "。！？!?；;"
INSUFFICIENT_RE = re.compile(
    r"(品宣)?资料(中|里)?(未涉及|未提及|没有涉及|没有提及|未覆盖|不足|没有相关|没有关于)|"
    r"未在(品宣)?资料中(出现|提及)|无法从(品宣)?资料|资料未(给出|披露|说明)|品宣未(涉及|提及|披露)"
)
# 铺垫式开头：FAQ 答案首句、Blog 导语首句出现它们，就不是「直接回答 / 结论先行」
PREAMBLE_RE = re.compile(
    r"^(首先|在.{0,14}(时代|背景下|今天|当下)|随着|当今|如今|近年来|众所周知|这是一个|关于这个问题|"
    r"这个问题|要回答|我们先|让我们|谈到|说到|提到)"
)


def norm(text: str) -> str:
    """比对用的规整：复用 B 组 canon（NFKC、全角转半角、去空白、千分位），再统一小写。"""
    return vchecks.canon(text).lower()


def strip_cites(text: str) -> str:
    return CITE_RE.sub("", text)


def parse_cites(text: str) -> list[str]:
    out: list[str] = []
    for m in CITE_RE.finditer(text):
        for part in re.split(r"[,，、;；]", m.group(1)):
            part = part.strip().upper()
            if part:
                sid = part if part.startswith("S") else f"S{part}"
                if sid not in out:
                    out.append(sid)
    return out


@dataclass
class Sent:
    text: str  # 含引用标记
    section: int  # 所在小节序号，-1 = 首个二级标题之前（标题 / 导语区）
    cites: list[str]

    @property
    def plain(self) -> str:
        return strip_cites(self.text).strip()

    def as_contract(self, idx: int = 0) -> CSentence:
        return CSentence(idx=idx, text=self.text, cites=self.cites)


def split_sentences(markdown: str) -> list[Sent]:
    """按 Markdown 行与中文句末标点切句；标题行不算句子，但记下句子属于第几节。

    与 B 组 verify.sentences 的差别只有「记节号」：评估要按节看导语、结尾、FAQ 答案首句。
    句末引用写在句号后（「…15倍。[S5]」）时挪回前一句，否则引用会算到下一句头上。
    """
    out: list[Sent] = []
    section = -1
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or set(line) <= set("-*_=| "):
            continue
        h = _HEADING_RE.match(line)
        if h:
            if len(h.group(1)) >= 2:
                section += 1
            continue
        line = _LIST_MARK_RE.sub("", line, count=1)
        line = re.sub(r"\*\*|__", "", line)
        pieces: list[str] = []
        buf = ""
        i = 0
        while i < len(line):
            buf += line[i]
            if line[i] in _SENT_END:
                while i + 1 < len(line) and line[i + 1] in "”’」』）)】》\"'":
                    i += 1
                    buf += line[i]
                pieces.append(buf)
                buf = ""
            i += 1
        if buf.strip():
            pieces.append(buf)
        for p in pieces:
            p = p.strip()
            m = re.match(r"^((?:\[\s*[Ss][^\]]*\]\s*)+)(.*)$", p)
            if m and out and out[-1].section == section:
                out[-1].text += m.group(1).strip()
                out[-1].cites = parse_cites(out[-1].text)
                p = m.group(2).strip()
            if not p or not strip_cites(p).strip(" 。！？!?；;"):
                continue
            out.append(Sent(text=p, section=section, cites=parse_cites(p)))
    return out


def says_insufficient(text: str) -> bool:
    return bool(INSUFFICIENT_RE.search(text))


def is_question(s: Sent | str) -> bool:
    return vchecks.is_question(s.text if isinstance(s, Sent) else s)


# 「据第 5 页图示」「最接近的资料在第 4、1、2 页」：页码是出处 / 导航，不是事实数字。
# 与 B 组 check_v2 的 _drop_page_refs 同一口径，并补上页码列表写法
_PAGE_LIST = re.compile(r"第\s*(\d{1,3}(?:\s*[、,，和与及]\s*\d{1,3})*)\s*页")


def drop_page_refs(text: str, cited_pages: set[int] | None) -> str:
    """cited_pages=None：句子无引用，页码全是导航，全部去掉；否则只去掉指向所引页的（指向别页的照常回查）。"""
    def sub(m: re.Match) -> str:
        pages = {int(x) for x in re.findall(r"\d+", m.group(1))}
        return " " if cited_pages is None or pages <= cited_pages else m.group(0)

    return _PAGE_LIST.sub(sub, text)


def is_fact_sentence(s: Sent) -> bool:
    """06 号的事实句定义 = B 组 V4 的 is_factual，再加一条：带引用的句子一律算事实句。

    加这一条是因为作者自己声明了它有出处，就该接受出处检查；
    设问句与「资料未涉及」声明不算 —— 后者是资料不足处理，单独计分。
    """
    p = s.plain
    if is_question(s) or says_insufficient(p) or len(p) < 4:
        return False
    return bool(s.cites) or vchecks.is_factual(drop_page_refs(p, None))


def numbers_of(text: str) -> list[str]:
    return vchecks.number_tokens(strip_cites(text))


def latin_of(text: str) -> list[str]:
    return vchecks.latin_names(strip_cites(text))


def number_present(value: str, text: str) -> bool:
    """数字写法在文本里出现（带边界：「5倍」不命中「15倍」）。复用 B 组 token_in_text。"""
    return vchecks.token_in_text(value, strip_cites(text))


# ── V2 / V3 ──────────────────────────────────────────────────
@dataclass
class NumberCheck:
    raw: str
    sentence: str
    cited_pages: list[int]
    ok: bool  # V2：出现在所引片段
    reason: str = ""
    object_ok: bool | None = None  # V3：None = 不适用（该数字不是所引页的配对数值）
    object_reason: str = ""


def check_numbers(sent: Sent, citations: dict[str, dict], corpus: Corpus | None = None) -> list[NumberCheck]:
    """逐个数字做 V2（在所引片段原文中）与 V3（挂对标签）。

    V2 逐数字计（数字一致率的分母是数字个数，不是句子数）；V3 调 B 组 check_v3，
    它是句级判定，所以同一句里的配对数字共享 V3 结论。
    例：第 10 页「AI驱动流量增长 8倍 / AI驱动订单增长 15倍」，句子写「AI 驱动流量增长 15 倍」
    ⇒ 15倍 的区分词「订单」缺席 ⇒ 挂错对象。
    """
    corpus = corpus or default_corpus()
    cmap = corpus.to_snippets(citations)
    cited = [cmap[c] for c in sent.cites if c in cmap]
    cited_pages = sorted({s.page_no for s in cited})
    body = drop_page_refs(sent.text, set(cited_pages) if sent.cites else None)
    toks = vchecks.number_tokens_detail(body)
    if not toks:
        return []
    material = "\n".join(vchecks.snippet_material(s) for s in cited)
    other = "\n".join(vchecks.snippet_material(s) for k, s in cmap.items() if k not in sent.cites)
    v3 = vchecks.check_v3(sent.as_contract(), cmap, corpus.page_metrics) if cited else None
    paired_vals = [m.value for p in cited_pages for m in corpus.page_metrics(p) if m.value]
    out: list[NumberCheck] = []
    for t in toks:
        nc = NumberCheck(raw=t.raw, sentence=sent.text, cited_pages=cited_pages, ok=False)
        if not sent.cites:
            nc.reason = "数字所在句没有引用，无从核对"
        elif not cited:
            nc.reason = "引用编号不在本轮映射表里"
        elif vchecks.token_in_text(t, material):
            nc.ok = True
        elif other and vchecks.token_in_text(t, other):
            nc.reason = "数字出现在本轮其他片段，但不在本句所引片段（引错片段）"
        else:
            nc.reason = "所引片段原文中没有这个数字（疑似编造或改写了数字）"
        if nc.ok and v3 is not None and any(vchecks.token_in_text(v, t.raw) for v in paired_vals):
            nc.object_ok = v3.ok
            nc.object_reason = "" if v3.ok else v3.reason
        out.append(nc)
    return out


@dataclass
class SupportCheck:
    """引用支持的确定性部分：数字与拉丁专名都在所引片段中（= B 组 V2）。"""

    ok: bool
    reason: str = ""


def deterministic_support(sent: Sent, citations: dict[str, dict], corpus: Corpus | None = None) -> SupportCheck:
    corpus = corpus or default_corpus()
    cmap = corpus.to_snippets(citations)
    if not any(c in cmap for c in sent.cites):
        return SupportCheck(False, "无有效引用")
    r = vchecks.check_v2(sent.as_contract(), cmap)
    return SupportCheck(r.ok, "" if r.ok else r.reason)


# ── 越界 ─────────────────────────────────────────────────────
def forbidden_hits(text: str, forbidden: Iterable[str]) -> list[str]:
    t = norm(text)
    return [f for f in forbidden if f and norm(f) in t]


def non_citable_cites(text: str, citations: dict[str, dict], corpus: Corpus | None = None) -> list[dict]:
    """正文里实际用到的引用中，状态为 image_only / unverified 的那些。

    状态先看 citations 自带的 status，没有再回语料查：编排层漏填状态不该让越界漏网。
    """
    corpus = corpus or default_corpus()
    out = []
    for sid in sorted(set(parse_cites(text))):
        c = citations.get(sid)
        if not c:
            continue
        status = c.get("status") or corpus.status_of(str(c.get("fact_id", "")))
        if status in NON_CITABLE_STATUSES:
            out.append({"sid": sid, "fact_id": c.get("fact_id"), "page_no": c.get("page_no"), "status": status})
    return out


# ── qa：与参考答案的要素匹配 ──────────────────────────────────
def answer_elements(reference: str, numbers: Iterable[str] = ()) -> list[str]:
    """参考答案的关键要素：用例显式给的 numbers + 参考答案里的数字 + 拉丁专名。

    不用 BLEU / ROUGE：中文短答里「8 倍」和「8倍」、「增长了 8 倍」和「8 倍增长」
    n-gram 重合度差很多但事实一样；反过来「8 倍」写成「15 倍」n-gram 几乎不变，事实全错。
    """
    els: list[str] = []
    for n in list(numbers) + numbers_of(reference or "") + latin_of(reference or ""):
        n = str(n)
        if n and not any(norm(n) == norm(e) for e in els):
            els.append(n)
    return els


def element_match(output: str, elements: list[str]) -> tuple[list[str], list[str]]:
    plain = strip_cites(output)
    hit, miss = [], []
    for e in elements:
        ok = number_present(e, plain) if numbers_of(e) else norm(e) in norm(plain)
        (hit if ok else miss).append(e)
    return hit, miss


def number_object_present(output: str, value: str, obj: str) -> bool:
    """expect.numbers 的 {value, object}：数字出现，且同一句里有它描述的对象关键词。"""
    for s in split_sentences(output):
        if number_present(value, s.plain) and norm(obj) in norm(s.plain):
            return True
    return False


def char_f1(output: str, reference: str) -> float:
    """字符级 F1，**仅供参考**，不进任何核心分数（理由同 answer_elements）。"""
    from collections import Counter

    a = Counter(norm(strip_cites(output)))
    b = Counter(norm(reference))
    common = sum((a & b).values())
    if not common:
        return 0.0
    p, r = common / sum(a.values()), common / sum(b.values())
    return 2 * p * r / (p + r)


# ── 要点覆盖（确定性一档）─────────────────────────────────────
_QWORDS = frozenset("什么 如何 为什么 哪些 怎么 怎样 是否 多少 哪个 哪里 需要 可以 能否".split())


def keywords(text: str) -> list[str]:
    """关键词：数字原样 + jieba 分词后长度 ≥2 的词（复用建库同一个分词器与专名词典）。"""
    from ragwriter.index.tokenize import tokenize

    kws: list[str] = list(numbers_of(text))
    for tok in tokenize(strip_cites(text)):
        if len(tok) >= 2 and not tok.isdigit() and tok not in _QWORDS and tok not in kws:
            kws.append(tok)
    # 搜索模式会同时产出长词和它的子词，留长词即可，避免「品牌价值蒸馏」被算成 4 个命中
    return [k for k in kws if not any(k != o and k in o for o in kws)]


def _kw_hit(k: str, plain: str) -> bool:
    return number_present(k, plain) if numbers_of(k) else norm(k) in norm(plain)


@dataclass
class KeyPointMatch:
    text: str
    page: int | None
    page_cited: bool
    kw_hit: list[str]
    kw_miss: list[str]
    status: str  # covered / uncertain / missed（确定性判定）

    @property
    def kw_ratio(self) -> float:
        tot = len(self.kw_hit) + len(self.kw_miss)
        return len(self.kw_hit) / tot if tot else 0.0


KP_KW_THRESHOLD = 0.5


def match_key_point(output: str, used_citations: dict[str, dict], kp: dict | str) -> KeyPointMatch:
    """要点所在页被引用 且 关键词命中过半 ⇒ 覆盖；两者只占其一 ⇒ 不确定（交模型）；都没有 ⇒ 未覆盖。

    为什么要两个条件同时满足：只看关键词，「AI」「品牌」这类词哪篇都有；
    只看页码，引了第 10 页却只写了 64% 购物意愿，15 倍订单增长这个要点仍然没覆盖。
    """
    if isinstance(kp, str):
        kp = {"text": kp}
    text = str(kp.get("text", ""))
    page = kp.get("page")
    pages = page if isinstance(page, list) else ([page] if page is not None else [])
    cited_pages = {int(c.get("page_no")) for c in used_citations.values() if c.get("page_no") is not None}
    page_cited = bool(pages) and any(int(p) in cited_pages for p in pages)
    plain = strip_cites(output)
    hit, miss = [], []
    for k in keywords(text):
        (hit if _kw_hit(k, plain) else miss).append(k)
    ratio = len(hit) / (len(hit) + len(miss)) if (hit or miss) else 0.0
    # 要点里的数字是硬要求：「AI 驱动流量增长 8 倍」和「订单增长 15 倍」共享一半关键词，
    # 只看比例会把前者算成覆盖了后者
    kw_ok = ratio >= KP_KW_THRESHOLD and not any(numbers_of(k) for k in miss)
    if not pages:
        status = "covered" if ratio >= 0.8 else ("uncertain" if kw_ok else "missed")
    elif page_cited and kw_ok:
        status = "covered"
    elif page_cited or kw_ok:
        status = "uncertain"
    else:
        status = "missed"
    return KeyPointMatch(text=text, page=pages[0] if pages else None, page_cited=page_cited,
                         kw_hit=hit, kw_miss=miss, status=status)


# ── 体裁结构 ─────────────────────────────────────────────────
_ENDING_HEAD_RE = re.compile(r"结语|总结|结论|写在最后|结尾|展望|小结|最后")
_ENDING_CUE_RE = re.compile(r"^(总之|综上|总的来说|归根结底|最后|未来|因此|所以|对于.{0,12}而言)")
_Q_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:Q\s*\d*|问\s*\d*)\s*[:：.、]", re.I)
_A_LINE_RE = re.compile(r"^\s*(?:\*\*)?\s*(?:A\s*\d*|答\s*\d*)\s*[:：.、]\s*(?:\*\*)?", re.I)

# 体裁的形状约定，来源 prompts/planner.md（D 组）：
#   blog          标题 + 导语 + 3–5 小节 + 结尾
#   faq           5–8 组问答，答案第一句直接回答
#   brand_intro   固定四节「是谁 → 做什么 → 凭什么 → 服务谁」
#   product_intro 3–5 节「产品是什么 → 核心能力（1–2 节）→ 效果数据 → 交付与服务方式」
# brand_intro / product_intro 的节目清单优先读 prompts/genres/<genre>.{yaml,yml,json,md}（D 组在写），
# 读不到才用下面的内置表，并在结构检查的 note 里写明「按 planner.md 推断」。
GENRE_SHAPE: dict[str, dict[str, Any]] = {
    "blog": {"title": True, "lede": True, "min_sections": 3, "max_sections": 5, "ending": True},
    "faq": {"min_pairs": 5, "max_pairs": 8},
    "brand_intro": {"title": True, "lede": True, "min_sections": 4, "max_sections": 4, "ending": False},
    "product_intro": {"title": True, "lede": True, "min_sections": 3, "max_sections": 5, "ending": False},
}
BUILTIN_SECTIONS: dict[str, list[dict[str, Any]]] = {
    "brand_intro": [
        {"name": "是谁", "keywords": ["是谁", "定位", "愿景", "团队", "关于", "简介", "创始"]},
        {"name": "做什么", "keywords": ["做什么", "业务", "服务内容", "方法论", "产品", "解决方案"]},
        {"name": "凭什么", "keywords": ["凭什么", "技术", "依据", "优势", "实力", "壁垒"]},
        {"name": "服务谁", "keywords": ["服务谁", "客户", "行业", "案例", "对象", "适用"]},
    ],
    "product_intro": [
        {"name": "产品是什么", "keywords": ["是什么", "定位", "概述", "简介", "产品"]},
        {"name": "核心能力", "keywords": ["能力", "技术", "功能", "方法", "如何", "体系"]},
        {"name": "效果数据", "keywords": ["效果", "数据", "案例", "指标", "成果"]},
        {"name": "交付与服务", "keywords": ["交付", "服务", "合作", "流程"], "optional": True},
    ],
}


def load_genre_sections(genre: str, genres_dir: Path = GENRES_DIR) -> tuple[list[dict[str, Any]], str]:
    """读 D 组的节目清单。返回 (清单, 来源说明)。

    容忍两种写法：yaml/json 的 `sections: [str | {name, keywords?, optional?}]`；
    md 里第一组列表项「- 是谁：品牌定位、愿景…」（冒号后的顿号分隔词当关键词）。
    """
    for ext in ("yaml", "yml", "json", "md"):
        p = genres_dir / f"{genre}.{ext}"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        items: list[dict[str, Any]] = []
        if ext == "md":
            for line in text.splitlines():
                m = re.match(r"^\s*(?:[-*]|\d+[.、])\s*(?:\*\*)?([^：:*（(]+?)(?:\*\*)?\s*[：:](.*)$", line)
                if m:
                    extra = [w.strip(" 。.") for w in re.split(r"[、，,；;]", m.group(2)) if 1 < len(w.strip(" 。.")) <= 8]
                    items.append({"name": m.group(1).strip(), "keywords": [m.group(1).strip(), *extra]})
                elif items and not line.strip():
                    break
        else:
            data = yaml.safe_load(text) if ext != "json" else json.loads(text)
            raw = (data or {}).get("sections", data) if isinstance(data, dict) else data
            for s in raw or []:
                if isinstance(s, str):
                    items.append({"name": s, "keywords": [s]})
                elif isinstance(s, dict):
                    name = str(s.get("name") or s.get("heading") or s.get("id") or "")
                    kws = [str(k) for k in (s.get("keywords") or s.get("aliases") or [])]
                    items.append({"name": name, "keywords": [name, *kws], "optional": bool(s.get("optional"))})
        if items:
            return items, f"prompts/genres/{p.name}"
    return BUILTIN_SECTIONS.get(genre, []), "内置表（按 prompts/planner.md 推断，genres/ 未落地）"


@dataclass
class StructureCheck:
    genre: str
    items: dict[str, bool]
    detail: dict[str, Any]
    note: str = ""

    @property
    def score(self) -> float:
        return sum(self.items.values()) / len(self.items) if self.items else 0.0

    @property
    def ok(self) -> bool:
        return all(self.items.values())


def faq_pairs(markdown: str) -> list[tuple[str, str]]:
    """识别 Q/A 组：「Q1：…」「**问：**…」「## …？」三种问句写法，问句之后到下一问句之间为答。"""
    pairs: list[tuple[str, str]] = []
    q: str | None = None
    ans: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        h = _HEADING_RE.match(line)
        is_q = bool(_Q_LINE_RE.match(line)) or bool(
            h and len(h.group(1)) >= 2 and strip_cites(h.group(2)).strip().endswith(("？", "?"))
        )
        if is_q:
            if q is not None:
                pairs.append((q, "\n".join(ans)))
            q, ans = line, []
            continue
        if h and len(h.group(1)) == 1:
            continue  # 一级标题是 FAQ 总标题
        if q is not None:
            ans.append(_A_LINE_RE.sub("", line))
    if q is not None:
        pairs.append((q, "\n".join(ans)))
    return pairs


def _clean_q(q: str) -> str:
    q = _HEADING_RE.sub(r"\2", q.strip())
    return re.sub(r"^(?:\*\*)?\s*(?:Q\s*\d*|问\s*\d*)\s*[:：.、]\s*(?:\*\*)?", "", q, flags=re.I).strip()


def answer_is_direct(question: str, answer: str) -> tuple[bool, str]:
    """FAQ 答案首句是否直接回答（确定性近似）：不以铺垫开头，且含问句里的核心名词。"""
    sents = split_sentences(answer)
    if not sents:
        return False, "无回答"
    first = sents[0].plain
    if PREAMBLE_RE.match(first):
        return False, f"首句以铺垫开头：{first[:20]}"
    kws = [k for k in keywords(_clean_q(question)) if k not in {"万悉", "万悉科技", "trendee"}]
    if kws and not any(_kw_hit(k, first) for k in kws):
        return False, f"首句不含问句核心词 {kws[:4]}"
    return True, ""


def check_jsonld(jsonld: Any, n_pairs: int) -> tuple[bool, str]:
    """done.data.jsonld 必须是 schema.org FAQPage，Question 数与正文 Q/A 数一致。"""
    if jsonld is None or jsonld == "":
        return False, "缺 jsonld"
    if isinstance(jsonld, str):
        try:
            jsonld = json.loads(jsonld)
        except json.JSONDecodeError:
            return False, "jsonld 不是合法 JSON"
    nodes = jsonld.get("@graph", [jsonld]) if isinstance(jsonld, dict) else jsonld
    page = next((n for n in nodes if isinstance(n, dict) and n.get("@type") == "FAQPage"), None)
    if page is None:
        return False, "jsonld 里没有 @type=FAQPage"
    ctx = str(page.get("@context") or (jsonld.get("@context") if isinstance(jsonld, dict) else "") or "")
    if "schema.org" not in ctx:
        return False, "@context 不是 schema.org"
    qs = page.get("mainEntity") or []
    qs = qs if isinstance(qs, list) else [qs]
    valid = [q for q in qs if isinstance(q, dict) and q.get("@type") == "Question"
             and isinstance(q.get("acceptedAnswer"), dict) and q["acceptedAnswer"].get("text")]
    if len(valid) != len(qs):
        return False, "存在缺 acceptedAnswer.text 的 Question"
    if len(qs) != n_pairs:
        return False, f"jsonld 有 {len(qs)} 个 Question，正文 {n_pairs} 组"
    return True, ""


def _sections(markdown: str) -> list[tuple[str, str]]:
    """[(二级标题, 该节正文)]。"""
    out: list[tuple[str, str]] = []
    cur: tuple[str, list[str]] | None = None
    for raw in markdown.splitlines():
        m = _HEADING_RE.match(raw.strip())
        if m and len(m.group(1)) == 2:
            if cur:
                out.append((cur[0], "\n".join(cur[1])))
            cur = (m.group(2).strip(), [])
        elif cur is not None and raw.strip():
            cur[1].append(raw)
    if cur:
        out.append((cur[0], "\n".join(cur[1])))
    return out


def check_structure(markdown: str, genre: str, *, jsonld: Any = None) -> StructureCheck:
    shape = GENRE_SHAPE.get(genre)
    if shape is None:
        return StructureCheck(genre=genre, items={}, detail={"error": f"未知体裁 {genre}"})
    if genre == "faq":
        pairs = faq_pairs(markdown)
        n = len(pairs)
        answered = [p for p in pairs if strip_cites(p[1]).strip()]
        direct = [(p[0], *answer_is_direct(*p)) for p in answered]
        not_direct = [{"q": _clean_q(q), "why": why} for q, ok, why in direct if not ok]
        jl_ok, jl_why = check_jsonld(jsonld, n)
        items = {
            f"Q/A 组数 {shape['min_pairs']}–{shape['max_pairs']}": shape["min_pairs"] <= n <= shape["max_pairs"],
            "每问都有答": n > 0 and len(answered) == n,
            "答案首句直接回答": bool(direct) and not not_direct,
            "JSON-LD 为 FAQPage 且组数一致": jl_ok,
        }
        return StructureCheck(genre=genre, items=items, detail={
            "pairs": n, "answered": len(answered), "not_direct": not_direct, "jsonld": jl_why})

    lines = [l for l in markdown.splitlines() if l.strip()]
    title = next((m.group(2).strip() for l in lines if (m := _HEADING_RE.match(l.strip())) and len(m.group(1)) == 1), None)
    lede_sents = [s for s in split_sentences(markdown) if s.section == -1]
    secs = _sections(markdown)
    ending_by_head = bool(secs) and bool(_ENDING_HEAD_RE.search(secs[-1][0]))
    body = secs[:-1] if ending_by_head else secs
    items: dict[str, bool] = {}
    detail: dict[str, Any] = {"title": title, "sections": [h for h, _ in body]}
    note = ""
    if shape.get("title"):
        items["有标题"] = bool(title)
    if shape.get("lede"):
        items["有导语"] = bool(lede_sents)
    if genre == "blog":
        first = lede_sents[0].plain if lede_sents else ""
        # 结论先行的确定性近似：导语首句不是铺垫、不是设问
        items["导语结论先行"] = bool(first) and not PREAMBLE_RE.match(first) and not is_question(first)
        detail["lede_first"] = first
    mn, mx = shape.get("min_sections") or 0, shape.get("max_sections")
    items[f"正文小节 {mn}–{mx or '∞'}"] = len(body) >= mn and (mx is None or len(body) <= mx)
    if shape.get("ending"):
        last_line = re.sub(r"\*\*", "", lines[-1].strip()) if lines else ""
        by_cue = bool(lines) and not _HEADING_RE.match(lines[-1].strip()) and bool(_ENDING_CUE_RE.search(last_line))
        items["有结尾"] = ending_by_head or by_cue
    if genre in BUILTIN_SECTIONS:
        spec, src = load_genre_sections(genre)
        note = f"节目清单来源：{src}"
        matched: list[int | None] = []
        for item in spec:
            kws = [norm(k) for k in item.get("keywords") or [item["name"]] if k]
            idx = next((i for i, (h, b) in enumerate(body)
                        if any(k in norm(h + " " + (split_sentences(b)[0].plain if split_sentences(b) else "")) for k in kws)),
                       None)
            matched.append(idx)
        required = [(it, ix) for it, ix in zip(spec, matched) if not it.get("optional")]
        missing = [it["name"] for it, ix in required if ix is None]
        order = [ix for _, ix in required if ix is not None]
        items["节目齐全"] = not missing
        items["节目顺序"] = order == sorted(order)
        detail.update({"section_spec": [it["name"] for it in spec], "missing_sections": missing})
    return StructureCheck(genre=genre, items=items, detail=detail, note=note)


# ── 资料不足处理 ─────────────────────────────────────────────
_GENERIC_TOPIC_WORDS = {"万悉", "万悉科技", "trendee", "品牌", "中国"}


def _related(subtopic: str) -> Callable[[str], bool]:
    kws = [k for k in keywords(subtopic) if k.lower() not in _GENERIC_TOPIC_WORDS] or keywords(subtopic)

    def rel(text: str) -> bool:
        t = norm(text)
        hits = sum(1 for k in kws if norm(k) in t)
        return bool(kws) and hits >= max(1, (len(kws) + 1) // 2)

    return rel


@dataclass
class InsufficientCheck:
    subtopic: str
    # qa：stated（正文明确写出）/ fabricated（疑似编出）/ skipped（静默跳过）
    # write：ok / missing_boundary（boundary 里没有）/ fabricated（正文编了相关数字）/ body_phrase（正文出现不足字样）
    status: str
    evidence: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("stated", "ok")


def check_insufficient_qa(markdown: str, subtopics: list[str]) -> list[InsufficientCheck]:
    """qa：每个子话题要么正文明确写出「资料未涉及」，要么就是编了或跳了。

    判「编出」：子话题关键词出现在一个事实句里、而该句没有不足声明 ——
    比如「万悉的融资」被写成「万悉已完成数千万元融资」。
    """
    sents = split_sentences(markdown)
    out = []
    for st in subtopics:
        rel = _related(st)
        stated = next((s for s in sents if says_insufficient(s.plain) and rel(s.plain)), None)
        if stated is None and sents and all(says_insufficient(s.plain) or not is_fact_sentence(s) for s in sents) \
                and any(says_insufficient(s.plain) for s in sents):
            stated = next(s for s in sents if says_insufficient(s.plain))  # 整答就是一句拒答
        if stated:
            out.append(InsufficientCheck(st, "stated", stated.plain))
            continue
        fab = next((s for s in sents if is_fact_sentence(s) and rel(s.plain)), None)
        out.append(InsufficientCheck(st, "fabricated", fab.text) if fab else InsufficientCheck(st, "skipped"))
    return out


def check_boundary_write(markdown: str, subtopics: list[str], boundary: list[dict]) -> list[InsufficientCheck]:
    """write（2026-10-22 产品评审后的口径）：资料不足不写进正文，而进 done.data.boundary。

    每个期望的不足主题：① 出现在 boundary 里；② 正文没有编出相关数字；
    ③ 正文没有「品宣资料未涉及」类字样（读者看到的是成稿，不该出现系统内部的边界声明）。
    """
    sents = split_sentences(markdown)
    phrase = next((s for s in sents if says_insufficient(s.plain)), None)
    b_text = [str(b.get("topic", "")) + " " + str(b.get("source", "")) for b in boundary]
    out = []
    for st in subtopics:
        rel = _related(st)
        fab = next((s for s in sents if numbers_of(s.plain) and rel(s.plain) and not says_insufficient(s.plain)), None)
        if fab:
            out.append(InsufficientCheck(st, "fabricated", fab.text))
        elif not any(rel(t) for t in b_text):
            out.append(InsufficientCheck(st, "missing_boundary", f"boundary={[b.get('topic') for b in boundary]}"))
        elif phrase:
            out.append(InsufficientCheck(st, "body_phrase", phrase.plain))
        else:
            out.append(InsufficientCheck(st, "ok"))
    return out


# ── clarify：建议主题是否在资料覆盖范围 ───────────────────────
def suggestion_pages(suggestion: str | dict, corpus: Corpus | None = None) -> list[int]:
    """建议主题能在哪些页找到依据。返回空 = 资料里没有，等于给用户挖坑。"""
    corpus = corpus or default_corpus()
    if isinstance(suggestion, dict):
        topic = str(suggestion.get("topic") or suggestion.get("text") or "")
        given = [int(p) for p in suggestion.get("pages") or []]
    else:
        topic, given = str(suggestion), []
    kws = [k for k in keywords(topic) if k.lower() not in _GENERIC_TOPIC_WORDS] or keywords(topic)
    if not kws:
        return []
    need = max(1, round(len(kws) * 0.6))
    found = []
    for p in given or corpus.all_page_nos():
        t = norm(corpus.page_text(p))
        if sum(1 for k in kws if norm(k) in t) >= need:
            found.append(p)
    return found


# ── 检索命中 ─────────────────────────────────────────────────
def _page_from_fact_id(fid: str) -> int | None:
    m = re.match(r"p(\d+)-", fid)
    return int(m.group(1)) if m else None


def retrieval_topk_pages(retrieval_events: list[dict], top_k: int = 5) -> list[list[int]] | None:
    """每个 retrieval 事件前 k 条命中的页码；事件里没有有序命中列表时返回 None（调用方退到并集口径）。"""
    out = []
    for e in retrieval_events:
        hits = e.get("hits") or e.get("results")
        if not isinstance(hits, list):
            return None
        pages = []
        for h in hits[:top_k]:
            if not isinstance(h, dict):
                continue
            p = h.get("page_no") or (h.get("fact") or {}).get("page_no")
            fid = h.get("fact_id") or (h.get("fact") or {}).get("fact_id")
            if p is None and fid:
                p = _page_from_fact_id(str(fid))
            if p is not None:
                pages.append(int(p))
        out.append(pages)
    return out


# ── 拦截可见性（注入故障用例）─────────────────────────────────
@dataclass
class InterceptionCheck:
    v3_fail_seen: bool
    patch_after: bool
    final_clean: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.v3_fail_seen and self.patch_after and self.final_clean


def _is_fail(d: dict) -> bool:
    r = d.get("result", d.get("ok"))
    return r in ("fail", "failed", False) or d.get("ok") is False


def check_interception(events: list[tuple[str, dict]], final_text: str, citations: dict[str, dict],
                       corpus: Corpus | None = None) -> InterceptionCheck:
    """inject_fault 用例：必须先有 V3 失败的 verify 事件，其后有 patch，且成稿里不再有挂错对象的数字。

    这是「校验闸真的在工作」的证据，不是质量分：闸不响，其他指标再好也说明不了不编造。
    """
    # verify 载荷（handlers._verify_payload）：result=pass|fail，failed=["V3", …]；也兼容单检查的 check 字段
    fail_idx = [i for i, (t, d) in enumerate(events)
                if t == "verify" and _is_fail(d)
                and ("V3" in [str(x).upper() for x in d.get("failed") or []] or str(d.get("check", "")).upper() == "V3")]
    patch_idx = [i for i, (t, _) in enumerate(events) if t == "patch"]
    patch_after = bool(fail_idx) and any(p > fail_idx[0] for p in patch_idx)
    bad = [nc for s in split_sentences(final_text) for nc in check_numbers(s, citations, corpus)
           if nc.object_ok is False]
    return InterceptionCheck(
        v3_fail_seen=bool(fail_idx), patch_after=patch_after, final_clean=not bad,
        detail="；".join(f"成稿仍有错配：{nc.sentence}" for nc in bad[:2]),
    )
