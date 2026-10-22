"""句级确定性校验 V1–V4。

模型在写作阶段最常见的三种错，恰好对应三道确定性闸：
  - 编引用：写了 [S9]，本次上下文里根本没有 S9                       → V1
  - 编数字：片段里是 61.66 亿，句子写成 62 亿；或凭记忆补一个专名      → V2
  - 挂错对象：数字是真的，但描述的是同页的另一个对象（8 倍↔15 倍对调）→ V3
另外还有一种是「该带引用却没带」，事实句裸奔                          → V4

这四道都不调模型：可复现、零成本、能在答辩时逐条解释。
V5（判别模型判支持度）是 P2，不在这里。

每条 CheckResult.reason 会原样显示在前端与校验报告里，所以写给人看，
要说清楚「哪个数字 / 哪个专名 / 哪一页 / 缺了什么」。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from ragwriter.contracts import CheckResult, Sentence, SentenceVerdict, Snippet
from ragwriter.ingest.crosscheck import normalize
from ragwriter.models import Metric
from ragwriter.verify.sentences import strip_cites

# ── 规范化 ─────────────────────────────────────────────────────
# 在 crosscheck.normalize（NFKC、全角转半角、去空白）之上再统一三件事：
#   破折号/波浪号都当区间符；千分位逗号去掉；「$5万亿」与「5万亿美元」视为同一写法
_DASHES = re.compile(r"[—–―~－‒]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DOLLAR = re.compile(r"\$(\d+(?:\.\d+)?(?:万亿|亿|万|千|百)?)")


def canon(s: str) -> str:
    s = normalize(s)
    s = _DASHES.sub("-", s)
    s = _THOUSANDS.sub("", s)
    return _DOLLAR.sub(r"\1美元", s)


# ── 数字 token ────────────────────────────────────────────────
_NUM = r"\d+(?:\.\d+)?"
_MAG = r"(?:万亿|亿|万|千|百)"
# 单位。多字单位放前面，避免「个月」被「个」先吃掉
_UNIT = r"(?:个月|小时|分钟|美元|%|倍|年|月|日|天|周|名|次|家|个|项|款|位|元|篇|条|秒|x|X)"
# 前不能贴着字母/数字/小数点：「GPT-4o」「B2B」里的数字算专名的一部分，不单独回查。
# 后面：带了量级或单位的，允许紧跟字母——规范化去空格后「15 倍 Shopify」变成「15倍Shopify」，
# 原先一律禁字母会回溯成只抽「15」，单位丢失（G 组评估时发现）；裸数字后才禁字母
_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9.])\+?{_NUM}(?:-{_NUM})?"
    rf"(?:(?:{_MAG}{_UNIT}?|{_UNIT})\+?(?![0-9.])|\+?(?![A-Za-z0-9.]))"
)
# 「2026年4月」整体当一个日期 token，否则「4月」会单独去片段里找、必然找不到
_DATE = re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?")
# 写作时常补的量词，不改变数字含义：片段「61.66亿」句子「61.66亿次」视为一致
_SOFT_UNIT = re.compile(r"(?<=[\d万亿千百])(?:次|个|名|家|位|款|项|条|篇)$")


@dataclass
class NumToken:
    raw: str          # 句中写法（规范化后）
    forms: list[str]  # 在片段里找到任一写法即算一致


def _date_forms(y: str, m: str, d: str | None) -> list[str]:
    mm = [m, m.zfill(2)] if len(m) == 1 else [m, m.lstrip("0") or m]
    out = []
    for mo in mm:
        out += [f"{y}年{mo}月", f"{y}.{mo}", f"{y}-{mo}", f"{y}/{mo}"]
    if d:
        out = [f"{y}年{m}月{d}日", f"{y}.{m}.{d}", f"{y}-{m}-{d}", f"{y}/{m}/{d}"] + out
    return list(dict.fromkeys(out))


def _token_forms(t: str) -> list[str]:
    forms = [t]
    core = t
    if core.startswith("+"):  # 「+27%」与「增长27%」同义
        core = core[1:]
        forms.append(core)
    if core.endswith("+"):  # 「300+」也接受片段写「300以上」
        forms.append(core[:-1] + "以上")
    if re.fullmatch(r"\d{4}年", core):  # 「2025年」：片段常写成「(2025)」
        forms.append(core[:-1])
    soft = _SOFT_UNIT.sub("", core)
    if soft != core:
        forms.append(soft)
    return list(dict.fromkeys(forms))


def number_tokens_detail(text: str) -> list[NumToken]:
    """句中所有需要回查的数字类 token（引用标记先剥掉）。"""
    s = canon(strip_cites(text))
    out: list[NumToken] = []
    for m in _DATE.finditer(s):
        out.append(NumToken(m.group(0), _date_forms(m.group(1), m.group(2), m.group(3))))
    s = _DATE.sub(" ", s)  # 日期已处理，从正文里挖掉
    for m in _TOKEN.finditer(s):
        t = m.group(0)
        out.append(NumToken(t, _token_forms(t)))
    return out


def number_tokens(text: str) -> list[str]:
    return [t.raw for t in number_tokens_detail(text)]


def _form_in(form: str, hay: str) -> bool:
    """带边界的子串匹配：「5倍」不能命中「15倍」，「1.66亿」不能命中「61.66亿」，
    「5万」不能命中「5万亿」。"""
    start = 0
    while True:
        i = hay.find(form, start)
        if i < 0:
            return False
        j = i + len(form)
        prev = hay[i - 1] if i > 0 else ""
        nxt = hay[j] if j < len(hay) else ""
        ok = not (prev.isdigit() or prev == ".")
        if ok and form[-1].isdigit():
            ok = not (nxt.isdigit() or (nxt == "." and j + 1 < len(hay) and hay[j + 1].isdigit()))
        if ok and nxt and nxt in "万亿千百":
            ok = False
        if ok:
            return True
        start = i + 1


def token_in_text(token: NumToken | str, text: str) -> bool:
    if isinstance(token, NumToken):
        tok = token
    else:
        # 字符串入参（如 Metric.value）也走同一套解析，日期、$ 等写法才能对上
        parsed = number_tokens_detail(token)
        c = canon(token)
        tok = parsed[0] if len(parsed) == 1 else NumToken(c, _token_forms(c))
    hay = canon(text)
    return any(_form_in(f, hay) for f in tok.forms)


# ── 拉丁专名 ──────────────────────────────────────────────────
_LATIN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-.&][A-Za-z0-9]+)*")
# 通用缩写与行业词：写作时会自然出现，不是需要出处的专名
GENERIC_LATIN = {
    "ai", "aigc", "geo", "seo", "sem", "llm", "llms", "app", "apps", "top", "kpi", "roi",
    "b2b", "b2c", "dtc", "saas", "api", "faq", "q", "a", "vs", "ok", "pc", "ui", "ux",
    "ceo", "cto", "pr", "kol", "koc", "sku", "gmv", "rag", "agent", "agents", "h5",
}


def latin_names(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", strip_cites(text))
    out: list[str] = []
    for m in _LATIN.finditer(s):
        w = m.group(0).rstrip(".")
        if len(w) < 2 or w.lower() in GENERIC_LATIN or w in out:
            continue
        out.append(w)
    return out


def _latin_in(name: str, text: str) -> bool:
    """大小写不敏感，且按词边界：「Chat」不能靠「ChatGPT」蒙混过关。"""
    hay = unicodedata.normalize("NFKC", text).lower()
    pat = rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])"
    return re.search(pat, hay) is not None


# ── 事实句判断（V4）───────────────────────────────────────────
# 断言动词与出处标记。刻意不收单字「是」「达」「超」：前者几乎每句都有，
# 后两个会误中「超级流量入口」这类修饰语
_ASSERT = re.compile(
    r"达到|增长|提升|提高|下降|降低|减少|超过|突破|占比|占据|显示|表明|证明|证实|"
    r"数据|调研|研究|报告|统计|预测|预计|位居|排名|领先|首个|首家|唯一|推出|发布|"
    r"上线|添加|获得|荣获|拥有|覆盖|入选|据|根据|来源|翻了|倍增"
)
_INSUFFICIENT = "品宣资料未涉及"


def is_question(text: str) -> bool:
    s = strip_cites(text).strip()
    return bool(re.match(r"^(?:Q|问)\s*[:：]", s)) or s.endswith(("？", "?"))


def is_factual(text: str) -> bool:
    """是否事实句：含数字 / 拉丁专名 / 断言动词。

    问句（FAQ 的「Q：」行）与「品宣资料未涉及」声明不是论断，不算事实句——
    前者本来就不需要出处，后者是写作约束要求的诚实表述，强行要求引用只会逼模型编。
    """
    s = strip_cites(text)
    if not s or is_question(s) or _INSUFFICIENT in s:
        return False
    return bool(number_tokens_detail(s) or latin_names(s) or _ASSERT.search(s))


# ── 片段原文 ──────────────────────────────────────────────────
def snippet_material(sn: Snippet) -> str:
    """「所引片段原文」= 片段正文 + 配对的标签/数值/出处口径。

    出处口径（如「Shopify官方数据(2025)」）单列在 source_note 里，
    「据 Shopify 2025 年数据」里的年份要能在这里找到。
    """
    parts = [sn.text]
    if sn.metric:
        parts += [sn.metric.label, sn.metric.value, sn.metric.source_note]
    return "\n".join(p for p in parts if p)


def _cited(sentence: Sentence, cmap: dict[str, Snippet]) -> list[Snippet]:
    return [cmap[c] for c in sentence.cites if c in cmap]


def _where(snips: list[Snippet]) -> str:
    return "、".join(f"{s.sid}·第{s.page_no}页" for s in snips)


# ── V1 ────────────────────────────────────────────────────────
def check_v1(sentence: Sentence, cmap: dict[str, Snippet]) -> CheckResult:
    missing = [c for c in sentence.cites if c not in cmap]
    if missing:
        return CheckResult(
            check="V1", ok=False,
            reason=f"引用编号 {'、'.join(missing)} 不在本次上下文映射表里（可用：{'、'.join(sorted(cmap, key=lambda x: int(x[1:]))) or '无'}）",
        )
    return CheckResult(check="V1", ok=True, reason="" if sentence.cites else "无引用")


# ── V2 ────────────────────────────────────────────────────────
# 「据第 5 页图示」是提示词要求的出处写法，其中的页码不是事实数字
_PAGE_REF = re.compile(r"第\s*(\d{1,3})\s*页")


def _drop_page_refs(text: str, pages: set[int]) -> str:
    """去掉指向**所引页**的页码表述再回查；指向未引用页的「第 N 页」保留，照常拦。"""
    return _PAGE_REF.sub(lambda m: " " if int(m.group(1)) in pages else m.group(0), text)


# 范围口径词：模型最爱把「全球市场」的数据写成「海外用户」、把一个案例写成「越来越多品牌」。
# 当作专名处理——句子里出现，所引原文里就必须有。10-22 真跑两次都犯（苏姐 M8 第 2、3 项）
SCOPE_TERMS = ("海外", "越来越多", "普遍", "绝大多数", "大多数企业", "所有品牌")
# 点明案例主体的词。引用案例页的事实不带这些词，就是把单个案例写成了普遍现象
# 不收「客户」「企业」「产品」：案例原文里本来就有（「客户流量持续减少」），收了等于没拦
CASE_MARKERS = ("案例", "某", "首饰", "牛仔", "母婴", "新能源", "工厂", "供应商", "OEM")


def scope_terms(text: str) -> list[str]:
    s = strip_cites(text)
    return [t for t in SCOPE_TERMS if t in s]


def check_v2(sentence: Sentence, cmap: dict[str, Snippet], *, case_pages: frozenset[int] = frozenset()
             ) -> CheckResult:
    snips = _cited(sentence, cmap)
    body = _drop_page_refs(sentence.text, {s.page_no for s in snips})
    nums = number_tokens_detail(body)
    names = latin_names(body)
    if not nums and not names and not scope_terms(body) and not ({s.page_no for s in snips} & case_pages):
        return CheckResult(check="V2", ok=True, reason="句中无需回查的数字或专名")
    if not snips:
        # 没有可回查的片段：无引用交给 V4，编号无效交给 V1，这里不重复计数
        return CheckResult(check="V2", ok=True, reason="无有效引用，跳过回查（由 V1/V4 判定）")
    material = "\n".join(snippet_material(s) for s in snips)
    bad_nums = [t.raw for t in nums if not token_in_text(t, material)]
    bad_names = [n for n in names if not _latin_in(n, material)]
    bad_scope = [t for t in scope_terms(body) if t not in material]
    if bad_nums or bad_names or bad_scope:
        parts = []
        if bad_nums:
            parts.append("数字 " + "、".join(f"「{x}」" for x in bad_nums))
        if bad_names:
            parts.append("专名 " + "、".join(f"「{x}」" for x in bad_names))
        if bad_scope:
            parts.append("范围口径 " + "、".join(f"「{x}」" for x in bad_scope))
        return CheckResult(
            check="V2", ok=False,
            reason=f"{'；'.join(parts)} 不在所引片段（{_where(snips)}）原文中",
        )
    case_hit = sorted({s.page_no for s in snips} & case_pages)
    if case_hit and not any(m in strip_cites(sentence.text) for m in CASE_MARKERS):
        return CheckResult(check="V2", ok=False,
                           reason=f"引用了第 {'、'.join(map(str, case_hit))} 页案例的事实，但句中没有点明是哪个案例，"
                                  "读起来像普遍现象")
    return CheckResult(check="V2", ok=True)


# ── V3 ────────────────────────────────────────────────────────
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CJK = re.compile(r"[一-鿿]")
# 区分词两端的标点、空白、加号都是 builder 字符差集的副产品（「 推荐排名」「+名」「味“）」）
_EDGE_JUNK = re.compile(r"^[\s\W_+]+|[\s\W_+]+$")
_FUNCTION_CHARS = set("的了和与及在为是")


def _is_partial_latin(frag: str, label: str) -> bool:
    """区分词是不是拉丁词的残片。

    builder.attach_discriminators 按**字符**求差集，对中文标签正好（「流量」「订单」），
    对英文标签会退化成「ChatGPT→h、T」「Claude AI→ud」这种残片。
    残片拿去做子串判断，「h」会误中 Shopify、「o」会误中 OpenAI，必须丢掉。
    """
    if not re.fullmatch(r"[A-Za-z0-9]+", frag):
        return False
    return not any(w.lower() == frag.lower() for w in _LATIN_WORD.findall(label))


def _clean(frags: list[str], m: Metric) -> list[str]:
    """把 builder 给的区分词洗成能拿去做子串判断的词。

    丢掉：拉丁残片；单个汉字（「开」会误中「开发」）；数值本身的一部分
    （第 2 页「50大」是数值，不是对象，拿它做区分词等于用答案验证答案）。
    """
    val = canon(m.value)
    out = []
    for d in frags:
        d = _EDGE_JUNK.sub("", d or "")
        if not d or _is_partial_latin(d, m.label):
            continue
        if not re.search(r"[A-Za-z一-鿿]", d):
            continue
        if len(d) == 1 and _CJK.match(d):
            continue
        if canon(d) in val:
            continue
        out.append(d)
    return list(dict.fromkeys(out))


def _char_frags(label: str, other_labels: list[str]) -> list[str]:
    """与 builder.attach_discriminators 同一算法：标签里兄弟标签都没有的字，按原顺序连成片段。"""
    others: set[str] = set()
    for o in other_labels:
        others |= set(o)
    frags, cur = [], ""
    for ch in label:
        if ch not in others:
            cur += ch
        elif cur:
            frags.append(cur)
            cur = ""
    if cur:
        frags.append(cur)
    return frags


def effective_discriminators(m: Metric, siblings: list[Metric]) -> list[str]:
    """实际用于 V3 的区分词，三级退路：

    1. builder 回填的区分词，洗掉残片后还有剩 → 用它
    2. 只跟**不同标签**的兄弟重算一次：第 9 页两条「DeepSeek」（Maomu 与 QuestMobile
       两个口径）互相抵消，builder 给出空表，但相对 ChatGPT、豆包它是能区分的
    3. 拉丁标签按整词：「Perplexity AI」→「Perplexity」（「AI」是兄弟共有的）
    三级都空 → 空表，V3 对该配对跳过并在 reason 里写明。
    """
    good = _clean(m.discriminators, m)
    if good:
        return good
    diff = [s for s in siblings if s.label != m.label]
    good = _clean(_char_frags(m.label, [s.label for s in diff]), m)
    if good:
        return good
    others = {w.lower() for s in diff for w in _LATIN_WORD.findall(s.label)}
    return [
        w for w in _LATIN_WORD.findall(m.label)
        if w.lower() not in others and w.lower() not in GENERIC_LATIN
    ]


def _disc_exact(d: str, body: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9]+", d):
        return _latin_in(d, body)
    return normalize(d).lower() in normalize(body).lower()


def _disc_loose(d: str, body: str) -> bool:
    """本数字的区分词「出现在句中」：长片段允许改写，命中其中任一实义二字词即可。

    builder 给的区分词常是长片段（「客户订单量显著提升」），写作时必然改写措辞，
    要求整段原样出现会大面积误拦。兄弟排除那一侧仍用严格整词匹配（见 check_v3），
    两侧不对称是有意的：放宽「必须出现」、收紧「不得出现」，两头都少误拦。
    """
    if _disc_exact(d, body):
        return True
    nd = normalize(d)
    if len(nd) < 3 or not _CJK.search(nd):
        return False
    nb = normalize(body)
    for i in range(len(nd) - 1):
        bg = nd[i : i + 2]
        if _CJK.fullmatch(bg[0]) and _CJK.fullmatch(bg[1]) and not (set(bg) & _FUNCTION_CHARS):
            if bg in nb:
                return True
    return False


def check_v3(
    sentence: Sentence,
    cmap: dict[str, Snippet],
    page_metrics: Callable[[int], list[Metric]],
) -> CheckResult:
    snips = _cited(sentence, cmap)
    pages = sorted({s.page_no for s in snips})
    if not pages:
        return CheckResult(check="V3", ok=True, reason="无有效引用，跳过")
    body = strip_cites(sentence.text)
    errors: list[str] = []
    skipped: list[str] = []
    checked = 0
    for p in pages:
        metrics = [m for m in page_metrics(p) if m.value]
        present = [m for m in metrics if token_in_text(m.value, body)]
        if not present:
            continue
        present_vals = {canon(m.value) for m in present}
        present_labels = {m.label for m in present}
        discs = {
            id(m): effective_discriminators(m, [s for s in metrics if s is not m]) for m in metrics
        }
        own_discs = {d for m in present for d in discs[id(m)]}
        for val in sorted(present_vals):
            # 同一数值可能对应多个配对，任一配对的区分词出现即算挂对
            group = [m for m in present if canon(m.value) == val]
            usable = [m for m in group if discs[id(m)]]
            head = group[0]
            if not usable:
                skipped.append(
                    f"第{p}页「{head.value}」（{head.label}）区分词为空，同页标签无法区分，跳过区分词检查"
                )
            else:
                checked += 1
                if not any(_disc_loose(d, body) for m in usable for d in discs[id(m)]):
                    want = "、".join(f"「{d}」" for m in usable for d in discs[id(m)])
                    errors.append(
                        f"「{head.value}」在第{p}页属于「{'/'.join(m.label for m in usable)}」，"
                        f"句中缺少区分词 {want}"
                    )
            # 兄弟排除：数值不在本句、标签也不同的配对，其区分词不得出现在本句。
            # 区分词为空的配对也要做这一步——「ChatGPT 3.55亿」里 3.55 亿自己无从区分，
            # 但「ChatGPT」是 61.66 亿那一组的区分词，照样能拦住
            for sib in metrics:
                if canon(sib.value) in present_vals or sib.label in present_labels:
                    continue
                hit = [d for d in discs[id(sib)] if d not in own_discs and _disc_exact(d, body)]
                if hit:
                    errors.append(
                        f"句中出现了同页「{sib.label}」（{sib.value}）的区分词 "
                        f"{'、'.join(f'「{d}」' for d in hit)}，但「{head.value}」不属于它"
                    )
    if errors:
        return CheckResult(
            check="V3", ok=False, reason="数字挂错对象：" + "；".join(dict.fromkeys(errors))
        )
    if skipped:
        return CheckResult(check="V3", ok=True, reason="；".join(skipped))
    return CheckResult(check="V3", ok=True, reason="" if checked else "句中无所引页的配对数值")


# ── V4 ────────────────────────────────────────────────────────
def check_v4(sentence: Sentence, *, strict: bool = False) -> CheckResult:
    if is_question(sentence.text):
        return CheckResult(check="V4", ok=True, reason="问句，免引用")
    if strict and not sentence.cites and len(strip_cites(sentence.text).strip()) >= 8:
        # 导语是全文的论点句，最容易写出「AI 搜索正在取代传统搜索成为第一入口」这种
        # 不含数字、不含断言词、却是实打实事实判断的句子——is_factual 抓不住，导语里一律要引用
        return CheckResult(check="V4", ok=False, reason="导语句须带引用（导语只能复述已核实的内容）")
    if not sentence.cites and (terms := scope_terms(sentence.text)):
        return CheckResult(check="V4", ok=False,
                           reason=f"含范围口径「{'、'.join(terms)}」的判断须带引用，且原文要有同样的口径")
    if not is_factual(sentence.text):
        return CheckResult(check="V4", ok=True, reason="非事实句")
    if not sentence.cites:
        s = strip_cites(sentence.text)
        why = []
        if nums := number_tokens(s):
            why.append("数字 " + "、".join(nums))
        if names := latin_names(s):
            why.append("专名 " + "、".join(names))
        if not why and (m := _ASSERT.search(s)):
            why.append(f"断言词「{m.group(0)}」")
        return CheckResult(check="V4", ok=False, reason=f"事实句未带引用（含{'；'.join(why)}）")
    return CheckResult(check="V4", ok=True)


# ── V5 简版：字面支持度 ─────────────────────────────────────
# 完整 V5 是模型判别（P2，未启用）。这里先用一个零成本的确定性近似兜住最常见的一类：
# 改写时为了过 V4 随手挂一个不相干的引用（10-22 真跑实例：「买家向 AI 提问……决定订单流向」
# 挂了第 32 页 OEM 案例的结构，句子与片段的二元组召回只有 0.05）。
# 阈值用同一次真跑的 33 句标定：问题句 0.05 / 0.08 / 0.08 / 0.14，正常句最低 0.23，取 0.15。
# 它判不了「字面像、意思错」（如把第 1 层的品牌价值蒸馏说成第 2 层），那要等模型版 V5。
LEXICAL_MIN = 0.15
_V5_STOP = set("的了和与及是在对让被将为等中把从到也并或而其之这一个们您我你他她它而且以及就都")
_V5_JUNK = re.compile(r"[\W_\d]+")


def _bigrams(text: str) -> set[str]:
    s = "".join(ch for ch in _V5_JUNK.sub("", text) if ch not in _V5_STOP)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def check_v5_lexical(sentence: Sentence, cmap: dict[str, Snippet]) -> CheckResult:
    snips = _cited(sentence, cmap)
    if not snips:
        return CheckResult(check="V5", ok=True, reason="无有效引用，跳过（由 V1/V4 判定）")
    body = _drop_page_refs(strip_cites(sentence.text), {s.page_no for s in snips})
    body = re.sub(r"据第\s*页图示", "", body)
    sb = _bigrams(body)
    if len(sb) < 4:
        return CheckResult(check="V5", ok=True, reason="句子过短，跳过")
    recall = len(sb & _bigrams("".join(snippet_material(s) for s in snips))) / len(sb)
    if recall < LEXICAL_MIN:
        return CheckResult(check="V5", ok=False,
                           reason=f"所引片段几乎不支持本句（字面支持度 {recall:.2f} < {LEXICAL_MIN}），像是为过校验硬挂的引用")
    return CheckResult(check="V5", ok=True, reason=f"字面支持度 {recall:.2f}")


def verify_sentence(
    sentence: Sentence,
    citation_map: dict[str, Snippet],
    page_metrics: Callable[[int], list[Metric]],
    *,
    strict_cite: bool = False,
    lexical: bool = True,
    case_pages: frozenset[int] = frozenset(),
) -> SentenceVerdict:
    """对一句跑 V1–V4（+ V5 简版），结果顺序固定为 V1、V2、V3、V4、V5。全跑，不短路——
    前端要一次看到所有失败原因，改写提示词也要一次给全。

    strict_cite：导语用，非问句一律要引用。lexical：V5 简版字面支持度，默认开。
    case_pages：案例页页码（标题以「案例」开头的页），引用它们的句子须点明案例主体。
    """
    results = [
        check_v1(sentence, citation_map),
        check_v2(sentence, citation_map, case_pages=case_pages),
        check_v3(sentence, citation_map, page_metrics),
        check_v4(sentence, strict=strict_cite),
    ]
    if lexical:
        results.append(check_v5_lexical(sentence, citation_map))
    return SentenceVerdict(sentence=sentence, results=results)
