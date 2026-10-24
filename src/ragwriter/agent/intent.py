"""意图识别：模型理解 → 程序护栏 → 槽位合并（10 号文档第 3 节）。

🔴 10-23 子曰定：**模型优先**。原先规则快路先判、命中即返回，「第 N 页」「问号结尾」一律当 qa，
模型看不到问句的真实目的，回答也只能套一个模板。现在每轮由模型理解用户要什么，并给出
「回答形状」（问数字 / 核实对错 / 列举 / 简述 / 解释）与「焦点」（到底问的是什么），
作答按形状裁剪，不回答没问的东西。别改回「规则先判」。

规则只剩两个角色：没有模型（或模型调用失败）时兜底路由；模型输出之后的程序护栏
（写作缺主题 → 继承或追问，改写 / 解释缺上一篇 → 追问）。

🔴 意图识别**不负责最终拒答**。「万悉 2025 年营收」看起来是普通 qa，拦住它的是检索后的
充分性判断；这里只预先挡掉明显越界的请求（写代码、闲聊、天气）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from ragwriter.agent.context import assemble, profile_slice
from ragwriter.contracts import AnswerShape, Genre, Intent

INTENTS: tuple[str, ...] = get_args(Intent)
GENRES: tuple[str, ...] = get_args(Genre)
SHAPES: tuple[str, ...] = get_args(AnswerShape)
# 本版本有处理器的意图；explain 识别出来后走 refuse，一句话说明依据在哪，不假装能做
SUPPORTED: frozenset[str] = frozenset({"write", "qa", "revise", "confirm", "chitchat", "out_of_scope", "clarify"})
# confirm 的动作：用户对「待确认写作」的回复。10-23 子曰定：每次写作先审主题、与用户确认，再检索
CONFIRM_ACTIONS: tuple[str, ...] = ("accept", "modify", "pick", "cancel")
# revise 的动作。改写链路按动作分流：前四种调模型改一块，delete 不调模型，add_section 补检索写新节，
# retitle 只改标题。动作由模型判断（「讲合规那节改口语」这种说法规则拆不出来），规则只兜底
REVISE_ACTIONS: tuple[str, ...] = ("shorten", "expand", "tone", "rewrite", "delete", "add_section", "retitle")
_REVISE_TARGET = re.compile(r"^(?:lede|ending|title|all|section:[1-9]\d*)$")
CONFIDENCE_FLOOR = 0.6

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "intent.md"

# 体裁词。顺序有意义：「产品介绍」要排在「介绍」类泛词之前（这里没有泛词，留作提醒）
_GENRE_WORDS: list[tuple[str, str]] = [
    (r"faq|常见问题|问答集", "faq"),
    (r"blog|博客|博文", "blog"),
    (r"品牌介绍|品牌简介", "brand_intro"),
    (r"产品介绍|产品简介", "product_intro"),
]
_WRITE_VERB = re.compile(r"写|生成|出一版|再出|来一篇|来一版|撰写|起草|产出")
_TOPIC = [
    re.compile(r"(?:主题(?:是|为)?|关于|围绕|讲讲?|聊聊)[：:\s]*(?P<t>.+)"),
    re.compile(r"[：:](?P<t>.+)"),
    # 「写一篇产品介绍，介绍 X」：逗号后接主题。10-23 业务实测漏掉这种说法，主题被上一轮继承覆盖
    re.compile(r"[，,]\s*(?:介绍|讲|说说|写)?(?:一下)?\s*(?P<t>.{4,})"),
]
# 「写一篇关于合规 GEO 的 Blog」：主题尾巴上的「的 + 体裁词」不属于主题
_TOPIC_TAIL = re.compile(r"\s*的?\s*(?:官网\s*)?(?:blog|博客|文章|faq|常见问题|品牌介绍|产品介绍)\s*$", re.IGNORECASE)
_PAGE_REF = re.compile(r"第\s*\d+\s*页")
_QUESTION_END = re.compile(r"[?？呢吗嘛]\s*$")
_QUESTION_WORD = re.compile(r"多少|几倍|几个|哪些|哪个|哪家|是什么|什么是|怎么|如何|为什么|是否|有没有|情况$|介绍一下")
# 🔴 越界 ≠ 资料不足（评审 M6）：这里只拦与品牌资料**无关**的请求。
# 「融资 / 营收 / 员工 / 价格」是正常的品牌问题，绝不能放进来——它们要走检索，
# 由充分性判断给出「资料不足 + 最接近的页」，评审才看得到拒答是有依据的
_OUT_OF_SCOPE = re.compile(
    r"python|java(?:script)?|sql|代码|脚本|写个程序|编程|bug|正则表达式|天气|气温|下雨|"
    r"讲个笑话|笑话|陪我聊|聊天|你是谁|彩票|星座|翻译(?:一下|成|下面)",
    re.IGNORECASE,
)

# clarify 的主题建议（评审 S2）。🔴 已用检索验证资料充分：三条都能命中多页 verified 事实，
# 不从 catalog 现挑，是因为页标题不等于「能写成一篇」；前端首页空态也用这三条（GET /v1/suggestions）
SUGGESTED_TOPICS: tuple[str, ...] = (
    "LLM-原生 GEO 与传统 GEO 有什么不同",
    "为什么品牌要被 AI 记住",
    "为什么 GEO 必须合规",
)
_GREETING = re.compile(r"^\s*(你好|您好|hi|hello|hey|在吗|哈喽|早上好|晚上好)[!！。.~\s]*$", re.IGNORECASE)
_REVISE = re.compile(r"改一下|改成|改得|改短|改长|缩短|短一点|长一点|精简|扩写|展开|换成?.{0,6}语气|口语|正式一点|"
                     r"加一节|新增一节|补一节|删掉|删除|去掉|重写|标题改|改标题")
# 「换成 FAQ / 改成产品介绍」是体裁变化 ⇒ 新的 write（主题继承），不是 revise
_GENRE_SWITCH = re.compile(r"换成|改成|改写成|转成|做成")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_CONFIRM_ACCEPT = re.compile(r"^\s*(确认|好的?|可以|行|没问题|就这样|开始写?(作)?吧?|写吧|ok|okay|yes|嗯+|对)[，,。.!！\s]*"
                             r"(开始(写作)?|写吧|就这么写)?[。.!！\s]*$", re.IGNORECASE)
_CONFIRM_CANCEL = re.compile(r"^\s*(取消|算了|不写了|不用了|先不写|不要了)")
_CONFIRM_MODIFY = re.compile(r"主题改为|改为[：:]|换个角度|换个主题|侧重|改成讲|不对，|不是这个")
_SECTION_NO = re.compile(r"第\s*([一二两三四五六七八九十]|\d+)\s*(?:节|部分|段|小节)")
_EXPLAIN = re.compile(r"依据|出处|为什么删|来源是|哪页说的|引用的哪")
_CONSTRAINT = re.compile(r"(不超过\s*\d+\s*字|\d+\s*字(?:以内|左右)|语气[^，,。]{1,8}|篇幅[^，,。]{1,8})")


@dataclass
class IntentResult:
    intent: str
    genre: str | None = None
    topic: str | None = None
    query: str | None = None
    constraints: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "rule"  # rule | model | forced | selected（写作台指定体裁与主题，不识别）
    reason: str = ""  # clarify / 暂不支持 的原因，refuse 处理器据此出文案
    # 本轮从槽位继承了什么（评审 S5），如 {"topic": "……"}；前端据此显示「沿用主题：…」
    inherited: dict[str, str] = field(default_factory=dict)
    answer_shape: str = "short"  # qa 的回答形状，见 contracts.AnswerShape
    focus: str = ""  # 用户到底在问什么（名词短语），拒答与追问文案用它，不复述整句原话
    # 检索用的补充改写（0–3 条）：带上品牌别名与关键词。10-23 实测「万悉是做什么的」，
    # 答案在「核心产品：Trendee 致力于……」，正文里没有「万悉」二字，只用原问题检索捞不到
    search_queries: list[str] = field(default_factory=list)
    # revise 专用：改哪块（lede / ending / title / all / section:N，N 从 1 起、按用户看到的正文小节数）、
    # 怎么改（REVISE_ACTIONS）、用户要求的要点。非 revise 时为空串
    revise_target: str = ""
    revise_action: str = ""
    revise_instruction: str = ""
    # confirm 专用：accept / modify / pick / cancel。pick 时 topic = 所选备选；modify 时修改意见放 revise_instruction
    confirm_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 规则快路 ──────────────────────────────────────────────
def detect_genre(message: str) -> str | None:
    for pat, genre in _GENRE_WORDS:
        if re.search(pat, message, re.IGNORECASE):
            return genre
    return None


def extract_topic(message: str) -> str | None:
    for pat in _TOPIC:
        m = pat.search(message)
        if m:
            t = m.group("t").strip().strip("「」“”\"'").rstrip("。.!！")
            t = _TOPIC_TAIL.sub("", t).strip()
            # 逗号后面也常是约束（「再写一篇，800 字以内」「语气正式一点」），那不是主题
            if pat is _TOPIC[-1] and (_CONSTRAINT.search(t) or re.search(r"语气|篇幅|字数|口吻|风格", t)):
                continue
            if len(t) >= 2:
                return t
    return None


def extract_constraints(message: str) -> list[str]:
    return [m.strip() for m in _CONSTRAINT.findall(message)]


def rule_classify(message: str, slots: dict[str, Any]) -> IntentResult | None:
    """返回 None 表示规则没命中。命中但槽位不全时返回 intent=clarify 且 reason 说明缺什么，
    由上层决定要不要再问模型。"""
    msg = message.strip()
    if not msg:
        return IntentResult(intent="clarify", reason="empty")
    if _GREETING.match(msg):
        return IntentResult(intent="chitchat", reason="greeting")

    if slots.get("pending_write"):
        # 待确认期间的短回复。只认明确说法；其余（问个问题、写别的）照常往下走，pending 保留
        if _CONFIRM_ACCEPT.match(msg):
            return IntentResult(intent="confirm", confirm_action="accept")
        if _CONFIRM_CANCEL.match(msg):
            return IntentResult(intent="confirm", confirm_action="cancel")
        if _CONFIRM_MODIFY.search(msg):
            return IntentResult(intent="confirm", confirm_action="modify", revise_instruction=msg)

    genre = detect_genre(msg)
    if genre and _WRITE_VERB.search(msg):
        topic = extract_topic(msg)
        # 「写一篇 FAQ：……」里冒号后面的才是主题；体裁词本身不能当主题
        if topic and detect_genre(topic) == genre and len(topic) <= 8:
            topic = None
        res = IntentResult(intent="write", genre=genre, topic=topic, constraints=extract_constraints(msg))
        if not topic and slots.get("topic"):
            res.topic = slots["topic"]
            res.inherited["topic"] = slots["topic"]
        if not res.topic:
            return IntentResult(intent="clarify", genre=genre, reason="missing_topic",
                                constraints=res.constraints)
        return res

    # 越界放在写作判断之后：「写一篇 Blog：AI 搜索如何改变代码检索」不该被「代码」误杀；
    # 但「帮我写个 Python 脚本」没有体裁词，会落到这里
    if _OUT_OF_SCOPE.search(msg):
        return IntentResult(intent="out_of_scope", reason="out_of_scope")

    if _EXPLAIN.search(msg):
        if slots.get("last_artifact_id"):
            return IntentResult(intent="explain", reason="unsupported")
        return IntentResult(intent="clarify", reason="no_artifact")
    if genre and _GENRE_SWITCH.search(msg):
        # 「换成 FAQ」：体裁变、主题沿用上一轮。必须排在 revise 之前，否则「改成」会被当成改写
        if slots.get("topic"):
            return IntentResult(intent="write", genre=genre, topic=slots["topic"], inherited={"topic": slots["topic"]},
                                constraints=extract_constraints(msg))
        return IntentResult(intent="clarify", genre=genre, reason="missing_topic")
    if _REVISE.search(msg):
        if slots.get("last_artifact_id"):
            target, action = rule_revise(msg)
            return IntentResult(intent="revise", constraints=extract_constraints(msg), revise_target=target,
                                revise_action=action, revise_instruction=msg)
        return IntentResult(intent="clarify", reason="no_artifact")

    if _WRITE_VERB.search(msg) and re.search(r"一篇|一版|文章|稿", msg):
        topic = extract_topic(msg)
        if topic:
            return IntentResult(intent="write", genre=slots.get("genre") or "blog", topic=topic,
                                constraints=extract_constraints(msg),
                                inherited={"genre": slots["genre"]} if slots.get("genre") else {})
        return IntentResult(intent="clarify", reason="missing_topic")

    if _PAGE_REF.search(msg) or _QUESTION_END.search(msg) or _QUESTION_WORD.search(msg):
        return IntentResult(intent="qa", query=msg)
    return None


def rule_revise(message: str) -> tuple[str, str]:
    """规则兜底拆 revise 的（目标, 动作）。只认最常见的说法；「讲合规那节」这类指代要靠模型对照大纲。"""
    msg = message
    target = ""
    m = _SECTION_NO.search(msg)
    if m:
        raw = m.group(1)
        target = f"section:{int(raw) if raw.isdigit() else _CN_NUM[raw]}"
    elif re.search(r"标题|题目", msg):
        target = "title"
    elif re.search(r"开头|导语|开篇|引言", msg):
        target = "lede"
    elif re.search(r"结尾|结语|收尾|末尾", msg):
        target = "ending"
    elif re.search(r"整篇|全文|通篇|全篇", msg):
        target = "all"

    if re.search(r"加一节|新增一节|补一节|增加一节|再加一节|多加一节", msg):
        action = "add_section"
    elif re.search(r"删掉|删除|去掉|不要了", msg):
        action = "delete"
    elif target == "title":
        action = "retitle"
    elif re.search(r"短|精简|压缩|缩|简洁|字以内|字内", msg):
        action = "shorten"
    elif re.search(r"长一点|展开|扩写|丰富|详细|充实", msg):
        action = "expand"
    elif re.search(r"语气|口语|正式|活泼|口吻|风格|轻松|严肃", msg):
        action = "tone"
    else:
        action = "rewrite"
    return normalize_revise(target, action)


def normalize_revise(target: str, action: str) -> tuple[str, str]:
    """目标与动作对齐：改标题一律 retitle；加一节不指定位置就放正文末尾（all）；其余缺目标视为整篇。"""
    target = target if _REVISE_TARGET.match(target or "") else ""
    action = action if action in REVISE_ACTIONS else ""
    if action == "retitle" or (target == "title" and action in ("", "shorten", "expand", "tone", "rewrite")):
        return "title", "retitle"
    if not action:
        action = "rewrite"
    if not target:
        target = "all"
    return target, action


# ── 模型分类 ──────────────────────────────────────────────
INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "genre", "topic", "query", "search_queries", "answer_shape", "focus", "constraints",
                 "revise_target", "revise_action", "revise_instruction", "confirm_action", "confidence"],
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "genre": {"type": "string", "enum": [*GENRES, "none"]},
        "topic": {"type": "string"},
        "query": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "answer_shape": {"type": "string", "enum": list(SHAPES)},
        "focus": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        # strict schema 里所有字段都必须 required，非 revise 时给空串 / none
        "revise_target": {"type": "string"},
        "revise_action": {"type": "string", "enum": [*REVISE_ACTIONS, "none"]},
        "revise_instruction": {"type": "string"},
        "confirm_action": {"type": "string", "enum": [*CONFIRM_ACTIONS, "none"]},
        "confidence": {"type": "number"},
    },
}


def load_prompt() -> str:
    """只取 `## System` 段作为系统提示词；文件头是给人看的说明，不送模型。"""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if "## System" in text:
        text = text.split("## System", 1)[1]
    return text.strip()


def model_classify(client: Any, message: str, slots: dict[str, Any], profile: dict,
                   window_text: str = "") -> IntentResult:
    import json

    ctx = assemble(
        l0=load_prompt(),
        l1=profile_slice(profile, "intent"),
        l3=window_text or ("槽位：" + json.dumps(slots, ensure_ascii=False)),
        task=f"用户消息：{message}",
        budget=2000,
    )
    out = client.complete_json(ctx.system, ctx.user, json_schema=INTENT_SCHEMA, max_tokens=400, tag="intent")
    intent = out.get("intent") if out.get("intent") in INTENTS else "clarify"
    genre = out.get("genre") if out.get("genre") in GENRES else None
    try:
        conf = float(out.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    shape = out.get("answer_shape") if out.get("answer_shape") in SHAPES else "short"
    return IntentResult(
        intent=intent, genre=genre,
        topic=(out.get("topic") or "").strip() or None,
        query=(out.get("query") or "").strip() or None,
        constraints=[c for c in out.get("constraints") or [] if isinstance(c, str) and c.strip()],
        confidence=conf, source="model", answer_shape=shape,
        focus=(out.get("focus") or "").strip(),
        search_queries=[q.strip() for q in out.get("search_queries") or [] if isinstance(q, str) and q.strip()][:3],
        revise_target=str(out.get("revise_target") or "").strip().replace("：", ":").replace(" ", ""),
        revise_action=str(out.get("revise_action") or "").strip(),
        revise_instruction=str(out.get("revise_instruction") or "").strip(),
        confirm_action=str(out.get("confirm_action") or "").strip(),
    )


# ── 总入口 ────────────────────────────────────────────────
def classify(message: str, slots: dict[str, Any], *, client: Any = None, profile: dict | None = None,
             window_text: str = "") -> IntentResult:
    """有模型 ⇒ 模型理解（每轮一次），再过程序护栏；没有模型或调用失败 ⇒ 规则兜底。"""
    rule = rule_classify(message, slots)
    if client is None:
        return rule or IntentResult(intent="clarify", reason="no_rule_no_model")

    try:
        res = model_classify(client, message, slots, profile or {}, window_text)
    except Exception as exc:  # noqa: BLE001 — 分类失败不该让整轮崩掉，退回规则结果或追问
        fallback = rule or IntentResult(intent="clarify", reason="model_failed")
        fallback.reason = fallback.reason or f"model_failed: {exc}"
        return fallback

    if res.confidence < CONFIDENCE_FLOOR:
        return IntentResult(intent="clarify", source="model", confidence=res.confidence,
                            reason="low_confidence", genre=res.genre)
    if res.intent == "write":
        if not res.genre and not (rule and rule.genre) and slots.get("genre"):
            res.inherited["genre"] = slots["genre"]
        res.genre = res.genre or (rule.genre if rule else None) or slots.get("genre") or "blog"
        if not res.topic and slots.get("topic"):
            res.topic = slots["topic"]
            res.inherited["topic"] = slots["topic"]
        if not res.topic:
            return IntentResult(intent="clarify", source="model", genre=res.genre, reason="missing_topic",
                                confidence=res.confidence)
    if res.intent == "qa" and not res.query:
        res.query = message
    if res.intent == "qa" and not res.focus:
        res.focus = res.query or message
    if res.intent in ("revise", "explain"):
        if not slots.get("last_artifact_id"):
            return IntentResult(intent="clarify", source="model", reason="no_artifact", confidence=res.confidence)
        if res.intent == "explain":
            res.reason = "unsupported"
    if res.intent == "confirm":
        if not slots.get("pending_write"):
            # 没有待确认的写作却说「确认」：多半是指代上一轮别的东西，追问而不是凭空开写
            return IntentResult(intent="clarify", source="model", reason="no_pending", confidence=res.confidence)
        if res.confirm_action not in CONFIRM_ACTIONS:
            res.confirm_action = (rule.confirm_action if rule and rule.intent == "confirm" else "") or "modify"
        if res.confirm_action == "pick" and not res.topic:
            res.confirm_action = "modify"
        if res.confirm_action == "modify":
            res.revise_instruction = res.revise_instruction or res.query or message
        res.revise_target = res.revise_action = ""
        if res.confirm_action != "modify":
            res.revise_instruction = ""
        return res
    res.confirm_action = ""
    if res.intent == "revise":
        # 模型给的目标 / 动作不合法时用规则拆一遍补上；指令为空就用原话，改写提示词至少有东西可依
        target, action = res.revise_target, res.revise_action
        if not _REVISE_TARGET.match(target or "") or action not in REVISE_ACTIONS:
            r_target, r_action = rule_revise(message)
            target = target if _REVISE_TARGET.match(target or "") else r_target
            action = action if action in REVISE_ACTIONS else r_action
        res.revise_target, res.revise_action = normalize_revise(target, action)
        res.revise_instruction = res.revise_instruction or message
        res.genre = None
    else:
        res.revise_target = res.revise_action = res.revise_instruction = ""
    if res.intent == "clarify" and not res.reason:
        res.reason = "model_clarify"
    if res.intent == "out_of_scope":
        res.reason = res.reason or "out_of_scope"
    return res


def merge_slots(slots: dict[str, Any], res: IntentResult) -> dict[str, Any]:
    """槽位由程序合并，不让模型写摘要（2.4 节：模型摘要写错一个数字会被后续每轮继承）。"""
    new = dict(slots)
    if res.intent == "write":
        new["genre"] = res.genre or new.get("genre")
        new["topic"] = res.topic or new.get("topic")
        if res.constraints:
            new["constraints"] = list(dict.fromkeys([*(new.get("constraints") or []), *res.constraints]))
    elif res.intent == "qa":
        new["last_query"] = res.query or new.get("last_query")
    return new


def suggest_topics(profile: dict | None = None, n: int = 3) -> list[str]:
    """clarify 时的主题建议。profile 参数保留是为了将来按品牌分区取，本版本用验证过的常量。"""
    return list(SUGGESTED_TOPICS[:n])
