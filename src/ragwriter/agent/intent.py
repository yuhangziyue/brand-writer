"""意图识别：规则快路 → 模型分类 → 槽位合并（10 号文档第 3 节）。

两级而不是只用模型：规则快路零成本、确定性，命中「写一篇 Blog：……」「第 10 页……」这种
高频说法时省掉一次模型调用，也让无 key 环境下最常见的请求照样能路由。

🔴 意图识别**不负责最终拒答**。「万悉 2025 年营收」看起来是普通 qa，拦住它的是检索后的
充分性判断；这里只预先挡掉明显越界的请求（写代码、闲聊、天气）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from ragwriter.agent.context import assemble, profile_slice
from ragwriter.contracts import Genre, Intent

INTENTS: tuple[str, ...] = get_args(Intent)
GENRES: tuple[str, ...] = get_args(Genre)
# 本版本有处理器的意图；revise / explain 识别出来后走 refuse，明说暂不支持，不假装能做
SUPPORTED: frozenset[str] = frozenset({"write", "qa", "out_of_scope", "clarify"})
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
]
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
_REVISE = re.compile(r"改一下|改成|改得|缩短|短一点|长一点|精简|换成?.{0,6}语气|口语|正式一点|加一节|删掉第|重写第")
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
    source: str = "rule"  # rule | model | forced
    reason: str = ""  # clarify / 暂不支持 的原因，refuse 处理器据此出文案
    # 本轮从槽位继承了什么（评审 S5），如 {"topic": "……"}；前端据此显示「沿用主题：…」
    inherited: dict[str, str] = field(default_factory=dict)

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
        return IntentResult(intent="out_of_scope", reason="chitchat")

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
    if _REVISE.search(msg):
        if slots.get("last_artifact_id"):
            return IntentResult(intent="revise", reason="unsupported", constraints=extract_constraints(msg))
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


# ── 模型分类 ──────────────────────────────────────────────
INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "genre", "topic", "query", "constraints", "confidence"],
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "genre": {"type": "string", "enum": [*GENRES, "none"]},
        "topic": {"type": "string"},
        "query": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
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
    return IntentResult(
        intent=intent, genre=genre,
        topic=(out.get("topic") or "").strip() or None,
        query=(out.get("query") or "").strip() or None,
        constraints=[c for c in out.get("constraints") or [] if isinstance(c, str) and c.strip()],
        confidence=conf, source="model",
    )


# ── 总入口 ────────────────────────────────────────────────
def classify(message: str, slots: dict[str, Any], *, client: Any = None, profile: dict | None = None,
             window_text: str = "") -> IntentResult:
    """规则命中且槽位齐全 ⇒ 直接返回；否则有模型就问模型；都不行 ⇒ clarify。"""
    rule = rule_classify(message, slots)
    if rule is not None and rule.intent != "clarify":
        return rule
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
    if res.intent in ("revise", "explain"):
        if not slots.get("last_artifact_id"):
            return IntentResult(intent="clarify", source="model", reason="no_artifact", confidence=res.confidence)
        res.reason = "unsupported"
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
