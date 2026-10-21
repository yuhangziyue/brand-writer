"""T1 交叉校验：页卡里的每条内容都要回到文字层对账。

这是整条链路上最重要的一道闸。模型在结构化阶段同样会编：
它可能把「17.6%」写成「17.6% 可见性提升」，也可能把两张卡片的标签对调。
凡是对不上文字层的，一律不进事实库。

三类检查：
  1. 数值：规范化后必须在该页文字层出现
  2. 标签：与该页某个文本块的相似度必须达标
  3. 配对：与几何配对的结果比对，不一致的降级，交由 V3 区分词在生成期再拦一道
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from ragwriter.ingest.pagecard import CardMetric, PageCard
from ragwriter.models import Metric, Page

LABEL_SIM_THRESHOLD = 0.8   # 滑窗局部匹配阈值
LABEL_RECALL_THRESHOLD = 0.9  # 退一步的字符召回阈值

# 规范化：全角转半角、去空白、统一百分号与加号
_FULLWIDTH_MAP = {"％": "%", "＋": "+", "－": "-", "＄": "$", "，": ",", "．": "."}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in _FULLWIDTH_MAP.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def value_in_text(value: str, page_text: str) -> bool:
    return normalize(value) in normalize(page_text)


def partial_ratio(label: str, text: str) -> float:
    """滑窗局部匹配：把标签长度的窗口在全文上滑一遍，取最高相似度。

    不能直接拿整块比：第 2 页是一整段人物介绍，块长 600 字，
    「论文被引次数」这种准确的短标签跟它比只有 0.06，会被误判成编造。
    """
    nl, nt = normalize(label), normalize(text)
    if not nl or not nt:
        return 0.0
    n = len(nl)
    if n >= len(nt):
        return SequenceMatcher(None, nl, nt).ratio()
    return max(
        SequenceMatcher(None, nl, nt[i : i + n]).ratio() for i in range(len(nt) - n + 1)
    )


def char_recall(label: str, text: str) -> float:
    """标签用到的字有多少出现在原文里。宽松，但配合「数值必须在原文」够用。"""
    nl, nt = set(normalize(label)), set(normalize(text))
    return len(nl & nt) / len(nl) if nl else 0.0


def label_matches_block(label: str, page: Page) -> float:
    """标签与该页文本的贴合度：块内包含直接算满分，否则取滑窗局部匹配。"""
    nl = normalize(label)
    if not nl:
        return 0.0
    for b in page.blocks:
        if nl in normalize(b.text):
            return 1.0
    return partial_ratio(label, page.text)


def geometry_verdict(cm: CardMetric, geo: list[Metric]) -> str:
    """几何配对对这个数值的态度：agree / conflict / absent。

    三态而不是两态，是因为「几何没给出配对」和「几何给了不同的配对」
    完全不是一回事。第 32 页竖排标签、第 33 页的 +27%，几何本来就配不出来；
    把「没结果」当成「冲突」去降级，等于拿已知更弱的那条路去否决更强的那条。
    """
    for g in geo:
        if normalize(g.value) != normalize(cm.value):
            continue
        if not g.label:
            return "absent"
        return "agree" if similarity(g.label, cm.label) >= LABEL_SIM_THRESHOLD else "conflict"
    return "absent"


def check_metric(cm: CardMetric, page: Page, geo: list[Metric]) -> tuple[Metric, str]:
    """返回 (带状态的 Metric, 判定说明)。说明会进校验报告，便于答辩时逐条解释。"""
    if not value_in_text(cm.value, page.text):
        # 文字层找不到这个数值：要么来自截图，要么是模型编的。两种都不能用
        return (
            Metric(value=cm.value, label=cm.label, source_note=cm.source_note, method="vlm"),
            "image_only:数值不在文字层",
        )
    label_sim = label_matches_block(cm.label, page)
    recall = char_recall(cm.label, page.text)
    geo_v = geometry_verdict(cm, geo)
    m = Metric(value=cm.value, label=cm.label, source_note=cm.source_note, method="llm")

    if label_sim < LABEL_SIM_THRESHOLD and recall < LABEL_RECALL_THRESHOLD:
        return m, f"unverified:标签贴合度不足 局部匹配{label_sim:.2f} 字符召回{recall:.2f}"
    if not cm.certain:
        return m, "unverified:模型自标不确定"
    if geo_v == "conflict":
        # 两条独立路径给出**不同**的配对。几何在竖排标签上本来就会错，
        # 所以不直接否定，降级为 unverified，交给生成期的 V3 区分词再判一次
        return m, "unverified:与几何配对冲突"
    if geo_v == "agree":
        return m, "verified:双路一致"
    gate = "局部匹配" if label_sim >= LABEL_SIM_THRESHOLD else "字符召回"
    return m, f"verified:单路+{gate}"


def check_page_card(card: PageCard, page: Page, geo: list[Metric]) -> dict:
    """对一页做完整交叉校验，返回结构化结果。"""
    metrics: list[tuple[Metric, str]] = [check_metric(cm, page, geo) for cm in card.metrics]
    points: list[tuple[str, str]] = []
    for p in card.points:
        # 要点允许改写措辞，但其中出现的数值必须在文字层里
        nums = re.findall(r"[+\-]?[$￥]?\d[\d,.]*\s*(?:万亿|亿|万|千|个月|%|％|倍|年|天|名|次)?", p)
        bad = [n for n in nums if n.strip() and not value_in_text(n, page.text)]
        points.append((p, "verified" if not bad else f"unverified:数值不在文字层 {bad}"))
    return {
        "page_no": page.page_no,
        "metrics": metrics,
        "points": points,
        "image_only_items": card.image_only_items,
    }
