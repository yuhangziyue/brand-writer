"""T1 退化路径：用版面坐标把数字和它描述的对象配起来。

为什么需要它：整页文本按行横扫，多栏版式下数字与标签会被拆散。
第 10 页抽出来是「2025年4月，OpenAI向 8倍 15倍 ChatGPT添加购物功能」，
而 8 倍属于「AI驱动流量增长」、15 倍属于「AI驱动订单增长」，与 OpenAI 那句无关。

为什么它只是退化路径：实跑 36 页发现版式至少三种——
  1. 标签在数字下方（第 10、23、26 页）
  2. 标签在数字上方（第 9 页：ChatGPT 在上，61.66亿 在下，再下面才是单位「月访问量」）
  3. 数字与标签在同一个文本块里（第 11 页「AI问答引擎2个月」、第 25 页「AI引用频率提升 120%+」）
还有第 32 页的竖排标签，任何「正上方/正下方」启发式都会配错。
所以主路是多模态页卡，本模块的产出默认标 unverified，只有被页卡印证后才升为 verified。
"""

from __future__ import annotations

import re
import statistics

from ragwriter.models import Block, Metric

# 数值 token：61.66亿 / 8倍 / 64% / $5万亿 / 2个月 / 2.5年 / 95%+ / 300+ / +27% / 5735万
VALUE_RE = re.compile(
    r"[+\-]?[$￥]?\d[\d,.]*\s*(?:万亿|亿|万|千|个月|小时|分钟|%|％|倍|年|天|名|次|人)?\+?"
)
# 出处口径块的特征：带括号或「来源/数据/预测/调研」。
# 只用于**找出处**，不能用来排除标签候选：第 10 页的真标签「全球AI电商市场规模预测」
# 自带「预测」二字，当初拿它排除候选，$5万亿 就被错配到了「AI驱动订单增长」。
SOURCE_HINT_RE = re.compile(r"[（(]|来源|数据|预测|调研|统计|官方")
# 排除标签候选用的窄规则：整块是括号注释，或明写「来源」
NOT_A_LABEL_RE = re.compile(r"^[（(].*[）)]$|来源")
# 单位尾注，不能当标签，如第 9 页数字下方的「月访问量」
UNIT_TAIL = {"月访问量", "月活跃用户", "历史最快", "移动爆款", "传统增长"}

MAX_PAIR_GAP = 80.0  # 纵向最大距离，超过就不算同一组
MIN_PAIR_GAP = -12.0  # 允许轻微重叠：第 9 页压 2pt，第 23 页「95%+」与标签压 10pt
HEADER_CUTOFF = 120.0  # 页眉区域的块是标题或章节号，不参与数值配对
MIN_H_OVERLAP = 0.5  # 横向重叠比例下限
MAX_VALUE_TEXT = 24  # 数值块的文本长度上限。放宽到 24 才装得下第 25 页「内容通过率 （去"AI味"）80%+」
LARGE_FONT_RATIO = 1.25  # 数值块字号相对页面中位字号的倍数下限


def _page_median_size(blocks: list[Block]) -> float:
    sizes = [b.size for b in blocks]
    return statistics.median(sizes) if sizes else 12.0


def _value_tokens(text: str) -> list[str]:
    return [m.group(0).strip() for m in VALUE_RE.finditer(text) if m.group(0).strip()]


# 纯序号：章节号 01 / 步骤号 1 / 小节号 2.1，没有单位，不是指标
ORDINAL_RE = re.compile(r"^\d{1,2}$|^0\d$|^\d\.\d$")
# 序号旁边出现这些词才说明它真是个名次指标，如「Top 1」；否则是流程步骤号
RANK_CONTEXT_RE = re.compile(r"排名|Top|TOP|第|名次")


def is_value_block(block: Block, median_size: float) -> bool:
    """判定一个块是不是「数值展示块」。

    靠字号而不是靠内容比例：正文里「2025年4月，OpenAI向ChatGPT添加购物功能」
    也含数字，但它是 14.1pt 的正文字号；真正的数值展示块都是放大的。
    """
    if len(block.text) > MAX_VALUE_TEXT:
        return False
    if block.y0 < HEADER_CUTOFF:
        return False
    tokens = _value_tokens(block.text)
    if not tokens:
        return False
    if len(block.text) <= 4 and ORDINAL_RE.match(block.text.strip()):
        return False
    return block.size >= max(14.0, median_size * LARGE_FONT_RATIO)


def _split_inline(block: Block) -> tuple[str, str] | None:
    """同一个块里既有数字又有标签时就地切开，如「AI引用频率提升 120%+」。

    这种情况最可靠：版面已经把它们放在一起了，不需要猜。
    """
    tokens = _value_tokens(block.text)
    if not tokens:
        return None
    value = max(tokens, key=len)
    label = block.text.replace(value, " ", 1).strip(" ·（）()　")
    label = re.sub(r"\s+", " ", label)
    if len(label) < 2:
        return None
    # 「01 / 核心优化策略」「1 信息检索」这种是流程步骤号，不是指标
    if ORDINAL_RE.match(value) and not RANK_CONTEXT_RE.search(label):
        return None
    return value, label


def _find_label(value_block: Block, blocks: list[Block]) -> tuple[Block | None, str]:
    """跨块找标签。返回 (标签块, 方向)。方向用于诊断，写进校验报告。"""
    below: list[tuple[float, Block]] = []
    above: list[tuple[float, Block]] = []
    for b in blocks:
        if b is value_block:
            continue
        if b.size >= value_block.size:  # 比数字还大的是标题，不是标签
            continue
        if value_block.h_overlap(b) < MIN_H_OVERLAP:
            continue
        if b.text in UNIT_TAIL or NOT_A_LABEL_RE.search(b.text):
            continue
        gap_below = b.y0 - value_block.y1
        gap_above = value_block.y0 - b.y1
        if MIN_PAIR_GAP <= gap_below <= MAX_PAIR_GAP:
            below.append((gap_below, b))
        elif MIN_PAIR_GAP <= gap_above <= MAX_PAIR_GAP:
            above.append((gap_above, b))
    if below:
        return min(below, key=lambda t: t[0])[1], "below"
    if above:
        return min(above, key=lambda t: t[0])[1], "above"
    return None, "none"


def _find_source_note(anchor: Block, blocks: list[Block]) -> str:
    cands = []
    for b in blocks:
        if b is anchor or b.size > anchor.size:
            continue
        if not SOURCE_HINT_RE.search(b.text):
            continue
        if anchor.h_overlap(b) < MIN_H_OVERLAP:
            continue
        gap = b.y0 - anchor.y1
        if -5 <= gap <= 40:
            cands.append((gap, b.text))
    return min(cands, key=lambda t: t[0])[1] if cands else ""


def pair_metrics(blocks: list[Block]) -> list[Metric]:
    """对一页做几何配对，返回候选 Metric 列表。

    产出一律 method="geometry"，交给 crosscheck 与页卡比对后才决定能不能用。
    """
    median = _page_median_size(blocks)
    out: list[Metric] = []
    for vb in blocks:
        if not is_value_block(vb, median):
            continue
        inline = _split_inline(vb)
        if inline:
            value, label = inline
            out.append(
                Metric(
                    value=value,
                    label=label,
                    source_note=_find_source_note(vb, blocks),
                    method="geometry",
                )
            )
            continue
        label_block, _direction = _find_label(vb, blocks)
        value = max(_value_tokens(vb.text), key=len)
        # 与 _split_inline 同一条规则：光杆序号旁边没有名次语境，就是流程步骤号
        if ORDINAL_RE.match(value) and not (
            label_block and RANK_CONTEXT_RE.search(label_block.text)
        ):
            continue
        if label_block is None:
            out.append(Metric(value=value, label="", method="geometry"))
            continue
        out.append(
            Metric(
                value=value,
                label=label_block.text,
                source_note=_find_source_note(label_block, blocks),
                method="geometry",
            )
        )
    return out
