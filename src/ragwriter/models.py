"""核心数据结构。

父子两级：Page 是父 chunk（引用单位），Fact 是子 chunk（检索单位）。
见架构文档 1.1 节。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 事实分三级，不是两级。
#   verified         文字层可逐字核验
#   image_structural 节点名文字层可核，但**关系**只在页面图示里（流程图的箭头、层级、对应）
#                    —— 可用，引用时须标注「结构依据页面图示」
#   image_only       文字层完全无法核验的事实（截图里的数字、图表占比、截图正文）
#                    —— 默认不可用
#   unverified       两条解析路径给出冲突结果，留待生成期的 V3 区分词再判
FactStatus = Literal["verified", "image_structural", "image_only", "unverified"]
FactKind = Literal["point", "metric", "definition", "case", "title", "flow"]


class Block(BaseModel):
    """文字层的一个文本块，带版面坐标。坐标用于几何配对与前端高亮。"""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float  # 块内最大字号，用于识别标题与大号数字

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    def h_overlap(self, other: "Block") -> float:
        """两块横向重叠宽度占较窄者的比例。"""
        inter = min(self.x1, other.x1) - max(self.x0, other.x0)
        if inter <= 0:
            return 0.0
        narrower = min(self.x1 - self.x0, other.x1 - other.x0)
        return inter / narrower if narrower > 0 else 0.0


class Metric(BaseModel):
    """一个数字与它描述的对象、出处的配对。

    这是本题解析的真正难点：多栏版式下整页文本会把数字和标签拆散，
    「8倍 15倍 AI驱动流量增长 AI驱动订单增长」必须还原成两组配对。
    """

    value: str  # 照抄文字层原文，如 "15倍"
    label: str  # 这个数字描述的对象，如 "AI驱动订单增长"
    source_note: str = ""  # 出处口径，如 "Shopify官方数据(2025)"
    # 配对来源：llm=按内容流块序的文字层结构化（主路）；vlm=额外送了渲染图；
    # geometry=纯版面坐标启发式（退化路径与对照组）
    method: Literal["llm", "vlm", "geometry", "none"] = "none"
    discriminators: list[str] = Field(default_factory=list)  # V3 区分词，建索引时回填


class Page(BaseModel):
    """父 chunk。36 页，页既是主题单位也是引用单位。"""

    page_no: int
    chapter: str
    title: str
    text: str
    blocks: list[Block] = Field(default_factory=list)
    image_path: str = ""
    is_divider: bool = False  # 纯章节页，基本无信息量


class Fact(BaseModel):
    """子 chunk。检索命中的单元，回填时拿它的 page_no 找父页。"""

    fact_id: str  # "p10-f03"
    page_no: int
    kind: FactKind
    text: str
    prefix: str  # 确定性上下文前缀，不调模型，由页面结构拼出
    status: FactStatus = "verified"
    metric: Metric | None = None

    @property
    def indexed_text(self) -> str:
        """送进检索的文本 = 前缀 + 正文。"""
        return f"{self.prefix}\n{self.text}"
