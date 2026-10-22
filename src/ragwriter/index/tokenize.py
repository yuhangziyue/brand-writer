"""T3 中文分词，BM25 的上游。

建库与查询必须走**同一个**分词函数，否则同一个词两边切得不一样，BM25 直接失配。

专名词典是这里唯一需要人维护的东西：jieba 默认会把「LLM-原生GEO」切成碎片、
把「万悉」切成「万」「悉」，而这些恰恰是品牌写作里最常被问到的词。
"""

from __future__ import annotations

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)  # 默认会往 stderr 打「Building prefix dict」

# 品宣里的专名与术语。新增语料时在这里补，不要散落到调用方
USER_WORDS = [
    "Trendee", "万悉", "万悉科技", "GEO", "LLM-原生GEO", "LLM-原生", "原生GEO",
    "传统GEO", "合规GEO", "投毒式GEO", "AI投毒", "JSON-LD", "RAG", "Schema",
    "ChatGPT", "DeepSeek", "Gemini", "Perplexity", "Claude", "Grok", "Kimi",
    "豆包", "通义千问", "元宝", "文心一言", "Shopify", "麦肯锡",
    "品牌价值蒸馏", "价值蒸馏", "问题预测", "用户语言转译", "语言转译",
    "AI记忆点", "深度记忆", "长期记忆", "先验知识", "思维链", "知识基建",
    "信任矩阵", "AI信任矩阵", "智能体", "可见性", "询盘", "出海", "跨境",
    "NO-MOQ", "OEM", "B端", "OPC",
]
for _w in USER_WORDS:
    jieba.add_word(_w, freq=200000)

# 只滤掉几乎每条都有的虚词。不做大停用词表：品宣短句里「AI」「品牌」都是信息词
STOPWORDS = frozenset("的 了 和 与 及 是 在 对 让 被 将 为 等 中 把 从 到 也 并 或 而 其 之".split())

_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """搜索模式分词：长词再切一遍短词，「品牌价值蒸馏」同时产出「品牌」「价值」「蒸馏」。"""
    out: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip().lower()
        if not tok or tok in STOPWORDS or _PUNCT.match(tok):
            continue
        out.append(tok)
    return out


def to_fts_doc(text: str) -> str:
    """建库：空格连接的词序列，交给 FTS5 的 unicode61 按空格切。"""
    return " ".join(tokenize(text))


def to_fts_query(text: str) -> str:
    """查询：每个词加引号再 OR 起来。

    加引号是为了让 FTS5 把「llm-原生」这类带符号的词当短语处理，
    而不是把「-」解析成 FTS5 的语法。OR 而不是 AND：写作型检索宁多勿漏，排序交给 BM25。
    """
    toks = dict.fromkeys(tokenize(text))  # 去重保序
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in toks)
