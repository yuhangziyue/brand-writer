"""混合检索：BM25 + 稠密，RRF 融合，再做充分性判断（05-architecture.md 第 3.2 节）。

为什么两路都要：品宣里大量是专名和数字（「LLM-原生GEO」「15倍」），BM25 对字面匹配最稳；
但写作子问题是改写过的自然语言（「AI 给电商带来了多少订单」），只有稠密路能接住同义改写。
RRF 只用名次不用分数，两路分数量纲完全不同也不需要归一化。

充分性 τ 不是拍的，是用下面的小标注集校准出来的（`calibrate()`，CLI 里 `ragwriter calibrate`）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ragwriter.contracts import USABLE_STATUSES, Hit, RetrievalResult
from ragwriter.index.tokenize import tokenize

RRF_K = 60

# ── 充分性阈值 ────────────────────────────────────────────────
# 主信号：稠密路 top1 余弦（bge-m3，USABLE_STATUSES 过滤后）。
# 校准（2026-10-22，corpus_sha dc583b28c427，`ragwriter calibrate` 可复现）：
#   有答案 10 条  0.593 0.609 0.634 0.644 0.662 0.663 0.672 0.687 0.692 0.722
#   陷阱题  8 条  0.526 0.535 0.554 0.559 0.568 0.592 0.594 0.599
#   **两组不可完全分开**：最高的陷阱题「GEO 服务的价格」0.599（撞上第 35 页「付费形式」），
#   高于最低的有答案题「为什么中国出海品牌需要GEO」0.593。后者是综合论证题，
#   资料里「出海」只出现一次、要从多页拼，单条事实的相似度天然偏低。
#   取 0.599 与 0.609 的中点 0.604：陷阱题全部拦下，代价是那道综合题判为不足（1 条漏判）。
#   间隙只有 0.01，语料或模型一换就必须重跑校准。
TAU_DENSE = 0.604
# 退化信号：BM25 前 3 条命中里，单条覆盖查询词的最大比例。只在稠密不可用时用。
#   有答案  0.43 0.56 0.67 0.67 0.67 0.75 0.80 1.00 1.00 1.00
#   陷阱题  0.20 0.25 0.25 0.33 0.40 0.50 0.60 0.67
#   重叠更严重（「万悉的上市计划」0.67：第 34 页「上市新能源企业……计划进军澳洲」字面命中「上市」「计划」）。
#   0.633 与 0.415 都错 3 条；取 0.633（错的是 2 条漏判 + 1 条放过），按「少放过陷阱」原则。
#   这条信号只是退化兜底，BM25 单路时资料不足的判断明显不可靠，答辩要讲明。
TAU_COVERAGE = 0.633

# 有答案的问题：答案在品宣里能找到（页码是人工标注的期望页，供 calibrate 顺带看召回）
CALIB_ANSWERABLE: list[tuple[str, list[int]]] = [
    ("AI驱动订单增长多少", [10]),
    ("GEO 三个阶段分别注入到哪", [5]),
    ("Trendee 的商业模式", [35]),
    ("为什么中国出海品牌需要GEO", [9, 10, 11, 30]),
    ("问题预测覆盖率", [23]),
    ("消费者用AI购物的意愿有多高", [10]),
    ("什么是AI投毒", [12]),
    ("品牌价值蒸馏模型是什么", [20]),
    ("GEO监控系统能追踪什么", [27]),
    ("牛仔服饰工厂案例的效果", [32]),
]
# 陷阱题：看着像品牌问题，但品宣里没有答案。写作时必须判「资料不足」
CALIB_TRAPS: list[str] = [
    "万悉科技的融资情况",
    "万悉2025年营收",
    "Trendee 和竞品 X 公司对比",
    "万悉有多少员工",
    "万悉的上市计划",
    "GEO 服务的价格",
    "万悉的办公地址在哪个城市",
    "Trendee 的客户续约率",
]


def _rrf(rank_lists: Iterable[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids in rank_lists:
        for r, fid in enumerate(ids, start=1):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + r)
    return scores


def _coverage(store, query: str, ids: list[str], top: int = 3) -> float:
    """查询词覆盖率：前几条命中里，单条事实覆盖了多大比例的（去重）查询词。

    BM25 分数本身没有绝对尺度（随语料与查询长度漂），不能直接卡阈值；
    覆盖率是 0–1 的比例，才能跨查询比较。取单条最大而不是并集：
    并集会让「万悉 + 营收」分别命中两条不相干的事实也算全覆盖。
    """
    qtok = set(tokenize(query))
    if not qtok or not ids:
        return 0.0
    best = 0.0
    for fid in ids[:top]:
        dtok = set(tokenize(store.fact(fid).indexed_text))
        best = max(best, len(qtok & dtok) / len(qtok))
    return best


def _dense_ready(store, embedder) -> bool:
    """稠密路能不能跑：库里有向量、调用方给了 embedder、两边模型一致。"""
    if embedder is None:
        return False
    meta = store.meta()
    return bool(meta.get("dense")) and meta.get("embed_model") == embedder.model_name


def _run(store, queries: list[str], embedder, statuses, k_each: int, k_final: int) -> RetrievalResult:
    statuses = tuple(statuses)
    use_dense = _dense_ready(store, embedder)
    qvecs = embedder.encode(queries) if use_dense else None

    lists: list[list[str]] = []
    best_bm25: dict[str, int] = {}
    best_dense: dict[str, int] = {}
    dense_top1 = -1.0
    coverage = 0.0
    for i, q in enumerate(queries):
        b = [fid for fid, _ in store.bm25(q, statuses=statuses, k=k_each)]
        lists.append(b)
        for r, fid in enumerate(b, start=1):
            best_bm25[fid] = min(best_bm25.get(fid, r), r)
        coverage = max(coverage, _coverage(store, q, b))
        if use_dense:
            d_hits = store.dense(np.asarray(qvecs[i]), statuses=statuses, k=k_each)
            d = [fid for fid, _ in d_hits]
            lists.append(d)
            for r, fid in enumerate(d, start=1):
                best_dense[fid] = min(best_dense.get(fid, r), r)
            if d_hits:
                dense_top1 = max(dense_top1, d_hits[0][1])

    fused = sorted(_rrf(lists).items(), key=lambda kv: (-kv[1], kv[0]))[:k_final]
    facts = store.facts([fid for fid, _ in fused])
    hits = [
        Hit(fact=f, rrf_score=s, bm25_rank=best_bm25.get(f.fact_id), dense_rank=best_dense.get(f.fact_id))
        for f, (_, s) in zip(facts, fused)
    ]

    # 多查询时取各查询里最好的那个信号：问答的「改写句 + 原话」只要有一路找到资料就算够
    if not hits:
        sufficient, signal = False, "两路均无命中"
    elif use_dense:
        sufficient = dense_top1 >= TAU_DENSE
        signal = "" if sufficient else f"稠密 top1 余弦 {dense_top1:.3f} < τ={TAU_DENSE}"
    else:
        sufficient = coverage >= TAU_COVERAGE
        signal = "" if sufficient else f"BM25 查询词覆盖率 {coverage:.2f} < τ={TAU_COVERAGE}（稠密未启用）"

    return RetrievalResult(
        query=" || ".join(queries),
        hits=hits,
        top_score=hits[0].rrf_score if hits else 0.0,
        sufficient=sufficient,
        signal=signal,
        dense_used=use_dense,
    )


def search(
    store, query: str, *, embedder=None, statuses=USABLE_STATUSES, k_each: int = 20, k_final: int = 12
) -> RetrievalResult:
    """单查询：BM25 与稠密各取 k_each，RRF 融合取 k_final。"""
    return _run(store, [query], embedder, statuses, k_each, k_final)


def search_many(
    store, queries: list[str], *, embedder=None, statuses=USABLE_STATUSES, k_each: int = 20, k_final: int = 12
) -> RetrievalResult:
    """多查询：每个查询各跑两路，所有排名列表一起进 RRF。

    不是把各查询的结果拼起来再排：同一条事实被多个查询、多路同时命中，
    RRF 会自然把它抬高，这正是「改写句 + 原话」双路查询想要的效果。
    Hit 上的两路名次取所有查询里最好的一次。
    """
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return RetrievalResult(query="", hits=[], top_score=0.0, sufficient=False, signal="空查询")
    return _run(store, queries, embedder, statuses, k_each, k_final)


# ── 校准 ──────────────────────────────────────────────────────
def _signals(store, query: str, embedder, statuses) -> dict:
    b = [fid for fid, _ in store.bm25(query, statuses=statuses, k=20)]
    out = {"coverage": _coverage(store, query, b), "dense_top1": None, "top_pages": []}
    if _dense_ready(store, embedder):
        d = store.dense(embedder.encode([query])[0], statuses=statuses, k=3)
        out["dense_top1"] = d[0][1] if d else None
    r = search(store, query, embedder=embedder, statuses=statuses)
    out["top_pages"] = [h.fact.page_no for h in r.hits[:3]]
    return out


def _split(pos: list[float], neg: list[float]) -> dict:
    """在两组分数之间找阈值。

    候选切点取相邻两个分数的中点（而不是分数本身，免得某条样本正好压在线上）。
    可分就取间隙中点；不可分就取错分最少的切点，平手时优先少放过陷阱题——
    陷阱题被判「够」会让模型硬写，代价远大于把一个难题判成「资料不足」。
    """
    vals = sorted(set(pos + neg))
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or vals
    best = None
    for t in cands:
        fn = sum(1 for x in pos if x < t)
        fp = sum(1 for x in neg if x >= t)
        key = (fn + fp, fp)
        if best is None or key < best[0]:
            best = (key, t, fn, fp)
    (err, _), t, fn, fp = best
    gap = round(min(pos) - max(neg), 3) if min(pos) > max(neg) else None
    return {"tau": round(t, 3), "separable": err == 0, "errors": err,
            "false_negative": fn, "false_positive": fp, "gap": gap}


def calibrate(store, embedder=None, statuses=USABLE_STATUSES) -> dict:
    """在标注集上跑两种信号，给出分布与建议 τ。"""
    pos = [(q, pages, _signals(store, q, embedder, statuses)) for q, pages in CALIB_ANSWERABLE]
    neg = [(q, None, _signals(store, q, embedder, statuses)) for q in CALIB_TRAPS]
    report: dict = {"answerable": [], "traps": []}
    for name, rows in (("answerable", pos), ("traps", neg)):
        for q, pages, s in rows:
            report[name].append({"query": q, "expected_pages": pages, **s})
    for sig in ("dense_top1", "coverage"):
        p = [s[sig] for _, _, s in pos if s[sig] is not None]
        n = [s[sig] for _, _, s in neg if s[sig] is not None]
        report[sig] = _split(p, n) if p and n else None
    return report
