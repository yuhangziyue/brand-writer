"""混合检索：BM25 单路必测；稠密在 sentence_transformers 可用时才跑。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragwriter.index.store import RagStore, build_index
from ragwriter.retrieve import hybrid
from ragwriter.retrieve.hybrid import search, search_many

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DOCS = ROOT / "docs"
ALL = ("verified", "image_structural", "image_only", "unverified")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("idx") / "rag.db"
    build_index(BUILD, path, embedder=None, docs_dir=DOCS)
    s = RagStore(path)
    yield s
    s.close()


def _ids(res, n=5):
    return [h.fact.fact_id for h in res.hits[:n]]


def test_order_growth_hits_p10_15x(store):
    res = search(store, "AI驱动订单增长")
    assert not res.dense_used
    top = res.hits[0]
    assert top.fact.fact_id == "p10-f02" and top.fact.metric.value == "15倍"
    assert top.bm25_rank == 1 and top.dense_rank is None
    assert res.top_score == top.rrf_score == pytest.approx(1 / 61)


def test_geo_three_phases_hits_p05_flow(store):
    res = search(store, "GEO三个阶段注入点")
    top5 = res.hits[:5]
    assert any(h.fact.page_no == 5 and h.fact.kind == "flow" for h in top5), _ids(res)


def test_default_excludes_unusable(store):
    for q in ("chatgpt.com 会话次数 收入", "AI平台引流数据表", "AI驱动订单增长"):
        res = search(store, q, k_final=30)
        assert all(h.fact.status in ("verified", "image_structural") for h in res.hits)
    res = search(store, "chatgpt.com 会话次数 收入", statuses=ALL, k_final=30)
    assert any(h.fact.status == "image_only" for h in res.hits)


def test_k_final_and_empty_query(store):
    assert len(search(store, "GEO", k_final=3).hits) == 3
    res = search(store, "的了和")
    assert res.hits == [] and res.sufficient is False and res.signal


def test_search_many_fuses_all_lists(store):
    single = search(store, "AI驱动订单增长")
    many = search_many(store, ["AI驱动订单增长", "Shopify 订单 倍数"])
    assert many.hits[0].fact.fact_id == "p10-f02"
    # 两个查询都命中它，RRF 分应高于单查询
    assert many.hits[0].rrf_score > single.hits[0].rrf_score
    assert search_many(store, ["", "  "]).hits == []


def test_bm25_fallback_sufficiency(store):
    assert search(store, "问题预测覆盖率").sufficient
    res = search(store, "万悉的办公地址在哪个城市")
    assert not res.sufficient and "覆盖率" in res.signal


def test_split_prefers_fewer_false_positives():
    r = hybrid._split([0.5, 0.7, 0.8], [0.3, 0.6])
    # 0.4 与 0.65 都只错 1 条；0.65 错的是漏判而非放过陷阱，应选它
    assert r["tau"] == pytest.approx(0.65) and r["false_positive"] == 0
    r = hybrid._split([0.7, 0.8], [0.3, 0.4])
    assert r["separable"] and r["tau"] == pytest.approx(0.55)


# ── 稠密（可选）──
@pytest.fixture(scope="module")
def dense_store(tmp_path_factory):
    pytest.importorskip("sentence_transformers")
    from ragwriter.index.embed import load_embedder

    emb = load_embedder(True)
    if emb is None:
        pytest.skip("bge-m3 不可用")
    path = tmp_path_factory.mktemp("dense") / "rag.db"
    build_index(BUILD, path, embedder=emb, docs_dir=DOCS)
    s = RagStore(path)
    yield s, emb
    s.close()


def test_dense_hybrid(dense_store):
    store, emb = dense_store
    assert store.meta()["dense"] and store.meta()["embed_dim"] == emb.dim
    res = search(store, "AI给电商带来了多少订单", embedder=emb)
    assert res.dense_used
    assert any(h.fact.fact_id == "p10-f02" for h in res.hits[:3]), _ids(res)
    assert any(h.dense_rank is not None and h.bm25_rank is not None for h in res.hits)
    # 稠密路同样受 status 过滤
    d = store.dense(emb.encode(["chatgpt.com 会话次数 收入"])[0], k=50)
    assert all(store.fact(fid).status in ("verified", "image_structural") for fid, _ in d)


def test_dense_tau_rejects_traps(dense_store):
    store, emb = dense_store
    for q in hybrid.CALIB_TRAPS:
        assert not search(store, q, embedder=emb).sufficient, q
    assert search(store, "AI驱动订单增长多少", embedder=emb).sufficient
