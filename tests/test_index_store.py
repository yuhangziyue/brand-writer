"""索引入库：用真实 build/ 产物建一个临时库（BM25 单路，保持测试快）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragwriter.index.store import RagStore, build_index

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DOCS = ROOT / "docs"


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "rag.db"
    build_index(BUILD, path, embedder=None, docs_dir=DOCS)
    return path


@pytest.fixture(scope="module")
def store(db):
    s = RagStore(db)
    yield s
    s.close()


def _jsonl(name: str) -> list[dict]:
    return [json.loads(x) for x in (BUILD / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def test_counts_match_build_artifacts(store):
    facts = _jsonl("facts.jsonl")
    meta = store.meta()
    assert meta["counts"]["facts"] == len(facts) == meta["counts"]["facts_fts"]
    assert meta["counts"]["pages"] == len(_jsonl("pages.jsonl")) == len(store.pages())
    assert meta["counts"]["metrics"] == sum(1 for f in facts if f["metric"])
    assert meta["counts"]["page_cards"] == len(list((BUILD / "cards").glob("p*.json")))
    assert meta["counts"]["vision_cards"] == len(list((BUILD / "vision-cards").glob("p*.json")))
    for st in ("verified", "image_structural", "image_only", "unverified"):
        assert meta["status_counts"].get(st, 0) == sum(1 for f in facts if f["status"] == st)
    assert meta["dense"] is False and meta["counts"]["embeddings"] == 0


def test_fact_roundtrip_and_order(store):
    f = store.fact("p10-f02")
    assert f.metric is not None and f.metric.value == "15倍" and f.status == "verified"
    ids = ["p10-f04", "p05-f01", "p10-f01"]
    assert [x.fact_id for x in store.facts(ids)] == ids
    with pytest.raises(KeyError):
        store.fact("p99-f01")


def test_page_facts_filters_status(store):
    usable = store.page_facts(10)
    assert usable and all(f.status in ("verified", "image_structural") for f in usable)
    everything = store.page_facts(10, statuses=("verified", "image_structural", "image_only", "unverified"))
    assert any(f.status == "image_only" for f in everything)


def test_page_metrics_p10_have_discriminators(store):
    ms = store.page_metrics(10)
    assert len(ms) == 4
    assert all(m.discriminators for m in ms)
    by_label = {m.label: m.discriminators for m in ms}
    assert by_label["AI驱动订单增长"] == ["订单"]
    assert by_label["AI驱动流量增长"] == ["流量"]


def test_bm25_excludes_unusable_by_default(store):
    # 第 31 页截图里的会话数只存在于 image_only 事实
    hits = store.bm25("chatgpt.com 会话次数 交易量 收入", k=50)
    assert all(store.fact(fid).status in ("verified", "image_structural") for fid, _ in hits)
    all_hits = store.bm25(
        "chatgpt.com 会话次数 交易量 收入",
        statuses=("verified", "image_structural", "image_only", "unverified"), k=50,
    )
    assert any(store.fact(fid).status == "image_only" for fid, _ in all_hits)


def test_bm25_scores_descending_and_empty_query(store):
    hits = store.bm25("AI驱动订单增长")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert store.bm25("的 了 和") == []  # 全是停用词
    assert store.bm25("，。！") == []


def test_dense_without_vectors_is_empty(store):
    assert store.dense(np.ones(1024, dtype=np.float32)) == []


def test_profile_rules_extracted(store):
    prof = store.profile()
    assert {"entities", "catalog", "rules"} <= prof.keys()
    assert len(prof["catalog"]) == 36
    corr = {c["wrong"]: c["right"] for c in prof["rules"]["corrections"]}
    assert corr.get("思维连注入") == "思维链注入"
    assert any("紫烟" in w for w in corr)
    forb = prof["rules"]["forbidden"]
    p31 = [r for r in forb if 31 in r["pages"]]
    assert p31 and "匿名" in p31[0]["rule"]
    # 档案会进提示词：被禁用的具体站点名不能借规则原文漏进去
    assert "Nihaojewelry" not in json.dumps(prof, ensure_ascii=False)
    assert any(30 in r["pages"] and "出海" in r["rule"] for r in forb)
    assert any(22 in r["pages"] for r in forb)
    names = json.dumps(prof["entities"], ensure_ascii=False)
    assert "Trendee" in names and "Huina Mao" in names and "Xin Shuai" in names


def test_profile_missing_rules_raises(tmp_path):
    (tmp_path / "structured-content.md").write_text("# 空文档\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_index(BUILD, tmp_path / "x.db", embedder=None, docs_dir=tmp_path)
    assert not (tmp_path / "x.db").exists()


class _ExplodingEmbedder:
    model_name = "boom"
    dim = 4

    def encode(self, texts):
        raise RuntimeError("嵌入中途失败")


def test_atomic_replace_keeps_old_db(tmp_path):
    path = tmp_path / "rag.db"
    build_index(BUILD, path, embedder=None, docs_dir=DOCS)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="嵌入中途失败"):
        build_index(BUILD, path, embedder=_ExplodingEmbedder(), docs_dir=DOCS)
    assert path.read_bytes() == before  # 旧库原封不动
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rag.db"]  # 没留临时文件
    RagStore(path).close()  # 旧库仍可正常打开


def test_atomic_no_partial_db_on_first_build(tmp_path):
    path = tmp_path / "sub" / "rag.db"
    with pytest.raises(RuntimeError):
        build_index(BUILD, path, embedder=_ExplodingEmbedder(), docs_dir=DOCS)
    assert list(path.parent.iterdir()) == []
