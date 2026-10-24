"""T3 索引入库与只读访问：SQLite 单文件 `build/index/rag.db`。

为什么是 SQLite 而不是向量库：约 400 条事实、36 页，FTS5 自带 BM25，
稠密向量整块载入 numpy 全量点积也只要亚毫秒。引入 Chroma / Qdrant
只会多一个服务或一个重依赖（选型见 10-agent-architecture.md 第 4.2 节）。

库是**建库产物**，可以整个删掉重建；所以写入走「临时文件 → os.replace」，
中途任何异常都不会留下半个库，在线链路要么读到旧库、要么读到完整的新库。

所有状态的事实都入库，可用性靠 status 过滤，而不是建库时丢掉：
image_only 在校验报告里还要能被指出来（「这个数字只在截图里」），删了就讲不清了。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ragwriter.chunk.builder import attach_discriminators, page_prefix
from ragwriter.contracts import USABLE_STATUSES
from ragwriter.index import tokenize as tok
from ragwriter.index.embed import Embedder
from ragwriter.index.profile import build_profile
from ragwriter.models import Fact, Metric, Page

# 表结构一变就加一，在线链路据此拒绝读旧库，而不是读出缺字段的数据
SCHEMA_VERSION = "1"

# FTS5 的列权重：fact_id 不参与打分；前缀（第几页 · 章 · 页标题）同页所有事实都一样，
# 权重给正文的 0.4，否则「第10页」这类前缀词会把整页事实一起抬上来，冲掉正文的区分度
BM25_WEIGHTS = (0.0, 0.4, 1.0)

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pages (
    page_no INTEGER PRIMARY KEY, chapter TEXT, title TEXT, text TEXT,
    is_divider INTEGER, image_path TEXT, blocks_json TEXT
);
CREATE TABLE facts (
    fact_id TEXT PRIMARY KEY, page_no INTEGER NOT NULL, kind TEXT NOT NULL,
    status TEXT NOT NULL, prefix TEXT NOT NULL, text TEXT NOT NULL
);
CREATE INDEX facts_page ON facts(page_no);
CREATE INDEX facts_status ON facts(status);
CREATE TABLE metrics (
    fact_id TEXT PRIMARY KEY REFERENCES facts(fact_id), value TEXT, label TEXT,
    source_note TEXT, method TEXT, discriminators_json TEXT
);
CREATE TABLE page_cards (page_no INTEGER PRIMARY KEY, card_json TEXT NOT NULL);
CREATE TABLE vision_cards (page_no INTEGER PRIMARY KEY, card_json TEXT NOT NULL);
CREATE TABLE brand_profile (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE VIRTUAL TABLE facts_fts USING fts5(fact_id UNINDEXED, prefix_tok, text_tok);
CREATE TABLE embeddings (fact_id TEXT PRIMARY KEY REFERENCES facts(fact_id), vec BLOB NOT NULL);
"""


# ── 建库 ──────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_cards(card_dir: Path) -> dict[int, str]:
    """按页读原始卡片 JSON。原样存，不做解析——它们是建库的「证据」，供答辩回溯。"""
    out: dict[int, str] = {}
    if not card_dir.is_dir():
        return out
    for p in sorted(card_dir.glob("p*.json")):
        raw = p.read_text(encoding="utf-8")
        out[int(json.loads(raw)["page_no"])] = raw
    return out


def _corpus_sha(paths: Iterable[Path]) -> str:
    """对全部输入文件算一个摘要。库和 build/ 产物对不上时，在线链路能一眼看出来。"""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _dict_hash() -> str:
    """分词词典的指纹。词典一改，FTS 里预分好的词就和查询切法不一致了，必须重建。"""
    payload = ("\n".join(tok.USER_WORDS) + "\x00" + " ".join(sorted(tok.STOPWORDS))
               + "\x00" + " ".join(f"{k}={v}" for k, v in sorted(tok.ALIASES.items())))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def apply_fact_patches(facts: list[Fact], path: Path) -> list[Fact]:
    """人工数据补丁：给表格单元格、人物卡片里的数字补上主语（10-24）。

    PDF 内容流里第 35 页三种商业模式的单元格与表头分离（工具智能体的表头甚至排在最后）、
    第 2 页创始人履历的数字不带人名、第 9 页两张榜单的标签不带榜单名——模型拿到「付费形式：Token 调用消耗」
    无从知道属于哪种模式，实测把 OPC 的软件订阅写成了工具智能体的、把创始人的论文数写成了团队的。
    每条补丁都依据 docs/structured-content.md 的逐页人工核对；根治在页卡解析（见 README 已知局限）。
    补丁文件不存在时原样返回。必须在 attach_discriminators 之前应用：V3 的区分词从标签算。
    """
    if not path.exists():
        return facts
    patches = json.loads(path.read_text(encoding="utf-8")).get("patches", {})
    out = []
    for f in facts:
        pt = patches.get(f.fact_id)
        if pt:
            upd: dict = {"text": pt["text"]}
            if f.metric is not None and pt.get("label"):
                upd["metric"] = f.metric.model_copy(update={"label": pt["label"]})
            f = f.model_copy(update=upd)
        out.append(f)
    return out


def title_facts(pages: list[Page], facts: list[Fact]) -> list[Fact]:
    """每个非分隔页的页标题补成一条可引用的事实（kind=title，id 形如 p02-t00）。

    10-23 业务实测：问「万悉科技的愿景」答「未涉及」。愿景「让好的品牌进入AI的深度记忆」
    **只在第 2 页标题里**，标题又只活在 prefix 中——检索靠 prefix 命中了第 2 页，
    但送进模型的片段只有正文（「万悉科技 (Trendee)」），模型看不到也引用不了。
    页标题是文字层原文，与正文同样可核，所以标 verified。正文里已有同文的就不重复补。
    """
    by_page: dict[int, set[str]] = {}
    for f in facts:
        by_page.setdefault(f.page_no, set()).add(f.text.strip())
    out = []
    for p in pages:
        title = (p.title or "").strip()
        if p.is_divider or len(title) < 4 or title in by_page.get(p.page_no, set()):
            continue
        out.append(Fact(fact_id=f"p{p.page_no:02d}-t00", page_no=p.page_no, kind="title",
                        text=title, prefix=page_prefix(p), status="verified"))
    return out


def build_index(
    build_dir: Path, db_path: Path, *, embedder: Embedder | None, docs_dir: Path
) -> dict:
    """从 build/ 产物建库，返回统计。embedder 为 None 时只建 BM25，meta.dense=0。"""
    t0 = time.perf_counter()
    build_dir, db_path, docs_dir = Path(build_dir), Path(db_path), Path(docs_dir)

    pages = [Page(**d) for d in _read_jsonl(build_dir / "pages.jsonl")]
    facts = [Fact(**d) for d in _read_jsonl(build_dir / "facts.jsonl")]
    facts = apply_fact_patches(facts, build_dir / "fact_patches.json")
    facts += title_facts(pages, facts)
    # 区分词在 ingest 时已算过；这里再算一遍是幂等的，防的是手工改过 facts.jsonl
    # 却忘了重算的情况——V3 闸完全依赖它，宁可多算一次
    attach_discriminators(facts)
    cards = _read_cards(build_dir / "cards")
    vcards = _read_cards(build_dir / "vision-cards")
    # 档案先建：抽不到规则会报错，早于耗时的向量编码失败
    profile = build_profile(pages, docs_dir)

    inputs = [build_dir / "pages.jsonl", build_dir / "facts.jsonl"]
    inputs += [p for p in [build_dir / "fact_patches.json"] if p.exists()]
    inputs += sorted((build_dir / "cards").glob("p*.json"))
    inputs += sorted((build_dir / "vision-cards").glob("p*.json"))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标同目录，os.replace 才是同一文件系统内的原子重命名；
    # 后缀保持 .db，被 .gitignore 的 build/index/*.db 一并覆盖
    tmp = db_path.with_name(f".{db_path.stem}-{os.getpid()}.tmp.db")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO pages VALUES (?,?,?,?,?,?,?)",
            [
                (
                    p.page_no, p.chapter, p.title, p.text, int(p.is_divider), p.image_path,
                    json.dumps([b.model_dump() for b in p.blocks], ensure_ascii=False),
                )
                for p in pages
            ],
        )
        conn.executemany(
            "INSERT INTO facts VALUES (?,?,?,?,?,?)",
            [(f.fact_id, f.page_no, f.kind, f.status, f.prefix, f.text) for f in facts],
        )
        conn.executemany(
            "INSERT INTO metrics VALUES (?,?,?,?,?,?)",
            [
                (
                    f.fact_id, f.metric.value, f.metric.label, f.metric.source_note,
                    f.metric.method, json.dumps(f.metric.discriminators, ensure_ascii=False),
                )
                for f in facts
                if f.metric is not None
            ],
        )
        conn.executemany(
            "INSERT INTO facts_fts VALUES (?,?,?)",
            [(f.fact_id, tok.to_fts_doc(f.prefix), tok.to_fts_doc(f.text)) for f in facts],
        )
        conn.executemany("INSERT INTO page_cards VALUES (?,?)", sorted(cards.items()))
        conn.executemany("INSERT INTO vision_cards VALUES (?,?)", sorted(vcards.items()))
        conn.executemany(
            "INSERT INTO brand_profile VALUES (?,?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in profile.items()],
        )

        embed_s = 0.0
        dim = 0
        if embedder is not None:
            te = time.perf_counter()
            vecs = np.asarray(embedder.encode([f.indexed_text for f in facts]), dtype=np.float32)
            embed_s = time.perf_counter() - te
            if vecs.shape[0] != len(facts):
                raise RuntimeError(f"向量条数 {vecs.shape[0]} 与事实数 {len(facts)} 不一致")
            dim = int(vecs.shape[1])
            conn.executemany(
                "INSERT INTO embeddings VALUES (?,?)",
                [(f.fact_id, vecs[i].tobytes()) for i, f in enumerate(facts)],
            )

        status_counts: dict[str, int] = {}
        for f in facts:
            status_counts[f.status] = status_counts.get(f.status, 0) + 1
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("pages", "facts", "metrics", "page_cards", "vision_cards",
                      "brand_profile", "facts_fts", "embeddings")
        }
        meta = {
            "schema_version": SCHEMA_VERSION,
            "corpus_sha": _corpus_sha(inputs),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embed_model": embedder.model_name if embedder is not None else "",
            "embed_dim": str(dim),
            "dense": "1" if embedder is not None else "0",
            "dict_hash": _dict_hash(),
            "counts": json.dumps(counts),
            "status_counts": json.dumps(status_counts),
        }
        conn.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
        conn.commit()
        conn.close()
        os.replace(tmp, db_path)
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise

    return {
        "db_path": str(db_path),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "embed_s": round(embed_s, 2),
        "size_bytes": db_path.stat().st_size,
        "dense": embedder is not None,
        "embed_model": meta["embed_model"],
        "embed_dim": dim,
        "corpus_sha": meta["corpus_sha"],
        "counts": counts,
        "status_counts": status_counts,
    }


# ── 只读访问 ──────────────────────────────────────────────────
class RagStore:
    """rag.db 的只读门面。

    事实和页在构造时整体读进内存（几百条，几 MB 以内）：在线链路一次请求要
    反复按 id 取事实、回填页，每次都打 SQL 没有意义。BM25 仍走 FTS5。
    """

    def __init__(self, db_path: Path | str):
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"索引库不存在：{path}（先跑 ragwriter index）")
        # 只读打开：在线链路不该有任何写；check_same_thread=False 让 FastAPI
        # 线程池共享一个连接，并发由下面的锁串行化（查询都是亚毫秒级）
        self._conn = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro", uri=True, check_same_thread=False
        )
        self._lock = threading.Lock()
        self._meta = dict(self._conn.execute("SELECT key, value FROM meta"))
        if self._meta.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"索引库 schema {self._meta.get('schema_version')} 与代码 {SCHEMA_VERSION} 不符，请重建"
            )
        if self._meta.get("dict_hash") != _dict_hash():
            # 词典改了却没重建，BM25 会静默失配——这里直接拦下
            raise RuntimeError("分词词典与建库时不一致，请重跑 ragwriter index")

        metrics: dict[str, Metric] = {}
        for fid, value, label, note, method, disc in self._conn.execute(
            "SELECT fact_id, value, label, source_note, method, discriminators_json FROM metrics"
        ):
            metrics[fid] = Metric(
                value=value, label=label, source_note=note, method=method,
                discriminators=json.loads(disc),
            )
        self._facts: dict[str, Fact] = {}
        for fid, page_no, kind, status, prefix, text in self._conn.execute(
            "SELECT fact_id, page_no, kind, status, prefix, text FROM facts ORDER BY rowid"
        ):
            self._facts[fid] = Fact(
                fact_id=fid, page_no=page_no, kind=kind, status=status,
                prefix=prefix, text=text, metric=metrics.get(fid),
            )
        self._pages: dict[int, Page] = {}
        for page_no, chapter, title, text, div, img, blocks in self._conn.execute(
            "SELECT page_no, chapter, title, text, is_divider, image_path, blocks_json "
            "FROM pages ORDER BY page_no"
        ):
            self._pages[page_no] = Page(
                page_no=page_no, chapter=chapter, title=title, text=text,
                is_divider=bool(div), image_path=img, blocks=json.loads(blocks),
            )
        self._profile = {
            k: json.loads(v)
            for k, v in self._conn.execute("SELECT key, value_json FROM brand_profile")
        }
        # 稠密矩阵按 status 组合缓存：(fact_ids, (n, dim) float32)
        self._dense_cache: dict[tuple[str, ...], tuple[list[str], np.ndarray]] = {}

    # ── 取数 ──
    def fact(self, fact_id: str) -> Fact:
        try:
            return self._facts[fact_id]
        except KeyError:
            raise KeyError(f"无此事实：{fact_id}") from None

    def facts(self, ids: list[str]) -> list[Fact]:
        """保持传入顺序：调用方传的是检索排名，顺序本身就是信息。"""
        return [self.fact(i) for i in ids]

    def page(self, page_no: int) -> Page:
        try:
            return self._pages[page_no]
        except KeyError:
            raise KeyError(f"无此页：{page_no}") from None

    def pages(self) -> list[Page]:
        return list(self._pages.values())

    def page_facts(self, page_no: int, statuses=USABLE_STATUSES) -> list[Fact]:
        allowed = set(statuses)
        return [f for f in self._facts.values() if f.page_no == page_no and f.status in allowed]

    def page_metrics(self, page_no: int) -> list[Metric]:
        """该页**全部**数字配对，不按 status 过滤。

        V3 要拿同页所有兄弟标签做对照：哪怕某个配对本身不可用，它的区分词
        仍然能证明「句子里的对象说错了」。
        """
        return [
            f.metric for f in self._facts.values()
            if f.page_no == page_no and f.metric is not None
        ]

    def profile(self) -> dict:
        return self._profile

    def meta(self) -> dict:
        out = dict(self._meta)
        out["dense"] = out.get("dense") == "1"
        out["embed_dim"] = int(out.get("embed_dim") or 0)
        for k in ("counts", "status_counts"):
            if k in out:
                out[k] = json.loads(out[k])
        return out

    # ── 两路检索 ──
    def bm25(
        self, query: str, *, statuses=USABLE_STATUSES, k: int = 20
    ) -> list[tuple[str, float]]:
        """FTS5 BM25。返回 (fact_id, 分数)，分数已取负，越大越相关。"""
        q = tok.to_fts_query(query)
        statuses = tuple(statuses)
        if not q or not statuses:
            return []  # 查询全是虚词或标点：没有可匹配的词，不是错误
        marks = ",".join("?" * len(statuses))
        w = ", ".join(str(x) for x in BM25_WEIGHTS)
        sql = (
            f"SELECT facts_fts.fact_id, bm25(facts_fts, {w}) AS s "
            f"FROM facts_fts JOIN facts ON facts.fact_id = facts_fts.fact_id "
            f"WHERE facts_fts MATCH ? AND facts.status IN ({marks}) "
            f"ORDER BY s LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (q, *statuses, k)).fetchall()
        return [(fid, -float(s)) for fid, s in rows]

    def _dense_matrix(self, statuses: tuple[str, ...]) -> tuple[list[str], np.ndarray]:
        key = tuple(sorted(set(statuses)))
        if key not in self._dense_cache:
            marks = ",".join("?" * len(key))
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT e.fact_id, e.vec FROM embeddings e JOIN facts f "
                    f"ON f.fact_id = e.fact_id WHERE f.status IN ({marks}) ORDER BY f.rowid",
                    key,
                ).fetchall()
            ids = [r[0] for r in rows]
            dim = self.meta()["embed_dim"]
            mat = (
                np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
                if rows else np.zeros((0, dim), dtype=np.float32)
            )
            self._dense_cache[key] = (ids, mat)
        return self._dense_cache[key]

    def dense(
        self, qvec: np.ndarray, *, statuses=USABLE_STATUSES, k: int = 20
    ) -> list[tuple[str, float]]:
        """全量点积。向量入库前已 L2 归一化，点积即余弦。库里没有向量时返回空。"""
        if not self._meta.get("dense") == "1" or not tuple(statuses):
            return []
        ids, mat = self._dense_matrix(tuple(statuses))
        if not ids:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != mat.shape[1]:
            raise ValueError(f"查询向量维度 {q.shape[0]} 与库内 {mat.shape[1]} 不符（换了嵌入模型？）")
        q = q / (np.linalg.norm(q) or 1.0)
        scores = mat @ q
        top = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in top]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "RagStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
