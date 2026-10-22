"""命令行入口：`ragwriter <子命令>`。

用 argparse 子命令结构，每个子命令一个 `_cmd_xxx(args)` 加一段注册代码；
其它模块要加命令（如 serve、eval）就在 `_build_parser` 里照样追加一段，不用改分发逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BUILD = "build"
DEFAULT_DB = "build/index/rag.db"
DEFAULT_DOCS = "docs"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _embedder_for(store, no_dense: bool):
    """库里有向量才加载模型：bge-m3 加载要十几秒，BM25 单路的库没必要等。"""
    if no_dense or not store.meta().get("dense"):
        return None
    from ragwriter.index.embed import load_embedder

    emb = load_embedder()
    if emb is None:
        print("[warn] 库里有向量但嵌入模型加载失败，退到 BM25 单路", file=sys.stderr)
    return emb


# ── index ──
def _cmd_index(args: argparse.Namespace) -> int:
    from ragwriter.index.store import build_index

    embedder = None
    if not args.no_dense:
        from ragwriter.index.embed import load_embedder

        embedder = load_embedder()
        if embedder is None:
            print("[warn] 稠密模型不可用（未装 dense 依赖或 USE_DENSE=0），只建 BM25", file=sys.stderr)
    stats = build_index(
        Path(args.build_dir), Path(args.db), embedder=embedder, docs_dir=Path(args.docs_dir)
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


# ── search ──
def _cmd_search(args: argparse.Namespace) -> int:
    from ragwriter.contracts import USABLE_STATUSES
    from ragwriter.index.store import RagStore
    from ragwriter.retrieve.hybrid import search

    statuses = ("verified", "image_structural", "image_only", "unverified") if args.all_status else USABLE_STATUSES
    with RagStore(args.db) as store:
        embedder = _embedder_for(store, args.no_dense)
        res = search(store, args.query, embedder=embedder, statuses=statuses, k_final=args.k)
        for i, h in enumerate(res.hits, start=1):
            f = h.fact
            text = f.text.replace("\n", " ")[:60]
            print(
                f"{i:>2}. {f.fact_id}  p{f.page_no:<2} {f.status:<16} "
                f"bm25={h.bm25_rank if h.bm25_rank is not None else '-':>2} "
                f"dense={h.dense_rank if h.dense_rank is not None else '-':>2} "
                f"rrf={h.rrf_score:.4f}  {text}"
            )
        print(f"sufficient={res.sufficient} top_score={res.top_score:.4f} dense_used={res.dense_used}")
        if res.signal:
            print(f"signal: {res.signal}")
    return 0


# ── calibrate ──
def _cmd_calibrate(args: argparse.Namespace) -> int:
    from ragwriter.index.store import RagStore
    from ragwriter.retrieve.hybrid import calibrate

    with RagStore(args.db) as store:
        embedder = _embedder_for(store, args.no_dense)
        rep = calibrate(store, embedder)
    for name in ("answerable", "traps"):
        print(f"== {name}")
        for r in rep[name]:
            d = "-" if r["dense_top1"] is None else f"{r['dense_top1']:.3f}"
            print(f"  dense_top1={d}  coverage={r['coverage']:.2f}  top_pages={r['top_pages']}  {r['query']}")
    print("dense_top1 split:", rep["dense_top1"])
    print("coverage   split:", rep["coverage"])
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ragwriter", description="万悉品宣 RAG 写作系统")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="从 build/ 产物建 SQLite 索引库")
    p.add_argument("--build-dir", default=DEFAULT_BUILD)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--docs-dir", default=DEFAULT_DOCS, help="structured-content.md 所在目录，品牌档案的规则来源")
    p.add_argument("--no-dense", action="store_true", help="不建向量，只建 BM25")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("search", help="混合检索，打印每条命中的两路名次")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8, help="融合后保留条数")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--all-status", action="store_true", help="包含 image_only / unverified")
    p.add_argument("--no-dense", action="store_true", help="即使库里有向量也只跑 BM25")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("calibrate", help="在内置标注集上校准充分性阈值 τ")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--no-dense", action="store_true")
    p.set_defaults(func=_cmd_calibrate)

    # 参数原样转给 eval.evaluate.main_eval：--runs X.jsonl | --live [--save-runs]、--no-judge、--strict、--out、--json
    p = sub.add_parser("eval", help="按意图评估多轮对话用例，写 docs/eval-report.md")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_eval)
    return ap


def _cmd_eval(args: argparse.Namespace) -> int:
    from ragwriter.eval.evaluate import main_eval  # 延迟导入：评估依赖编排器全链路

    return int(main_eval(args.rest) or 0)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        # argparse 的 REMAINDER 接不住以 -- 开头的参数，评估的参数原样转交
        return _cmd_eval(argparse.Namespace(rest=argv[1:]))
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
