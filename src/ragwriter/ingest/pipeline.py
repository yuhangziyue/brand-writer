"""离线建库主流程：PDF → 页 → 页卡 → 交叉校验 → 事实库。

结果全部入库（`build/`），评审人可以直接用，不必自己跑模型。
页卡按页缓存，重跑只补缺的那几页。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ragwriter.chunk.builder import attach_discriminators, build_facts, build_vision_facts
from ragwriter.ingest.crosscheck import check_page_card
from ragwriter.ingest.extract import extract_pages
from ragwriter.ingest.geometry import pair_metrics
from ragwriter.ingest.pagecard import PageCard, build_page_card
from ragwriter.llm.client import LLMClient
from ragwriter.models import Fact, Page

MAX_WORKERS = 6


def _card_path(build_dir: Path, page_no: int) -> Path:
    return build_dir / "cards" / f"p{page_no:02d}.json"


def load_or_build_card(
    client: LLMClient, page: Page, build_dir: Path, *, use_vision: bool, refresh: bool
) -> PageCard:
    cache = _card_path(build_dir, page.page_no)
    if cache.exists() and not refresh:
        return PageCard(**json.loads(cache.read_text(encoding="utf-8")))
    card = build_page_card(client, page, use_vision=use_vision)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return card


def run_build(
    pdf_path: str | Path,
    build_dir: str | Path,
    client: LLMClient,
    *,
    use_vision: bool = False,
    refresh: bool = False,
) -> tuple[list[Page], list[Fact], dict]:
    build_dir = Path(build_dir)
    pages = extract_pages(pdf_path, render_dir=build_dir / "pages")

    def work(page: Page) -> tuple[Page, PageCard]:
        if page.is_divider:
            return page, PageCard(page_no=page.page_no, page_topic=page.title)
        return page, load_or_build_card(
            client, page, build_dir, use_vision=use_vision, refresh=refresh
        )

    cards: dict[int, PageCard] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for page, card in pool.map(work, pages):
            cards[page.page_no] = card

    all_facts: list[Fact] = []
    report: dict = {"pages": [], "totals": {}}
    vision_dir = build_dir / "vision-cards"
    for page in pages:
        geo = pair_metrics(page.blocks)
        checked = check_page_card(cards[page.page_no], page, geo)
        facts = build_facts(page, checked)
        # 视觉复核的产出：流程图关系与图内事实。文件存在才合并，
        # 没有视觉 provider 的评审人跑 make build 也不会报错，只是少这部分
        vcard_path = vision_dir / f"p{page.page_no:02d}.json"
        if vcard_path.exists():
            vcard = json.loads(vcard_path.read_text(encoding="utf-8"))
            facts.extend(build_vision_facts(page, vcard, start_seq=len(facts)))
        all_facts.extend(facts)
        report["pages"].append(
            {
                "page_no": page.page_no,
                "title": page.title,
                "chapter": page.chapter,
                "is_divider": page.is_divider,
                "geometry_metrics": [
                    {"value": m.value, "label": m.label} for m in geo
                ],
                "card_metrics": [
                    {"value": m.value, "label": m.label, "verdict": v}
                    for m, v in checked["metrics"]
                ],
                "point_verdicts": [v for _, v in checked["points"]],
                "image_only_items": checked["image_only_items"],
                "fact_count": len(facts),
            }
        )
    attach_discriminators(all_facts)

    report["totals"] = {
        "pages": len(pages),
        "facts": len(all_facts),
        "verified": sum(1 for f in all_facts if f.status == "verified"),
        "image_structural": sum(1 for f in all_facts if f.status == "image_structural"),
        "image_only": sum(1 for f in all_facts if f.status == "image_only"),
        "unverified": sum(1 for f in all_facts if f.status == "unverified"),
        "metric_facts": sum(1 for f in all_facts if f.kind == "metric"),
        "flow_facts": sum(1 for f in all_facts if f.kind == "flow"),
    }

    build_dir.mkdir(parents=True, exist_ok=True)
    with (build_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(p.model_dump_json() + "\n")
    with (build_dir / "facts.jsonl").open("w", encoding="utf-8") as f:
        for fact in all_facts:
            f.write(fact.model_dump_json() + "\n")
    (build_dir / "corpus-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pages, all_facts, report
