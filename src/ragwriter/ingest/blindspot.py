"""盲区探测：量化「图里有、文字层没有」的内容。

## 为什么需要它

系统只认文字层为事实源，图里的东西天然进不来。真正的风险不是「漏了」，
而是**漏了却不知道**——于是在该说「资料未涉及」的地方沉默，或者漏掉一个本该有的论点。

## 为什么不用视觉模型

不需要。要回答的是**计数问题**（图里有没有我们拿不到的事实），不是**理解问题**。
本机 OCR 就够，而且它离线、免费、确定性可复现——比调一次视觉模型更适合写进交付。

## 判据：靠几何，不靠启发式

一个数字串算「真盲区」，要同时满足三条：

1. 文字层里找不到它（按纯数字连写比对，躲开千分位与空格差异）
2. 它的位置落在某张内嵌图片的矩形内，重叠 > 50%
3. 它不压在任何文字块上（重叠 < 30%）——否则那是文字层文本被 OCR 误读

第 3 条是关键。只做前两条会有大量假阳性：英文 OCR 读中文会出乱码，
乱码里常偶然拼出数字串。但那些乱码的位置恰恰压在文字块上，几何能把它们摘干净。

## 声明的能力边界（必须随结果一起给出）

本机 Vision OCR 在开发机上只支持 `en-US` 与五种欧洲语言，**不支持中文**
（三个 revision 全部如此，经 `supportedRecognitionLanguagesForTextRecognitionLevel` 实测）。
所以探测器只回答「图里有没有拿不到的**数字与拉丁专名**」，回答不了中文要点。

这个边界是可接受的，理由：要点允许改写，漏一条不致命；
**数字不允许改写，漏一个就是品牌事故**。探测器正好覆盖了致命的那一类。

想要中文覆盖，两条路（都需要人工决定，本模块不自作主张）：
在「语言与地区」里加入中文让系统下载中文 OCR 模型，或安装 tesseract 的 `chi_sim`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ragwriter.models import Page

NUM_RE = re.compile(r"\d[\d,.]{2,}")  # 3 位及以上的数字串
MIN_IMAGE_OVERLAP = 0.5
MAX_TEXT_OVERLAP = 0.3


class OCRUnavailable(RuntimeError):
    """本机没有可用的 OCR。调用方应跳过盲区探测并在报告里写明。"""


@dataclass
class BlindSpot:
    page_no: int
    tokens: list[str]
    ocr_line: str
    image_overlap: float
    text_overlap: float


def ocr_available() -> bool:
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_languages() -> list[str]:
    """本机 OCR 支持的语言。写进报告，让读者知道结论的边界在哪。"""
    import Vision

    req = Vision.VNRecognizeTextRequest.alloc().init()
    langs, _err = (
        Vision.VNRecognizeTextRequest
        .supportedRecognitionLanguagesForTextRecognitionLevel_revision_error_(
            1, req.revision(), None
        )
    )
    return list(langs or [])


def _ocr_boxes(png: str | Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """返回 [(文本, 归一化 bbox(x, y, w, h))]，原点在左下（Vision 的约定）。"""
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(png))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        raise OCRUnavailable(f"读不出图片：{png}")
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(1)  # Accurate
    req.setUsesLanguageCorrection_(False)  # 关纠错：它会把数字「修」成单词
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise OCRUnavailable(str(err))
    out = []
    for obs in req.results() or []:
        bb = obs.boundingBox()
        out.append(
            (
                obs.topCandidates_(1)[0].string(),
                (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height),
            )
        )
    return out


def _to_page_rect(bb, width: float, height: float):
    x, y, w, h = bb
    return (x * width, (1 - y - h) * height, (x + w) * width, (1 - y) * height)


def _overlap_ratio(a, b) -> float:
    """b 覆盖了 a 的多少比例。"""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy) / area if area > 0 else 0.0


def detect(pdf_path: str | Path, pages: list[Page]) -> list[BlindSpot]:
    """逐页探测。pages 必须已渲染（image_path 有值）。"""
    if not ocr_available():
        raise OCRUnavailable("未安装 pyobjc-framework-Vision（本机 OCR 仅 macOS 可用）")

    doc = pymupdf.open(pdf_path)
    found: list[BlindSpot] = []
    for page in pages:
        if not page.image_path:
            continue
        pg = doc[page.page_no - 1]
        width, height = float(pg.rect.width), float(pg.rect.height)
        image_rects = [
            (r.x0, r.y0, r.x1, r.y1)
            for im in pg.get_images(full=True)
            for r in pg.get_image_rects(im[0])
        ]
        text_rects = [(b.x0, b.y0, b.x1, b.y1) for b in page.blocks]
        layer_digits = re.sub(r"\D", "", page.text)

        for line, bb in _ocr_boxes(page.image_path):
            tokens = [
                t
                for t in NUM_RE.findall(line)
                if re.sub(r"\D", "", t) and re.sub(r"\D", "", t) not in layer_digits
            ]
            if not tokens:
                continue
            rect = _to_page_rect(bb, width, height)
            in_image = max((_overlap_ratio(rect, ir) for ir in image_rects), default=0.0)
            on_text = max((_overlap_ratio(rect, tr) for tr in text_rects), default=0.0)
            if in_image > MIN_IMAGE_OVERLAP and on_text < MAX_TEXT_OVERLAP:
                found.append(
                    BlindSpot(
                        page_no=page.page_no,
                        tokens=sorted(set(tokens)),
                        ocr_line=line.strip(),
                        image_overlap=round(in_image, 2),
                        text_overlap=round(on_text, 2),
                    )
                )
    doc.close()
    return found
