"""断句与引用解析：校验的最小单位是句子。

V1–V4 都是句级的，断句错一步，后面全错：
  - 引用 `[S4]` 写在句号前还是句号后，模型两种都会写。挂错句子，
    带引用的句子会被判成「无引用」（V4 误拦），没带的反而蒙混过去。
  - 小数点、`2026.4` 这类日期里的「.」不是句末，只认中文句末标点与换行。
  - 列表序号「1. 」会被数字正则当成数字去回查，必须先剥掉。
  - Markdown 标题不是论断，不进校验。
"""

from __future__ import annotations

import re

from ragwriter.contracts import Sentence

# 一组引用：[S4] / [S4][S5] / [S4, S5] / [S4，S5] / [S4、S5]
_CITE_GROUP = re.compile(r"\[\s*[Ss]\d+(?:\s*[,，、;；]\s*[Ss]\d+)*\s*\]")
_SID = re.compile(r"[Ss](\d+)")

# 句末标点。英文句点不算：小数、日期、域名里到处都是
_TERMINATORS = "。！？；!?"
# 句末标点后紧跟的收尾符号仍归本句
_CLOSERS = "”’」』）)】》\"'"

# 列表项标记：- / * / + / 1. / 1、/ 1) / > 引用块
# 「1.」后必须有空格：否则「3.55亿……」开头的句子会被剥掉「3.」
_LIST_MARK = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}\.\s+|\d{1,3}[、)）]\s*|>\s*)")


def parse_cites(text: str) -> list[str]:
    """按出现顺序取出全部引用编号，去重。"S4" 统一成大写。"""
    out: list[str] = []
    for g in _CITE_GROUP.finditer(text):
        for n in _SID.findall(g.group(0)):
            sid = f"S{int(n)}"
            if sid not in out:
                out.append(sid)
    return out


def strip_cites(text: str) -> str:
    """去掉引用标记，供数字回查与事实句判断用。

    「[S4]」里的 4 不能当成句中数字去回查，否则每句都会被 V2 拦。
    """
    s = _CITE_GROUP.sub("", text)
    s = re.sub(r"[ \t]+([。！？；，,!?])", r"\1", s)  # 「8 倍 [S5]。」去引用后留下的空格
    return s.strip()


_OPENERS = "“「『《（(‘"
_QUOTE_CLOSERS = "”」』》）)’"


def _split_line(line: str) -> list[str]:
    """一行内按句末标点切；标点后紧跟的收尾符号与引用组归前一句。

    引号、书名号、括号里的问号叹号不算句末（10-24 实测）：「第一层是“让AI记住什么？”，由……回答」
    被切成两截，前半截改写后后半截原样留下，成稿里出现「……回答[S41]。，由……回答」的乱码句。
    """
    pieces: list[str] = []
    i, start, n = 0, 0, len(line)
    depth = 0
    while i < n:
        ch = line[i]
        if ch in _OPENERS:
            depth += 1
        elif ch in _QUOTE_CLOSERS and depth:
            depth -= 1
        if ch in _TERMINATORS and depth == 0:
            j = i + 1
            while j < n and line[j] in _TERMINATORS:  # 「？！」连用
                j += 1
            while j < n and line[j] in _CLOSERS:
                j += 1
            # 「。[S4][S5]」「。 [S4, S5]」——句号后的引用属于前一句
            while True:
                k = j
                while k < n and line[k] in " \t":
                    k += 1
                m = _CITE_GROUP.match(line, k)
                if not m:
                    break
                j = m.end()
            pieces.append(line[start:j])
            start = i = j
            continue
        i += 1
    if start < n:
        pieces.append(line[start:])
    return [p.strip() for p in pieces if p.strip()]


def split_sentences(text: str) -> list[Sentence]:
    """把一节正文切成句子。

    规则：
      - 换行一定断句；空行、`#` 开头的标题行不算句子
      - 列表项各自成句，列表标记（`- ` `1. `）从句子文本里剥掉
      - 只有引用、没有正文的碎片（如换行后单独一行 `[S4]`）并回前一句
      - idx 从 0 起，按本段文本内的顺序
    """
    raw: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body = _LIST_MARK.sub("", line, count=1)
        for piece in _split_line(body):
            if not strip_cites(piece) and raw:
                raw[-1] = f"{raw[-1]}{piece}"
                continue
            raw.append(piece)
    return [Sentence(idx=i, text=s, cites=parse_cites(s)) for i, s in enumerate(raw)]
