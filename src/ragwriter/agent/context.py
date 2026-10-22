"""单次模型调用的上下文组装与 token 预算（10 号文档 5.2 节）。

为什么要程序强制截断，而不是交给模型「自己忽略多余内容」：
上下文一旦超长，中转要么报错要么静默截掉尾部——尾部恰好常是本轮任务。
所以这里按优先级从低往高砍：L3 工作窗口 → L1 品牌档案切片 → 检索片段，L0 与本轮任务永不砍。

只有检索片段可被引用；L1、L3 只做导航与对话状态（10 号文档第 2 节核心原则）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 粗估：一个中文字符 ≈ 1 token，一个英文词或数字串 ≈ 1 token。
# 不引 tiktoken：各家中转的分词器本来就不同，这里只需要量级正确、偏保守
_CJK = re.compile(r"[㐀-鿿豈-﫿　-〿＀-￯]")
_WORD = re.compile(r"[A-Za-z0-9_]+(?:[.\-'][A-Za-z0-9_]+)*")

DEFAULT_BUDGET = 5000  # 单次调用上限，见 5.2 节


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_CJK.findall(text)) + len(_WORD.findall(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按行保留到预算为止；单行本身超预算时再按字符切。

    按行切是因为检索片段与窗口都是一行一条，整行丢比半行残留更不容易误导模型。
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        cost = estimate_tokens(line)
        if used + cost <= max_tokens:
            kept.append(line)
            used += cost
            continue
        remain = max_tokens - used
        if remain > 0 and not kept:
            # 第一行就放不下：按字符切，至少给模型一点东西
            out = []
            for ch in line:
                if estimate_tokens("".join(out) + ch) > remain:
                    break
                out.append(ch)
            kept.append("".join(out))
        break
    return "\n".join(kept)


@dataclass
class CallContext:
    """组装结果。dropped 记下被砍掉的层，进运行日志，方便事后解释「模型为什么没看到 X」。"""

    system: str
    user: str
    tokens: int
    dropped: list[str]


def assemble(
    *,
    l0: str,
    task: str,
    snippets: str = "",
    l1: str = "",
    l3: str = "",
    budget: int = DEFAULT_BUDGET,
) -> CallContext:
    """按 5.2 节的版式拼 system / user，并按优先级 L0 > 检索片段 > L1 > L3 截断。"""
    dropped: list[str] = []
    fixed = estimate_tokens(l0) + estimate_tokens(task)
    remain = max(budget - fixed, 0)

    parts: dict[str, str] = {}
    # 从高到低分配剩余预算：高优先级先拿，低优先级拿剩下的
    for name, text in (("snippets", snippets), ("l1", l1), ("l3", l3)):
        if not text:
            parts[name] = ""
            continue
        cut = truncate_to_tokens(text, remain)
        if cut != text:
            dropped.append(name if not cut else f"{name}(部分)")
        parts[name] = cut
        remain -= estimate_tokens(cut)

    system = l0 if not parts["l1"] else f"{l0}\n\n# 品牌档案（仅作导航，不可引用）\n{parts['l1']}"
    user_blocks = []
    if parts["l3"]:
        user_blocks.append(f"# 对话状态（仅供理解指代，不可引用）\n{parts['l3']}")
    if parts["snippets"]:
        user_blocks.append(f"# 检索片段（唯一可引用的事实来源）\n{parts['snippets']}")
    user_blocks.append(f"# 本轮任务\n{task}")
    user = "\n\n".join(user_blocks)
    return CallContext(system=system, user=user,
                       tokens=estimate_tokens(system) + estimate_tokens(user), dropped=dropped)


# ── L1 品牌档案按节点切片（2.2 节末：各节点只看自己需要的部分）──────────────
def profile_slice(profile: dict, node: str) -> str:
    """意图识别只看 entities 与章节名；规划看全部 catalog；写作与改写只看 rules。"""
    profile = profile or {}
    lines: list[str] = []
    if node == "intent":
        ents = profile.get("entities") or {}
        if ents:
            lines.append(f"实体：{_flat(ents)}")
        chapters: list[str] = []
        for row in profile.get("catalog") or []:
            ch = row.get("chapter", "")
            if ch and ch not in chapters:
                chapters.append(ch)
        if chapters:
            lines.append("资料章节：" + "；".join(chapters))
    elif node == "planner":
        # 36 行全量约 2000 token，超预算；紧凑成「p9 标题」一行一页，章节名不重复写
        for row in profile.get("catalog") or []:
            lines.append(f"p{row.get('page_no')} {row.get('title', '')}")
    else:
        rules = profile.get("rules") or {}
        for key in ("corrections", "forbidden", "status_policy"):
            if rules.get(key):
                lines.append(f"{key}：{_flat(rules[key])}")
    return "\n".join(lines)


def _flat(obj: object) -> str:
    if isinstance(obj, dict):
        return "；".join(f"{k}={_flat(v)}" for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return "、".join(_flat(x) for x in obj)
    return str(obj)
