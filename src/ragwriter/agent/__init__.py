"""会话 Agent 编排层（10 号文档）：意图识别 → 路由处理器 → SSE 事件流 → 会话回写。

对外入口只有 `Orchestrator(deps).handle(message, session_id=..., options=...)`；
依赖经 `Deps` 注入，生产用 `Deps.from_env()`，测试用 `ragwriter.agent.testing.make_fake_deps()`。
"""

from ragwriter.agent.intent import SUGGESTED_TOPICS, IntentResult, classify
from ragwriter.agent.memory import SessionStore, WorkingWindow
from ragwriter.agent.orchestrator import Deps, Orchestrator

__all__ = ["Deps", "Orchestrator", "SessionStore", "WorkingWindow", "IntentResult", "classify",
           "SUGGESTED_TOPICS"]
