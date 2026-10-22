"""T3 稠密向量。

bge-m3 是可选依赖（`pip install -e '.[dense]'`，模型约 2GB）。没装或 USE_DENSE=0 时
`load_embedder` 返回 None，整条链路退到 BM25 单路，并在库的 meta 表里记下 dense=0，
检索时据此跳过稠密路——**退化必须是显式的**，不能让评审人以为跑的是混合检索。
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 (n, dim) float32，已 L2 归一化，点积即余弦。"""
        ...


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer  # 可选依赖，延迟导入

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # sentence-transformers 5.x 改名为 get_embedding_dimension，旧名仍可用但会告警
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        # bge-m3 的稠密检索不需要 bge-zh 那种「为这个句子生成表示」的查询指令前缀
        vecs = self._model.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def load_embedder(enabled: bool | None = None) -> Embedder | None:
    """按环境变量决定是否加载。返回 None 表示走 BM25 单路。"""
    if enabled is None:
        enabled = os.getenv("USE_DENSE", "1").strip() not in {"0", "false", ""}
    if not enabled:
        return None
    try:
        return BgeM3Embedder(os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip())
    except ImportError:
        return None
