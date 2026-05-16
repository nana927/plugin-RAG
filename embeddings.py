from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .config import RagConfig


if TYPE_CHECKING:
    import numpy as np


def _np():
    """延迟导入 numpy，避免未安装时直接报错。"""
    import numpy as np

    return np


def _normalize(vectors):
    """
    对向量做 L2 归一化，避免向量长度差异影响相似度。

    Args:
        vectors: shape = (N, dim) 的向量矩阵。

    Returns:
        归一化后的向量矩阵。
    """
    np = _np()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


class EmbeddingProvider(ABC):
    """Embedding 抽象接口，约定输入文本 -> 向量输出。"""
    @abstractmethod
    def embed_texts(self, texts: List[str]):
        """
        批量文本向量化。

        Args:
            texts: 文本列表。

        Returns:
            shape=(N, dim) 的向量矩阵。
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str):
        """
        单条查询向量化。

        Args:
            text: 查询文本。

        Returns:
            shape=(dim,) 的向量。
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class DashScopeEmbedding(EmbeddingProvider):
    """基于 DashScope (OpenAI 兼容) 的 embedding 实现。"""
    def __init__(self, config: RagConfig) -> None:
        from openai import OpenAI

        if not config.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for dashscope embedding.")

        self._client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
        )
        self._model = config.embedding_model
        self._dimensions = config.embedding_dimensions

    @property
    def dim(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: List[str]):
        """调用 DashScope 批量生成向量，并做归一化。"""
        np = _np()
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
            encoding_format="float",
        )
        vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """单条查询向量化。"""
        return self.embed_texts([text])[0]


class HashEmbedding(EmbeddingProvider):
    """
    Dependency-light fallback embedding.

    This is not as semantically strong as DashScope or sentence-transformers,
    but it lets the RAG package run locally without API keys or model downloads.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]):
        return [self._normalize_list(self._embed_one(text)) for text in texts]

    def embed_query(self, text: str):
        return self.embed_texts([text])[0]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self._dim
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        return vector

    def _tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        return tokens or list(text)

    def _normalize_list(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return vector
        return [value / norm for value in vector]


class LocalEmbedding(EmbeddingProvider):
    """基于 sentence-transformers 的本地 embedding 实现。"""
    def __init__(self, config: RagConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.local_embedding_model)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]):
        """本地模型批量向量化（已归一化）。"""
        np = _np()
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def build_embedding(config: RagConfig) -> EmbeddingProvider:
    """
    根据配置创建 embedding 实例。

    Args:
        config: RagConfig 配置。

    Returns:
        EmbeddingProvider 实例。
    """
    if config.embedding == "hash":
        return HashEmbedding()
    if config.embedding == "dashscope":
        return DashScopeEmbedding(config)
    if config.embedding == "local":
        return LocalEmbedding(config)
    raise ValueError(f"Unsupported embedding type: {config.embedding}")
