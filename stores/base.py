from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..types import ChunkRecord, SearchResult


class VectorStore(ABC):
    """
    向量存储抽象层：对接 FAISS/Milvus 等后端。

    约定：
    - upsert: 写入/追加向量与元数据
    - search: 向量检索
    - clear: 清理某个知识库的数据
    """
    @abstractmethod
    def upsert(self, records: List[ChunkRecord], vectors, knowledge_base: str) -> None:
        """
        写入向量及元数据。

        Args:
            records: ChunkRecord 列表。
            vectors: shape=(N, dim) 的向量矩阵。
            knowledge_base: 知识库标识。
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        """Delete one document and all its chunks from a knowledge base."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector,
        knowledge_base: str,
        top_k: int,
        query_text: Optional[str] = None,
        vector_weight: float = 0.7,
        candidate_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        向量检索。

        Args:
            query_vector: shape=(dim,) 的查询向量。
            knowledge_base: 知识库标识。
            top_k: 返回条数。

        Returns:
            SearchResult 列表。
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, knowledge_base: str) -> None:
        """删除指定知识库的数据。"""
        raise NotImplementedError
