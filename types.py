from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ChunkRecord:
    """
    文档分片数据结构（入库用）。

    Fields:
        chunk_id: 分片唯一 ID。
        doc_id: 文档 ID。
        doc_name: 文档名。
        text: 分片文本。
        metadata: 额外元信息（可选）。
    """
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """
    检索结果结构。

    Fields:
        chunk_id: 命中的分片 ID。
        doc_id: 文档 ID。
        file_name: 文档名。
        text: 命中的文本。
        score: 相似度分数。
        vector_score: 向量相似度分数（混合检索时）。
        term_score: 词面相似度分数（混合检索时）。
        metadata: 额外元信息（可选）。
    """
    chunk_id: str
    doc_id: str
    file_name: str
    text: str
    score: float
    vector_score: Optional[float] = None
    term_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
