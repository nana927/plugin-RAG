from __future__ import annotations

import warnings
import re
from typing import List, Optional, Set

from .base import VectorStore
from ..types import ChunkRecord, SearchResult


def _np():
    import numpy as np

    return np


def _expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """
    Milvus VARCHAR 的 max_length 按 UTF-8 字节长度校验。
    中文字符可能占多个字节，所以不能只用 Python 字符数切片。
    """
    raw = (value or "").encode("utf-8")
    if len(raw) <= max_bytes:
        return value or ""
    return raw[:max_bytes].decode("utf-8", errors="ignore")


class MilvusStore(VectorStore):
    """
    Milvus vector store.

    pymilvus is imported lazily here, so importing RAG or using the FAISS backend
    does not require Milvus dependencies.
    """

    def __init__(
        self,
        host: str,
        port: str,
        collection: str,
        text_max_len: int = 4096,
        debug: bool = False,
    ) -> None:
        self._milvus = self._load_milvus()
        self.host = host
        self.port = port
        self.collection_name = collection
        self.text_max_len = text_max_len
        self.debug = debug
        self._collection = None

    def _load_milvus(self):
        try:
            warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
            try:
                from pymilvus.exceptions import PyMilvusDeprecationWarning

                warnings.filterwarnings("ignore", category=PyMilvusDeprecationWarning)
            except Exception:
                pass
        except ImportError as exc:
            raise ImportError(
                "Milvus backend requires pymilvus. Install it with: pip install pymilvus"
            ) from exc

        return {
            "Collection": Collection,
            "CollectionSchema": CollectionSchema,
            "DataType": DataType,
            "FieldSchema": FieldSchema,
            "connections": connections,
            "utility": utility,
        }

    def _connect(self) -> None:
        self._milvus["connections"].connect(host=self.host, port=self.port)

    def _ensure_collection(self, dim: int):
        if self._collection is not None:
            return self._collection

        self._connect()
        if self._milvus["utility"].has_collection(self.collection_name):
            self._collection = self._milvus["Collection"](self.collection_name)
            return self._collection

        FieldSchema = self._milvus["FieldSchema"]
        CollectionSchema = self._milvus["CollectionSchema"]
        DataType = self._milvus["DataType"]
        Collection = self._milvus["Collection"]

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self.text_max_len),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields)
        self._collection = Collection(self.collection_name, schema, consistency_level="Strong")
        self._collection.create_index("vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
        self._collection.load()
        return self._collection

    def upsert(self, records: List[ChunkRecord], vectors, knowledge_base: str) -> None:
        if not records:
            return
        np = _np()
        vectors = np.asarray(vectors, dtype=np.float32)
        col = self._ensure_collection(vectors.shape[1])
        entities = [
            [_truncate_utf8(r.chunk_id, 128) for r in records],
            [_truncate_utf8(knowledge_base, 128) for _ in records],
            [_truncate_utf8(r.doc_id, 128) for r in records],
            [_truncate_utf8(r.doc_name, 256) for r in records],
            [_truncate_utf8(r.text, self.text_max_len) for r in records],
            vectors,
        ]
        col.insert(entities)
        col.flush()

    def search(
        self,
        query_vector,
        knowledge_base: str,
        top_k: int,
        query_text: Optional[str] = None,
        vector_weight: float = 0.7,
        candidate_k: Optional[int] = None,
    ) -> List[SearchResult]:
        np = _np()
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        col = self._ensure_collection(q.shape[1])
        expr = f"kb_id == '{knowledge_base}'"
        recall_k = candidate_k or top_k
        recall_k = max(top_k, recall_k)
        results = col.search(
            q,
            anns_field="vector",
            limit=recall_k,
            param={"metric_type": "IP", "params": {}},
            expr=expr,
            output_fields=["chunk_id", "doc_id", "doc_name", "text"],
        )[0]

        query_terms = self._extract_keywords(query_text or "")
        vector_weight = self._clamp_weight(vector_weight)
        term_weight = 1.0 - vector_weight

        raw_candidates = []
        for hit in results:
            fields = hit.fields
            raw_candidates.append(
                {
                    "fields": fields,
                    "vector_score": float(hit.score),
                    "term_score": self._keyword_score(query_terms, fields.get("text") or ""),
                }
            )

            if self.debug:
                preview = (fields.get("text") or "").replace("\n", " ")[:120]
                print(
                    "[retrieval-debug] candidate "
                    f"doc={fields.get('doc_name')} vector={float(hit.score):.4f} "
                    f"term={raw_candidates[-1]['term_score']:.4f} text={preview!r}"
                )

        if not raw_candidates:
            return []

        normalized_vector_scores = self._normalize_scores(
            [item["vector_score"] for item in raw_candidates]
        )

        out: List[SearchResult] = []
        for item, normalized_vector_score in zip(raw_candidates, normalized_vector_scores):
            fields = item["fields"]
            if query_terms:
                final_score = (
                    vector_weight * normalized_vector_score
                    + term_weight * item["term_score"]
                )
            else:
                final_score = item["vector_score"]

            result = SearchResult(
                chunk_id=fields.get("chunk_id"),
                doc_id=fields.get("doc_id"),
                file_name=fields.get("doc_name"),
                text=fields.get("text"),
                score=final_score,
                metadata=None,
            )
            result.vector_score = item["vector_score"]
            result.normalized_vector_score = normalized_vector_score
            result.term_score = item["term_score"]
            out.append(result)

            if self.debug:
                print(
                    "[retrieval-debug] final "
                    f"doc={result.file_name} score={result.score:.4f} "
                    f"normalized_vector={normalized_vector_score:.4f} term={item['term_score']:.4f}"
                )

        out.sort(key=lambda item: item.score, reverse=True)
        return out[:top_k]

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def _extract_keywords(query: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query.lower())
            if token
        }

    @classmethod
    def _keyword_score(cls, query_terms: Set[str], text: str) -> float:
        if not query_terms or not text:
            return 0.0
        normalized_text = text.lower()
        matched = sum(1 for term in query_terms if term in normalized_text)
        return matched / len(query_terms)

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight

    def has_document(self, knowledge_base: str, doc_id: str, doc_name: str) -> bool:
        self._connect()
        if not self._milvus["utility"].has_collection(self.collection_name):
            return False
        col = self._milvus["Collection"](self.collection_name)
        try:
            col.load()
            kb = _expr_value(knowledge_base)
            safe_doc_id = _expr_value(doc_id)
            expr = f"kb_id == '{kb}' and doc_id == '{safe_doc_id}'"
            return bool(col.query(expr=expr, output_fields=["chunk_id"], limit=1))
        except Exception:
            return False

    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        self._connect()
        if not self._milvus["utility"].has_collection(self.collection_name):
            return
        kb = _expr_value(knowledge_base)
        safe_doc_id = _expr_value(doc_id)
        col = self._milvus["Collection"](self.collection_name)
        col.delete(expr=f"kb_id == '{kb}' and doc_id == '{safe_doc_id}'")
        col.flush()

    def clear(self, knowledge_base: str) -> None:
        self._connect()
        if not self._milvus["utility"].has_collection(self.collection_name):
            return
        col = self._milvus["Collection"](self.collection_name)
        col.delete(expr=f"kb_id == '{knowledge_base}'")
        col.flush()
