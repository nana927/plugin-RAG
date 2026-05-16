from __future__ import annotations

import re
from typing import List

from .config import RagConfig
from .embeddings import build_embedding
from .pre_process.qa_generator import QA_RECORD_TYPE, QA_TEXT_PREFIX
from .pre_process.query_decomposer import decompose_query
from .stores import Bm25Store, FaissStore, VectorStore
from .types import ChunkRecord, SearchResult


class RagService:
    """
    RAG 服务入口：统一封装 embedding + 向量库后端。

    使用方式：
    - add_documents: 入库文档分片
    - retrieve: 查询检索
    - clear: 清理指定知识库
    """
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.embedding = build_embedding(config)
        self.store = self._build_store(config)
        self.bm25_store = Bm25Store(
            config.storage_dir,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )

    def _build_store(self, config: RagConfig) -> VectorStore:
        """根据配置初始化向量库后端。"""
        if config.backend == "faiss":
            return FaissStore(config.storage_dir, debug=config.retrieval_debug)
        if config.backend == "milvus":
            from .stores import MilvusStore

            return MilvusStore(
                host=config.milvus_host,
                port=config.milvus_port,
                collection=config.milvus_collection,
                debug=config.retrieval_debug,
            )
        raise ValueError(f"Unsupported backend: {config.backend}")

    def add_documents(self, records: List[ChunkRecord], knowledge_base: str) -> None:
        """
        入库文档分片。

        Args:
            records: ChunkRecord 列表。
            knowledge_base: 知识库标识。
        """
        texts = [record.text for record in records]
        vectors = self.embedding.embed_texts(texts)
        self.store.upsert(records, vectors, knowledge_base)
        self.bm25_store.upsert(records, knowledge_base)

    def retrieve(self, query: str, knowledge_base: str, top_k: int = 5) -> List[SearchResult]:
        """
        检索接口。

        Args:
            query: 查询文本。
            knowledge_base: 知识库标识。
            top_k: 返回条数。

        Returns:
            SearchResult 列表。
        """
        subqueries = decompose_query(query, self.config)
        if len(subqueries) <= 1:
            return self._retrieve_one(query, knowledge_base, top_k)

        results: List[SearchResult] = []
        seen = set()
        per_query_k = top_k
        for subquery in subqueries:
            for item in self._retrieve_one(subquery, knowledge_base, per_query_k):
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                metadata = dict(item.metadata or {})
                metadata["matched_subquery"] = subquery
                metadata["original_query"] = query
                item.metadata = metadata
                results.append(item)
        return results[: max(top_k, len(subqueries))]

    def _retrieve_one(self, query: str, knowledge_base: str, top_k: int) -> List[SearchResult]:
        query_vector = self.embedding.embed_query(query)
        candidates = self._retrieve_base(query, query_vector, knowledge_base, top_k)
        return [self._format_qa_result(item) if self._is_qa_result(item) else item for item in candidates]

    def _retrieve_base(
        self,
        query: str,
        query_vector,
        knowledge_base: str,
        top_k: int,
    ) -> List[SearchResult]:
        if not self.config.hybrid_enabled:
            return self.store.search(query_vector, knowledge_base, top_k)

        candidate_k = max(top_k, self.config.hybrid_candidate_k)
        vector_results = self.store.search(query_vector, knowledge_base, candidate_k)
        bm25_results = self.bm25_store.search(query, knowledge_base, candidate_k)
        return self._merge_hybrid_results(
            vector_results,
            bm25_results,
            top_k,
            self.config.hybrid_vector_weight,
        )

    @staticmethod
    def _is_qa_result(item: SearchResult) -> bool:
        metadata = item.metadata or {}
        return metadata.get("record_type") == QA_RECORD_TYPE or item.text.startswith(QA_TEXT_PREFIX)

    @staticmethod
    def _format_qa_result(item: SearchResult) -> SearchResult:
        metadata = item.metadata or {}
        question = metadata.get("qa_question")
        answer = metadata.get("qa_answer")
        if question and answer:
            item.text = f"{QA_TEXT_PREFIX}\n命中的预生成问题：{question}\n可加入上下文的答案：{answer}"
        return item

    def clear(self, knowledge_base: str) -> None:
        """清理指定知识库的数据。"""
        self.store.clear(knowledge_base)
        self.bm25_store.clear(knowledge_base)

    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        """Delete one document from the knowledge base by doc_id."""
        self.store.delete_document(knowledge_base, doc_id)
        self.bm25_store.delete_document(knowledge_base, doc_id)

    @classmethod
    def _merge_hybrid_results(
        cls,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int,
        vector_weight: float,
    ) -> List[SearchResult]:
        vector_scores = cls._normalize_scores([item.score for item in vector_results])
        bm25_scores = cls._normalize_scores([item.score for item in bm25_results])
        merged = {}

        for item, normalized_score in zip(vector_results, vector_scores):
            merged[item.chunk_id] = {
                "result": item,
                "vector_score": item.score,
                "normalized_vector_score": normalized_score,
                "bm25_score": 0.0,
                "normalized_bm25_score": 0.0,
            }

        for item, normalized_score in zip(bm25_results, bm25_scores):
            if item.chunk_id not in merged:
                merged[item.chunk_id] = {
                    "result": item,
                    "vector_score": 0.0,
                    "normalized_vector_score": 0.0,
                    "bm25_score": item.score,
                    "normalized_bm25_score": normalized_score,
                }
            else:
                merged[item.chunk_id]["bm25_score"] = item.score
                merged[item.chunk_id]["normalized_bm25_score"] = normalized_score

        vector_weight = cls._clamp_weight(vector_weight)
        bm25_weight = 1.0 - vector_weight
        results: List[SearchResult] = []
        for item in merged.values():
            result = item["result"]
            result.vector_score = item["vector_score"]
            result.term_score = item["bm25_score"]
            result.normalized_vector_score = item["normalized_vector_score"]
            result.normalized_term_score = item["normalized_bm25_score"]
            result.score = (
                vector_weight * item["normalized_vector_score"]
                + bm25_weight * item["normalized_bm25_score"]
            )
            results.append(result)

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

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
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())

    @staticmethod
    def _term_similarity(query_tokens: List[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(RagService._tokenize(text))
        if not text_tokens:
            return 0.0
        overlap = 0
        for token in query_tokens:
            if token in text_tokens:
                overlap += 1
        return overlap / max(len(query_tokens), 1)

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight


def get_default_rag_service() -> RagService:
    """使用默认配置创建 RagService。"""
    return RagService(RagConfig())
