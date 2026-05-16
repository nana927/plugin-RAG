from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Set

import faiss
import numpy as np

from .base import VectorStore
from ..types import ChunkRecord, SearchResult


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _kb_dir(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(base_dir, knowledge_base)


def _meta_path(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(_kb_dir(base_dir, knowledge_base), "metadata.json")


def _vector_path(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(_kb_dir(base_dir, knowledge_base), "vectors.npy")


def _index_path(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(_kb_dir(base_dir, knowledge_base), "index.faiss")


def _load_metadata(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class FaissStore(VectorStore):
    """
    Local FAISS vector store.

    This implementation assumes both faiss and numpy are installed. Search can
    optionally rerank FAISS candidates with LLM-extracted query keywords.
    """

    def __init__(self, storage_dir: str, debug: bool = False) -> None:
        self.storage_dir = storage_dir
        self.debug = debug

    def _load_index(self, knowledge_base: str, dim: int):
        index_file = _index_path(self.storage_dir, knowledge_base)
        if os.path.exists(index_file):
            return faiss.read_index(index_file)
        return faiss.IndexFlatIP(dim)

    def _save_index(self, knowledge_base: str, index) -> None:
        faiss.write_index(index, _index_path(self.storage_dir, knowledge_base))

    def upsert(self, records: List[ChunkRecord], vectors, knowledge_base: str) -> None:
        if not records:
            return

        kb_dir = _kb_dir(self.storage_dir, knowledge_base)
        _ensure_dir(kb_dir)

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")

        metadata_path = _meta_path(self.storage_dir, knowledge_base)
        vector_path = _vector_path(self.storage_dir, knowledge_base)

        existing_meta = _load_metadata(metadata_path)
        if os.path.exists(vector_path):
            existing_vecs = np.load(vector_path)
        else:
            existing_vecs = np.zeros((0, vectors.shape[1]), dtype=np.float32)

        new_meta = existing_meta + [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "doc_name": record.doc_name,
                "text": record.text,
                "metadata": record.metadata,
            }
            for record in records
        ]
        new_vecs = np.vstack([existing_vecs, vectors])

        np.save(vector_path, new_vecs)
        _save_metadata(metadata_path, new_meta)

        index = self._load_index(knowledge_base, vectors.shape[1])
        index.add(vectors)
        self._save_index(knowledge_base, index)


    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """
        将一组分数归一化到 0~1 区间。

        Args:
            scores: 原始分数列表。

        Returns:
            归一化后的分数列表。
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def search(
        self,
        query_vector,
        knowledge_base: str,
        top_k: int,
        query_text: Optional[str] = None,
        vector_weight: float = 0.7,
        candidate_k: Optional[int] = 30,
    ) -> List[SearchResult]:
        """
        在指定知识库中进行向量召回，并可选地结合关键词匹配进行重排序。

        检索流程：
        1. 使用 FAISS 根据 query_vector 召回 candidate_k 个候选文本块；
        2. 对候选文本块的向量相似度分数 vector_score 做 0~1 归一化；
        3. 如果提供 query_text，则提取查询关键词，并计算关键词匹配分数 term_score；
        4. 将归一化后的向量分数和关键词分数加权融合，得到最终分数；
        5. 按最终分数降序排序，返回 top_k 个结果。

        最终分数计算方式：
            final_score = vector_weight * normalized_vector_score
                        + (1 - vector_weight) * term_score

        注意：
        这里属于“向量召回 + 关键词重排”，不是完整的 BM25 混合检索。
        """

        vector_path = _vector_path(self.storage_dir, knowledge_base)
        metadata_path = _meta_path(self.storage_dir, knowledge_base)

        if not os.path.exists(vector_path) or not os.path.exists(metadata_path):
            return []

        vectors = np.load(vector_path)
        metadata = _load_metadata(metadata_path)
        if vectors.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        index = self._load_index(knowledge_base, vectors.shape[1])

        # FAISS 先召回更多候选，再进行关键词重排
        recall_k = candidate_k or top_k
        recall_k = max(top_k, min(recall_k, len(metadata)))

        scores, indices = index.search(query, recall_k)

        query_terms = self._extract_keywords(query_text or "")
        if self.debug:
            print(
                "[retrieval-debug] "
                f"kb={knowledge_base} top_k={top_k} recall_k={recall_k} "
                f"query={query_text!r} terms={sorted(query_terms)}"
            )
        vector_weight = self._clamp_weight(vector_weight)
        term_weight = 1.0 - vector_weight

        raw_candidates = []

        # 先收集候选结果，不立即计算最终分数
        for rank, index_id in enumerate(indices[0]):
            if index_id == -1:
                continue

            meta = metadata[index_id]
            vector_score = float(scores[0][rank])
            term_score = self._keyword_score(query_terms, meta.get("text", ""))

            raw_candidates.append({
                "meta": meta,
                "vector_score": vector_score,
                "term_score": term_score,
            })

            if self.debug:
                preview = meta.get("text", "").replace("\n", " ")[:120]
                print(
                    "[retrieval-debug] candidate "
                    f"rank={rank + 1} doc={meta.get('doc_name')} "
                    f"vector={vector_score:.4f} term={term_score:.4f} "
                    f"text={preview!r}"
                )

        if not raw_candidates:
            return []

        # 对 FAISS 返回的向量分数做候选集内 0~1 归一化
        vector_scores = [item["vector_score"] for item in raw_candidates]
        normalized_vector_scores = self._normalize_scores(vector_scores)

        candidates = []

        for item, normalized_vector_score in zip(raw_candidates, normalized_vector_scores):
            if query_terms:
                final_score = (
                    vector_weight * normalized_vector_score
                    + term_weight * item["term_score"]
                )
            else:
                # 没有关键词时，仍然使用原始向量分数排序
                final_score = item["vector_score"]

            result = self._to_result(item["meta"], final_score)

            # 保留中间分数，方便调试和分析检索结果
            result.vector_score = item["vector_score"]
            result.normalized_vector_score = normalized_vector_score
            result.term_score = item["term_score"]

            candidates.append(result)

            if self.debug:
                print(
                    "[retrieval-debug] final "
                    f"doc={result.file_name} score={result.score:.4f} "
                    f"normalized_vector={normalized_vector_score:.4f} "
                    f"term={item['term_score']:.4f}"
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def clear(self, knowledge_base: str) -> None:
        kb_dir = _kb_dir(self.storage_dir, knowledge_base)
        if not os.path.exists(kb_dir):
            return
        for name in ["metadata.json", "vectors.npy", "index.faiss"]:
            path = os.path.join(kb_dir, name)
            if os.path.exists(path):
                os.remove(path)

    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        kb_dir = _kb_dir(self.storage_dir, knowledge_base)
        metadata_path = _meta_path(self.storage_dir, knowledge_base)
        vector_path = _vector_path(self.storage_dir, knowledge_base)
        index_path = _index_path(self.storage_dir, knowledge_base)
        if not os.path.exists(metadata_path) or not os.path.exists(vector_path):
            return

        metadata = _load_metadata(metadata_path)
        vectors = np.load(vector_path)
        keep_indices = [
            idx
            for idx, item in enumerate(metadata)
            if item.get("doc_id") != doc_id
        ]
        if len(keep_indices) == len(metadata):
            return

        if not keep_indices:
            for path in [metadata_path, vector_path, index_path]:
                if os.path.exists(path):
                    os.remove(path)
            return

        new_metadata = [metadata[idx] for idx in keep_indices]
        new_vectors = vectors[keep_indices]
        _ensure_dir(kb_dir)
        _save_metadata(metadata_path, new_metadata)
        np.save(vector_path, new_vectors)

        index = faiss.IndexFlatIP(new_vectors.shape[1])
        index.add(new_vectors.astype(np.float32))
        self._save_index(knowledge_base, index)

    @staticmethod
    def _to_result(meta: dict, score: float) -> SearchResult:
        return SearchResult(
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            file_name=meta["doc_name"],
            text=meta["text"],
            score=score,
            metadata=meta.get("metadata"),
        )

    @staticmethod
    def _extract_keywords(query: str) -> Set[str]:
        """
        默认使用大模型抽取关键字，
        fall_back:_extract_keywords_regex
        """
        if not query:
            return set()
        try:
            return FaissStore._extract_keywords_with_llm(query)
        except Exception:
            return FaissStore._extract_keywords_regex(query)

    @staticmethod
    def _extract_keywords_with_llm(query: str) -> Set[str]:
        """
        用大模型抽取query关键字

        环境变量依赖:
            RAG_KEYWORD_API_KEY: keyword extraction API key.
            RAG_KEYWORD_BASE_URL: OpenAI-compatible base URL.
            RAG_KEYWORD_MODEL: model name, default qwen-plus.
        """
        api_key = (
            os.getenv("RAG_KEYWORD_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("Missing keyword extraction API key.")

        base_url = os.getenv(
            "RAG_KEYWORD_BASE_URL",
            os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        model = os.getenv("RAG_KEYWORD_MODEL", "qwen-plus")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是检索关键词提取器。只输出 JSON 数组，不要解释。"
                        "关键词应保留业务名词、指标名词、英文缩写、产品名、文件名、"
                        "缺陷类型、算法术语。去掉虚词和泛化问法。"
                    ),
                },
                {"role": "user", "content": f"从下面查询中提取 3 到 8 个检索关键词：\n{query}"},
            ],
        )
        content = response.choices[0].message.content or "[]"
        return FaissStore._parse_llm_keywords(content)

    @staticmethod
    def _parse_llm_keywords(content: str) -> Set[str]:
        """从文本中解析关键字列表"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", content)
            if not match:
                return FaissStore._extract_keywords_regex(content)
            data = json.loads(match.group(0))

        if not isinstance(data, list):
            return set()
        return {str(item).strip().lower() for item in data if str(item).strip()}

    @staticmethod
    def _extract_keywords_regex(query: str) -> Set[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query.lower())
        stopwords = {"的", "了", "是", "和", "与", "及", "在", "有", "吗", "么", "什么", "如何"}
        return {token for token in tokens if token and token not in stopwords}

    @classmethod
    def _keyword_score(cls, query_terms: Set[str], text: str) -> float:
        """
        计算keyword_score
        text 不 tokenize, 用 substring match
        """
        if not query_terms or not text:
            return 0.0

        normalized_text = text.lower()

        matched = sum(
            1 for term in query_terms
            if term in normalized_text
        )

        return matched / len(query_terms)

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight
