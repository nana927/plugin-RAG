from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from ..types import ChunkRecord, SearchResult


STOPWORDS = {
    "的",
    "了",
    "和",
    "或",
    "及",
    "以及",
    "与",
    "在",
    "是",
    "为",
    "下",
    "中",
    "上",
    "对",
    "按",
    "把",
    "请",
    "根据",
    "这个",
    "本次",
}

SPECIAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_./%*-]+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _kb_dir(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(base_dir, knowledge_base)


def _index_path(base_dir: str, knowledge_base: str) -> str:
    return os.path.join(_kb_dir(base_dir, knowledge_base), "bm25_index.json")


def _jieba_cut(text: str) -> List[str]:
    try:
        import jieba

        return list(jieba.cut(text))
    except ImportError as exc:
        raise ImportError("BM25 Chinese tokenization requires jieba. Install it with: pip install jieba") from exc


def _is_valid_token(token: str) -> bool:
    if not token or token in STOPWORDS or token.isspace():
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", token))


def _tokenize(text: str) -> List[str]:
    text = text or ""
    lowered = text.lower()

    special_tokens = [token.lower() for token in SPECIAL_TOKEN_RE.findall(lowered)]
    jieba_tokens = [
        token.strip().lower()
        for token in _jieba_cut(text)
        if token.strip()
    ]
    chinese_bigrams = []
    for match in CHINESE_RE.findall(text):
        chinese_bigrams.extend(
            match[idx : idx + 2]
            for idx in range(len(match) - 1)
        )

    tokens = special_tokens + jieba_tokens + chinese_bigrams
    return [token for token in tokens if _is_valid_token(token)]


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"records": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Bm25Store:
    """
    Local BM25 lexical index.

    It is intentionally independent from FAISS/Milvus, so the same BM25 recall
    path can be used with either vector backend.
    """

    def __init__(self, storage_dir: str, k1: float = 1.5, b: float = 0.75) -> None:
        self.storage_dir = storage_dir
        self.k1 = k1
        self.b = b

    def upsert(self, records: List[ChunkRecord], knowledge_base: str) -> None:
        if not records:
            return

        kb_dir = _kb_dir(self.storage_dir, knowledge_base)
        _ensure_dir(kb_dir)
        path = _index_path(self.storage_dir, knowledge_base)
        data = _load_json(path)

        by_chunk_id = {
            item["chunk_id"]: item
            for item in data.get("records", [])
            if item.get("chunk_id")
        }
        for record in records:
            tokens = _tokenize(record.text)
            by_chunk_id[record.chunk_id] = {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "doc_name": record.doc_name,
                "text": record.text,
                "metadata": record.metadata,
                "tokens": tokens,
                "token_counts": dict(Counter(tokens)),
                "length": len(tokens),
            }

        _save_json(path, {"records": list(by_chunk_id.values())})

    def search(
        self,
        query: str,
        knowledge_base: str,
        top_k: int,
    ) -> List[SearchResult]:
        path = _index_path(self.storage_dir, knowledge_base)
        data = _load_json(path)
        records = data.get("records", [])
        if not records:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        total_docs = len(records)
        avg_len = sum(item.get("length", 0) for item in records) / max(total_docs, 1)
        df = self._document_frequency(records)
        query_counts = Counter(query_terms)

        scored = []
        for item in records:
            score = self._score_record(item, query_counts, df, total_docs, avg_len)
            if score <= 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchResult(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                file_name=item["doc_name"],
                text=item["text"],
                score=float(score),
                vector_score=None,
                term_score=float(score),
                metadata=item.get("metadata"),
            )
            for score, item in scored[:top_k]
        ]

    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        path = _index_path(self.storage_dir, knowledge_base)
        if not os.path.exists(path):
            return
        data = _load_json(path)
        records = [
            item
            for item in data.get("records", [])
            if item.get("doc_id") != doc_id
        ]
        if records:
            _save_json(path, {"records": records})
        else:
            os.remove(path)

    def clear(self, knowledge_base: str) -> None:
        path = _index_path(self.storage_dir, knowledge_base)
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _document_frequency(records: List[dict]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for item in records:
            for token in set(item.get("tokens", [])):
                df[token] = df.get(token, 0) + 1
        return df

    def _score_record(
        self,
        record: dict,
        query_counts: Counter,
        df: Dict[str, int],
        total_docs: int,
        avg_len: float,
    ) -> float:
        score = 0.0
        doc_len = record.get("length", 0)
        token_counts = record.get("token_counts", {})
        for term, query_tf in query_counts.items():
            tf = token_counts.get(term, 0)
            if tf <= 0:
                continue
            idf = math.log(1.0 + (total_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            norm = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(avg_len, 1.0))
            score += query_tf * idf * (tf * (self.k1 + 1.0)) / norm
        return score
