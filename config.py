from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_inline_comment(value.strip()).strip('"').strip("'")
            os.environ.setdefault(key, value)


def _strip_inline_comment(value: str) -> str:
    quote = None
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None:
            return value[:idx].strip()
    return value.strip()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_optional_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() == "true"


_rag_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_rag_dir)
_load_env_file(os.path.join(_backend_dir, ".env"))
_load_env_file(os.path.join(_rag_dir, ".env"))


@dataclass
class RagConfig:
    """
    RAG 配置对象。

    字段说明：
    - storage_dir: FAISS 本地存储目录（保存向量与元信息）。
    - backend: 向量库后端，支持 "faiss" / "milvus"。
    - embedding: 向量生成方式，支持 "dashscope" / "local"。

    DashScope 相关：
    - embedding_model: DashScope embedding 模型名。
    - embedding_dimensions: 期望的向量维度（DashScope embedding 可指定）。
    - dashscope_api_key: DashScope API Key（从环境变量读取）。
    - dashscope_base_url: DashScope 兼容 API 地址。

    本地 embedding 相关：
    - local_embedding_model: sentence-transformers 模型名称或路径。

    Milvus 相关：
    - milvus_host/milvus_port: Milvus 服务地址。
    - milvus_collection: Milvus collection 名称。

    Hybrid 检索相关：
    - hybrid_enabled: 是否启用混合检索（向量 + 词面相似度）。
    - hybrid_vector_weight: 向量相似度权重（0~1）。
    - hybrid_candidate_k: 先召回的候选数，用于混合重排。
    """
    storage_dir: str = _env_str("RAG_STORAGE_DIR", "RAG/storage")
    knowledge_base: str = _env_str("RAG_KNOWLEDGE_BASE", "resume_kb")
    demo_knowledge_base: str = _env_str("RAG_DEMO_KNOWLEDGE_BASE", "demo_kb")
    backend: str = _env_str("RAG_BACKEND", "faiss")  # faiss | milvus
    embedding: str = _env_str("RAG_EMBEDDING", "hash")  # hash | dashscope | local
    chunker: str = _env_str("RAG_CHUNKER", "simple")  # simple | deepdoc
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 120)

    # Embedding settings
    embedding_model: str = _env_str("RAG_EMBEDDING_MODEL", "text-embedding-v3")
    embedding_dimensions: int = _env_int("RAG_EMBEDDING_DIM", 1024)
    dashscope_api_key: Optional[str] = _env_optional_str("DASHSCOPE_API_KEY")
    dashscope_base_url: str = _env_str(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Local embedding
    local_embedding_model: str = _env_str(
        "RAG_LOCAL_EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5",
    )

    # Milvus settings
    milvus_host: str = _env_str("MILVUS_HOST", "localhost")
    milvus_port: str = _env_str("MILVUS_PORT", "19530")
    milvus_collection: str = _env_str("MILVUS_COLLECTION", "rag_chunks")

    # Hybrid retrieval
    hybrid_enabled: bool = _env_bool("RAG_HYBRID_ENABLED", False)
    hybrid_vector_weight: float = _env_float("RAG_HYBRID_VECTOR_WEIGHT", 0.7)
    hybrid_candidate_k: int = _env_int("RAG_HYBRID_CANDIDATE_K", 20)
    bm25_k1: float = _env_float("RAG_BM25_K1", 1.5)
    bm25_b: float = _env_float("RAG_BM25_B", 0.75)
    retrieval_debug: bool = _env_bool("RAG_RETRIEVAL_DEBUG", False)

    # Query decomposition
    query_decomposition_enabled: bool = _env_bool("RAG_QUERY_DECOMPOSITION_ENABLED", True)
    query_decomposition_api_key: Optional[str] = _env_optional_str("RAG_QUERY_DECOMPOSITION_API_KEY")
    query_decomposition_base_url: str = _env_str(
        "RAG_QUERY_DECOMPOSITION_BASE_URL",
        _env_str("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    query_decomposition_model: str = _env_str("RAG_QUERY_DECOMPOSITION_MODEL", "qwen-plus")
    query_decomposition_max_subqueries: int = _env_int("RAG_QUERY_DECOMPOSITION_MAX_SUBQUERIES", 4)

    # Generated QA index
    qa_generation_enabled: bool = _env_bool("RAG_QA_GENERATION_ENABLED", False)
    qa_generation_api_key: Optional[str] = _env_optional_str("RAG_QA_GENERATION_API_KEY")
    qa_generation_base_url: str = _env_str(
        "RAG_QA_GENERATION_BASE_URL",
        _env_str("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    qa_generation_model: str = _env_str("RAG_QA_GENERATION_MODEL", "qwen-max")
    qa_generation_temperature: float = _env_float("RAG_QA_GENERATION_TEMPERATURE", 0.2)
    qa_generation_max_context_chars: int = _env_int("RAG_QA_GENERATION_MAX_CONTEXT_CHARS", 8000)
    qa_generation_max_pairs: int = _env_int("RAG_QA_GENERATION_MAX_PAIRS", 20)
    qa_pdf_preview_summary_enabled: bool = _env_bool("RAG_QA_PDF_PREVIEW_SUMMARY_ENABLED", True)
    qa_pdf_preview_pages: int = _env_int("RAG_QA_PDF_PREVIEW_PAGES", 4)
    qa_pdf_preview_max_chars: int = _env_int("RAG_QA_PDF_PREVIEW_MAX_CHARS", 12000)
    qa_pdf_preview_summary_max_chars: int = _env_int("RAG_QA_PDF_PREVIEW_SUMMARY_MAX_CHARS", 1200)
