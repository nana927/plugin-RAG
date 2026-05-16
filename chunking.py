from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import os
from typing import Any, Dict, List

import importlib
import sys

from .config import RagConfig


class Chunker(ABC):
    """
    文档切分抽象层。

    约定：
    - chunk_file 返回原始 DeepDoc/RAGFlow 风格的 chunks 列表
    """

    @abstractmethod
    def chunk_file(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        对指定文件进行解析与切分。

        Args:
            file_path: 文件路径
            **kwargs: 解析器额外参数（如 layout_recognize）

        Returns:
            List[dict]，每个元素包含 content_with_weight 等字段
        """
        raise NotImplementedError


class DeepDocChunker(Chunker):
    """
    DeepDoc 切分实现：复用项目内的 DeepDoc 解析逻辑。

    使用 service.core.rag.app.manual.chunk 作为默认切分方式，
    行为与原项目保持一致。
    """

    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def chunk_file(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        try:
            rag_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(rag_dir)
            app_dir = os.path.join(backend_dir, "app")
            for path in (rag_dir, app_dir):
                if path not in sys.path:
                    sys.path.insert(0, path)
            chunk = importlib.import_module("parser.manual").chunk
        except Exception as exc:
            raise ImportError(
                "DeepDoc chunker is optional and could not be imported. "
                "Install DeepDoc dependencies and use Python 3.8+ before setting RAG_CHUNKER=deepdoc."
            ) from exc

        def _dummy(prog=None, msg=""):
            return None

        return chunk(file_path, callback=_dummy, **kwargs)


class SimpleChunker(Chunker):
    """
    Lightweight default chunker.

    Supports text-like files without importing DeepDOC, OCR, ONNX or beartype.
    It is intended for local smoke tests and simple knowledge-base documents.
    """

    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def chunk_file(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        text = self._read_file(file_path)
        chunks = self._split_text(text)
        doc_id = hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()
        doc_name = os.path.basename(file_path)
        return [
            {
                "chunk_id": hashlib.sha1(f"{doc_id}:{idx}:{chunk[:128]}".encode("utf-8")).hexdigest(),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "content_with_weight": chunk,
                "metadata": {"source_path": file_path, "parser": "simple"},
            }
            for idx, chunk in enumerate(chunks)
        ]

    def _read_file(self, file_path: str) -> str:
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), ensure_ascii=False, indent=2)
        if suffix in {".txt", ".md", ".markdown", ".py", ".sql", ".csv"}:
            return self._read_text(file_path)
        raise ValueError(
            f"SimpleChunker only supports text-like files. Use RAG_CHUNKER=deepdoc for: {file_path}"
        )

    def _read_text(self, file_path: str) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, min(self.config.chunk_overlap, chunk_size - 1))
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]


def build_chunker(config: RagConfig) -> Chunker:
    if config.chunker == "simple":
        return SimpleChunker(config)
    if config.chunker == "deepdoc":
        return DeepDocChunker(config)
    raise ValueError(f"Unsupported chunker in config: {config.chunker}")
