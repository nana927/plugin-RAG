from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from .chunking import build_chunker
from .config import RagConfig
from .rag_service import RagService
from .pre_process.qa_generator import generate_qa_records
from .pre_process.table_utils import contains_html_table, extract_html_tables, html_tables_to_structured_text
from .types import ChunkRecord


def stable_id(*parts: str) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def file_doc_id(file_path: str) -> str:
    return stable_id(_normalized_path(file_path))


def chunks_to_records(
    chunks: List[Dict[str, Any]],
    file_path: str,
    knowledge_base: str,
) -> List[ChunkRecord]:
    file_name = os.path.basename(file_path)
    doc_id = file_doc_id(file_path)
    records: List[ChunkRecord] = []
    chunks = merge_table_chunks(chunks)

    for idx, item in enumerate(chunks):
        text = item.get("content_with_weight", "").strip()
        if not text:
            continue

        metadata = dict(item.get("metadata") or {})
        metadata.update(
            {
                "source_path": file_path,
                "docnm_kwd": item.get("docnm_kwd", file_name),
                "title_tks": item.get("title_tks"),
                "page_num_int": item.get("page_num_int"),
                "position_int": item.get("position_int"),
            }
        )
        if contains_html_table(text):
            tables = extract_html_tables(text)
            metadata.update(
                {
                    "record_type": "table",
                    "table_count": len(tables),
                    "tables": tables,
                }
            )
            text = html_tables_to_structured_text(text)

        records.append(
            ChunkRecord(
                chunk_id=item.get("chunk_id") or stable_id(doc_id, knowledge_base, str(idx), text[:128]),
                doc_id=item.get("doc_id") or doc_id,
                doc_name=item.get("doc_name") or file_name,
                text=text,
                metadata=metadata,
            )
        )

    return records


def merge_table_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    作用：
        合并 DeepDoc 输出中被拆散的 HTML 表格片段，保证一张表尽量作为一个 record 入库。
    入参：
        chunks: DeepDoc/simple chunker 输出的 chunk 字典列表。
    出参：
        合并后的 chunks 列表。
    主要逻辑：
        遇到包含 caption/table/tr/td/th 的 chunk 后进入表格缓冲；
        连续表格片段会被拼接到同一个 chunk；
        遇到 </table> 或下一个非表格 chunk 时结束合并；
        metadata 保留第一段为主，并记录 merged_table_chunk_count。
    """
    merged: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []

    for item in chunks:
        text = (item.get("content_with_weight") or "").strip()
        if _is_table_fragment(text):
            buffer.append(item)
            if _table_fragment_is_complete(text):
                merged.append(_flush_table_buffer(buffer))
                buffer = []
            continue

        if buffer:
            merged.append(_flush_table_buffer(buffer))
            buffer = []
        merged.append(item)

    if buffer:
        merged.append(_flush_table_buffer(buffer))
    return merged


def _is_table_fragment(text: str) -> bool:
    return contains_html_table(text)


def _table_fragment_is_complete(text: str) -> bool:
    return "</table" in text.lower()


def _flush_table_buffer(buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not buffer:
        return {}
    if len(buffer) == 1:
        item = dict(buffer[0])
        metadata = dict(item.get("metadata") or {})
        metadata["merged_table_chunk_count"] = 1
        item["metadata"] = metadata
        return item

    first = dict(buffer[0])
    metadata = dict(first.get("metadata") or {})
    metadata["merged_table_chunk_count"] = len(buffer)
    first["metadata"] = metadata
    first["content_with_weight"] = "\n".join(
        (item.get("content_with_weight") or "").strip()
        for item in buffer
        if (item.get("content_with_weight") or "").strip()
    )
    first.pop("chunk_id", None)
    return first


def build_records_from_file(
    file_path: str,
    knowledge_base: str,
    config: RagConfig | None = None,
    **kwargs,
) -> List[ChunkRecord]:
    config = config or RagConfig()
    chunker = build_chunker(config)
    chunks = chunker.chunk_file(file_path, **kwargs)
    return chunks_to_records(chunks, file_path, knowledge_base)


def knowledge_base_has_file(
    file_path: str,
    knowledge_base: str,
    config: RagConfig | None = None,
    store: Optional[Any] = None,
) -> bool:
    config = config or RagConfig()
    abs_file_path = _normalized_path(file_path)
    file_name = os.path.basename(file_path)
    doc_id = stable_id(abs_file_path)

    if config.backend == "milvus":
        if store is not None and hasattr(store, "has_document"):
            return bool(store.has_document(knowledge_base, doc_id, file_name))
        return False

    metadata_path = os.path.join(config.storage_dir, knowledge_base, "metadata.json")
    if not os.path.exists(metadata_path):
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    for item in metadata:
        item_metadata = item.get("metadata") or {}
        source_path = item_metadata.get("source_path")
        if item.get("doc_id") == doc_id:
            return True
        if source_path and _normalized_path(source_path) == abs_file_path:
            return True
    return False


def ingest_file(
    service: RagService,
    file_path: str,
    knowledge_base: str,
    ask_on_existing: bool = True,
    **kwargs,
) -> List[ChunkRecord]:
    doc_id = file_doc_id(file_path)
    if knowledge_base_has_file(file_path, knowledge_base, service.config, service.store):
        if ask_on_existing:
            try:
                answer = input(
                    f"File already exists in knowledge base '{knowledge_base}'. "
                    "Delete this document and re-import it? [y/N]: "
                ).strip().lower()
            except EOFError:
                answer = ""
            if answer not in {"y", "yes"}:
                print("Skipped import.")
                return []
        service.delete_document(knowledge_base, doc_id)

    records = build_records_from_file(file_path, knowledge_base, service.config, **kwargs)
    qa_records = generate_qa_records(records, knowledge_base, service.config)
    all_records = records + qa_records
    service.add_documents(all_records, knowledge_base=knowledge_base)
    if qa_records:
        print(f"Generated {len(qa_records)} QA records for retrieval enhancement.")
    return all_records


def build_file_service(
    file_path: str,
    knowledge_base: str = "resume_kb",
    chunker: str = "deepdoc",
    config: RagConfig | None = None,
    ask_on_existing: bool = True,
    **kwargs,
) -> RagService:
    config = config or RagConfig(chunker=chunker)
    service = RagService(config)
    records = ingest_file(
        service,
        file_path,
        knowledge_base,
        ask_on_existing=ask_on_existing,
        **kwargs,
    )
    if records:
        print(f"Inserted {len(records)} chunks into knowledge base: {knowledge_base}")
    return service
