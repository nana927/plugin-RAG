from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from ..config import RagConfig
from ..types import ChunkRecord


QA_RECORD_TYPE = "generated_qa"
QA_TEXT_PREFIX = "【预生成QA】"
logger = logging.getLogger(__name__)


def stable_id(*parts: str) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def generate_qa_records(
    records: List[ChunkRecord],
    knowledge_base: str,
    config: RagConfig,
) -> List[ChunkRecord]:
    """
    作用：
        根据入库文档 chunks 预生成固定 QA，并转换成可写入向量库的 ChunkRecord。
    入参：
        records: 原始文档 chunks 转成的 ChunkRecord 列表。
        knowledge_base: 知识库名称，用于稳定生成 chunk_id。
        config: RAG 配置，包含 QA 生成模型、数量、上下文长度等。
    出参：
        List[ChunkRecord]，每条记录是一组“问题-答案”。
    主要逻辑：
        先写入文件名、文件路径等文档信息；
        如果是 PDF，额外读取前 4 页原始文本，并先用 LLM 提炼成摘要作为补充内容；
        再取文档 chunks 前若干字符作为生成上下文；
        调用 OpenAI-compatible chat 模型生成 JSON QA 列表；
        把问题和答案写入 text，并在 metadata 中标记 record_type=generated_qa。
    """
    if not records or not config.qa_generation_enabled:
        return []

    client = QaGenerationClient(config)
    if not client.enabled:
        print("QA generation skipped: missing RAG_QA_GENERATION_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        return []

    logger.info(
        "[qa-generation] start doc=%s chunks=%s max_pairs=%s model=%s",
        records[0].doc_name,
        len(records),
        config.qa_generation_max_pairs,
        config.qa_generation_model,
    )
    pdf_preview_summary, pdf_metadata = _build_pdf_preview_summary(records, client, config)
    context = _records_to_context(
        records,
        config.qa_generation_max_context_chars,
        pdf_preview_summary,
        pdf_metadata,
        config.qa_pdf_preview_pages,
    )
    if not context:
        return []
    logger.info("[qa-generation] context_chars=%s", len(context))

    pairs = client.generate(context, config.qa_generation_max_pairs)
    if not pairs:
        logger.info("[qa-generation] no qa pairs generated")
        return []

    doc_id = records[0].doc_id
    doc_name = records[0].doc_name
    source_path = (records[0].metadata or {}).get("source_path")
    qa_records: List[ChunkRecord] = []

    for idx, pair in enumerate(pairs):
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        if not question or not answer:
            continue
        chunk_id = stable_id(doc_id, knowledge_base, QA_RECORD_TYPE, question)
        qa_records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                text=f"{QA_TEXT_PREFIX}\n问题：{question}\n答案：{answer}",
                metadata={
                    "record_type": QA_RECORD_TYPE,
                    "qa_question": question,
                    "qa_answer": answer,
                    "source_path": source_path,
                    "generated_by": config.qa_generation_model,
                    "pdf_preview_metadata": pdf_metadata,
                },
            )
        )

    logger.info("[qa-generation] generated_records=%s", len(qa_records))
    return qa_records


class QaGenerationClient:
    def __init__(self, config: RagConfig) -> None:
        self.api_key = (
            config.qa_generation_api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = config.qa_generation_base_url
        self.model = config.qa_generation_model
        self.temperature = config.qa_generation_temperature
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.enabled:
            raise RuntimeError("Missing QA generation API key.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, context: str, max_pairs: int) -> List[Dict[str, str]]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是文档入库前的 QA 生成器。请严格依据给定文档内容生成高价值问答，"
                        "覆盖定义、指标、参数、流程、约束、结论和容易被用户询问的事实。"
                        "不要编造文档没有的信息。只输出 JSON 数组，数组元素格式为 "
                        "{\"question\":\"...\",\"answer\":\"...\"}。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"最多生成 {max_pairs} 组 QA。\n\n文档内容：\n{context}",
                },
            ],
        )
        return _parse_pairs(response.choices[0].message.content or "", max_pairs)

    def summarize_pdf_preview(self, text: str, max_chars: int) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是文档首页信息提炼器。请严格依据给定 PDF 前几页文本，"
                        "提炼对后续生成 QA 最有帮助的信息。重点包括：文档名称、版本号、日期、"
                        "算法/系统名称、需求范围、核心指标、测试要求、角色、关键约束。"
                        "不要编造。输出简洁中文要点。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"请将下面内容压缩到 {max_chars} 字以内：\n\n{text}",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()[:max_chars]


def _build_pdf_preview_summary(
    records: List[ChunkRecord],
    client: QaGenerationClient,
    config: RagConfig,
) -> Tuple[str, Dict[str, Any]]:
    first_metadata = records[0].metadata or {}
    source_path = str(first_metadata.get("source_path") or "")
    pdf_preview, pdf_metadata = _read_pdf_first_pages(source_path, page_count=config.qa_pdf_preview_pages)
    if not pdf_preview:
        logger.info("[qa-generation] pdf_preview skipped path=%s", source_path or "<empty>")
        return "", pdf_metadata

    pdf_preview = pdf_preview[: config.qa_pdf_preview_max_chars]
    logger.info(
        "[qa-generation] pdf_preview chars=%s metadata_keys=%s summary_enabled=%s",
        len(pdf_preview),
        sorted(pdf_metadata.keys()),
        config.qa_pdf_preview_summary_enabled,
    )
    if not config.qa_pdf_preview_summary_enabled:
        return pdf_preview[: config.qa_pdf_preview_summary_max_chars], pdf_metadata

    try:
        summary = client.summarize_pdf_preview(
            pdf_preview,
            config.qa_pdf_preview_summary_max_chars,
        )
    except Exception:
        logger.warning("[qa-generation] pdf_preview_summary failed, fallback to truncated preview", exc_info=True)
        return pdf_preview[: config.qa_pdf_preview_summary_max_chars], pdf_metadata
    logger.info("[qa-generation] pdf_preview_summary chars=%s", len(summary))
    return summary or pdf_preview[: config.qa_pdf_preview_summary_max_chars], pdf_metadata


def _records_to_context(
    records: List[ChunkRecord],
    max_chars: int,
    pdf_preview_summary: str = "",
    pdf_metadata: Dict[str, Any] | None = None,
    pdf_preview_pages: int = 4,
) -> str:
    first = records[0]
    first_metadata = first.metadata or {}
    source_path = str(first_metadata.get("source_path") or "")
    header = _document_header(first.doc_name, source_path)

    pieces: List[str] = []
    total = 0
    total = _append_piece(pieces, header, total, max_chars)
    metadata_text = _format_pdf_metadata(pdf_metadata or {})
    total = _append_piece(
        pieces,
        f"[PDF 前 {pdf_preview_pages} 页结构化 metadata]\n{metadata_text}" if metadata_text else "",
        total,
        max_chars,
    )
    total = _append_piece(
        pieces,
        f"[PDF 前 {pdf_preview_pages} 页提炼信息]\n{pdf_preview_summary}" if pdf_preview_summary else "",
        total,
        max_chars,
    )

    for idx, record in enumerate(records, start=1):
        text = record.text.strip()
        if not text:
            continue
        piece = f"[chunk {idx}]\n{text}\n"
        total = _append_piece(pieces, piece, total, max_chars)
        if total >= max_chars:
            break
    return "\n".join(pieces)


def _append_piece(pieces: List[str], piece: str, total: int, max_chars: int) -> int:
    if not piece or total >= max_chars:
        return total
    if total + len(piece) > max_chars:
        piece = piece[: max(0, max_chars - total)]
    if piece:
        pieces.append(piece)
        total += len(piece)
    return total


def _document_header(doc_name: str, source_path: str) -> str:
    lines = ["[文档信息]", f"文件名：{doc_name}"]
    if source_path:
        lines.append(f"文件路径：{source_path}")
    return "\n".join(lines)


def _read_pdf_first_pages(file_path: str, page_count: int = 4) -> Tuple[str, Dict[str, Any]]:
    if not file_path or not file_path.lower().endswith(".pdf") or not os.path.exists(file_path):
        return "", {}
    try:
        import pdfplumber
    except ImportError:
        return "", {}

    texts: List[str] = []
    tables: List[List[List[str]]] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages[:page_count], start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    texts.append(f"[page {page_idx}]\n{text}")
                for table in page.extract_tables() or []:
                    normalized_table = _normalize_table(table)
                    if normalized_table:
                        tables.append(normalized_table)
    except Exception:
        logger.warning("[qa-generation] pdfplumber failed path=%s", file_path, exc_info=True)
        return "", {}
    metadata = _extract_pdf_preview_metadata(tables)
    logger.info(
        "[qa-generation] pdfplumber pages_with_text=%s tables=%s requested_pages=%s",
        len(texts),
        len(tables),
        page_count,
    )
    return "\n\n".join(texts).strip(), metadata


def _normalize_table(table: List[List[Any]]) -> List[List[str]]:
    
    """

    return [
        [["文档版本号", "V1.0", "文档编号", "..."], ...],
        [["序号", "版本", "作者", "修改时间", "修改说明"], ...]
    ]
    """
    normalized: List[List[str]] = []
    for row in table:
        cells = [_clean_cell(cell) for cell in row]
        if any(cells):
            normalized.append(cells)
    return normalized


def _clean_cell(value: Any) -> str:
    """_clean_cell() 会把单元格转成字符串，并把多个空白字符合并成一个空格"""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _extract_pdf_preview_metadata(tables: List[List[List[str]]]) -> Dict[str, Any]:
    """
    遍历前几页抽到的所有表格并结构化表格
    """
    
    metadata: Dict[str, Any] = {}
    revisions: List[Dict[str, str]] = []

    for table in tables:
        if not table:
            continue
        # 取第一行作为表头
        header = [_clean_key(cell) for cell in table[0]]
        if {"序号", "版本", "作者", "修改时间"}.issubset(set(header)):
            revisions.extend(_extract_revision_rows(table, header))
            continue
        metadata.update(_extract_key_value_table(table))

    if revisions:
        metadata["修订历史"] = revisions
    return {key: value for key, value in metadata.items() if not _is_empty_metadata_value(value)}


def _extract_key_value_table(table: List[List[str]]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    pending_key = ""
    for row in table:
        cells = row + [""] * (4 - len(row))
        for idx in range(0, len(cells), 2):
            key = _clean_key(cells[idx])
            value = cells[idx + 1].strip() if idx + 1 < len(cells) else ""
            if key:
                pending_key = key
                if value:
                    data[key] = _merge_value(data.get(key, ""), value)
            elif pending_key and value:
                data[pending_key] = _merge_value(data.get(pending_key, ""), value)
    return data


def _extract_revision_rows(table: List[List[str]], header: List[str]) -> List[Dict[str, str]]:
    revisions: List[Dict[str, str]] = []
    for row in table[1:]:
        item: Dict[str, str] = {}
        for idx, key in enumerate(header):
            if not key:
                continue
            value = row[idx].strip() if idx < len(row) else ""
            if value:
                item[key] = value
        if item and any(value for key, value in item.items() if key != "序号"):
            revisions.append(item)
    return revisions


def _clean_key(value: str) -> str:
    return re.sub(r"\s+", "", value or "").strip("：:")


def _merge_value(old: str, new: str) -> str:
    if not old:
        return new.strip()
    if not new:
        return old
    return f"{old}{new.strip()}"


def _format_pdf_metadata(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    lines: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, list):
            lines.append(f"{key}：")
            for item in value:
                if isinstance(item, dict):
                    lines.append("  - " + "；".join(f"{k}={v}" for k, v in item.items()))
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}：{value}")
    return "\n".join(lines)


def _is_empty_metadata_value(value: Any) -> bool:
    return value == "" or value == [] or value == {}


def _parse_pairs(text: str, max_pairs: int) -> List[Dict[str, str]]:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    pairs: List[Dict[str, str]] = []
    for item in data[:max_pairs]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs
