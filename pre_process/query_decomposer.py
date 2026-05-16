from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List

from ..config import RagConfig


logger = logging.getLogger(__name__)


def decompose_query(query: str, config: RagConfig) -> List[str]:
    """
    作用：
        将用户的复合问题拆成多个可独立检索的子问题。
    入参：
        query: 用户原始问题。
        config: RAG 配置，控制是否启用 LLM 拆解、模型和最大子问题数。
    出参：
        子问题列表；如果无法拆解，则返回只包含原问题的列表。
    主要逻辑：
        启用时优先调用大模型拆解；
        大模型不可用或失败时，使用本地规则兜底；
        最后去重、限数，并保证至少返回原问题。
    """
    query = query.strip()
    if not query:
        return []
    if not config.query_decomposition_enabled:
        return [query]

    subqueries: List[str] = []
    client = QueryDecompositionClient(config)
    if client.enabled:
        try:
            subqueries = client.decompose(query, config.query_decomposition_max_subqueries)
        except Exception:
            logger.warning("[query-decomposition] llm failed, fallback to rules", exc_info=True)

    if not subqueries:
        subqueries = _decompose_by_rules(query)

    subqueries = _clean_subqueries(subqueries, query, config.query_decomposition_max_subqueries)
    logger.info("[query-decomposition] query=%s subqueries=%s", query, subqueries)
    return subqueries


class QueryDecompositionClient:
    def __init__(self, config: RagConfig) -> None:
        self.api_key = (
            config.query_decomposition_api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = config.query_decomposition_base_url
        self.model = config.query_decomposition_model
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.enabled:
            raise RuntimeError("Missing query decomposition API key.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def decompose(self, query: str, max_subqueries: int) -> List[str]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 检索前的问题拆解器。请把用户复合问题拆成可以独立检索的子问题。"
                        "要求：不回答问题；不扩展文档外信息；保留原问题中的限定对象；"
                        "如果只有一个问题，就返回一个元素。只输出 JSON 数组字符串。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"最多拆成 {max_subqueries} 个子问题：\n{query}",
                },
            ],
        )
        return _parse_subqueries(response.choices[0].message.content or "")


def _parse_subqueries(text: str) -> List[str]:
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
    return [str(item).strip() for item in data if str(item).strip()]


def _decompose_by_rules(query: str) -> List[str]:
    context_prefix, query_body = _split_context_prefix(query)
    pieces = re.split(r"[？?；;。]\s*", query_body)
    candidates: List[str] = []
    for piece in pieces:
        piece = piece.strip(" ，,")
        if not piece:
            continue
        subpieces = re.split(r"(?:，|,|并且|以及|同时|另外|还有)", piece)
        if len(subpieces) == 1:
            candidates.append(piece)
        else:
            prefix = _extract_topic_prefix(piece)
            for subpiece in subpieces:
                subpiece = subpiece.strip(" ，,")
                if not subpiece:
                    continue
                if context_prefix:
                    candidates.append(f"{context_prefix}{subpiece}")
                elif prefix and not subpiece.startswith(prefix) and len(subpiece) < 16:
                    candidates.append(f"{prefix}{subpiece}")
                else:
                    candidates.append(subpiece)
    return candidates


def _split_context_prefix(query: str) -> tuple[str, str]:
    match = re.match(r"^((?:根据|依据).{1,30}?[，,])(.+)$", query.strip())
    if not match:
        return "", query
    return match.group(1), match.group(2)


def _extract_topic_prefix(text: str) -> str:
    match = re.match(r"(.{1,30}?(?:这个|该|本|根据|依据)?(?:PRD|文档|需求|文件))", text, flags=re.I)
    if match:
        return match.group(1)
    if "，" in text:
        return text.split("，", 1)[0]
    return ""


def _clean_subqueries(subqueries: List[str], original: str, max_subqueries: int) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in subqueries:
        item = re.sub(r"\s+", " ", item).strip(" ，,。？?")
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
        if len(cleaned) >= max_subqueries:
            break
    return cleaned or [original]
