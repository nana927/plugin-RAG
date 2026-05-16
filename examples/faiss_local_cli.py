from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

from RAG.config import RagConfig
from RAG.ingestion import build_file_service
from RAG.rag_service import RagService
from RAG.types import ChunkRecord, SearchResult


def build_demo_service(config: RagConfig) -> tuple[RagService, str]:
    service = RagService(config)
    knowledge_base = config.demo_knowledge_base

    records = [
        ChunkRecord(
            chunk_id="chunk_001",
            doc_id="doc_001",
            doc_name="demo.txt",
            text=(
                "Algorithm testing agents can generate test cases, retrieve RAG "
                "knowledge, calculate metrics, and create test reports."
            ),
            metadata={"source": "example"},
        ),
        ChunkRecord(
            chunk_id="chunk_002",
            doc_id="doc_002",
            doc_name="metric.txt",
            text=(
                "Model evaluation usually focuses on Precision, Recall, F1, "
                "confusion matrices, and bad case analysis."
            ),
            metadata={"source": "example"},
        ),
        ChunkRecord(
            chunk_id="chunk_003",
            doc_id="doc_003",
            doc_name="rag.txt",
            text=(
                "RAG combines document chunking, embedding, vector retrieval, "
                "keyword scoring, reranking, and final answer generation."
            ),
            metadata={"source": "example"},
        ),
    ]

    service.clear(knowledge_base)
    service.add_documents(records, knowledge_base=knowledge_base)
    return service, knowledge_base


def build_service_from_file(args: argparse.Namespace, config: RagConfig) -> tuple[RagService, str]:
    service = build_file_service(
        str(Path(args.file).expanduser()),
        knowledge_base=config.knowledge_base,
        config=config,
        ask_on_existing=not args.yes,
    )
    return service, config.knowledge_base


def print_config(config: RagConfig, knowledge_base: str) -> None:
    print(
        "Config: "
        f"backend={config.backend}, "
        f"embedding={config.embedding}, "
        f"chunker={config.chunker}, "
        f"kb={knowledge_base}, "
        f"hybrid={config.hybrid_enabled}, "
        f"debug={config.retrieval_debug}"
    )


def build_context(results: List[SearchResult], max_chars: int = 5000) -> str:
    pieces: List[str] = []
    total = 0
    for idx, item in enumerate(results, start=1):
        piece = f"[{idx}] {item.file_name} score={item.score:.4f}\n{item.text.strip()}\n"
        if total + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - total)]
        if piece:
            pieces.append(piece)
            total += len(piece)
        if total >= max_chars:
            break
    return "\n".join(pieces)


def generate_answer(query: str, results: List[SearchResult]) -> str:
    api_key = os.getenv("RAG_EVAL_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "未配置大模型 API Key，暂不生成最终回答。"

    try:
        from openai import OpenAI
    except ImportError:
        return "未安装 openai，暂不生成最终回答。"

    base_url = os.getenv(
        "RAG_EVAL_BASE_URL",
        os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    model = os.getenv("RAG_EVAL_MODEL", os.getenv("RAG_KEYWORD_MODEL", "qwen-plus"))
    context = build_context(results)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的 RAG 问答助手。只能依据给定上下文回答。"
                    "如果问题包含多个子问题，要逐一回答，不要漏答。"
                    "如果上下文不足，就说明未找到明确依据。"
                ),
            },
            {"role": "user", "content": f"问题：{query}\n\n上下文：\n{context}\n\n请给出最终回答："},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def print_results(query: str, results: List[SearchResult]) -> None:
    if not results:
        print("No results found.")
        return

    top_results = results[:3]
    print("\nAnswer:")
    print(generate_answer(query, top_results))
    print("\nTop matches:")
    for idx, item in enumerate(top_results, start=1):
        print(f"\n[{idx}] {item.file_name} score={item.score:.4f}")
        metadata = item.metadata or {}
        if metadata.get("matched_subquery"):
            print(f"matched_subquery: {metadata['matched_subquery']}")
        print(_preview_text(item.text))


def _preview_text(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def interactive_loop(service: RagService, knowledge_base: str, top_k: int) -> None:
    print("RAG demo is ready. Type your question, or type exit/quit/q to stop.")
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except EOFError:
            print("\nBye.")
            break
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if not query:
            continue

        results = service.retrieve(query, knowledge_base=knowledge_base, top_k=top_k)
        print_results(query, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG demo. Runtime settings are read from .env.")
    parser.add_argument("--file", help="Optional file path to ingest before asking questions.")
    parser.add_argument("-y", "--yes", action="store_true", help="Re-import without asking on existing files.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    config = RagConfig()
    if args.file:
        service, knowledge_base = build_service_from_file(args, config)
    else:
        service, knowledge_base = build_demo_service(config)
    print_config(config, knowledge_base)
    interactive_loop(service, knowledge_base, top_k=3)


if __name__ == "__main__":
    main()
