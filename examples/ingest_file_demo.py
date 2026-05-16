from __future__ import annotations

import argparse
from pathlib import Path

from RAG.config import RagConfig
from RAG.ingestion import ingest_file
from RAG.rag_service import RagService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest one file. Runtime settings are read from .env.")
    parser.add_argument("file", help="File path to ingest.")
    parser.add_argument("-y", "--yes", action="store_true", help="Re-import without asking on existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RagConfig()
    service = RagService(config)
    file_path = str(Path(args.file).expanduser())

    records = ingest_file(
        service,
        file_path,
        config.knowledge_base,
        ask_on_existing=not args.yes,
    )
    if records:
        print(f"Inserted {len(records)} chunks into knowledge base: {config.knowledge_base}")
    else:
        print("No records inserted.")


if __name__ == "__main__":
    main()
