from .base import VectorStore
from .bm25_store import Bm25Store
from .faiss_store import FaissStore


def __getattr__(name):
    if name == "MilvusStore":
        from .milvus_store import MilvusStore

        return MilvusStore
    raise AttributeError(name)

__all__ = ["VectorStore", "Bm25Store", "FaissStore", "MilvusStore"]
