"""Step 4 - persist vectors and search them (ChromaDB, cosine similarity)."""
import chromadb

from . import config
from .chunker import Chunk


def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def add(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    _collection().upsert(
        ids=[f"{c.source}:{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )


def query(embedding: list[float], k: int) -> list[dict]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[embedding], n_results=min(k, col.count()))
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "score": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def list_sources() -> list[str]:
    metas = _collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def delete_source(source: str) -> None:
    _collection().delete(where={"source": source})


def reset() -> None:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
