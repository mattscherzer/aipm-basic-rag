"""Glue: ingest a file, or answer a question (retrieve -> prompt -> generate)."""
from collections.abc import Iterator
from pathlib import Path

import ollama

from . import config, store
from .chunker import chunk_pages
from .embedder import embed
from .loader import load_file
from .prompt import build_messages


def ingest(path: Path) -> int:
    """Index a file; returns the number of chunks stored."""
    path = Path(path)
    pages = load_file(path)
    chunks = chunk_pages(pages, path.name, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return 0
    store.delete_source(path.name)  # re-indexing replaces old chunks
    store.add(chunks, embed([c.text for c in chunks]))
    return len(chunks)


def retrieve(question: str, k: int | None = None) -> list[dict]:
    return store.query(embed([question])[0], k or config.TOP_K)


def stream_answer(question: str, hits: list[dict], model: str | None = None) -> Iterator[str]:
    stream = ollama.chat(
        model=model or config.CHAT_MODEL, messages=build_messages(question, hits), stream=True
    )
    for part in stream:
        yield part["message"]["content"]
