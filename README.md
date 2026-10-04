# Local RAG (Ollama + Streamlit)

Minimal retrieval-augmented generation app that runs fully offline. No RAG framework: each step is one small file in `rag/`.

## Setup
```bash
ollama pull llama3.1:8b        # ~5 GB; use llama3.2:3b on low-RAM machines
ollama pull nomic-embed-text   # ~270 MB
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
.venv/bin/streamlit run app.py
```
Change models in `rag/config.py` (or the chat model box in the sidebar).

## How it works
**Ingest** (`pipeline.ingest`): `loader` extracts text per page -> `chunker` cuts ~800-char overlapping chunks -> `embedder` turns each into a vector (Ollama) -> `store` saves vectors + file/page metadata in ChromaDB (`data/chroma`).

**Query** (`pipeline.retrieve` + `stream_answer`): embed the question -> top-4 nearest chunks by cosine similarity -> `prompt` puts them in the context with "answer only from this" -> Ollama chat streams the answer. Retrieved chunks and scores show under each answer.

## Tests
```bash
.venv/bin/python -m pytest
```

## Ideas to extend
Reranking, hybrid BM25 search, query rewriting with chat history, OCR for scanned PDFs, an eval question set.
