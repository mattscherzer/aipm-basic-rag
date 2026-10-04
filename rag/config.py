"""All tunable settings in one place."""
from pathlib import Path

CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # chunks retrieved per question

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "documents"
