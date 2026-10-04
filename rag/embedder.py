"""Step 3 - turn text into vectors with a local Ollama embedding model."""
import ollama

from . import config


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    response = ollama.embed(model=model or config.EMBED_MODEL, input=texts)
    return [list(v) for v in response["embeddings"]]
