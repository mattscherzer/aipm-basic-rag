"""Step 2 - split page text into overlapping chunks.

Overlap keeps sentences that straddle a boundary retrievable from either side.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int  # position within the file


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = " ".join(text.split())  # normalise whitespace
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to cut at a space instead of mid-word
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_pages(pages: list[tuple[int, str]], source: str, size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for page, text in pages:
        for piece in split_text(text, size, overlap):
            out.append(Chunk(piece, source, page, len(out)))
    return out
