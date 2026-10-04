"""Step 1 - read a file and return its text as (page_number, text) pairs."""
from pathlib import Path

from pypdf import PdfReader


def load_file(path: Path) -> list[tuple[int, str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    elif suffix in {".txt", ".md"}:
        pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return [(n, text) for n, text in pages if text.strip()]
