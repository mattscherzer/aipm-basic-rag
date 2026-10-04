import pytest

from rag.chunker import chunk_pages, split_text


def test_short_text_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # neighbouring chunks share content
    assert chunks[1].split()[0] in chunks[0]


def test_all_content_covered():
    text = " ".join(f"w{i}" for i in range(300))
    joined = " ".join(split_text(text, 80, 10))
    assert all(f"w{i}" in joined for i in range(300))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_page_metadata_kept():
    chunks = chunk_pages([(1, "alpha " * 50), (2, "beta " * 50)], "f.pdf", 100, 10)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.index for c in chunks] == list(range(len(chunks)))
