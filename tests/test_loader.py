import pytest

from rag.loader import load_file


def test_txt_loaded_as_single_page(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert load_file(p) == [(1, "hello")]


def test_unsupported_type(tmp_path):
    p = tmp_path / "a.docx"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_file(p)
