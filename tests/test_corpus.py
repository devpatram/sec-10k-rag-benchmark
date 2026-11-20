import pytest

pymupdf = pytest.importorskip("pymupdf")

from ragbench.corpus import evidence_pages, load_pages  # noqa: E402


@pytest.fixture
def tiny_pdf(tmp_path):
    doc = pymupdf.open()
    for text in ["Revenue was $215,938\nmillion this year.", "We repurchased 112 million shares.",
                 "We repurchased 12 million shares."]:
        doc.new_page().insert_text((72, 72), text)
    doc.save(tmp_path / "TEST_10-K.pdf")
    return tmp_path


def test_one_chunk_per_page_with_metadata(tiny_pdf):
    pages = load_pages(tiny_pdf, documents=("TEST_10-K",))
    assert [(p.document, p.page) for p in pages] == [("TEST_10-K", 1), ("TEST_10-K", 2), ("TEST_10-K", 3)]
    assert pages[2].label() == "TEST_10-K, page 3"


def test_anchor_survives_a_line_break(tiny_pdf):
    pages = load_pages(tiny_pdf, documents=("TEST_10-K",))
    assert evidence_pages(pages, "TEST_10-K", "215,938 million") == {1}


def test_anchor_does_not_match_inside_a_bigger_number(tiny_pdf):
    pages = load_pages(tiny_pdf, documents=("TEST_10-K",))
    assert evidence_pages(pages, "TEST_10-K", "12 million shares") == {3}
    assert evidence_pages(pages, "TEST_10-K", "nothing here|12 million shares") == {3}


def test_missing_pdf_says_where_to_look(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_pages(tmp_path, documents=("NOPE",))
