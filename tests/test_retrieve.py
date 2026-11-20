import pytest

from ragbench.corpus import Page
from ragbench.retrieve import DecompositionRetriever, HybridRetriever, parse_subquestions

PAGES = [Page(0, "D", 1, "employees human capital workforce headcount"),
         Page(1, "D", 2, "revenue net revenue total revenue fiscal year"),
         Page(2, "D", 3, "research and development expenses")]


class Fixed:
    def __init__(self, name, order):
        self.name, self.order = name, order

    def rank(self, query, n=50):
        return self.order[:n]


def test_bm25_finds_the_page_with_the_words():
    pytest.importorskip("rank_bm25")
    from ragbench.retrieve import BM25Retriever

    assert BM25Retriever(PAGES).rank("total revenue", n=1) == [1]


def test_hybrid_fuses_both_rankings():
    hybrid = HybridRetriever(Fixed("Dense", [0, 2, 1]), Fixed("BM25", [2, 1, 0]))
    assert hybrid.rank("q")[0] == 2  # 2nd and 1st beats 1st and 3rd


def test_decomposition_pools_every_subquestion():
    base = Fixed("BM25", [])
    base.rank = lambda q, n=50: {"NVIDIA revenue": [1, 0], "AMD revenue": [2, 0]}[q][:n]
    decomposed = DecompositionRetriever(base, split=lambda q: ["NVIDIA revenue", "AMD revenue"])
    ranked = decomposed.rank("Compare NVIDIA and AMD revenue")
    assert ranked[0] == 0  # second for both sub-questions: 2/62 beats a single 1/61
    assert set(ranked[1:]) == {1, 2}  # each sub-question's best page is in the pool
    assert decomposed.name == "Decomposition + BM25"
    assert decomposed.last_subquestions == ["NVIDIA revenue", "AMD revenue"]


def test_decomposition_falls_back_to_the_question():
    base = Fixed("BM25", [0])
    decomposed = DecompositionRetriever(base, split=lambda q: [])
    decomposed.rank("What was revenue?")
    assert decomposed.last_subquestions == ["What was revenue?"]


def test_parse_subquestions_strips_markers_and_caps_at_four():
    raw = "1. NVIDIA revenue fiscal 2026\n- AMD net revenue 2025\n* Intel net revenue 2025\n\nok\n2) a\n5. extra query here"
    assert parse_subquestions(raw) == ["NVIDIA revenue fiscal 2026", "AMD net revenue 2025",
                                       "Intel net revenue 2025", "extra query here"]


def test_parse_subquestions_drops_labels_and_repeats():
    raw = "Question: NVIDIA revenue 2026\nQuery: NVIDIA revenue 2026\nSub-question 2: AMD revenue 2025"
    assert parse_subquestions(raw) == ["NVIDIA revenue 2026", "AMD revenue 2025"]
