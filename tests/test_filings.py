"""Runs only when the three PDFs are in data/pdfs: every evidence anchor must land on at least one page."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "pdfs"
pytestmark = pytest.mark.skipif(not all((PDFS / f"{d}.pdf").exists() for d in
                                        ("NVIDIA_FY2026_10-K", "AMD_FY2025_10-K", "Intel_FY2025_10-K")),
                                reason="the 10-K PDFs are not in data/pdfs")


def test_every_anchor_is_found_and_pages_are_455():
    from ragbench.corpus import load_pages
    from ragbench.run import load_questions

    pages = load_pages(PDFS)
    assert len(pages) == 455
    questions = load_questions(ROOT / "data" / "questions.json", pages)  # raises if an anchor is on no page
    assert sum(len(q["groups"]) for q in questions) == 27  # 13 single-fact questions + 14 facts in 5 comparisons
