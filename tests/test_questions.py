import json
import re
from collections import Counter
from pathlib import Path

from ragbench.corpus import DOCUMENTS, anchor_pattern

QUESTIONS = json.loads((Path(__file__).resolve().parents[1] / "data" / "questions.json").read_text(encoding="utf-8"))


def test_shape_of_the_question_set():
    assert [q["id"] for q in QUESTIONS] == list(range(1, 21))
    assert Counter(q["difficulty"] for q in QUESTIONS) == {"easy": 8, "medium": 5, "hard": 5, "unanswerable": 2}


def test_evidence_matches_the_difficulty():
    for q in QUESTIONS:
        if q["difficulty"] == "unanswerable":
            assert q["evidence"] == [], q["id"]
        else:
            assert q["evidence"], q["id"]
            assert all(e["document"] in DOCUMENTS for e in q["evidence"]), q["id"]
            anchor_pattern(q["evidence"][0]["anchor"])  # compiles
    hard = [q for q in QUESTIONS if q["difficulty"] == "hard"]
    assert all(len({e["document"] for e in q["evidence"]}) >= 2 for q in hard)  # hard = across filings


def test_reference_answers_carry_the_anchored_figures():
    """The figure the retrieval metric looks for is the figure the reference answer gives."""
    for q in QUESTIONS:
        for e in q["evidence"]:
            first = e["anchor"].split("|")[0].replace("\\", "")
            number = re.search(r"\d[\d,.]*", first).group()
            assert number in q["answer"], (q["id"], number)
