"""Retrieval metrics where one fact can be printed on several pages.

A question has evidence groups: one per fact it needs (one per company in a comparison). A group is found when any
page that carries that fact is retrieved, so a figure repeated in the summary table and in the financial statements
does not halve recall.
"""
from __future__ import annotations

Key = tuple[str, int]  # (document, page)


def reciprocal_rank(retrieved: list[Key], groups: list[set[Key]]) -> float:
    """1 / rank of the first retrieved page that belongs to any evidence group; 0 if none does."""
    relevant = set().union(*groups) if groups else set()
    for rank, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def recall(retrieved: list[Key], groups: list[set[Key]]) -> float:
    """Share of evidence groups with at least one page in the retrieved list."""
    if not groups:
        raise ValueError("recall is undefined for a question with no evidence")
    got = set(retrieved)
    return sum(bool(g & got) for g in groups) / len(groups)


def rrf(rankings: list[list[int]], c: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (c + rank of d). Ties keep first-seen order."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=lambda d: -scores[d])
