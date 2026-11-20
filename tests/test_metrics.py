import pytest

from ragbench.metrics import rrf, recall, reciprocal_rank

N, A = "NVIDIA_FY2026_10-K", "AMD_FY2025_10-K"


def test_a_fact_printed_on_two_pages_counts_once():
    groups = [{(N, 53), (N, 71)}]
    assert recall([(N, 71), (N, 2)], groups) == 1.0
    assert recall([(N, 53), (N, 71)], groups) == 1.0  # both copies retrieved is still one fact found


def test_comparison_needs_every_company():
    groups = [{(N, 71)}, {(A, 73)}]
    assert recall([(N, 71), (N, 72)], groups) == 0.5
    assert recall([(A, 73), (N, 71)], groups) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_page():
    groups = [{(N, 71)}, {(A, 73)}]
    assert reciprocal_rank([(N, 1), (A, 73), (N, 71)], groups) == 0.5
    assert reciprocal_rank([(N, 1), (N, 2)], groups) == 0.0


def test_recall_is_undefined_without_evidence():
    with pytest.raises(ValueError):
        recall([(N, 1)], [])


def test_rrf_rewards_agreement():
    # c = 60: item 7 is 2nd in both lists (2/62) and beats item 1, which is 1st in one list only (1/61).
    assert rrf([[1, 7, 3], [9, 7, 4]], c=60)[0] == 7
    assert set(rrf([[1, 2], [3]], c=60)) == {1, 2, 3}
