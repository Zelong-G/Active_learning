import pytest

from active_learning.scoring import AgreementScore
from active_learning.selection import select_samples


def score(name: str, uncertainty: float, *, empty: bool = False) -> AgreementScore:
    agreement = 1.0 - uncertainty
    return AgreementScore(
        name, agreement, agreement, agreement, agreement, uncertainty,
        1, 3, empty, True,
    )


def test_uncertainty_ranking_excludes_labeled_and_empty() -> None:
    rows = [score("a", 0.7), score("b", 0.95, empty=True), score("c", 0.2)]
    chosen = select_samples(rows, 1, labeled_ids=("c",))
    assert [row.image_id for row in chosen] == ["a"]


def test_random_seed_is_independent_of_row_order() -> None:
    rows = [score(str(index), index / 10) for index in range(8)]
    first = select_samples(rows, 3, strategy="random", seed=9)
    second = select_samples(list(reversed(rows)), 3, strategy="random", seed=9)
    assert [r.image_id for r in first] == [r.image_id for r in second]


def test_duplicate_ids_and_oversized_budgets_rejected() -> None:
    with pytest.raises(ValueError):
        select_samples([score("a", 0.3), score("a", 0.4)], 1)
    with pytest.raises(ValueError):
        select_samples([score("a", 0.3)], 2)
