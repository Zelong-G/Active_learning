import numpy as np
import pytest

from active_learning.scoring import score_prediction_set
from active_learning.types import Instance, Prediction


def example_pass(label: int = 1) -> Prediction:
    mask = np.array([[1, 1], [1, 1]], dtype=bool)
    return Prediction((Instance((0, 0, 2, 2), label, mask),))


def test_identical_predictions_have_full_agreement_and_no_variation() -> None:
    score = score_prediction_set("same", [example_pass(), example_pass()])
    assert score.mean_agreement == pytest.approx(1.0)
    assert score.uncertainty_score == pytest.approx(0.0)
    assert score.has_variation is False


def test_class_disagreement_and_missing_instance_lower_agreement() -> None:
    disagreement = score_prediction_set(
        "class", [example_pass(1), example_pass(2)]
    )
    assert disagreement.classification_agreement == 0.0
    assert disagreement.mean_agreement == pytest.approx(2 / 3)
    missing = score_prediction_set("missing", [example_pass(), Prediction()])
    assert missing.uncertainty_score == pytest.approx(1.0)


def test_all_empty_is_flagged_not_misrepresented_as_uncertain() -> None:
    score = score_prediction_set("empty", [Prediction(), Prediction()])
    assert score.all_empty
    assert score.anchor_instances == 0
    assert score.uncertainty_score == 0.0


def test_requires_repeated_predictions() -> None:
    with pytest.raises(ValueError):
        score_prediction_set("one", [example_pass()])
