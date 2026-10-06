import numpy as np

from active_learning.matching import match_instances
from active_learning.types import Instance, Prediction


def test_matching_does_not_reuse_a_single_detection() -> None:
    mask = np.ones((4, 4), dtype=bool)
    anchors = Prediction((
        Instance((0, 0, 3, 3), 1, mask),
        Instance((0, 0, 3, 3), 1, mask),
    ))
    single = Prediction((Instance((0, 0, 3, 3), 1, mask),))
    assert len(match_instances(anchors, single)) == 1


def test_disjoint_boxes_do_not_match() -> None:
    mask = np.ones((4, 4), dtype=bool)
    a = Prediction((Instance((0, 0, 1, 1), 1, mask),))
    b = Prediction((Instance((2, 2, 3, 3), 1, mask),))
    assert match_instances(a, b) == {}
