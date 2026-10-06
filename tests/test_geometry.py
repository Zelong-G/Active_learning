import numpy as np
import pytest

from active_learning.geometry import box_iou, mask_dice


def test_box_iou_for_identity_disjoint_and_partial_overlap() -> None:
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1.0)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_mask_dice_and_validation() -> None:
    mask = np.array([[1, 0], [1, 0]], dtype=bool)
    assert mask_dice(mask, mask) == 1.0
    assert mask_dice(mask, ~mask) == 0.0
    assert mask_dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
    with pytest.raises(ValueError):
        mask_dice(mask, np.ones((3, 3)))
