"""Deterministic one-to-one alignment across stochastic predictions."""
from __future__ import annotations

from active_learning.geometry import box_iou
from active_learning.types import Prediction


def match_instances(
    anchor: Prediction,
    candidate: Prediction,
    *,
    min_box_iou: float = 0.1,
) -> dict[int, int]:
    """Greedy global-IoU matching with one-to-one constraints.

    Returns anchor-index to candidate-index assignments. Unmatched anchors are
    omitted and receive zero agreement downstream. Matches do not require equal
    labels, so class disagreement can be measured separately.
    """
    if not 0 <= min_box_iou <= 1:
        raise ValueError("min_box_iou must lie in [0, 1]")
    possible: list[tuple[float, int, int]] = []
    for i, first in enumerate(anchor.instances):
        for j, second in enumerate(candidate.instances):
            overlap = box_iou(first.box, second.box)
            if overlap > 0 and overlap >= min_box_iou:
                possible.append((-overlap, i, j))
    possible.sort()
    used_anchor: set[int] = set()
    used_candidate: set[int] = set()
    matches: dict[int, int] = {}
    for _, i, j in possible:
        if i not in used_anchor and j not in used_candidate:
            matches[i] = j
            used_anchor.add(i)
            used_candidate.add(j)
    return matches
