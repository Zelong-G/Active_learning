"""Numerically safe 2D overlap metrics."""
from __future__ import annotations

import numpy as np


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of XYXY boxes; disjoint boxes return zero."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection *= max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Sørensen-Dice similarity of full-image binary masks."""
    first, second = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("both masks must be 2D and share a shape")
    size = int(first.sum()) + int(second.sum())
    if size == 0:
        return 1.0
    return 2.0 * float(np.logical_and(first, second).sum()) / size
