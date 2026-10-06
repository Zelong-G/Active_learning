"""Small, framework-independent prediction contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Instance:
    """One detected instance with an XYXY box and a binary full-image mask."""

    box: tuple[float, float, float, float]
    label: int
    mask: np.ndarray
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.box) != 4 or not all(np.isfinite(self.box)):
            raise ValueError("box must contain four finite XYXY coordinates")
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        if self.label < 1:
            raise ValueError("instance label must be a positive category ID")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be a finite number in [0, 1]")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("mask must be a nonempty 2D array")
        object.__setattr__(self, "mask", mask.copy())


@dataclass(frozen=True)
class Prediction:
    """All retained instances from one stochastic prediction pass."""

    instances: tuple[Instance, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
