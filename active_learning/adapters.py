"""Optional adapter for torchvision-style instance-segmentation outputs."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from active_learning.types import Instance, Prediction


def _numpy(value: object) -> np.ndarray:
    """Convert NumPy or Torch-like tensors without importing Torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def from_torchvision_output(
    output: Mapping[str, object],
    *,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> Prediction:
    """Convert a single Mask R-CNN output into the public Prediction contract.

    The caller is responsible for producing *stochastic* passes. Calling a
    deterministic model in eval mode repeatedly does not estimate uncertainty.
    """
    if not 0 <= score_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("thresholds must be in [0, 1]")
    boxes = _numpy(output["boxes"])
    labels = _numpy(output["labels"])
    scores = _numpy(output["scores"])
    masks = _numpy(output["masks"])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or masks.ndim != 3:
        raise ValueError("expected boxes [N,4] and masks [N,H,W]")
    count = len(scores)
    if not (len(boxes) == len(labels) == len(masks) == count):
        raise ValueError("inconsistent number of boxes, labels, masks and scores")
    instances = tuple(
        Instance(
            tuple(float(x) for x in boxes[i]),
            int(labels[i]),
            masks[i] >= mask_threshold,
            float(scores[i]),
        )
        for i in range(count)
        if float(scores[i]) >= score_threshold
    )
    return Prediction(instances)
