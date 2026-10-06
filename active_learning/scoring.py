"""Agreement-based acquisition scores from repeated stochastic predictions."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from collections.abc import Sequence

import numpy as np

from active_learning.geometry import box_iou, mask_dice
from active_learning.matching import match_instances
from active_learning.types import Prediction


@dataclass(frozen=True)
class AgreementScore:
    image_id: str
    detection_agreement: float
    segmentation_agreement: float
    classification_agreement: float
    mean_agreement: float
    uncertainty_score: float
    anchor_instances: int
    passes: int
    all_empty: bool
    has_variation: bool


def _same_prediction(a: Prediction, b: Prediction) -> bool:
    if len(a.instances) != len(b.instances):
        return False
    return all(
        x.box == y.box
        and x.label == y.label
        and np.array_equal(x.mask, y.mask)
        for x, y in zip(a.instances, b.instances)
    )


def score_prediction_set(
    image_id: str,
    passes: Sequence[Prediction],
    *,
    min_box_iou: float = 0.1,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AgreementScore:
    """Summarize consistency across at least two stochastic inference passes.

    Lower agreement means a higher acquisition priority. These heuristic scores
    are *not* calibrated posterior uncertainty estimates. A stochastic
    prediction mechanism (for example feature DropBlock) is required to make
    repeat-pass disagreement meaningful.
    """
    if not image_id.strip() or len(passes) < 2:
        raise ValueError("nonempty image ID and at least two passes are required")
    if len(weights) != 3 or any(not np.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must contain three finite nonnegative numbers")
    if sum(weights) <= 0:
        raise ValueError("at least one weight must be positive")
    anchor_idx = max(range(len(passes)), key=lambda i: len(passes[i].instances))
    anchor = passes[anchor_idx]
    varying = any(not _same_prediction(passes[0], p) for p in passes[1:])

    if not anchor.instances:
        # No predicted object: this score is undefined, not a high-confidence
        # or guaranteed-uncertain image. Exclude it by default at selection.
        return AgreementScore(
            image_id, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, len(passes), True, varying,
        )

    detection: list[float] = []
    segmentation: list[float] = []
    classification: list[float] = []

    for idx, other in enumerate(passes):
        if idx == anchor_idx:
            continue
        matches = match_instances(anchor, other, min_box_iou=min_box_iou)
        for anchor_index, source in enumerate(anchor.instances):
            candidate_index = matches.get(anchor_index)
            if candidate_index is None:
                detection.append(0.0)
                segmentation.append(0.0)
                classification.append(0.0)
                continue
            target = other.instances[candidate_index]
            detection.append(box_iou(source.box, target.box))
            segmentation.append(mask_dice(source.mask, target.mask))
            classification.append(float(source.label == target.label))

    det, seg, cls = mean(detection), mean(segmentation), mean(classification)
    overall = sum(x * w for x, w in zip((det, seg, cls), weights)) / sum(weights)
    return AgreementScore(
        image_id=image_id,
        detection_agreement=det,
        segmentation_agreement=seg,
        classification_agreement=cls,
        mean_agreement=overall,
        uncertainty_score=1.0 - overall,
        anchor_instances=len(anchor.instances),
        passes=len(passes),
        all_empty=False,
        has_variation=varying,
    )
