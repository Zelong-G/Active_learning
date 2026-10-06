"""Selection methods that return manifests without moving source files."""
from __future__ import annotations

import random
from typing import Literal, Sequence

from active_learning.scoring import AgreementScore


def select_samples(
    scores: Sequence[AgreementScore],
    budget: int,
    *,
    strategy: Literal["uncertainty", "random"] = "uncertainty",
    seed: int = 0,
    labeled_ids: Sequence[str] = (),
    include_empty: bool = False,
) -> list[AgreementScore]:
    """Pick unlabeled images using uncertainty ranking or a seeded baseline."""
    if budget < 0:
        raise ValueError("budget cannot be negative")
    if strategy not in ("uncertainty", "random"):
        raise ValueError("strategy must be 'uncertainty' or 'random'")
    identifiers = [row.image_id for row in scores]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate image IDs in the candidate pool")
    already_labeled = set(labeled_ids)
    pool = [
        row for row in scores
        if row.image_id not in already_labeled
        and (include_empty or not row.all_empty)
    ]
    if budget > len(pool):
        raise ValueError("budget exceeds available unlabeled candidates")
    if strategy == "random":
        # Sorting first makes seeded sampling invariant to input CSV order.
        return random.Random(seed).sample(
            sorted(pool, key=lambda row: row.image_id), budget
        )
    return sorted(
        pool,
        key=lambda row: (-row.uncertainty_score, row.image_id),
    )[:budget]
