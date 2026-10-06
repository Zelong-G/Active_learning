# Reproducibility and experiment protocol

This repository's executable demonstrations use synthetic predictions only. They verify algorithmic contracts and deterministic selection behavior; they **do not** recreate the original RBC/WBC benchmark metrics.

## Data and provenance

For a new real-data experiment, document the dataset name/version, access/license, annotation format, image-level splits, class mapping and any patient/study-level leakage constraints. The public repository contains no microscopy imagery or original model checkpoints.

## Suggested comparison

Use the **same** starting labeled set, model initialization and unlabeled pool for:

- agreement-based uncertainty selection; and
- seeded random selection.

At each acquisition round:

1. Train or fine-tune the chosen instance-segmentation model on the current labeled set.
2. Freeze the trained model snapshot for scoring.
3. Generate repeated **stochastic** predictions over the remaining unlabeled pool.
4. Compute agreement scores, then select the same fixed number of samples for each strategy.
5. Add only those newly annotated images to the labeled pool.
6. Evaluate on an unchanged held-out validation/test split.

Report the initial labeled count, per-round budget, training schedule, optimizer, score/mask/IoU thresholds, number of stochastic passes and selection seed. Use multiple independent seeds; do not choose the strongest trial after seeing final scores.

## Metrics

Report separate instance-segmentation and detection metrics where applicable:

- mask AP / AP50 / AP75 (with precise IoU and class-averaging conventions);
- detection AP / AP50 if analyzed;
- per-class statistics and rare-class results only when supported by the data;
- performance versus **number of labeled images** (annotation budget).

Mean ± standard deviation over pre-defined seeds is preferable to selectively reporting a favorable run. Compare at identical budgets.

## Current public verification

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo.py --output-dir outputs/demo
```

The source example is artificially constructed, and all resulting scores are **illustrative only**. Unavailable historic dataset logs and checkpoints are not reconstructed or guessed. Original source implementations remain part of the pre-cleanup Git history for provenance but are not present in the new top-level source tree.

## Known limitations

- Greedy one-to-one box alignment may be suboptimal when instances overlap heavily.
- Single-anchor alignment is sensitive to missing or spurious detections.
- High agreement is not necessarily correctness; consistent systematic errors can look stable.
- An empty prediction set does not mean high confidence.
- Model-derived masks must share a consistent image-space coordinate system.
- Scoring synthetic repeated predictions does not validate an actual stochastic model.
