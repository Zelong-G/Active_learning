# Method: agreement-driven acquisition for instance segmentation

## Historical motivation

The earlier RBC/WBC experimentation used Mask R-CNN-style inference with stochastic feature perturbations in a custom feature pyramid network. Each image was evaluated repeatedly; detection boxes, instance masks and class labels were compared across passes. Low similarity was used as an indicator for adding the image to an annotation pool.

The current public implementation preserves the algorithmic question and makes scoring and sampling self-contained. It is a **modernized reconstruction**, not a bit-for-bit reproduction of all historical behaviors.

## Prediction format

An image has at least two stochastic inference passes. Each pass contains zero or more detections:

```json
{
  "images": [
    {
      "image_id": "example",
      "passes": [
        [{"box": [0, 0, 2, 2], "label": 1, "mask": [[1, 1], [1, 1]]}],
        [{"box": [0, 0, 2, 2], "label": 1, "mask": [[1, 1], [1, 1]]}]
      ]
    }
  ]
}
```

Boxes use XYXY pixel coordinates, labels are positive integers, and masks are full-image binary arrays with the same image shape. An optional `confidence` field can be supplied; for upstream models, filter low-confidence detections **before** scoring, using a fixed threshold across acquisition rounds.

## Matching

1. Choose the pass with the most retained detections as anchor (first pass wins ties).
2. For every other pass, compute IoU between each anchor box and candidate boxes.
3. Greedily match pairs in descending IoU order with one-to-one constraints and a configurable IoU threshold; this avoids matching the same candidate to several anchors.
4. An unmatched anchor gets zero agreement in that pass. Cross-pass matching does not require equal class labels, because class consistency is separately measured.

Greedy matching is simple and deterministic; it is not globally optimal Hungarian assignment.

## Agreement scores

For every matched anchor-to-candidate pair:

- **Detection agreement:** box IoU in [0,1].
- **Segmentation agreement:** binary-mask Dice similarity in [0,1].
- **Classification agreement:** 1 if class labels match, otherwise 0.

Unmatched anchors contribute zero to all three terms. Each term is averaged over all anchor instances and the non-anchor passes. With user-configurable nonnegative weights `w`:

```text
A = (w_box * IoU + w_mask * Dice + w_class * class_match) / sum(w)
acquisition_priority = 1 - A
```

The default weights are equal. Higher acquisition priority means lower cross-pass agreement.

For an image where all passes contain no detections, the statistic is undefined. The output explicitly sets `all_empty=true`, and the selector excludes these images by default. A project-specific empty-image sampling policy can be implemented separately.

## Interpretation

Repeated predictions must be **stochastic** (for example, a properly configured DropBlock or Monte-Carlo-style inference mechanism). Identical passes yield no useful disagreement evidence. The output flag `has_variation=false` helps detect this mistake.

Agreement is a **heuristic acquisition score**. It is not model calibration, a Bayesian predictive entropy estimate, or proof of improved downstream segmentation accuracy. A real evaluation must compare equal annotation budgets with random acquisition over multiple seeds.

## Safety and reproducibility

Selection never moves source images or annotations: only CSV manifests are emitted. Supply a stable labeled-set manifest, keep validation/test data out of the acquisition pool, and record the stochasticity mechanism, inference seed, IoU threshold, weights, and mask threshold for each run. See [Reproducibility](REPRODUCIBILITY.md).
