"""Score repeated instance-segmentation predictions from a JSON file."""
from __future__ import annotations

import argparse
from pathlib import Path

from active_learning.io import score_file, write_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-box-iou", type=float, default=0.1)
    args = parser.parse_args()
    rows = score_file(args.input, min_box_iou=args.min_box_iou)
    write_scores(args.output, rows)
    for row in sorted(rows, key=lambda x: (-x.uncertainty_score, x.image_id)):
        print(
            f"{row.image_id}: agreement={row.mean_agreement:.3f}, "
            f"uncertainty={row.uncertainty_score:.3f}, "
            f"variation={row.has_variation}, empty={row.all_empty}"
        )
    print(f"Wrote {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
