"""Dataset-free demonstration of uncertainty ranking and random baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

from active_learning.io import score_file, write_scores, write_selection
from active_learning.selection import select_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "predictions.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores = score_file(args.input)
    write_scores(args.output_dir / "scores.csv", scores)
    for strategy in ("uncertainty", "random"):
        selected = select_samples(
            scores, args.budget, strategy=strategy, seed=args.seed
        )
        write_selection(args.output_dir / f"{strategy}_selection.csv", selected)
        print(f"{strategy}: {[row.image_id for row in selected]}")
    print(f"Synthetic manifests written to {args.output_dir}")


if __name__ == "__main__":
    main()
