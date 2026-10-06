"""Create active-learning query manifests without mutating any dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from active_learning.io import read_scores, write_selection
from active_learning.selection import select_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("uncertainty", "random"),
        default="uncertainty",
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labeled", type=Path, help="Optional text file: one already-labeled image ID per line")
    parser.add_argument("--include-empty", action="store_true")
    args = parser.parse_args()
    labeled = (
        args.labeled.read_text(encoding="utf-8").splitlines()
        if args.labeled else []
    )
    chosen = select_samples(
        read_scores(args.scores),
        args.budget,
        strategy=args.strategy,
        seed=args.seed,
        labeled_ids=[line.strip() for line in labeled if line.strip()],
        include_empty=args.include_empty,
    )
    write_selection(args.output, chosen)
    print(f"{args.strategy}: selected {len(chosen)} images -> {args.output}")


if __name__ == "__main__":
    main()
