from pathlib import Path

from active_learning.io import (
    read_scores,
    score_file,
    write_scores,
    write_selection,
)
from active_learning.selection import select_samples


def test_synthetic_json_to_score_and_selection_manifests(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "examples/predictions.json"
    rows = score_file(source)
    assert len(rows) == 3
    assert {row.image_id for row in rows} == {
        "synthetic_stable", "synthetic_ambiguous", "synthetic_moderate"
    }
    target = tmp_path / "scores.csv"
    write_scores(target, rows)
    restored = read_scores(target)
    assert restored == rows

    selected = select_samples(restored, 2, strategy="uncertainty")
    output = tmp_path / "selected.csv"
    write_selection(output, selected)
    assert "synthetic_ambiguous" in output.read_text(encoding="utf-8")
