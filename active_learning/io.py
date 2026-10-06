"""Portable JSON prediction input and CSV acquisition manifests."""
from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from active_learning.scoring import AgreementScore, score_prediction_set
from active_learning.types import Instance, Prediction


def load_prediction_sets(path: Path) -> dict[str, list[Prediction]]:
    """Load one or more images with repeated predictions from public JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise TypeError("expected an object with an 'images' list")
    output: dict[str, list[Prediction]] = {}
    for entry in payload["images"]:
        image_id = str(entry["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate image ID: {image_id}")
        passes: list[Prediction] = []
        for detections in entry["passes"]:
            instances = tuple(
                Instance(
                    box=tuple(float(v) for v in instance["box"]),
                    label=int(instance["label"]),
                    mask=np.asarray(instance["mask"], dtype=bool),
                    confidence=float(instance.get("confidence", 1.0)),
                )
                for instance in detections
            )
            passes.append(Prediction(instances))
        output[image_id] = passes
    return output


def score_file(path: Path, *, min_box_iou: float = 0.1) -> list[AgreementScore]:
    return [
        score_prediction_set(image_id, passes, min_box_iou=min_box_iou)
        for image_id, passes in load_prediction_sets(path).items()
    ]


def write_scores(path: Path, rows: list[AgreementScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(AgreementScore)]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in columns})


def read_scores(path: Path) -> list[AgreementScore]:
    columns = {field.name for field in fields(AgreementScore)}
    rows: list[AgreementScore] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not columns.issubset(reader.fieldnames or []):
            raise ValueError("score CSV is missing required columns")
        for row in reader:
            def as_bool(value: str) -> bool:
                if value not in ("True", "False"):
                    raise ValueError("boolean CSV fields must be True or False")
                return value == "True"
            rows.append(
                AgreementScore(
                    image_id=row["image_id"],
                    detection_agreement=float(row["detection_agreement"]),
                    segmentation_agreement=float(row["segmentation_agreement"]),
                    classification_agreement=float(row["classification_agreement"]),
                    mean_agreement=float(row["mean_agreement"]),
                    uncertainty_score=float(row["uncertainty_score"]),
                    anchor_instances=int(row["anchor_instances"]),
                    passes=int(row["passes"]),
                    all_empty=as_bool(row["all_empty"]),
                    has_variation=as_bool(row["has_variation"]),
                )
            )
    return rows


def write_selection(path: Path, rows: list[AgreementScore]) -> None:
    """Write an audit-friendly manifest; never copy or move image files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["rank", "image_id", "uncertainty_score"])
        for rank, row in enumerate(rows, 1):
            writer.writerow([rank, row.image_id, f"{row.uncertainty_score:.6f}"])
