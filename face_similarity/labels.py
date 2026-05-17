from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelEntry:
    path: Path
    rating: float


def load_labels(labels_path: str | Path) -> list[LabelEntry]:
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels_file}")

    entries: list[LabelEntry] = []
    base_dir = labels_file.parent

    with labels_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Labels file is empty")
        required = {"path", "rating"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Labels file is missing required column(s): {missing_fields}")

        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            raw_rating = (row.get("rating") or "").strip()
            if not raw_path:
                raise ValueError(f"Row {row_number}: path is required")
            try:
                rating = float(raw_rating)
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: rating must be a number") from exc
            if not 0 <= rating <= 100:
                raise ValueError(f"Row {row_number}: rating must be between 0 and 100")

            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Row {row_number}: image does not exist: {image_path}")
            entries.append(LabelEntry(path=image_path, rating=rating))

    if not entries:
        raise ValueError("Labels file must contain at least one labeled image")
    return entries

