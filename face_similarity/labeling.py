from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_FIELDS = ["path", "rating_1_5", "rating"]
RATING_TO_SCORE = {
    1: 0,
    2: 25,
    3: 50,
    4: 75,
    5: 100,
}


@dataclass(frozen=True)
class LabelRow:
    path: str
    rating_1_5: int
    rating: int


def rating_to_score(rating_1_5: int) -> int:
    if rating_1_5 not in RATING_TO_SCORE:
        raise ValueError("rating_1_5 must be between 1 and 5")
    return RATING_TO_SCORE[rating_1_5]


def discover_images(source_dir: str | Path) -> list[str]:
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {root}")

    paths = [
        _csv_path(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_sort_key)


def natural_sort_key(value: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def load_label_rows(csv_path: str | Path) -> list[LabelRow]:
    path = Path(csv_path)
    if not path.exists():
        return []

    rows: list[LabelRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        for row in reader:
            try:
                image_path = (row.get("path") or "").strip()
                rating_1_5 = int((row.get("rating_1_5") or "").strip())
                rating = int(float((row.get("rating") or "").strip()))
            except ValueError:
                continue
            if not image_path:
                continue
            if rating_to_score(rating_1_5) != rating:
                continue
            rows.append(LabelRow(path=image_path, rating_1_5=rating_1_5, rating=rating))
    return rows


def labeled_paths(csv_path: str | Path) -> set[str]:
    return {row.path for row in load_label_rows(csv_path)}


def upsert_label(csv_path: str | Path, image_path: str | Path, rating_1_5: int) -> LabelRow:
    path = Path(csv_path)
    row = LabelRow(path=_csv_path(image_path), rating_1_5=rating_1_5, rating=rating_to_score(rating_1_5))
    rows = {existing.path: existing for existing in load_label_rows(path)}
    rows[row.path] = row

    ordered_rows = sorted(rows.values(), key=lambda item: natural_sort_key(item.path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in ordered_rows:
            writer.writerow(
                {
                    "path": item.path,
                    "rating_1_5": item.rating_1_5,
                    "rating": item.rating,
                }
            )
    return row


def next_unlabeled_path(paths: list[str], labeled: set[str], skipped: set[str]) -> str | None:
    for path in paths:
        if path not in labeled and path not in skipped:
            return path
    return None


def _csv_path(path: str | Path) -> str:
    return Path(path).as_posix()
