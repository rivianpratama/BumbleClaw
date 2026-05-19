from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_similarity.store import load_store


OUTPUT_FIELDS = [
    "path",
    "rating",
    "original_rating",
    "component_id",
    "component_size",
    "component_mean",
    "component_min",
    "component_max",
]


@dataclass(frozen=True)
class ComponentStats:
    component_id: int
    size: int
    mean: float
    minimum: float
    maximum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize labels for face-only scoring by averaging near-identical faces.")
    parser.add_argument("--labels", default="dataset_labels.csv", help="Input label CSV")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Embedding store matching the labels")
    parser.add_argument("--output", default="dataset_labels_normalized.csv", help="Normalized output CSV")
    parser.add_argument("--report", default="results/label_normalization.csv", help="Changed-row report CSV")
    parser.add_argument("--similarity-threshold", type=float, default=0.90, help="Cosine similarity used to group faces")
    parser.add_argument("--min-component-size", type=int, default=2, help="Minimum group size to normalize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_label_dicts(args.labels)
    store = load_store(args.store)
    normalized_by_path, stats_by_path = normalized_ratings_by_path(
        store.embeddings,
        store.ratings,
        store.paths,
        similarity_threshold=args.similarity_threshold,
        min_component_size=args.min_component_size,
    )
    changed_rows = write_normalized_labels(args.output, rows, normalized_by_path, stats_by_path)
    write_report(args.report, changed_rows)

    print(f"Read labels: {len(rows)}")
    print(f"Embedded labels considered: {len(store.paths)}")
    print(f"Similarity threshold: {args.similarity_threshold:g}")
    print(f"Rows changed: {len(changed_rows)}")
    print(f"Saved normalized labels to {args.output}")
    print(f"Saved normalization report to {args.report}")
    return 0


def load_label_dicts(path: str | Path) -> list[dict[str, str]]:
    label_path = Path(path)
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Labels file is empty")
        if "path" not in reader.fieldnames or "rating" not in reader.fieldnames:
            raise ValueError("Labels file must contain path and rating columns")
        return [dict(row) for row in reader]


def normalized_ratings_by_path(
    embeddings: np.ndarray,
    ratings: np.ndarray,
    paths: list[str],
    *,
    similarity_threshold: float,
    min_component_size: int,
) -> tuple[dict[str, float], dict[str, ComponentStats]]:
    if len(embeddings) != len(ratings) or len(paths) != len(ratings):
        raise ValueError("embeddings, ratings, and paths must have the same length")

    parent = list(range(len(paths)))
    normalized = normalize_embeddings(embeddings)
    similarities = normalized @ normalized.T
    for index in range(len(paths) - 1):
        matches = np.flatnonzero(similarities[index, index + 1 :] >= similarity_threshold)
        for offset in matches:
            union(parent, index, index + 1 + int(offset))

    groups: dict[int, list[int]] = {}
    for index in range(len(paths)):
        groups.setdefault(find(parent, index), []).append(index)

    normalized_by_path: dict[str, float] = {}
    stats_by_path: dict[str, ComponentStats] = {}
    component_id = 1
    for indices in groups.values():
        if len(indices) < min_component_size:
            continue
        group_ratings = ratings[indices].astype(np.float32)
        mean = float(np.mean(group_ratings))
        stats = ComponentStats(
            component_id=component_id,
            size=len(indices),
            mean=mean,
            minimum=float(np.min(group_ratings)),
            maximum=float(np.max(group_ratings)),
        )
        for index in indices:
            key = path_key(paths[index])
            normalized_by_path[key] = mean
            stats_by_path[key] = stats
        component_id += 1
    return normalized_by_path, stats_by_path


def write_normalized_labels(
    output_path: str | Path,
    rows: list[dict[str, str]],
    normalized_by_path: dict[str, float],
    stats_by_path: dict[str, ComponentStats],
) -> list[dict[str, str]]:
    changed_rows: list[dict[str, str]] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            original_rating = float(row["rating"])
            key = path_key(row["path"])
            rating = normalized_by_path.get(key, original_rating)
            stats = stats_by_path.get(key)
            output_row = normalized_row(row["path"], original_rating, rating, stats)
            writer.writerow(output_row)
            if abs(rating - original_rating) > 1e-6:
                changed_rows.append(output_row)
    return changed_rows


def normalized_row(path: str, original_rating: float, rating: float, stats: ComponentStats | None) -> dict[str, str]:
    return {
        "path": path,
        "rating": f"{rating:.6f}",
        "original_rating": f"{original_rating:.6f}",
        "component_id": "" if stats is None else str(stats.component_id),
        "component_size": "" if stats is None else str(stats.size),
        "component_mean": "" if stats is None else f"{stats.mean:.6f}",
        "component_min": "" if stats is None else f"{stats.minimum:.6f}",
        "component_max": "" if stats is None else f"{stats.maximum:.6f}",
    }


def write_report(path: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def union(parent: list[int], left: int, right: int) -> None:
    left_root = find(parent, left)
    right_root = find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def path_key(path: str) -> str:
    return path.replace("\\", "/").lower()


if __name__ == "__main__":
    raise SystemExit(main())
