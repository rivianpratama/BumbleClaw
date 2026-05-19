from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_similarity.store import load_store


AUDIT_FIELDS = [
    "path_a",
    "rating_a",
    "path_b",
    "rating_b",
    "rating_gap",
    "similarity",
]


@dataclass(frozen=True)
class ConflictPair:
    path_a: str
    rating_a: float
    path_b: str
    rating_b: float
    rating_gap: float
    similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find highly similar face embeddings with conflicting labels.")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Reference store path")
    parser.add_argument("--output", default="results/label_audit.csv", help="Audit CSV output path")
    parser.add_argument("--min-gap", type=float, default=50.0, help="Minimum rating difference to report")
    parser.add_argument("--min-similarity", type=float, default=0.55, help="Minimum cosine similarity to report")
    parser.add_argument("--neighbors", type=int, default=25, help="Nearest neighbors checked per image")
    parser.add_argument("--limit", type=int, default=500, help="Maximum rows to write")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = load_store(args.store)
    pairs = find_conflicting_pairs(
        store.embeddings,
        store.ratings,
        store.paths,
        min_gap=args.min_gap,
        min_similarity=args.min_similarity,
        neighbors=args.neighbors,
        limit=args.limit,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(conflict_row(pair))

    print(f"Checked {len(store.paths)} embedded labels")
    print(f"Found {len(pairs)} suspicious pair(s)")
    print(f"Saved audit to {output}")
    if pairs:
        print("Top conflicts:")
        for pair in pairs[:10]:
            print(
                f"{pair.rating_a:.0f} vs {pair.rating_b:.0f} | "
                f"gap={pair.rating_gap:.0f} sim={pair.similarity:.4f} | "
                f"{Path(pair.path_a).name} <-> {Path(pair.path_b).name}"
            )
    return 0


def find_conflicting_pairs(
    embeddings: np.ndarray,
    ratings: np.ndarray,
    paths: list[str],
    *,
    min_gap: float,
    min_similarity: float,
    neighbors: int,
    limit: int,
) -> list[ConflictPair]:
    if len(embeddings) != len(ratings) or len(paths) != len(ratings):
        raise ValueError("embeddings, ratings, and paths must have the same length")
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    normalized = normalize_embeddings(embeddings)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)

    neighbor_count = min(neighbors, max(len(ratings) - 1, 1))
    pairs_by_key: dict[tuple[int, int], ConflictPair] = {}
    for index in range(len(ratings)):
        nearest_indices = top_indices(similarities[index], neighbor_count)
        for neighbor_index in nearest_indices:
            similarity = float(similarities[index, neighbor_index])
            if similarity < min_similarity:
                continue
            gap = float(abs(float(ratings[index]) - float(ratings[neighbor_index])))
            if gap < min_gap:
                continue

            left, right = sorted((index, int(neighbor_index)))
            key = (left, right)
            pairs_by_key[key] = ConflictPair(
                path_a=paths[left],
                rating_a=float(ratings[left]),
                path_b=paths[right],
                rating_b=float(ratings[right]),
                rating_gap=gap,
                similarity=similarity,
            )

    pairs = sorted(
        pairs_by_key.values(),
        key=lambda pair: (-pair.rating_gap, -pair.similarity, pair.path_a, pair.path_b),
    )
    return pairs[:limit]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    if count >= len(values):
        return np.argsort(values)[::-1]
    unordered = np.argpartition(values, -count)[-count:]
    return unordered[np.argsort(values[unordered])[::-1]]


def conflict_row(pair: ConflictPair) -> dict[str, str]:
    return {
        "path_a": pair.path_a,
        "rating_a": f"{pair.rating_a:.0f}",
        "path_b": pair.path_b,
        "rating_b": f"{pair.rating_b:.0f}",
        "rating_gap": f"{pair.rating_gap:.0f}",
        "similarity": f"{pair.similarity:.6f}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
