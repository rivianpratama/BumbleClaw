from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ScoreResult:
    rating: float
    max_similarity: float
    mean_similarity: float
    nearest_indices: list[int]
    nearest_similarities: list[float]


def score_embedding(
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_ratings: np.ndarray,
    *,
    k: int = 5,
) -> ScoreResult:
    if reference_embeddings.ndim != 2:
        raise ValueError("reference_embeddings must be a 2D array")
    if len(reference_embeddings) == 0:
        raise ValueError("reference_embeddings must not be empty")
    if len(reference_embeddings) != len(reference_ratings):
        raise ValueError("reference_embeddings and reference_ratings must have the same length")
    if k < 1:
        raise ValueError("k must be at least 1")

    query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    similarities = cosine_similarity(query, reference_embeddings)[0]
    top_count = min(k, len(similarities))
    top_indices = np.argsort(similarities)[-top_count:][::-1]
    top_similarities = similarities[top_indices]
    top_ratings = reference_ratings[top_indices]

    weights = np.maximum(top_similarities, 0) ** 2
    if float(weights.sum()) == 0:
        rating = float(np.mean(top_ratings))
    else:
        rating = float(np.average(top_ratings, weights=weights))

    rating = float(np.clip(rating, 0, 100))
    return ScoreResult(
        rating=rating,
        max_similarity=float(np.max(similarities)),
        mean_similarity=float(np.mean(similarities)),
        nearest_indices=[int(index) for index in top_indices],
        nearest_similarities=[float(score) for score in top_similarities],
    )

