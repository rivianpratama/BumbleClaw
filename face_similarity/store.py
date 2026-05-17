from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceStore:
    embeddings: np.ndarray
    paths: list[str]
    ratings: np.ndarray
    backend: str
    model_name: str
    provider: str
    det_size: int
    det_thresh: float


def save_store(
    output_path: str | Path,
    *,
    embeddings: list[np.ndarray],
    paths: list[str],
    ratings: list[float],
    backend: str,
    model_name: str,
    provider: str,
    det_size: int,
    det_thresh: float,
) -> None:
    if not embeddings:
        raise ValueError("Cannot save a reference store with no embeddings")
    if not (len(embeddings) == len(paths) == len(ratings)):
        raise ValueError("embeddings, paths, and ratings must have the same length")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        paths=np.asarray(paths, dtype=str),
        ratings=np.asarray(ratings, dtype=np.float32),
        backend=np.asarray(backend),
        model_name=np.asarray(model_name),
        provider=np.asarray(provider),
        det_size=np.asarray(det_size, dtype=np.int32),
        det_thresh=np.asarray(det_thresh, dtype=np.float32),
    )


def load_store(store_path: str | Path) -> ReferenceStore:
    path = Path(store_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference store does not exist: {path}")

    data = np.load(path, allow_pickle=False)
    required = {"embeddings", "paths", "ratings", "backend", "model_name", "provider", "det_size"}
    missing = required.difference(data.files)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(
            "Reference store is missing required field(s): "
            f"{missing_fields}. Rebuild it with build_references.py."
        )

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    ratings = np.asarray(data["ratings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Reference store embeddings must be a 2D array")
    if len(embeddings) == 0:
        raise ValueError("Reference store is empty")
    if len(embeddings) != len(ratings):
        raise ValueError("Reference store embeddings and ratings length mismatch")

    return ReferenceStore(
        embeddings=embeddings,
        paths=[str(value) for value in data["paths"]],
        ratings=ratings,
        backend=str(data["backend"]),
        model_name=str(data["model_name"]),
        provider=str(data["provider"]),
        det_size=int(data["det_size"]),
        det_thresh=float(data["det_thresh"]) if "det_thresh" in data.files else 0.5,
    )
