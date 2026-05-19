from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class ClipStore:
    embeddings: np.ndarray
    paths: list[str]
    model_name: str
    device: str


def save_clip_store(
    output_path: str | Path,
    *,
    embeddings: list[np.ndarray],
    paths: list[str],
    model_name: str,
    device: str,
) -> None:
    if not embeddings:
        raise ValueError("Cannot save a CLIP store with no embeddings")
    if len(embeddings) != len(paths):
        raise ValueError("embeddings and paths must have the same length")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        paths=np.asarray(paths, dtype=str),
        model_name=np.asarray(model_name),
        device=np.asarray(device),
    )


def load_clip_store(store_path: str | Path) -> ClipStore:
    path = Path(store_path)
    if not path.exists():
        raise FileNotFoundError(f"CLIP store does not exist: {path}")

    data = np.load(path, allow_pickle=False)
    required = {"embeddings", "paths", "model_name", "device"}
    missing = required.difference(data.files)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"CLIP store is missing required field(s): {missing_fields}")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("CLIP store embeddings must be a 2D array")
    if len(embeddings) == 0:
        raise ValueError("CLIP store is empty")

    paths = [str(value) for value in data["paths"]]
    if len(paths) != len(embeddings):
        raise ValueError("CLIP store embeddings and paths length mismatch")

    return ClipStore(
        embeddings=embeddings,
        paths=paths,
        model_name=str(data["model_name"]),
        device=str(data["device"]),
    )


def normalize_path_key(path: str | Path) -> str:
    return Path(path).as_posix().lower()
