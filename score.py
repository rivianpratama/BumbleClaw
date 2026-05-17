from __future__ import annotations

import argparse
import csv
from pathlib import Path

from face_similarity.embedding import get_face_embedding
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score face images against a labeled reference store.")
    parser.add_argument("target", nargs="?", default="test_images", help="Image file or folder to score")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Reference store path")
    parser.add_argument("--k", type=int, default=11, help="Number of nearest references to use")
    parser.add_argument("--lenient", action="store_true", help="Do not require a detected face")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path")
    return parser.parse_args()


def image_paths(target: str) -> list[Path]:
    path = Path(target)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Target does not exist: {path}")
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    ]


def score_path(path: Path, store, *, k: int, enforce_detection: bool) -> dict[str, str]:
    embedding = get_face_embedding(
        path,
        model_name=store.model_name,
        provider=store.provider,
        det_size=store.det_size,
        det_thresh=store.det_thresh,
        enforce_detection=enforce_detection,
    )
    result = score_embedding(embedding, store.embeddings, store.ratings, k=k)
    nearest = [
        f"{Path(store.paths[index]).name}:{similarity:.4f}"
        for index, similarity in zip(result.nearest_indices, result.nearest_similarities)
    ]
    return {
        "file": str(path),
        "rating": f"{result.rating:.1f}",
        "max_similarity": f"{result.max_similarity:.4f}",
        "mean_similarity": f"{result.mean_similarity:.4f}",
        "nearest": "; ".join(nearest),
        "error": "",
    }


def main() -> int:
    args = parse_args()
    store = load_store(args.store)
    paths = image_paths(args.target)
    rows = []

    print(f"Scoring {len(paths)} image(s) against {len(store.embeddings)} reference(s)")
    print(
        f"Using backend={store.backend}, model={store.model_name}, provider={store.provider}, "
        f"det_thresh={store.det_thresh:g}"
    )
    print(f"{'File':<32} {'Rating':>8} {'Max':>8} {'Mean':>8}  Nearest")
    print("-" * 88)

    for path in paths:
        try:
            row = score_path(path, store, k=args.k, enforce_detection=not args.lenient)
            print(
                f"{Path(row['file']).name:<32} {row['rating']:>8} "
                f"{row['max_similarity']:>8} {row['mean_similarity']:>8}  {row['nearest']}"
            )
        except Exception as exc:
            row = {
                "file": str(path),
                "rating": "",
                "max_similarity": "",
                "mean_similarity": "",
                "nearest": "",
                "error": str(exc),
            }
            print(f"{path.name:<32} ERROR: {exc}")
        rows.append(row)

    if args.csv_path:
        output = Path(args.csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["file", "rating", "max_similarity", "mean_similarity", "nearest", "error"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
