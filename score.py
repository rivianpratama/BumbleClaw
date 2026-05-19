from __future__ import annotations

import argparse
import csv
from pathlib import Path

from face_similarity.clip_runtime import ensure_clip_runtime
from face_similarity.regressor import DEFAULT_REGRESSOR_PATH, load_regressor
from face_similarity.prediction import (
    DEFAULT_FACE_BIAS_WEIGHT,
    DEFAULT_MULTIMODAL_REGRESSOR_PATH,
    PREDICTION_METHODS,
    predict_image_rating,
)
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score face images against a labeled reference store.")
    parser.add_argument("target", nargs="?", default="test_images", help="Image file or folder to score")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Reference store path")
    parser.add_argument("--regressor", default=DEFAULT_REGRESSOR_PATH, help="Face-only regressor path")
    parser.add_argument("--multimodal-regressor", default=DEFAULT_MULTIMODAL_REGRESSOR_PATH, help="Multimodal regressor path")
    parser.add_argument("--method", choices=PREDICTION_METHODS, default="face_biased", help="Scoring method")
    parser.add_argument("--face-weight", type=float, default=DEFAULT_FACE_BIAS_WEIGHT, help="Face-only weight for face_biased method")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest references to use")
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


def score_path(
    path: Path,
    store,
    *,
    k: int,
    method: str,
    face_weight: float,
    face_regressor=None,
    multimodal_regressor=None,
    enforce_detection: bool = True,
) -> dict[str, str]:
    prediction = predict_image_rating(
        path,
        store=store,
        method=method,
        k=k,
        provider=store.provider,
        face_regressor=face_regressor,
        multimodal_regressor=multimodal_regressor,
        face_weight=face_weight,
        enforce_detection=enforce_detection,
    )
    knn_result = score_embedding_from_path(path, store, k=k, enforce_detection=enforce_detection)
    nearest = [
        f"{Path(store.paths[index]).name}:{similarity:.4f}"
        for index, similarity in zip(knn_result.nearest_indices, knn_result.nearest_similarities)
    ]
    return {
        "file": str(path),
        "rating": f"{prediction.rating:.1f}",
        "max_similarity": f"{knn_result.max_similarity:.4f}",
        "mean_similarity": f"{knn_result.mean_similarity:.4f}",
        "nearest": "; ".join(nearest),
        "error": "",
    }


def score_embedding_from_path(path: Path, store, *, k: int, enforce_detection: bool):
    from face_similarity.embedding import get_face_embedding

    embedding = get_face_embedding(
        path,
        model_name=store.model_name,
        provider=store.provider,
        det_size=store.det_size,
        det_thresh=store.det_thresh,
        enforce_detection=enforce_detection,
    )
    return score_embedding(embedding, store.embeddings, store.ratings, k=k)


def main() -> int:
    args = parse_args()
    ensure_clip_runtime(args.method)
    store = load_store(args.store)
    face_regressor = load_regressor(args.regressor) if args.method in {"regressor", "face_biased"} else None
    multimodal_regressor = load_regressor(args.multimodal_regressor) if args.method in {"multimodal", "face_biased"} else None
    paths = image_paths(args.target)
    rows = []

    print(f"Scoring {len(paths)} image(s) against {len(store.embeddings)} reference(s)")
    print(
        f"Using backend={store.backend}, model={store.model_name}, provider={store.provider}, "
        f"det_thresh={store.det_thresh:g}"
    )
    print(f"Using method={args.method}")
    if face_regressor is not None:
        print(f"Using face regressor={face_regressor.model_name} from {args.regressor}")
    if multimodal_regressor is not None:
        print(f"Using multimodal regressor={multimodal_regressor.model_name} from {args.multimodal_regressor}")
    print(f"{'File':<32} {'Rating':>8} {'Max':>8} {'Mean':>8}  Nearest")
    print("-" * 88)

    for path in paths:
        try:
            row = score_path(
                path,
                store,
                k=args.k,
                method=args.method,
                face_weight=args.face_weight,
                face_regressor=face_regressor,
                multimodal_regressor=multimodal_regressor,
                enforce_detection=not args.lenient,
            )
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
