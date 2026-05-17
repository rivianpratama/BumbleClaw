from __future__ import annotations

import argparse

from tqdm import tqdm

from face_similarity.embedding import (
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    get_face_app,
    get_face_embedding,
)
from face_similarity.labels import load_labels
from face_similarity.store import save_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled face embedding reference store.")
    parser.add_argument("--labels", default="labels.csv", help="CSV with columns: path,rating")
    parser.add_argument("--output", default="embeddings/reference_store.npz", help="Output .npz store path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="InsightFace model name")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE, help="Face detector input size")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_DET_THRESH, help="Face detector threshold")
    parser.add_argument("--lenient", action="store_true", help="Do not require a detected face")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = load_labels(args.labels)
    _, active_provider = get_face_app(
        model_name=args.model,
        provider=args.provider,
        det_size=args.det_size,
        det_thresh=args.det_thresh,
    )
    print(
        f"Using backend=insightface, model={args.model}, provider={active_provider}, "
        f"det_size={args.det_size}, det_thresh={args.det_thresh}"
    )

    embeddings = []
    paths = []
    ratings = []

    progress = tqdm(entries, desc="Embedding references")
    for entry in progress:
        progress.set_postfix(file=entry.path.name)
        try:
            embedding = get_face_embedding(
                entry.path,
                model_name=args.model,
                provider=args.provider,
                det_size=args.det_size,
                det_thresh=args.det_thresh,
                enforce_detection=not args.lenient,
            )
        except Exception as exc:
            print(f"SKIP {entry.path}: {exc}")
            continue
        embeddings.append(embedding)
        paths.append(str(entry.path))
        ratings.append(entry.rating)

    save_store(
        args.output,
        embeddings=embeddings,
        paths=paths,
        ratings=ratings,
        backend="insightface",
        model_name=args.model,
        provider=active_provider,
        det_size=args.det_size,
        det_thresh=args.det_thresh,
    )
    print(f"Saved {len(embeddings)} reference embedding(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
