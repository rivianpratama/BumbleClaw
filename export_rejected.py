from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy labeled images that fail face detection into a review folder.")
    parser.add_argument("--labels", default="dataset_labels.csv", help="CSV with columns: path,rating")
    parser.add_argument("--output-dir", default="results/rejected_faces", help="Folder to copy rejected images into")
    parser.add_argument("--report", default="results/rejected_faces.csv", help="CSV report path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="InsightFace model name")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE, help="Face detector input size")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_DET_THRESH, help="Face detector threshold")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in tqdm(entries, desc="Checking labeled images"):
        try:
            get_face_embedding(
                entry.path,
                model_name=args.model,
                provider=args.provider,
                det_size=args.det_size,
                det_thresh=args.det_thresh,
                enforce_detection=True,
            )
        except Exception as exc:
            copied_path = copy_rejected(entry.path, output_dir)
            rows.append(
                {
                    "path": str(entry.path),
                    "copied_path": str(copied_path),
                    "rating": entry.rating,
                    "error": str(exc),
                }
            )

    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "copied_path", "rating", "error"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Using backend=insightface, model={args.model}, provider={active_provider}, "
        f"det_size={args.det_size}, det_thresh={args.det_thresh}"
    )
    print(f"Rejected images: {len(rows)}")
    print(f"Copied to: {output_dir}")
    print(f"Report: {report_path}")
    return 0


def copy_rejected(source: Path, output_dir: Path) -> Path:
    target = output_dir / source.name
    if target.exists():
        stem = source.stem
        suffix = source.suffix
        index = 2
        while True:
            candidate = output_dir / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            index += 1
    shutil.copy2(source, target)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
