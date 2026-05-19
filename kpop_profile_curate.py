from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from face_similarity.embedding import (
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    get_face_app,
)
from face_similarity.labeling import IMAGE_EXTENSIONS, natural_sort_key
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store
from gender_cleanup import bbox_text, gender_label


REPORT_FIELDS = [
    "path",
    "identity",
    "status",
    "gender",
    "age",
    "face_count",
    "face_area_ratio",
    "predicted_rating",
    "max_similarity",
    "selected_path",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    path: Path
    identity: str
    embedding: np.ndarray
    gender: str
    age: str
    face_count: int
    face_area_ratio: float
    bbox: str
    predicted_rating: float
    max_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Bumble-like K-pop profile candidates for manual labeling.")
    parser.add_argument("--source-dir", default=r"D:\KPOP dataset", help="K-pop profile dataset folder")
    parser.add_argument("--output-dir", default="references/Kpop_Profile_curated", help="Curated copy destination")
    parser.add_argument("--report", default="results/kpop_profile_curate.csv", help="CSV report path")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Reference store used for scoring")
    parser.add_argument("--target-count", type=int, default=400, help="Maximum images to copy")
    parser.add_argument("--max-per-identity", type=int, default=14, help="Maximum selected images per identity")
    parser.add_argument("--top-per-identity-pool", type=int, default=35, help="Highest-scoring images kept per identity before diversity selection")
    parser.add_argument("--k", type=int, default=16, help="Number of nearest references to use for scoring")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="InsightFace model name")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE, help="Face detector input size")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_DET_THRESH, help="Face detector threshold")
    parser.add_argument("--dry-run", action="store_true", help="Write report without copying files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    image_paths = discover_images(source_dir)
    unique_paths, duplicate_paths = dedupe_paths(image_paths)
    store = load_store(args.store)

    app, active_provider = get_face_app(
        model_name=args.model,
        provider=args.provider,
        det_size=args.det_size,
        det_thresh=args.det_thresh,
    )
    cv2 = _import_cv2()

    rows_by_path = {path: base_report_row(path, source_dir) for path in image_paths}
    for path in duplicate_paths:
        rows_by_path[path]["status"] = "rejected_exact_duplicate"

    candidates: list[Candidate] = []
    for image_path in tqdm(unique_paths, desc="Scanning K-pop profile faces"):
        row = rows_by_path[image_path]
        candidate, error = inspect_candidate(
            image_path,
            source_dir,
            app,
            cv2,
            store_embeddings=store.embeddings,
            store_ratings=store.ratings,
            k=args.k,
        )
        if candidate is None:
            row["status"] = error or "rejected"
            row["error"] = error or ""
            continue

        row.update(candidate_report_values(candidate))
        if candidate.gender != "female":
            row["status"] = "rejected_not_female"
        else:
            row["status"] = "eligible"
            candidates.append(candidate)

    selected = select_balanced(
        candidates,
        target_count=args.target_count,
        max_per_identity=args.max_per_identity,
        top_per_identity_pool=args.top_per_identity_pool,
    )
    selected_paths = {candidate.path: candidate for candidate in selected}

    copied = 0
    for candidate in selected:
        row = rows_by_path[candidate.path]
        target = output_dir / candidate.path.relative_to(source_dir)
        row["status"] = "selected"
        row["selected_path"] = target.as_posix()
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.path, target)
            copied += 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_path[path] for path in image_paths]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = count_statuses(rows)
    print(
        f"Using backend=insightface, model={args.model}, provider={active_provider}, "
        f"det_size={args.det_size}, det_thresh={args.det_thresh:g}, k={args.k}"
    )
    print(f"Scanned: {len(image_paths)}")
    print(f"Unique images checked: {len(unique_paths)}")
    print(f"Eligible female single-face images: {len(candidates)}")
    print(f"Selected: {len(selected_paths)}")
    print(f"Copied: {copied if not args.dry_run else 0}")
    print(f"Identity buckets selected: {len({candidate.identity for candidate in selected})}")
    print(f"Rejected/no-face: {status_counts.get('rejected_no_face', 0)}")
    print(f"Rejected/not female: {status_counts.get('rejected_not_female', 0)}")
    print(f"Rejected/multi-face: {status_counts.get('rejected_multi_face', 0)}")
    print(f"Rejected/exact duplicate: {status_counts.get('rejected_exact_duplicate', 0)}")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    return 0


def discover_images(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: natural_sort_key(path.as_posix()))


def identity_key(path: Path, source_dir: Path) -> str:
    parts = path.relative_to(source_dir).parts
    if len(parts) >= 3 and parts[0] in {"1", "2"}:
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return "_root"


def dedupe_paths(paths: list[Path]) -> tuple[list[Path], set[Path]]:
    seen_hashes: set[str] = set()
    unique_paths = []
    duplicate_paths = set()
    for path in paths:
        digest = sha256_file(path)
        if digest in seen_hashes:
            duplicate_paths.add(path)
            continue
        seen_hashes.add(digest)
        unique_paths.append(path)
    return unique_paths, duplicate_paths


def base_report_row(path: Path, source_dir: Path) -> dict[str, str | int | float]:
    return {
        "path": path.as_posix(),
        "identity": identity_key(path, source_dir),
        "status": "",
        "gender": "",
        "age": "",
        "face_count": 0,
        "face_area_ratio": "",
        "predicted_rating": "",
        "max_similarity": "",
        "selected_path": "",
        "error": "",
    }


def inspect_candidate(
    path: Path,
    source_dir: Path,
    app: Any,
    cv2: Any,
    *,
    store_embeddings: np.ndarray,
    store_ratings: np.ndarray,
    k: int,
) -> tuple[Candidate | None, str | None]:
    image = cv2.imread(str(path))
    if image is None:
        return None, "rejected_unreadable"

    faces = app.get(image)
    if not faces:
        return None, "rejected_no_face"
    if len(faces) > 1:
        return None, "rejected_multi_face"

    face = faces[0]
    embedding = np.asarray(face.embedding, dtype=np.float32)
    score = score_embedding(embedding, store_embeddings, store_ratings, k=k)
    height, width = image.shape[:2]
    age = "" if getattr(face, "age", None) is None else str(int(face.age))
    return (
        Candidate(
            path=path,
            identity=identity_key(path, source_dir),
            embedding=embedding,
            gender=gender_label(getattr(face, "gender", None)),
            age=age,
            face_count=len(faces),
            face_area_ratio=face_area_ratio(face.bbox, width=width, height=height),
            bbox=bbox_text(face.bbox),
            predicted_rating=score.rating,
            max_similarity=score.max_similarity,
        ),
        None,
    )


def candidate_report_values(candidate: Candidate) -> dict[str, str | int | float]:
    return {
        "gender": candidate.gender,
        "age": candidate.age,
        "face_count": candidate.face_count,
        "face_area_ratio": f"{candidate.face_area_ratio:.6f}",
        "predicted_rating": f"{candidate.predicted_rating:.4f}",
        "max_similarity": f"{candidate.max_similarity:.4f}",
    }


def face_area_ratio(bbox: Any, *, width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    image_area = max(width * height, 1)
    return float(area / image_area)


def select_balanced(
    candidates: list[Candidate],
    *,
    target_count: int,
    max_per_identity: int,
    top_per_identity_pool: int,
) -> list[Candidate]:
    by_identity: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_identity.setdefault(candidate.identity, []).append(candidate)

    per_identity = {}
    for identity, items in sorted(by_identity.items(), key=lambda item: natural_sort_key(item[0])):
        ranked = sorted(items, key=lambda item: (-item.predicted_rating, -item.max_similarity, natural_sort_key(item.path.as_posix())))
        pool = ranked[:top_per_identity_pool]
        per_identity[identity] = select_diverse(pool, max_per_identity)
    return round_robin_take(per_identity, target_count)


def select_diverse(candidates: list[Candidate], limit: int) -> list[Candidate]:
    if len(candidates) <= limit:
        return candidates

    embeddings = np.vstack([normalize(item.embedding) for item in candidates])
    selected_indices = [0]
    while len(selected_indices) < limit:
        selected = embeddings[selected_indices]
        similarities = embeddings @ selected.T
        max_similarity = similarities.max(axis=1)
        max_similarity[selected_indices] = np.inf
        next_index = int(np.argmin(max_similarity))
        selected_indices.append(next_index)
    return [candidates[index] for index in selected_indices]


def round_robin_take(per_identity: dict[str, list[Candidate]], target_count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    offset = 0
    identities = list(per_identity)
    while len(selected) < target_count:
        added = False
        for identity in identities:
            items = per_identity[identity]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        offset += 1
    return selected


def normalize(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0:
        return embedding
    return embedding / norm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_statuses(rows: list[dict[str, str | int | float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required. Run: pip install -r requirements.txt") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
