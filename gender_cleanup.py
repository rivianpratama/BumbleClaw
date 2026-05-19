from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from face_similarity.embedding import (
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    get_face_app,
    select_largest_face,
)
from face_similarity.labeling import IMAGE_EXTENSIONS, natural_sort_key


REPORT_FIELDS = [
    "path",
    "action",
    "gender",
    "age",
    "face_count",
    "bbox",
    "moved_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarantine male-predicted images from a face dataset.")
    parser.add_argument("--source-dir", default="references/Selfie", help="Folder to scan")
    parser.add_argument("--output-dir", default="references/Selfie_removed_men", help="Folder for quarantined images")
    parser.add_argument("--report", default="results/selfie_gender_cleanup.csv", help="CSV report path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="InsightFace model name")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE, help="Face detector input size")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_DET_THRESH, help="Face detector threshold")
    parser.add_argument("--dry-run", action="store_true", help="Write report without moving files")
    parser.add_argument("--move-unknown", action="store_true", help="Also quarantine images with no usable gender result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    image_paths = discover_images(source_dir)

    app, active_provider = get_face_app(
        model_name=args.model,
        provider=args.provider,
        det_size=args.det_size,
        det_thresh=args.det_thresh,
    )
    cv2 = _import_cv2()

    rows = []
    moved_count = 0
    for image_path in tqdm(image_paths, desc="Checking gender"):
        row = inspect_image(image_path, app, cv2)
        move_action = quarantine_action(row, move_unknown=args.move_unknown, dry_run=args.dry_run)
        if move_action is not None:
            row["action"] = move_action
            if not args.dry_run:
                row["moved_path"] = str(move_to_quarantine(image_path, source_dir, output_dir))
                moved_count += 1
        rows.append(row)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = count_actions(rows)
    print(
        f"Using backend=insightface, model={args.model}, provider={active_provider}, "
        f"det_size={args.det_size}, det_thresh={args.det_thresh:g}"
    )
    print(f"Scanned: {len(rows)}")
    print(f"Female kept: {counts.get('keep', 0)}")
    male_action = "would_move_male" if args.dry_run else "moved_male"
    unknown_action = "would_move_unknown" if args.dry_run else "moved_unknown"
    print(f"Male {'would move' if args.dry_run else 'moved'}: {counts.get(male_action, 0)}")
    print(f"Unknown/no-face {'would move' if args.dry_run and args.move_unknown else 'moved' if args.move_unknown else 'kept'}: {counts.get(unknown_action, counts.get('unknown', 0))}")
    if not args.dry_run:
        print(f"Moved files: {moved_count}")
        print(f"Quarantine: {output_dir}")
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


def inspect_image(image_path: Path, app: Any, cv2: Any) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "path": image_path.as_posix(),
        "action": "unknown",
        "gender": "unknown",
        "age": "",
        "face_count": 0,
        "bbox": "",
        "moved_path": "",
        "error": "",
    }
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            row["error"] = "Could not read image"
            return row

        faces = app.get(image)
        row["face_count"] = len(faces)
        if not faces:
            row["error"] = "No face detected"
            return row

        face = select_largest_face(faces)
        row["bbox"] = bbox_text(face.bbox)
        row["age"] = "" if getattr(face, "age", None) is None else int(face.age)
        row["gender"] = gender_label(getattr(face, "gender", None))
        row["action"] = "keep" if row["gender"] == "female" else "unknown"
    except Exception as exc:
        row["error"] = str(exc)
    return row


def gender_label(gender: Any) -> str:
    if gender is None:
        return "unknown"
    return "male" if int(gender) == 1 else "female"


def quarantine_action(row: dict[str, str | int], *, move_unknown: bool, dry_run: bool) -> str | None:
    prefix = "would_move" if dry_run else "moved"
    if row["gender"] == "male":
        return f"{prefix}_male"
    if move_unknown and row["action"] == "unknown":
        return f"{prefix}_unknown"
    return None


def bbox_text(bbox: Any) -> str:
    return ",".join(str(round(float(value), 2)) for value in bbox)


def move_to_quarantine(source: Path, source_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(source_root)
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target = unused_path(target)
    shutil.move(str(source), str(target))
    return target


def unused_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def count_actions(rows: list[dict[str, str | int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row["action"])
        counts[action] = counts.get(action, 0) + 1
    return counts


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required. Run: pip install -r requirements.txt") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
