from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PIL import Image

from face_similarity.prediction import DEFAULT_FACE_BIAS_WEIGHT, RatingPrediction, biased_multimodal_score

DEFAULT_BUMBLE_LOG_DIR = Path(r"D:\BumbleLog")
DEFAULT_LOG_FORMAT = "webp"
LOG_FORMATS = {"jpg", "jpeg", "webp", "avif"}
LOG_FIELDS = [
    "timestamp",
    "screenshot",
    "method",
    "action",
    "score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
]


def save_profile_log(
    source_image: str | Path,
    *,
    prediction: RatingPrediction,
    action: str,
    log_dir: str | Path = DEFAULT_BUMBLE_LOG_DIR,
    quality: int = 45,
    max_width: int = 720,
    image_format: str = DEFAULT_LOG_FORMAT,
) -> Path:
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_format = normalize_log_format(image_format)
    image_name = f"profile_{timestamp}.{image_format}"
    image_path = output_dir / image_name
    save_compressed_image(source_image, image_path, quality=quality, max_width=max_width, image_format=image_format)
    append_score_row(output_dir / "scores.csv", image_name, prediction=prediction, action=action, timestamp=timestamp)
    return image_path


def save_compressed_image(
    source_image: str | Path,
    output_path: str | Path,
    *,
    quality: int,
    max_width: int,
    image_format: str,
) -> None:
    image_format = normalize_log_format(image_format)
    with Image.open(source_image) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = max(1, int(image.height * (max_width / image.width)))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(output_path, format=pillow_format(image_format), quality=quality, optimize=True)


def normalize_log_format(image_format: str) -> str:
    normalized = image_format.lower().lstrip(".")
    if normalized == "jpeg":
        return "jpg"
    if normalized not in LOG_FORMATS:
        allowed = ", ".join(sorted(LOG_FORMATS))
        raise ValueError(f"log image format must be one of: {allowed}")
    return normalized


def pillow_format(image_format: str) -> str:
    if image_format == "jpg":
        return "JPEG"
    return image_format.upper()


def append_score_row(
    csv_path: str | Path,
    screenshot_name: str,
    *,
    prediction: RatingPrediction,
    action: str,
    timestamp: str,
) -> None:
    output = Path(csv_path)
    write_header = not output.exists()
    face_biased = None
    if prediction.face_rating is not None and prediction.multimodal_rating is not None:
        face_biased = biased_multimodal_score(
            prediction.face_rating,
            prediction.multimodal_rating,
            face_weight=DEFAULT_FACE_BIAS_WEIGHT,
        )
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "screenshot": screenshot_name,
                "method": prediction.method,
                "action": action,
                "score": format_score(prediction.rating),
                "face_biased": format_score(face_biased),
                "multimodal": format_score(prediction.multimodal_rating),
                "ridge": format_score(prediction.face_rating),
                "knn": format_score(prediction.knn_rating),
            }
        )


def format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"
