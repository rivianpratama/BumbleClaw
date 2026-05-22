from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping

from PIL import Image

from face_similarity.prediction import DEFAULT_FACE_BIAS_WEIGHT, RatingPrediction, biased_multimodal_score

DEFAULT_BUMBLE_LOG_DIR = Path(r"D:\BumbleLog")
DEFAULT_LOG_FORMAT = "webp"
LOG_FORMATS = {"jpg", "jpeg", "webp", "avif"}
LOG_FIELDS = [
    "timestamp",
    "screenshot",
    "setup_name",
    "method",
    "action",
    "score",
    "final_score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
    "store_path",
    "regressor_path",
    "multimodal_regressor_path",
    "threshold",
    "decision_mode",
    "preference_model_path",
    "preference_threshold",
    "preference_probability",
    "dynamic_enabled",
    "dynamic_mode",
    "dynamic_window",
    "dynamic_target_right_rate",
    "dynamic_percentile",
    "dynamic_min_history",
    "dynamic_min_threshold",
    "dynamic_max_threshold",
    "dynamic_preference_enabled",
    "dynamic_preference_mode",
    "dynamic_preference_window",
    "dynamic_preference_target_right_rate",
    "dynamic_preference_percentile",
    "dynamic_preference_min_history",
    "dynamic_preference_min_threshold",
    "dynamic_preference_max_threshold",
    "face_weight",
    "k",
    "provider",
    "delay",
    "mode_247",
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
    config: Mapping[str, object] | None = None,
) -> Path:
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_format = normalize_log_format(image_format)
    image_name = f"profile_{timestamp}.{image_format}"
    image_path = output_dir / image_name
    save_compressed_image(source_image, image_path, quality=quality, max_width=max_width, image_format=image_format)
    append_score_row(
        output_dir / "scores.csv",
        image_name,
        prediction=prediction,
        action=action,
        timestamp=timestamp,
        config=config,
    )
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
    config: Mapping[str, object] | None = None,
) -> None:
    output = Path(csv_path)
    ensure_csv_header(output)
    write_header = not output.exists()
    config = config or {}
    face_biased = None
    if prediction.face_rating is not None and prediction.multimodal_rating is not None:
        face_weight = parse_config_float(config.get("face_weight"), DEFAULT_FACE_BIAS_WEIGHT)
        face_biased = biased_multimodal_score(
            prediction.face_rating,
            prediction.multimodal_rating,
            face_weight=face_weight,
        )
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "screenshot": screenshot_name,
                "setup_name": format_config(config.get("setup_name")),
                "method": prediction.method,
                "action": action,
                "score": format_score(prediction.rating),
                "final_score": format_score(final_score(prediction, config)),
                "face_biased": format_score(face_biased),
                "multimodal": format_score(prediction.multimodal_rating),
                "ridge": format_score(prediction.face_rating),
                "knn": format_score(prediction.knn_rating),
                "store_path": format_config(config.get("store_path")),
                "regressor_path": format_config(config.get("regressor_path")),
                "multimodal_regressor_path": format_config(config.get("multimodal_regressor_path")),
                "threshold": format_config(config.get("threshold")),
                "decision_mode": format_config(config.get("decision_mode")),
                "preference_model_path": format_config(config.get("preference_model_path")),
                "preference_threshold": format_config(config.get("preference_threshold")),
                "preference_probability": format_config(config.get("preference_probability")),
                "dynamic_enabled": format_config(config.get("dynamic_enabled")),
                "dynamic_mode": format_config(config.get("dynamic_mode")),
                "dynamic_window": format_config(config.get("dynamic_window")),
                "dynamic_target_right_rate": format_config(config.get("dynamic_target_right_rate")),
                "dynamic_percentile": format_config(config.get("dynamic_percentile")),
                "dynamic_min_history": format_config(config.get("dynamic_min_history")),
                "dynamic_min_threshold": format_config(config.get("dynamic_min_threshold")),
                "dynamic_max_threshold": format_config(config.get("dynamic_max_threshold")),
                "dynamic_preference_enabled": format_config(config.get("dynamic_preference_enabled")),
                "dynamic_preference_mode": format_config(config.get("dynamic_preference_mode")),
                "dynamic_preference_window": format_config(config.get("dynamic_preference_window")),
                "dynamic_preference_target_right_rate": format_config(config.get("dynamic_preference_target_right_rate")),
                "dynamic_preference_percentile": format_config(config.get("dynamic_preference_percentile")),
                "dynamic_preference_min_history": format_config(config.get("dynamic_preference_min_history")),
                "dynamic_preference_min_threshold": format_config(config.get("dynamic_preference_min_threshold")),
                "dynamic_preference_max_threshold": format_config(config.get("dynamic_preference_max_threshold")),
                "face_weight": format_config(config.get("face_weight")),
                "k": format_config(config.get("k")),
                "provider": format_config(config.get("provider")),
                "delay": format_config(config.get("delay")),
                "mode_247": format_config(config.get("mode_247")),
            }
        )


def ensure_csv_header(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == LOG_FIELDS:
            return
        rows = list(reader)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def final_score(prediction: RatingPrediction, config: Mapping[str, object]) -> float:
    if str(config.get("decision_mode", "")).lower() != "preference":
        return prediction.rating
    preference_probability = parse_optional_float(config.get("preference_probability"))
    if preference_probability is None:
        return prediction.rating
    return preference_probability * 100.0


def format_config(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def parse_config_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
