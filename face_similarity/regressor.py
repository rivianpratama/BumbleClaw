from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REGRESSOR_PATH = "models/rating_regressor_bumble_combined_round2.joblib"


@dataclass(frozen=True)
class RatingRegressor:
    estimator: Any
    model_name: str
    metrics: dict[str, float]
    metadata: dict[str, str]


def load_regressor(path: str | Path = DEFAULT_REGRESSOR_PATH) -> RatingRegressor:
    joblib = _import_joblib()
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Regressor model does not exist: {model_path}")

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ValueError(f"Invalid regressor model file: {model_path}")

    return RatingRegressor(
        estimator=payload["estimator"],
        model_name=str(payload.get("model_name", "regressor")),
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
        metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
    )


def save_regressor(
    path: str | Path,
    *,
    estimator: Any,
    model_name: str,
    metrics: dict[str, float],
    metadata: dict[str, str] | None = None,
) -> None:
    joblib = _import_joblib()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "model_name": model_name,
            "metrics": metrics,
            "metadata": metadata or {},
        },
        output,
    )


def predict_rating(regressor: RatingRegressor, embedding: np.ndarray) -> float:
    query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    rating = float(regressor.estimator.predict(query)[0])
    return float(np.clip(rating, 0, 100))


def predict_multimodal_rating(
    regressor: RatingRegressor,
    *,
    face_embedding: np.ndarray,
    clip_embedding: np.ndarray,
) -> float:
    feature_mode = regressor.metadata.get("feature_mode", "face_clip")
    if feature_mode == "face_only":
        features = np.asarray(face_embedding, dtype=np.float32).reshape(1, -1)
    elif feature_mode == "clip_only":
        features = np.asarray(clip_embedding, dtype=np.float32).reshape(1, -1)
    elif feature_mode == "face_clip":
        face = np.asarray(face_embedding, dtype=np.float32).reshape(1, -1)
        clip = np.asarray(clip_embedding, dtype=np.float32).reshape(1, -1)
        features = np.concatenate([face, clip], axis=1)
    else:
        raise ValueError(f"Unknown multimodal feature mode: {feature_mode}")
    rating = float(regressor.estimator.predict(features)[0])
    return float(np.clip(rating, 0, 100))


def _import_joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required. Run: pip install -r requirements.txt") from exc
    return joblib
