from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from face_similarity.prediction import RatingPrediction, biased_multimodal_score


FEATURE_FIELDS = [
    "score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
    "threshold",
    "face_weight",
    "component_spread",
    "ridge_minus_multimodal",
    "distance_from_threshold",
    "score_bucket",
    "is_round2",
    "is_round3",
]

SCORE_BUCKETS = [0, 20, 30, 40, 50, 60, 70, 80, 100]


@dataclass(frozen=True)
class PreferenceModel:
    estimator: Any
    model_name: str
    feature_fields: list[str]
    threshold: float
    metrics: dict[str, float]


class BucketLikeRateClassifier:
    def __init__(self, *, score_index: int, buckets: list[float] | None = None, smoothing: float = 2.0) -> None:
        self.score_index = score_index
        self.buckets = buckets or SCORE_BUCKETS
        self.smoothing = smoothing
        self.bucket_rates_: list[float] = []
        self.global_rate_: float = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BucketLikeRateClassifier":
        y = np.asarray(y, dtype=np.float32)
        self.global_rate_ = float(y.mean()) if len(y) else 0.5
        rates = []
        for lower, upper in zip(self.buckets, self.buckets[1:]):
            scores = x[:, self.score_index]
            mask = (scores >= lower) & (scores < upper)
            positives = float(y[mask].sum())
            total = float(mask.sum())
            rate = (positives + self.smoothing * self.global_rate_) / (total + self.smoothing)
            rates.append(float(rate))
        self.bucket_rates_ = rates
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = []
        for row in np.asarray(x):
            score = row[self.score_index]
            rate = self.global_rate_
            for index, (lower, upper) in enumerate(zip(self.buckets, self.buckets[1:])):
                if lower <= score < upper:
                    rate = self.bucket_rates_[index]
                    break
            probabilities.append([1.0 - rate, rate])
        return np.asarray(probabilities, dtype=np.float32)


def load_preference_model(path: str | Path) -> PreferenceModel:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ValueError(f"Invalid preference model file: {path}")
    return PreferenceModel(
        estimator=payload["estimator"],
        model_name=str(payload.get("model_name", "preference")),
        feature_fields=list(payload.get("feature_fields", FEATURE_FIELDS)),
        threshold=float(payload.get("threshold", 0.5)),
        metrics=dict(payload.get("metrics", {})),
    )


def save_preference_model(
    path: str | Path,
    *,
    estimator: Any,
    model_name: str,
    threshold: float,
    metrics: Mapping[str, float],
    feature_fields: list[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "model_name": model_name,
            "threshold": float(threshold),
            "metrics": dict(metrics),
            "feature_fields": feature_fields or FEATURE_FIELDS,
        },
        output,
    )


def preference_probability(model: PreferenceModel, features: Mapping[str, float]) -> float:
    row = np.asarray([[float(features.get(field, 0.0)) for field in model.feature_fields]], dtype=np.float32)
    probability = model.estimator.predict_proba(row)[0][1]
    return float(probability)


def features_from_prediction(
    prediction: RatingPrediction,
    *,
    threshold: float,
    face_weight: float,
    regressor_path: str | Path = "",
    multimodal_regressor_path: str | Path = "",
) -> dict[str, float]:
    face_biased = prediction.rating
    if prediction.face_rating is not None and prediction.multimodal_rating is not None:
        face_biased = biased_multimodal_score(
            prediction.face_rating,
            prediction.multimodal_rating,
            face_weight=face_weight,
        )
    return make_features(
        {
            "score": prediction.rating,
            "face_biased": face_biased,
            "multimodal": prediction.multimodal_rating,
            "ridge": prediction.face_rating,
            "knn": prediction.knn_rating,
            "threshold": threshold,
            "face_weight": face_weight,
            "regressor_path": str(regressor_path),
            "multimodal_regressor_path": str(multimodal_regressor_path),
        }
    )


def make_features(row: Mapping[str, object]) -> dict[str, float]:
    score = parse_float(row.get("score"))
    face_biased = parse_float(row.get("face_biased"), score)
    multimodal = parse_float(row.get("multimodal"), score)
    ridge = parse_float(row.get("ridge"), score)
    knn = parse_float(row.get("knn"), score)
    threshold = parse_float(row.get("threshold"), 55.0)
    face_weight = parse_float(row.get("face_weight"), 0.44)
    values = [value for value in (face_biased, multimodal, ridge, knn) if value is not None]
    component_spread = max(values) - min(values) if values else 0.0
    ridge_minus_multimodal = (ridge or 0.0) - (multimodal or 0.0)
    paths = f"{row.get('regressor_path', '')} {row.get('multimodal_regressor_path', '')}".lower()
    return {
        "score": score or 0.0,
        "face_biased": face_biased or 0.0,
        "multimodal": multimodal or 0.0,
        "ridge": ridge or 0.0,
        "knn": knn or 0.0,
        "threshold": threshold or 0.0,
        "face_weight": face_weight or 0.0,
        "component_spread": component_spread,
        "ridge_minus_multimodal": ridge_minus_multimodal,
        "distance_from_threshold": (score or 0.0) - (threshold or 0.0),
        "score_bucket": float(score_bucket(score or 0.0)),
        "is_round2": float("round2" in paths),
        "is_round3": float("round3" in paths),
    }


def feature_vector(features: Mapping[str, float], fields: list[str] | None = None) -> list[float]:
    fields = fields or FEATURE_FIELDS
    return [float(features.get(field, 0.0)) for field in fields]


def parse_float(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_bucket(score: float) -> int:
    for index, (lower, upper) in enumerate(zip(SCORE_BUCKETS, SCORE_BUCKETS[1:])):
        if lower <= score < upper:
            return index
    return len(SCORE_BUCKETS) - 2
