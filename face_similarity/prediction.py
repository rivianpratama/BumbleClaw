from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from face_similarity.clip_embedding import get_clip_embedding
from face_similarity.embedding import get_face_embedding
from face_similarity.regressor import RatingRegressor, predict_multimodal_rating, predict_rating
from face_similarity.scoring import score_embedding
from face_similarity.store import ReferenceStore

DEFAULT_MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal_bumble_combined_round3.joblib"
DEFAULT_FACE_BIAS_WEIGHT = 0.44
PREDICTION_METHODS = ("face_biased", "multimodal", "regressor", "knn")


@dataclass(frozen=True)
class RatingPrediction:
    rating: float
    method: str
    face_rating: float | None
    multimodal_rating: float | None
    knn_rating: float


def biased_multimodal_score(face_rating: float, multimodal_rating: float, *, face_weight: float) -> float:
    return float(face_weight * face_rating + (1 - face_weight) * multimodal_rating)


def predict_image_rating(
    image_path: str | Path,
    *,
    store: ReferenceStore,
    method: str,
    k: int,
    provider: str,
    face_regressor: RatingRegressor | None,
    multimodal_regressor: RatingRegressor | None,
    face_weight: float = DEFAULT_FACE_BIAS_WEIGHT,
    enforce_detection: bool = True,
    include_components: bool = False,
) -> RatingPrediction:
    embedding = get_face_embedding(
        image_path,
        model_name=store.model_name,
        provider=provider,
        det_size=store.det_size,
        det_thresh=store.det_thresh,
        enforce_detection=enforce_detection,
    )
    knn_result = score_embedding(embedding, store.embeddings, store.ratings, k=k)
    knn_rating = knn_result.rating
    face_rating = predict_rating(face_regressor, embedding) if face_regressor is not None else None
    multimodal_rating = None

    if multimodal_regressor is not None and (include_components or method in {"multimodal", "face_biased"}):
        clip_model = multimodal_regressor.metadata.get("clip_model", "openai/clip-vit-base-patch32")
        clip_embedding = get_clip_embedding(image_path, model_name=clip_model, provider=provider)
        multimodal_rating = predict_multimodal_rating(
            multimodal_regressor,
            face_embedding=embedding,
            clip_embedding=clip_embedding,
        )

    if method == "knn":
        return RatingPrediction(knn_rating, method, face_rating, multimodal_rating, knn_rating)

    if method == "regressor":
        if face_rating is None:
            raise ValueError("Face regressor is required for --method regressor")
        return RatingPrediction(face_rating, method, face_rating, multimodal_rating, knn_rating)

    if method in {"multimodal", "face_biased"}:
        if multimodal_rating is None:
            raise ValueError(f"Multimodal regressor is required for --method {method}")
        if method == "multimodal":
            return RatingPrediction(multimodal_rating, method, face_rating, multimodal_rating, knn_rating)
        if face_rating is None:
            raise ValueError("Face regressor is required for --method face_biased")
        rating = biased_multimodal_score(face_rating, multimodal_rating, face_weight=face_weight)
        return RatingPrediction(rating, method, face_rating, multimodal_rating, knn_rating)

    raise ValueError(f"Unknown prediction method: {method}")
