from __future__ import annotations

from face_similarity.prediction import RatingPrediction

METHOD = "multimodalx"
METHOD2 = "multimodalx2"
METHOD5 = "multimodalx5"
METHODS = (METHOD, METHOD2, METHOD5)
PREFERENCE_FEATURE_THRESHOLD = 55.0
RIDGE_WEIGHT = 0.73
MULTIMODAL_WEIGHT = 0.20
P_LIKE_WEIGHT = 0.07
MULTIMODALX2_RIDGE_WEIGHT = 0.12
MULTIMODALX2_MULTIMODAL_WEIGHT = 0.05
MULTIMODALX2_P_LIKE_WEIGHT = 0.53
MULTIMODALX2_KNN_WEIGHT = 0.30
MULTIMODALX5_RIDGE_WEIGHT = 0.60
MULTIMODALX5_MULTIMODAL_WEIGHT = 0.05
MULTIMODALX5_P_LIKE_WEIGHT = 0.15
MULTIMODALX5_KNN_WEIGHT = 0.20
MULTIMODALX5_OLD_P_LIKE_FACE_WEIGHT = 0.30


def score(ridge: float, multimodal: float, p_like: float, *, knn: float | None = None, method: str = METHOD) -> float:
    if method == METHOD:
        return float(
            RIDGE_WEIGHT * ridge
            + MULTIMODAL_WEIGHT * multimodal
            + P_LIKE_WEIGHT * p_like * 100.0
        )
    if method == METHOD2:
        if knn is None:
            raise ValueError("MultimodalX2 requires KNN score")
        return float(
            MULTIMODALX2_RIDGE_WEIGHT * ridge
            + MULTIMODALX2_MULTIMODAL_WEIGHT * multimodal
            + MULTIMODALX2_P_LIKE_WEIGHT * p_like * 100.0
            + MULTIMODALX2_KNN_WEIGHT * knn
        )
    if method == METHOD5:
        if knn is None:
            raise ValueError("MultimodalX5 requires KNN score")
        return float(
            MULTIMODALX5_RIDGE_WEIGHT * ridge
            + MULTIMODALX5_MULTIMODAL_WEIGHT * multimodal
            + MULTIMODALX5_P_LIKE_WEIGHT * p_like * 100.0
            + MULTIMODALX5_KNN_WEIGHT * knn
        )
    raise ValueError(f"Unknown MultimodalX method: {method}")


def prediction(base: RatingPrediction, p_like: float, *, method: str = METHOD) -> RatingPrediction:
    if base.face_rating is None or base.multimodal_rating is None:
        raise ValueError("MultimodalX requires ridge and multimodal scores")
    return RatingPrediction(
        rating=score(
            base.face_rating,
            base.multimodal_rating,
            p_like,
            knn=base.knn_rating,
            method=method,
        ),
        method=method,
        face_rating=base.face_rating,
        multimodal_rating=base.multimodal_rating,
        knn_rating=base.knn_rating,
    )


def old_p_like_face_weight(method: str, face_weight: float) -> float:
    if method == METHOD5:
        return MULTIMODALX5_OLD_P_LIKE_FACE_WEIGHT
    return face_weight
