from __future__ import annotations

from dataclasses import dataclass

ORIGINAL_NAME = "Original"
ORIGINAL_SETUP = "original"
ROUND1_NAME = "Round1"
ROUND1_SETUP = "round1"
ROUND2_NAME = "Round2"
ROUND2_SETUP = "round2"
ROUND3_NAME = "Round3"
ROUND3_SETUP = "round3"
EXPERIMENTAL1_NAME = "Experimental1"
EXPERIMENTAL1_SETUP = "experimental1"
EXPERIMENTAL2_NAME = "Experimental2"
EXPERIMENTAL2_SETUP = "experimental2"
EXPERIMENTAL3_NAME = "Experimental3"
EXPERIMENTAL3_SETUP = "experimental3"
MULTIMODALX_NAME = "MultimodalX"
MULTIMODALX_SETUP = "multimodalx"
MULTIMODALX2_NAME = "MultimodalX2"
MULTIMODALX2_SETUP = "multimodalx2"
MULTIMODALX3_NAME = "MultimodalX3"
MULTIMODALX3_SETUP = "multimodalx3"
MULTIMODALX4_NAME = "MultimodalX4"
MULTIMODALX4_SETUP = "multimodalx4"
MULTIMODALX5_NAME = "MultimodalX5"
MULTIMODALX5_SETUP = "multimodalx5"
MULTIMODALX6_NAME = "MultimodalX6"
MULTIMODALX6_SETUP = "multimodalx6"


@dataclass(frozen=True)
class ExperimentalSetup:
    setup_name: str
    store: str
    regressor: str
    multimodal_regressor: str
    method: str
    face_weight: float
    threshold: float
    decision_mode: str
    preference_model: str
    preference_threshold: float
    dynamic_preference_mode: str
    dynamic_preference_percentile: float
    dynamic_preference_window: int
    dynamic_preference_min_history: int
    dynamic_preference_min_threshold: float
    dynamic_preference_max_threshold: float
    blend_preference_model: str = ""
    dynamic_mode: str = ""
    dynamic_percentile: float | None = None
    dynamic_window: int = 200
    dynamic_min_history: int = 50
    dynamic_min_threshold: float = 48.0
    dynamic_max_threshold: float = 70.0
    k: int = 11


ORIGINAL = ExperimentalSetup(
    setup_name=ORIGINAL_NAME,
    store="embeddings/reference_store.npz",
    regressor="models/rating_regressor.joblib",
    multimodal_regressor="models/rating_regressor_multimodal.joblib",
    method="face_biased",
    face_weight=0.50,
    threshold=55.0,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

ROUND1 = ExperimentalSetup(
    setup_name=ROUND1_NAME,
    store="embeddings/reference_store_bumble_combined.npz",
    regressor="models/rating_regressor_bumble_combined.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined.joblib",
    method="face_biased",
    face_weight=0.50,
    threshold=55.0,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

ROUND2 = ExperimentalSetup(
    setup_name=ROUND2_NAME,
    store="embeddings/reference_store_bumble_combined_round2.npz",
    regressor="models/rating_regressor_bumble_combined_round2.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round2.joblib",
    method="face_biased",
    face_weight=0.22,
    threshold=55.0,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

ROUND3 = ExperimentalSetup(
    setup_name=ROUND3_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="face_biased",
    face_weight=0.44,
    threshold=55.0,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

EXPERIMENTAL1 = ExperimentalSetup(
    setup_name=EXPERIMENTAL1_NAME,
    store="embeddings/reference_store_bumble_combined_round2.npz",
    regressor="models/rating_regressor_bumble_combined_round2.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round2.joblib",
    method="face_biased",
    face_weight=0.30,
    threshold=55.0,
    decision_mode="preference",
    preference_model="models/bumble_preference_experimental1.joblib",
    preference_threshold=0.556059,
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

EXPERIMENTAL2 = ExperimentalSetup(
    setup_name=EXPERIMENTAL2_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="face_biased",
    face_weight=0.30,
    threshold=55.0,
    decision_mode="preference",
    preference_model="models/bumble_preference_experimental2.joblib",
    preference_threshold=0.593991,
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

EXPERIMENTAL3 = ExperimentalSetup(
    setup_name=EXPERIMENTAL3_NAME,
    store="embeddings/reference_store.npz",
    regressor="models/rating_regressor.joblib",
    multimodal_regressor="models/rating_regressor_multimodal.joblib",
    method="multimodalx_original",
    face_weight=0.50,
    threshold=67.342307,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=20,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=85.0,
)

MULTIMODALX = ExperimentalSetup(
    setup_name=MULTIMODALX_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="multimodalx",
    face_weight=0.30,
    threshold=56.863525,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=50,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=70.0,
)

MULTIMODALX2 = ExperimentalSetup(
    setup_name=MULTIMODALX2_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="multimodalx2",
    face_weight=0.30,
    threshold=55.835886,
    decision_mode="threshold",
    preference_model="models/bumble_preference_classifier.joblib",
    preference_threshold=0.558638,
    dynamic_preference_mode="",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=50,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=70.0,
    k=9,
)

MULTIMODALX3 = ExperimentalSetup(
    setup_name=MULTIMODALX3_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="face_biased",
    face_weight=0.44,
    threshold=55.0,
    decision_mode="preference",
    preference_model="models/bumble_preference_multimodalx3.joblib",
    preference_threshold=0.493593,
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=50,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=70.0,
)

MULTIMODALX4 = ExperimentalSetup(
    setup_name=MULTIMODALX4_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="multimodalx2",
    face_weight=0.30,
    threshold=55.835886,
    decision_mode="preference",
    preference_model="models/bumble_preference_multimodalx4.joblib",
    preference_threshold=0.519971,
    blend_preference_model="models/bumble_preference_classifier.joblib",
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=85.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=50,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=70.0,
    k=9,
)

MULTIMODALX5 = ExperimentalSetup(
    setup_name=MULTIMODALX5_NAME,
    store="embeddings/reference_store_bumble_combined_round3.npz",
    regressor="models/rating_regressor_bumble_combined_round3.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round3.joblib",
    method="multimodalx5",
    face_weight=0.60 / 0.65,
    threshold=55.0,
    decision_mode="preference",
    preference_model="models/bumble_preference_multimodalx5.joblib",
    preference_threshold=0.489530,
    blend_preference_model="models/bumble_preference_classifier.joblib",
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.0,
    dynamic_preference_max_threshold=0.75,
    dynamic_mode="from_logs",
    dynamic_percentile=80.0,
    dynamic_window=200,
    dynamic_min_history=50,
    dynamic_min_threshold=48.0,
    dynamic_max_threshold=70.0,
    k=11,
)

MULTIMODALX6 = ExperimentalSetup(
    setup_name=MULTIMODALX6_NAME,
    store="embeddings/reference_store_bumble_combined_round2.npz",
    regressor="models/rating_regressor_bumble_combined_round2.joblib",
    multimodal_regressor="models/rating_regressor_multimodal_bumble_combined_round2.joblib",
    method="face_biased",
    face_weight=0.30,
    threshold=55.0,
    decision_mode="preference",
    preference_model="models/bumble_preference_round2_veto.joblib",
    preference_threshold=0.527431,
    dynamic_preference_mode="from_logs",
    dynamic_preference_percentile=80.0,
    dynamic_preference_window=200,
    dynamic_preference_min_history=50,
    dynamic_preference_min_threshold=0.45,
    dynamic_preference_max_threshold=0.75,
)

EXPERIMENTAL_SETUPS = {
    EXPERIMENTAL1_SETUP: EXPERIMENTAL1,
    EXPERIMENTAL2_SETUP: EXPERIMENTAL2,
    EXPERIMENTAL3_SETUP: EXPERIMENTAL3,
    MULTIMODALX_SETUP: MULTIMODALX,
    MULTIMODALX2_SETUP: MULTIMODALX2,
    MULTIMODALX3_SETUP: MULTIMODALX3,
    MULTIMODALX4_SETUP: MULTIMODALX4,
    MULTIMODALX5_SETUP: MULTIMODALX5,
    MULTIMODALX6_SETUP: MULTIMODALX6,
}

AUTOMATION_SETUPS = {
    ORIGINAL_SETUP: ORIGINAL,
    ROUND1_SETUP: ROUND1,
    ROUND2_SETUP: ROUND2,
    ROUND3_SETUP: ROUND3,
    **EXPERIMENTAL_SETUPS,
}
