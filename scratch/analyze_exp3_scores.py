"""Analyze the distribution of Experimental3 scores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_similarity.dynamic_threshold import (
    recent_values,
    quantile,
)

CSV_PATH = Path(r"D:\BumbleLog\scores.csv")

current_config = {
    "setup_name": "Experimental3",
    "method": "multimodalx_original",
    "store_path": Path("embeddings/reference_store.npz"),
    "regressor_path": Path("models/rating_regressor.joblib"),
    "multimodal_regressor_path": Path("models/rating_regressor_multimodal.joblib"),
    "threshold": 67.342307,
    "decision_mode": "threshold",
    "preference_model_path": Path("models/bumble_preference_classifier.joblib"),
    "face_weight": 0.50,
}

scores = recent_values(CSV_PATH, value_field="score", limit=200, current_config=current_config)
print(f"Matching scores: {len(scores)}")

if scores:
    print(f"Min: {min(scores):.4f}")
    for p in [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]:
        q = quantile(scores, p)
        print(f"Quantile {p*100:.0f}%: {q:.4f}")
    print(f"Max: {max(scores):.4f}")
