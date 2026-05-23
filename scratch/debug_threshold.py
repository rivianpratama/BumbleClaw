"""Verify Experimental3 dynamic threshold fix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_similarity.dynamic_threshold import (
    recent_values,
    threshold_from_scores,
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

# With old min_history=50
old = threshold_from_scores(scores, fixed_threshold=67.342307, target_right_rate=0.20,
                            min_history=50, min_threshold=48.0, max_threshold=70.0)
# With new min_history=20
new = threshold_from_scores(scores, fixed_threshold=67.342307, target_right_rate=0.20,
                            min_history=20, min_threshold=48.0, max_threshold=70.0)

print(f"Old (min_history=50): {old:.4f} {'(STUCK - fixed)' if old == 67.342307 else '(dynamic)'}")
print(f"New (min_history=20): {new:.4f} {'(STUCK - fixed)' if new == 67.342307 else '(dynamic YES)'}")
