"""Analyze the preference probability of Experimental3 rows in the CSV."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_similarity.dynamic_threshold import (
    recent_values,
    quantile,
)

CSV_PATH = Path(r"D:\BumbleLog\scores.csv")

# Let's see what config parameters recent_values checks
# row_matches_config matches:
#   setup_name
#   method
#   face_weight
#   regressor_path
#   multimodal_regressor_path
#   decision_mode
#   preference_model_path

# The user is running with:
# --setup experimental3 --provider cuda --loop --delay 0 --log-quality 50 --log-max-width 720
# --dynamic-preference-from-logs --dynamic-preference-percentile 80 --dynamic-preference-window 20 --dynamic-preference-min-history 10

# Let's simulate this configuration:
current_config_default_decision = {
    "setup_name": "Experimental3",
    "method": "multimodalx_original",
    "store_path": Path("embeddings/reference_store.npz"),
    "regressor_path": Path("models/rating_regressor.joblib"),
    "multimodal_regressor_path": Path("models/rating_regressor_multimodal.joblib"),
    "threshold": 67.342307,
    "decision_mode": "threshold", # Since they didn't override --decision-mode
    "preference_model_path": Path("models/bumble_preference_classifier.joblib"),
    "face_weight": 0.50,
}

# What if they set --decision-mode preference?
current_config_pref_decision = {
    "setup_name": "Experimental3",
    "method": "multimodalx_original",
    "store_path": Path("embeddings/reference_store.npz"),
    "regressor_path": Path("models/rating_regressor.joblib"),
    "multimodal_regressor_path": Path("models/rating_regressor_multimodal.joblib"),
    "threshold": 67.342307,
    "decision_mode": "preference",
    "preference_model_path": Path("models/bumble_preference_classifier.joblib"),
    "face_weight": 0.50,
}

print("--- Default Decision Mode (threshold) ---")
scores_default = recent_values(CSV_PATH, value_field="score", limit=200, current_config=current_config_default_decision)
print(f"Matching rows count: {len(scores_default)}")
if scores_default:
    print(f"Last 20 scores: {[round(s, 2) for s in scores_default[-20:]]}")
    print(f"80th percentile of all matched: {quantile(scores_default, 0.80):.4f}")
    if len(scores_default) >= 20:
        print(f"80th percentile of last 20: {quantile(scores_default[-20:], 0.80):.4f}")

print("\n--- Preference Probability (from matches) ---")
# Let's extract the actual preference probabilities from the matched rows
pref_probs = recent_values(CSV_PATH, value_field="preference_probability", limit=200, current_config=current_config_default_decision)
print(f"Matching preference_probability count: {len(pref_probs)}")
if pref_probs:
    print(f"Last 20 preference_probabilities: {[round(p, 4) for p in pref_probs[-20:]]}")
    print(f"80th percentile of all matched: {quantile(pref_probs, 0.80):.4f}")
    if len(pref_probs) >= 20:
        print(f"80th percentile of last 20: {quantile(pref_probs[-20:], 0.80):.4f}")
