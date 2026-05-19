from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from face_similarity.clip_store import ClipStore, load_clip_store, normalize_path_key
from face_similarity.regressor import save_regressor
from face_similarity.store import ReferenceStore, load_store


BASELINE_MAE = 16.5903
BASELINE_RMSE = 20.7472
RESULT_FIELDS = [
    "validation_mode",
    "feature_mode",
    "model",
    "mae",
    "rmse",
    "bias",
    "pred_mean",
    "test_count",
]
FEATURE_MODES = ("face_only", "clip_only", "face_clip")
MODEL_NAMES = ("ridge", "logistic_expected", "mlp")


@dataclass(frozen=True)
class AlignedFeatures:
    face_embeddings: np.ndarray
    clip_embeddings: np.ndarray
    ratings: np.ndarray
    paths: list[str]


@dataclass(frozen=True)
class EvalResult:
    validation_mode: str
    feature_mode: str
    model_name: str
    estimator: Any
    mae: float
    rmse: float
    bias: float
    pred_mean: float
    test_count: int


class ExpectedScoreClassifier:
    def __init__(self, random_state: int = 42) -> None:
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExpectedScoreClassifier":
        self.pipeline.fit(x, y.astype(np.int32))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        classes = self.pipeline.named_steps["logisticregression"].classes_.astype(np.float32)
        probabilities = self.pipeline.predict_proba(x)
        return probabilities @ classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare face, CLIP, and multimodal swipe regressors.")
    parser.add_argument("--face-store", default="embeddings/reference_store.npz", help="InsightFace reference store")
    parser.add_argument("--clip-store", default="embeddings/clip_store.npz", help="CLIP store aligned by path")
    parser.add_argument("--output", default="models/rating_regressor_multimodal.joblib", help="Saved best model path")
    parser.add_argument("--report", default="results/multimodal_regressor_eval.csv", help="Evaluation CSV report")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for random split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--group-similarity-threshold", type=float, default=0.90, help="Face similarity threshold for leak-aware groups")
    parser.add_argument("--save-feature-mode", choices=FEATURE_MODES, help="Save this leak-aware feature mode instead of the best MAE result")
    parser.add_argument("--save-model", choices=MODEL_NAMES, help="Save this leak-aware model instead of the best MAE result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    face_store = load_store(args.face_store)
    clip_store = load_clip_store(args.clip_store)
    aligned = align_stores(face_store, clip_store)
    groups = face_similarity_groups(aligned.face_embeddings, threshold=args.group_similarity_threshold)
    results = evaluate_all(
        aligned,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    write_report(args.report, results)

    best = selected_leak_aware_result(
        results,
        feature_mode=args.save_feature_mode,
        model_name=args.save_model,
    )
    final_features = feature_matrix(aligned, best.feature_mode)
    final_estimator = make_model(best.model_name, random_state=args.random_state)
    final_estimator.fit(final_features, aligned.ratings)
    save_regressor(
        args.output,
        estimator=final_estimator,
        model_name=f"{best.feature_mode}_{best.model_name}",
        metrics={
            "mae": best.mae,
            "rmse": best.rmse,
            "bias": best.bias,
            "pred_mean": best.pred_mean,
            "test_count": float(best.test_count),
        },
        metadata={
            "feature_mode": best.feature_mode,
            "face_model": face_store.model_name,
            "clip_model": clip_store.model_name,
            "validation_mode": best.validation_mode,
            "beats_baseline": str(best.mae < BASELINE_MAE and best.rmse < BASELINE_RMSE),
        },
    )

    print(f"Aligned examples: {len(aligned.ratings)}")
    print(f"Best leak-aware model: {best.feature_mode}/{best.model_name}")
    print(f"MAE={best.mae:.4f} RMSE={best.rmse:.4f} bias={best.bias:.4f}")
    print(f"Baseline MAE={BASELINE_MAE:.4f} RMSE={BASELINE_RMSE:.4f}")
    print(f"Beats baseline: {best.mae < BASELINE_MAE and best.rmse < BASELINE_RMSE}")
    print(f"Saved model to {args.output}")
    print(f"Saved report to {args.report}")
    return 0


def align_stores(face_store: ReferenceStore, clip_store: ClipStore) -> AlignedFeatures:
    clip_by_path = {normalize_path_key(path): index for index, path in enumerate(clip_store.paths)}
    face_indices = []
    clip_indices = []
    missing = []
    for index, path in enumerate(face_store.paths):
        key = normalize_path_key(path)
        if key not in clip_by_path:
            missing.append(path)
            continue
        face_indices.append(index)
        clip_indices.append(clip_by_path[key])

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"CLIP store is missing {len(missing)} face-store path(s): {preview}")
    if not face_indices:
        raise ValueError("No aligned rows found between face and CLIP stores")

    return AlignedFeatures(
        face_embeddings=face_store.embeddings[face_indices],
        clip_embeddings=clip_store.embeddings[clip_indices],
        ratings=face_store.ratings[face_indices],
        paths=[face_store.paths[index] for index in face_indices],
    )


def evaluate_all(
    aligned: AlignedFeatures,
    *,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> list[EvalResult]:
    labels = aligned.ratings.astype(np.int32)
    random_split = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    leak_split = leak_aware_split(labels, groups, random_state=random_state)

    results = []
    for validation_mode, split in (("random", random_split), ("leak_aware", leak_split)):
        train_indices, test_indices = split
        for feature_mode in FEATURE_MODES:
            features = feature_matrix(aligned, feature_mode)
            for model_name in MODEL_NAMES:
                estimator = make_model(model_name, random_state=random_state)
                estimator.fit(features[train_indices], aligned.ratings[train_indices])
                predictions = np.clip(estimator.predict(features[test_indices]), 0, 100)
                results.append(
                    evaluate_model(
                        validation_mode,
                        feature_mode,
                        model_name,
                        estimator,
                        aligned.ratings[test_indices],
                        predictions,
                    )
                )
    return results


def feature_matrix(aligned: AlignedFeatures, feature_mode: str) -> np.ndarray:
    if feature_mode == "face_only":
        return aligned.face_embeddings
    if feature_mode == "clip_only":
        return aligned.clip_embeddings
    if feature_mode == "face_clip":
        return np.concatenate([aligned.face_embeddings, aligned.clip_embeddings], axis=1)
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def make_model(model_name: str, *, random_state: int) -> Any:
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))
    if model_name == "logistic_expected":
        return ExpectedScoreClassifier(random_state=random_state)
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 32),
                alpha=0.01,
                early_stopping=True,
                max_iter=500,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    validation_mode: str,
    feature_mode: str,
    model_name: str,
    estimator: Any,
    true: np.ndarray,
    pred: np.ndarray,
) -> EvalResult:
    errors = pred - true
    return EvalResult(
        validation_mode=validation_mode,
        feature_mode=feature_mode,
        model_name=model_name,
        estimator=estimator,
        mae=float(mean_absolute_error(true, pred)),
        rmse=float(mean_squared_error(true, pred) ** 0.5),
        bias=float(np.mean(errors)),
        pred_mean=float(np.mean(pred)),
        test_count=len(true),
    )


def leak_aware_split(labels: np.ndarray, groups: np.ndarray, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_indices, test_indices = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return train_indices, test_indices


def face_similarity_groups(embeddings: np.ndarray, *, threshold: float) -> np.ndarray:
    normalized = normalize_embeddings(embeddings)
    similarities = normalized @ normalized.T
    parent = list(range(len(embeddings)))
    for index in range(len(embeddings) - 1):
        matches = np.flatnonzero(similarities[index, index + 1 :] >= threshold)
        for offset in matches:
            union(parent, index, index + 1 + int(offset))
    roots: dict[int, int] = {}
    groups = np.empty(len(embeddings), dtype=np.int32)
    for index in range(len(embeddings)):
        root = find(parent, index)
        if root not in roots:
            roots[root] = len(roots)
        groups[index] = roots[root]
    return groups


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def union(parent: list[int], left: int, right: int) -> None:
    left_root = find(parent, left)
    right_root = find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def best_leak_aware_result(results: list[EvalResult]) -> EvalResult:
    leak_results = [result for result in results if result.validation_mode == "leak_aware"]
    if not leak_results:
        raise ValueError("No leak-aware results were produced")
    return min(leak_results, key=lambda result: (result.mae, result.rmse))


def selected_leak_aware_result(
    results: list[EvalResult],
    *,
    feature_mode: str | None,
    model_name: str | None,
) -> EvalResult:
    if feature_mode is None and model_name is None:
        return best_leak_aware_result(results)

    matches = [
        result
        for result in results
        if result.validation_mode == "leak_aware"
        and (feature_mode is None or result.feature_mode == feature_mode)
        and (model_name is None or result.model_name == model_name)
    ]
    if not matches:
        raise ValueError(f"No leak-aware result matched feature_mode={feature_mode!r}, model={model_name!r}")
    return min(matches, key=lambda result: (result.mae, result.rmse))


def write_report(path: str | Path, results: list[EvalResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.validation_mode, item.mae, item.rmse)):
            writer.writerow(
                {
                    "validation_mode": result.validation_mode,
                    "feature_mode": result.feature_mode,
                    "model": result.model_name,
                    "mae": f"{result.mae:.6f}",
                    "rmse": f"{result.rmse:.6f}",
                    "bias": f"{result.bias:.6f}",
                    "pred_mean": f"{result.pred_mean:.6f}",
                    "test_count": str(result.test_count),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
