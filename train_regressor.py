from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from face_similarity.regressor import save_regressor
from face_similarity.store import load_store


RESULT_FIELDS = ["model", "mae", "rmse", "bias", "pred_mean", "test_count"]


@dataclass(frozen=True)
class ModelResult:
    name: str
    estimator: Any
    mae: float
    rmse: float
    bias: float
    pred_mean: float
    test_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised 0-100 rating regressor from face embeddings.")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Reference store path")
    parser.add_argument("--output", default="models/rating_regressor.joblib", help="Saved regressor path")
    parser.add_argument("--report", default="results/regressor_eval.csv", help="Evaluation CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--include-xgboost", action="store_true", help="Try XGBoost if the package is installed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = load_store(args.store)
    results = train_and_evaluate(
        store.embeddings,
        store.ratings,
        test_size=args.test_size,
        random_state=args.random_state,
        include_xgboost=args.include_xgboost,
    )
    best = min(results, key=lambda result: (result.mae, result.rmse))

    final_estimator = make_model(best.name, random_state=args.random_state)
    final_estimator.fit(store.embeddings, store.ratings)
    save_regressor(
        args.output,
        estimator=final_estimator,
        model_name=best.name,
        metrics={
            "mae": best.mae,
            "rmse": best.rmse,
            "bias": best.bias,
            "pred_mean": best.pred_mean,
            "test_count": float(best.test_count),
        },
    )
    write_report(args.report, results)

    print(f"Trained on {len(store.embeddings)} embedding(s)")
    print(f"Best model: {best.name}")
    print(f"MAE={best.mae:.4f} RMSE={best.rmse:.4f} bias={best.bias:.4f}")
    print(f"Saved model to {args.output}")
    print(f"Saved report to {args.report}")
    return 0


def train_and_evaluate(
    embeddings: np.ndarray,
    ratings: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    include_xgboost: bool,
) -> list[ModelResult]:
    labels = ratings.astype(np.int32)
    train_x, test_x, train_y, test_y = train_test_split(
        embeddings,
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    results = []
    for name in model_names(include_xgboost):
        estimator = make_model(name, random_state=random_state)
        estimator.fit(train_x, train_y)
        pred = np.clip(estimator.predict(test_x), 0, 100)
        results.append(evaluate_model(name, estimator, test_y, pred))
    return results


def model_names(include_xgboost: bool) -> list[str]:
    names = ["ridge", "random_forest", "hist_gradient_boosting"]
    if include_xgboost and has_xgboost():
        names.append("xgboost")
    return names


def make_model(name: str, *, random_state: int) -> Any:
    if name == "ridge":
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if name == "xgboost":
        xgboost = _import_xgboost()
        return xgboost.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(name: str, estimator: Any, true: np.ndarray, pred: np.ndarray) -> ModelResult:
    errors = pred - true
    return ModelResult(
        name=name,
        estimator=estimator,
        mae=float(mean_absolute_error(true, pred)),
        rmse=float(mean_squared_error(true, pred) ** 0.5),
        bias=float(np.mean(errors)),
        pred_mean=float(np.mean(pred)),
        test_count=len(true),
    )


def write_report(path: str | Path, results: list[ModelResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.mae, item.rmse)):
            writer.writerow(
                {
                    "model": result.name,
                    "mae": f"{result.mae:.6f}",
                    "rmse": f"{result.rmse:.6f}",
                    "bias": f"{result.bias:.6f}",
                    "pred_mean": f"{result.pred_mean:.6f}",
                    "test_count": str(result.test_count),
                }
            )


def has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def _import_xgboost() -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("xgboost is optional. Install it before using --include-xgboost.") from exc
    return xgboost


if __name__ == "__main__":
    raise SystemExit(main())
