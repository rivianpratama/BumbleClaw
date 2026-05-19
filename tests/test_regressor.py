from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor

from face_similarity.regressor import load_regressor, predict_rating, save_regressor
from train_regressor import train_and_evaluate, write_report


class RegressorTests(unittest.TestCase):
    def test_save_load_and_predict_clips_rating(self) -> None:
        estimator = DummyRegressor(strategy="constant", constant=125.0)
        estimator.fit(np.asarray([[0.0], [1.0]], dtype=np.float32), np.asarray([0.0, 100.0], dtype=np.float32))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_regressor(path, estimator=estimator, model_name="dummy", metrics={"mae": 1.5})
            loaded = load_regressor(path)

        self.assertEqual(loaded.model_name, "dummy")
        self.assertAlmostEqual(loaded.metrics["mae"], 1.5)
        self.assertEqual(predict_rating(loaded, np.asarray([1.0], dtype=np.float32)), 100.0)

    def test_train_and_evaluate_returns_model_results(self) -> None:
        embeddings = np.asarray([[float(index), float(index % 3)] for index in range(30)], dtype=np.float32)
        ratings = np.asarray([0, 25, 50, 75, 100] * 6, dtype=np.float32)

        results = train_and_evaluate(
            embeddings,
            ratings,
            test_size=0.2,
            random_state=1,
            include_xgboost=False,
        )

        self.assertEqual({result.name for result in results}, {"ridge", "random_forest", "hist_gradient_boosting"})
        self.assertTrue(all(result.test_count == 6 for result in results))

    def test_write_report_preserves_expected_columns(self) -> None:
        embeddings = np.asarray([[float(index), float(index % 3)] for index in range(30)], dtype=np.float32)
        ratings = np.asarray([0, 25, 50, 75, 100] * 6, dtype=np.float32)
        results = train_and_evaluate(
            embeddings,
            ratings,
            test_size=0.2,
            random_state=1,
            include_xgboost=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            write_report(path, results)
            header = path.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(header, "model,mae,rmse,bias,pred_mean,test_count")


if __name__ == "__main__":
    unittest.main()
