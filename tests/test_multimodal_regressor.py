from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_similarity.clip_store import save_clip_store
from face_similarity.regressor import load_regressor, predict_multimodal_rating, save_regressor
from face_similarity.store import save_store
from train_multimodal_regressor import (
    ExpectedScoreClassifier,
    align_stores,
    best_leak_aware_result,
    evaluate_model,
    feature_matrix,
    selected_leak_aware_result,
    write_report,
)


class MultimodalRegressorTests(unittest.TestCase):
    def test_align_stores_rejects_missing_clip_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            face_path = Path(tmp) / "face.npz"
            clip_path = Path(tmp) / "clip.npz"
            save_store(
                face_path,
                embeddings=[np.asarray([1.0, 0.0], dtype=np.float32)],
                paths=["references/a.jpg"],
                ratings=[75],
                backend="insightface",
                model_name="buffalo_l",
                provider="cuda",
                det_size=640,
                det_thresh=0.25,
            )
            save_clip_store(
                clip_path,
                embeddings=[np.asarray([0.0, 1.0], dtype=np.float32)],
                paths=["references/b.jpg"],
                model_name="clip",
                device="cpu",
            )

            from face_similarity.clip_store import load_clip_store
            from face_similarity.store import load_store

            with self.assertRaisesRegex(ValueError, "missing 1 face-store"):
                align_stores(load_store(face_path), load_clip_store(clip_path))

    def test_feature_matrix_concatenates_face_and_clip_features(self) -> None:
        class Aligned:
            face_embeddings = np.asarray([[1.0, 2.0]], dtype=np.float32)
            clip_embeddings = np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32)

        combined = feature_matrix(Aligned, "face_clip")

        self.assertEqual(combined.shape, (1, 5))
        self.assertTrue(np.allclose(combined[0], [1, 2, 3, 4, 5]))

    def test_logistic_expected_score_returns_continuous_score(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]] * 4, dtype=np.float32)
        y = np.asarray([0, 25, 50, 75, 100] * 4, dtype=np.float32)

        model = ExpectedScoreClassifier(random_state=1).fit(x, y)
        pred = model.predict(np.asarray([[4.0]], dtype=np.float32))

        self.assertEqual(pred.shape, (1,))
        self.assertGreaterEqual(float(pred[0]), 0)
        self.assertLessEqual(float(pred[0]), 100)

    def test_best_leak_aware_result_ignores_random_results(self) -> None:
        low_random = evaluate_model("random", "face_only", "ridge", None, np.asarray([0.0]), np.asarray([0.0]))
        leak = evaluate_model("leak_aware", "face_clip", "ridge", None, np.asarray([0.0]), np.asarray([25.0]))

        self.assertEqual(best_leak_aware_result([low_random, leak]).validation_mode, "leak_aware")

    def test_selected_leak_aware_result_can_pin_model(self) -> None:
        mlp = evaluate_model("leak_aware", "face_clip", "mlp", None, np.asarray([0.0]), np.asarray([0.0]))
        ridge = evaluate_model("leak_aware", "face_clip", "ridge", None, np.asarray([0.0]), np.asarray([25.0]))

        selected = selected_leak_aware_result(
            [mlp, ridge],
            feature_mode="face_clip",
            model_name="ridge",
        )

        self.assertEqual(selected.model_name, "ridge")

    def test_report_contains_feature_modes(self) -> None:
        result = evaluate_model("leak_aware", "face_clip", "ridge", None, np.asarray([0.0]), np.asarray([25.0]))

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.csv"
            write_report(report, [result])
            text = report.read_text(encoding="utf-8")

        self.assertIn("validation_mode,feature_mode,model", text)
        self.assertIn("face_clip", text)

    def test_predict_multimodal_uses_feature_mode_metadata(self) -> None:
        from sklearn.dummy import DummyRegressor

        estimator = DummyRegressor(strategy="constant", constant=80.0)
        estimator.fit(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), np.asarray([80.0], dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            save_regressor(
                model_path,
                estimator=estimator,
                model_name="face_clip_ridge",
                metrics={"mae": 1.0},
                metadata={"feature_mode": "face_clip"},
            )
            regressor = load_regressor(model_path)

        score = predict_multimodal_rating(
            regressor,
            face_embedding=np.asarray([1.0], dtype=np.float32),
            clip_embedding=np.asarray([2.0, 3.0], dtype=np.float32),
        )

        self.assertEqual(score, 80.0)


if __name__ == "__main__":
    unittest.main()
