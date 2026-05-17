from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from face_similarity.embedding import active_provider_from_app, get_face_embedding, resolve_providers, select_largest_face


class EmbeddingBackendTests(unittest.TestCase):
    def test_auto_provider_prefers_cuda_when_available(self) -> None:
        with patch(
            "face_similarity.embedding.available_onnx_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            with patch("face_similarity.embedding.cuda_provider_works", return_value=True):
                providers, active = resolve_providers("auto")

        self.assertEqual(active, "cuda")
        self.assertEqual(providers[0], "CUDAExecutionProvider")

    def test_cuda_provider_fails_clearly_when_unavailable(self) -> None:
        with patch("face_similarity.embedding.available_onnx_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider is not available"):
                resolve_providers("cuda")

    def test_auto_provider_falls_back_when_cuda_is_listed_but_unusable(self) -> None:
        with patch(
            "face_similarity.embedding.available_onnx_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            with patch("face_similarity.embedding.cuda_provider_works", return_value=False):
                providers, active = resolve_providers("auto")

        self.assertEqual(active, "cpu")
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_selects_largest_face(self) -> None:
        small = SimpleNamespace(bbox=np.asarray([0, 0, 10, 10]), embedding=np.asarray([1, 0]))
        large = SimpleNamespace(bbox=np.asarray([0, 0, 20, 20]), embedding=np.asarray([0, 1]))

        selected = select_largest_face([small, large])

        self.assertIs(selected, large)

    def test_reads_actual_provider_from_model_sessions(self) -> None:
        cuda_session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        model = SimpleNamespace(session=cuda_session)
        app = SimpleNamespace(models={"recognition": model})

        self.assertEqual(active_provider_from_app(app), "cuda")

    def test_get_face_embedding_uses_largest_face_embedding(self) -> None:
        class FakeCv2:
            @staticmethod
            def imread(path: str) -> np.ndarray:
                return np.zeros((10, 10, 3), dtype=np.uint8)

        small = SimpleNamespace(bbox=np.asarray([0, 0, 10, 10]), embedding=np.asarray([1, 0]))
        large = SimpleNamespace(bbox=np.asarray([0, 0, 20, 20]), embedding=np.asarray([0, 1]))
        fake_app = SimpleNamespace(get=lambda image: [small, large])

        with patch("face_similarity.embedding._import_cv2", return_value=FakeCv2):
            with patch("face_similarity.embedding.get_face_app", return_value=(fake_app, "cpu")):
                embedding = get_face_embedding("image.jpg", provider="cpu", det_thresh=0.25)

        np.testing.assert_array_equal(embedding, np.asarray([0, 1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
