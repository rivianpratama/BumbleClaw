from __future__ import annotations

import unittest

import numpy as np

from face_similarity.scoring import score_embedding


class ScoreEmbeddingTests(unittest.TestCase):
    def test_weighted_top_k_rating_prefers_nearest_references(self) -> None:
        refs = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        ratings = np.asarray([90.0, 50.0, 5.0], dtype=np.float32)

        result = score_embedding(np.asarray([1.0, 0.0], dtype=np.float32), refs, ratings, k=2)

        self.assertGreater(result.rating, 80)
        self.assertEqual(result.nearest_indices[0], 0)
        self.assertAlmostEqual(result.max_similarity, 1.0)

    def test_uses_average_when_top_weights_are_zero(self) -> None:
        refs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ratings = np.asarray([20.0, 80.0], dtype=np.float32)

        result = score_embedding(np.asarray([-1.0, -1.0], dtype=np.float32), refs, ratings, k=2)

        self.assertAlmostEqual(result.rating, 50.0)

    def test_rejects_empty_reference_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            score_embedding(
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()

