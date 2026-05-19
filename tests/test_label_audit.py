from __future__ import annotations

import unittest

import numpy as np

from label_audit import conflict_row, find_conflicting_pairs, normalize_embeddings


class LabelAuditTests(unittest.TestCase):
    def test_finds_similar_embeddings_with_large_rating_gap(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        ratings = np.asarray([0.0, 100.0, 50.0], dtype=np.float32)
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        pairs = find_conflicting_pairs(
            embeddings,
            ratings,
            paths,
            min_gap=50,
            min_similarity=0.9,
            neighbors=2,
            limit=10,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].path_a, "a.jpg")
        self.assertEqual(pairs[0].path_b, "b.jpg")
        self.assertAlmostEqual(pairs[0].rating_gap, 100.0)

    def test_does_not_report_reversed_duplicates(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ratings = np.asarray([0.0, 100.0], dtype=np.float32)
        paths = ["a.jpg", "b.jpg"]

        pairs = find_conflicting_pairs(
            embeddings,
            ratings,
            paths,
            min_gap=50,
            min_similarity=0.9,
            neighbors=1,
            limit=10,
        )

        self.assertEqual(len(pairs), 1)

    def test_sorts_by_gap_then_similarity(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )
        ratings = np.asarray([0.0, 100.0, 75.0], dtype=np.float32)
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        pairs = find_conflicting_pairs(
            embeddings,
            ratings,
            paths,
            min_gap=50,
            min_similarity=0.8,
            neighbors=2,
            limit=10,
        )

        self.assertEqual(pairs[0].rating_gap, 100.0)
        self.assertEqual(pairs[0].path_b, "b.jpg")

    def test_normalize_handles_zero_vectors(self) -> None:
        normalized = normalize_embeddings(np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))

        self.assertTrue(np.allclose(normalized[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(normalized[1], [0.6, 0.8]))

    def test_formats_csv_row(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ratings = np.asarray([25.0, 100.0], dtype=np.float32)
        pair = find_conflicting_pairs(
            embeddings,
            ratings,
            ["a.jpg", "b.jpg"],
            min_gap=50,
            min_similarity=0.9,
            neighbors=1,
            limit=1,
        )[0]

        self.assertEqual(conflict_row(pair)["rating_gap"], "75")
        self.assertEqual(conflict_row(pair)["similarity"], "1.000000")


if __name__ == "__main__":
    unittest.main()
