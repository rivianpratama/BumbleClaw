from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from normalize_labels import (
    ComponentStats,
    normalized_ratings_by_path,
    normalized_row,
    path_key,
    write_normalized_labels,
)


class NormalizeLabelsTests(unittest.TestCase):
    def test_groups_high_similarity_faces_and_uses_mean_rating(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        ratings = np.asarray([25.0, 75.0, 100.0], dtype=np.float32)
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        normalized, stats = normalized_ratings_by_path(
            embeddings,
            ratings,
            paths,
            similarity_threshold=0.90,
            min_component_size=2,
        )

        self.assertAlmostEqual(normalized["a.jpg"], 50.0)
        self.assertAlmostEqual(normalized["b.jpg"], 50.0)
        self.assertNotIn("c.jpg", normalized)
        self.assertEqual(stats["a.jpg"].size, 2)

    def test_path_key_matches_slash_styles(self) -> None:
        self.assertEqual(path_key(r"references\Female Faces\0.jpg"), "references/female faces/0.jpg")

    def test_write_normalized_labels_reports_changed_rows(self) -> None:
        rows = [
            {"path": "a.jpg", "rating": "25"},
            {"path": "b.jpg", "rating": "75"},
        ]
        stats = ComponentStats(component_id=1, size=2, mean=50.0, minimum=25.0, maximum=75.0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.csv"
            changed = write_normalized_labels(
                output,
                rows,
                {"a.jpg": 50.0, "b.jpg": 50.0},
                {"a.jpg": stats, "b.jpg": stats},
            )
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(changed), 2)
        self.assertEqual(lines[0], "path,rating,original_rating,component_id,component_size,component_mean,component_min,component_max")

    def test_normalized_row_leaves_singleton_metadata_blank(self) -> None:
        row = normalized_row("a.jpg", 25.0, 25.0, None)

        self.assertEqual(row["component_id"], "")
        self.assertEqual(row["rating"], "25.000000")


if __name__ == "__main__":
    unittest.main()
