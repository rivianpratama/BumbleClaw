from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from face_similarity.labeling import (
    discover_images,
    labeled_paths,
    next_unlabeled_path,
    rating_to_score,
    upsert_label,
)


class LabelingTests(unittest.TestCase):
    def test_rating_to_score_maps_five_point_scale(self) -> None:
        self.assertEqual(rating_to_score(1), 0)
        self.assertEqual(rating_to_score(2), 25)
        self.assertEqual(rating_to_score(3), 50)
        self.assertEqual(rating_to_score(4), 75)
        self.assertEqual(rating_to_score(5), 100)

    def test_rating_to_score_rejects_invalid_rating(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            rating_to_score(6)

    def test_discovers_supported_images_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Female Faces"
            nested = source / "nested"
            nested.mkdir(parents=True)
            (source / "2.jpg").write_bytes(b"x")
            (source / "10.png").write_bytes(b"x")
            (nested / "1.jpeg").write_bytes(b"x")
            (source / "ignore.txt").write_text("x", encoding="utf-8")

            paths = discover_images(source)

            self.assertEqual(len(paths), 3)
            self.assertTrue(any(path.endswith("nested/1.jpeg") for path in paths))
            self.assertTrue(any(path.endswith("2.jpg") for path in paths))
            self.assertTrue(any(path.endswith("10.png") for path in paths))
            self.assertFalse(any(path.endswith("ignore.txt") for path in paths))

    def test_upsert_label_writes_expected_columns_and_updates_existing_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dataset_labels.csv"

            upsert_label(csv_path, "references/Female Faces/a.jpg", 5)
            upsert_label(csv_path, "references/Female Faces/a.jpg", 2)

            content = csv_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(content[0], "path,rating_1_5,rating")
            self.assertEqual(content[1], "references/Female Faces/a.jpg,2,25")
            self.assertEqual(labeled_paths(csv_path), {"references/Female Faces/a.jpg"})

    def test_next_unlabeled_path_skips_labeled_and_session_skipped(self) -> None:
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        result = next_unlabeled_path(paths, labeled={"a.jpg"}, skipped={"b.jpg"})

        self.assertEqual(result, "c.jpg")


if __name__ == "__main__":
    unittest.main()
