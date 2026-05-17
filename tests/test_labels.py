from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from face_similarity.labels import load_labels


class LoadLabelsTests(unittest.TestCase):
    def test_loads_relative_paths_and_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "references" / "a.jpg"
            image.parent.mkdir()
            image.write_bytes(b"not a real image")
            labels = root / "labels.csv"
            labels.write_text("path,rating\nreferences/a.jpg,87.5\n", encoding="utf-8")

            entries = load_labels(labels)

            self.assertEqual(entries[0].path, image)
            self.assertEqual(entries[0].rating, 87.5)

    def test_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels.csv"
            labels.write_text("file,score\nx.jpg,50\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required"):
                load_labels(labels)

    def test_rejects_out_of_range_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            image.write_bytes(b"x")
            labels = root / "labels.csv"
            labels.write_text("path,rating\na.jpg,101\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                load_labels(labels)


if __name__ == "__main__":
    unittest.main()

