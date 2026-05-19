from __future__ import annotations

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import label_app


class LabelAppTests(unittest.TestCase):
    def tearDown(self) -> None:
        label_app.configure(source_dirs=None, output_csv="dataset_labels.csv", port=7861)

    def test_initialize_randomizes_only_unlabeled_queue(self) -> None:
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        with patch("label_app.discover_all_images", return_value=paths):
            with patch("label_app.labeled_paths", return_value={"b.jpg"}):
                with patch("label_app.random.shuffle", side_effect=lambda items: items.reverse()):
                    state, image, _, _ = label_app.initialize()

        self.assertEqual(state["paths"], paths)
        self.assertEqual(state["queue"], ["c.jpg", "a.jpg"])
        self.assertEqual(image, "c.jpg")

    def test_configure_overrides_source_output_and_port(self) -> None:
        label_app.configure(source_dirs=["D:/BumbleTrain/selected"], output_csv="D:/BumbleTrain/labels/bumble_labels.csv", port=7863)

        self.assertEqual(label_app.SOURCE_DIRS[0].as_posix(), "D:/BumbleTrain/selected")
        self.assertEqual(label_app.OUTPUT_CSV.as_posix(), "D:/BumbleTrain/labels/bumble_labels.csv")
        self.assertEqual(label_app.SERVER_PORT, 7863)

    def test_allowed_paths_includes_existing_source_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "selected"
            source.mkdir()
            missing = Path(tmp) / "missing"
            label_app.configure(source_dirs=[str(source), str(missing)], output_csv="labels.csv", port=7863)

            self.assertEqual(label_app.allowed_paths(), [str(source.resolve())])


if __name__ == "__main__":
    unittest.main()
