from __future__ import annotations

import unittest
from unittest.mock import patch

import label_app


class LabelAppTests(unittest.TestCase):
    def test_initialize_randomizes_only_unlabeled_queue(self) -> None:
        paths = ["a.jpg", "b.jpg", "c.jpg"]

        with patch("label_app.discover_all_images", return_value=paths):
            with patch("label_app.labeled_paths", return_value={"b.jpg"}):
                with patch("label_app.random.shuffle", side_effect=lambda items: items.reverse()):
                    state, image, _, _ = label_app.initialize()

        self.assertEqual(state["paths"], paths)
        self.assertEqual(state["queue"], ["c.jpg", "a.jpg"])
        self.assertEqual(image, "c.jpg")


if __name__ == "__main__":
    unittest.main()
