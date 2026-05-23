from __future__ import annotations

import unittest

from app import biased_multimodal_score, comparison_text


class AppTests(unittest.TestCase):

    def test_biased_multimodal_score_blends_face_and_multimodal(self) -> None:
        self.assertEqual(
            biased_multimodal_score(40.0, 80.0, face_weight=0.5),
            60.0,
        )
        self.assertIsNone(biased_multimodal_score(None, 80.0, face_weight=0.5))
        self.assertIsNone(biased_multimodal_score(40.0, None, face_weight=0.5))

    def test_comparison_text_uses_ridge_as_reference(self) -> None:
        self.assertEqual(comparison_text(70.0, 55.0), "+15.0 vs Ridge")
        self.assertEqual(comparison_text(40.0, 55.0), "-15.0 vs Ridge")
        self.assertEqual(comparison_text(None, 55.0), "no comparison")


if __name__ == "__main__":
    unittest.main()
