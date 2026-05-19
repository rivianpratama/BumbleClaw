from __future__ import annotations

import unittest

from app import biased_multimodal_score, comparison_text, empty_scores, score_cards


class AppTests(unittest.TestCase):
    def test_empty_scores_contains_four_score_slots(self) -> None:
        html = empty_scores()

        self.assertIn("Face-Biased", html)
        self.assertIn("Multimodal", html)
        self.assertIn("Ridge", html)
        self.assertIn("KNN", html)

    def test_score_cards_falls_back_when_multimodal_missing(self) -> None:
        html = score_cards(
            knn_rating=60.0,
            regressor_rating=55.0,
            regressor_name="ridge",
            multimodal_rating=None,
            multimodal_name="Not trained",
            max_similarity=0.7,
        )

        self.assertIn("Not trained", html)
        self.assertIn("55.0", html)
        self.assertIn("60.0", html)
        self.assertIn("+5.0 vs Ridge", html)

    def test_biased_multimodal_score_blends_face_and_multimodal(self) -> None:
        self.assertEqual(
            biased_multimodal_score(40.0, 80.0, face_weight=0.5),
            60.0,
        )
        self.assertIsNone(biased_multimodal_score(None, 80.0))
        self.assertIsNone(biased_multimodal_score(40.0, None))

    def test_comparison_text_uses_ridge_as_reference(self) -> None:
        self.assertEqual(comparison_text(70.0, 55.0), "+15.0 vs Ridge")
        self.assertEqual(comparison_text(40.0, 55.0), "-15.0 vs Ridge")
        self.assertEqual(comparison_text(None, 55.0), "no comparison")


if __name__ == "__main__":
    unittest.main()
