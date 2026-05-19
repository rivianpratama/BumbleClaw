from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from face_similarity.automation_log import save_profile_log
from face_similarity.prediction import RatingPrediction


class AutomationLogTests(unittest.TestCase):
    def test_save_profile_log_writes_compressed_webp_and_csv(self) -> None:
        prediction = RatingPrediction(
            rating=60.0,
            method="face_biased",
            face_rating=50.0,
            multimodal_rating=70.0,
            knn_rating=55.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (1200, 800), color=(255, 0, 0)).save(source)

            output = save_profile_log(
                source,
                prediction=prediction,
                action="right",
                log_dir=root / "log",
                quality=30,
                max_width=600,
                image_format="webp",
            )

            self.assertEqual(output.suffix, ".webp")
            self.assertLess(output.stat().st_size, source.stat().st_size)
            with Image.open(output) as image:
                self.assertEqual(image.width, 600)

            with (root / "log" / "scores.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["screenshot"], output.name)
            self.assertEqual(rows[0]["score"], "60.0000")
            self.assertEqual(rows[0]["face_biased"], "60.0000")
            self.assertEqual(rows[0]["multimodal"], "70.0000")
            self.assertEqual(rows[0]["ridge"], "50.0000")
            self.assertEqual(rows[0]["knn"], "55.0000")

    def test_save_profile_log_can_write_avif(self) -> None:
        prediction = RatingPrediction(
            rating=60.0,
            method="face_biased",
            face_rating=50.0,
            multimodal_rating=70.0,
            knn_rating=55.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (400, 400), color=(0, 255, 0)).save(source)

            output = save_profile_log(source, prediction=prediction, action="right", log_dir=root / "log", image_format="avif")

            self.assertEqual(output.suffix, ".avif")
            with Image.open(output) as image:
                self.assertEqual(image.format, "AVIF")


if __name__ == "__main__":
    unittest.main()
