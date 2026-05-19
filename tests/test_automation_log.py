from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from face_similarity.automation_log import LOG_FIELDS, save_profile_log
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
                config={
                    "store_path": "embeddings/reference_store_bumble_combined_round2.npz",
                    "regressor_path": "models/rating_regressor_bumble_combined_round2.joblib",
                    "multimodal_regressor_path": "models/rating_regressor_multimodal_bumble_combined_round2.joblib",
                    "threshold": 62.34,
                    "face_weight": 0.22,
                    "k": 11,
                    "provider": "cuda",
                },
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
            self.assertEqual(rows[0]["face_biased"], "65.6000")
            self.assertEqual(rows[0]["multimodal"], "70.0000")
            self.assertEqual(rows[0]["ridge"], "50.0000")
            self.assertEqual(rows[0]["knn"], "55.0000")
            self.assertEqual(rows[0]["regressor_path"], "models/rating_regressor_bumble_combined_round2.joblib")
            self.assertEqual(rows[0]["multimodal_regressor_path"], "models/rating_regressor_multimodal_bumble_combined_round2.joblib")
            self.assertEqual(rows[0]["threshold"], "62.34")
            self.assertEqual(rows[0]["face_weight"], "0.22")
            self.assertEqual(rows[0]["k"], "11")
            self.assertEqual(rows[0]["provider"], "cuda")

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

    def test_save_profile_log_upgrades_existing_csv_header(self) -> None:
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
            log_dir = root / "log"
            log_dir.mkdir()
            Image.new("RGB", (400, 400), color=(0, 0, 255)).save(source)
            (log_dir / "scores.csv").write_text(
                "timestamp,screenshot,method,action,score,face_biased,multimodal,ridge,knn\n"
                "old,old.webp,face_biased,right,60.0000,60.0000,70.0000,50.0000,55.0000\n",
                encoding="utf-8",
            )

            save_profile_log(
                source,
                prediction=prediction,
                action="right",
                log_dir=log_dir,
                config={"threshold": 62.34},
            )

            with (log_dir / "scores.csv").open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, LOG_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["threshold"], "")
            self.assertEqual(rows[1]["threshold"], "62.34")


if __name__ == "__main__":
    unittest.main()
