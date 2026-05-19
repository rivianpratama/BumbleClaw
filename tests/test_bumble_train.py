from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

import bumble_train


class BumbleTrainTests(unittest.TestCase):
    def test_prepare_selects_train_and_validation_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "log"
            output = root / "train"
            source.mkdir()
            rows = []
            for index in range(12):
                name = f"profile_{index:02d}.webp"
                image = Image.new("RGB", (100, 200), color=(index * 12, index * 7, index * 3))
                for x in range(100):
                    for y in range(100):
                        if ((x // 12) + (y // 12) + index) % 3 == 0:
                            image.putpixel((x, y), (255 - index * 10, index * 13, 80))
                image.save(source / name)
                score = 20 + index * 6
                rows.append(
                    {
                        "timestamp": str(index),
                        "screenshot": name,
                        "method": "face_biased",
                        "action": "right" if score >= 62.34 else "left",
                        "score": str(score),
                        "face_biased": str(score),
                        "multimodal": str(score + 5),
                        "ridge": str(score - 5),
                        "knn": str(score),
                    }
                )
            write_csv(source / "scores.csv", rows, bumble_train.SELECTION_FIELDS[2:14])

            args = Namespace(
                source=str(source),
                output=str(output),
                target_count=8,
                val_count=4,
                threshold=62.34,
                seed=1,
                crop_left=0.0,
                crop_top=0.0,
                crop_right=1.0,
                crop_bottom=0.5,
                dedupe_hamming=-1,
                no_mask_share_icon=True,
            )

            bumble_train.prepare_workspace(args)

            manifest = list(csv.DictReader((output / "manifests" / "selection.csv").open(encoding="utf-8")))
            self.assertEqual(len(manifest), 8)
            self.assertEqual(sum(row["split"] == "validation" for row in manifest), 4)
            self.assertEqual(sum(row["split"] == "train" for row in manifest), 4)
            self.assertTrue((output / "raw" / "scores.csv").exists())

    def test_combine_labels_excludes_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            bumble_labels = root / "bumble.csv"
            manifest = root / "selection.csv"
            output = root / "combined.csv"

            write_csv(base, [{"path": "base.jpg", "rating_1_5": "3", "rating": "50"}], bumble_train.COMBINED_LABEL_FIELDS)
            (root / "base.jpg").write_bytes(b"x")
            write_csv(
                bumble_labels,
                [
                    {"path": (root / "selected" / "train" / "a.jpg").as_posix(), "rating_1_5": "4", "rating": "75"},
                    {"path": (root / "selected" / "validation" / "b.jpg").as_posix(), "rating_1_5": "5", "rating": "100"},
                ],
                bumble_train.COMBINED_LABEL_FIELDS,
            )
            write_csv(
                manifest,
                [
                    manifest_row("train", root / "selected" / "train" / "a.jpg"),
                    manifest_row("validation", root / "selected" / "validation" / "b.jpg"),
                ],
                bumble_train.SELECTION_FIELDS,
            )

            bumble_train.combine_labels(
                Namespace(base=str(base), bumble_labels=str(bumble_labels), manifest=str(manifest), output=str(output))
            )

            rows = list(csv.DictReader(output.open(encoding="utf-8")))
            self.assertEqual(
                [row["path"] for row in rows],
                [(root / "base.jpg").resolve().as_posix(), (root / "selected" / "train" / "a.jpg").resolve().as_posix()],
            )

    def test_evaluate_writes_error_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.csv"
            manifest = root / "selection.csv"
            report = root / "report.csv"
            selected_path = root / "selected" / "validation" / "a.jpg"
            write_csv(labels, [{"path": selected_path.as_posix(), "rating_1_5": "4", "rating": "75"}], bumble_train.COMBINED_LABEL_FIELDS)
            row = manifest_row("validation", selected_path)
            row.update({"score": "60", "face_biased": "60", "multimodal": "70", "ridge": "50", "knn": "55"})
            write_csv(manifest, [row], bumble_train.SELECTION_FIELDS)

            bumble_train.evaluate_labels(
                Namespace(labels=str(labels), manifest=str(manifest), split="validation", threshold=62.34, output=str(report))
            )

            rows = list(csv.DictReader(report.open(encoding="utf-8")))
            score_row = next(row for row in rows if row["metric"] == "score")
            self.assertEqual(score_row["count"], "1")
            self.assertEqual(score_row["mae"], "15.000000")
            self.assertEqual(score_row["swipe_error_rate"], "1.000000")

    def test_evaluate_prediction_csv_from_score_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.csv"
            predictions = root / "predictions.csv"
            report = root / "report.csv"
            image_path = root / "selected" / "validation" / "a.jpg"
            write_csv(labels, [{"path": image_path.as_posix(), "rating_1_5": "5", "rating": "100"}], bumble_train.COMBINED_LABEL_FIELDS)
            write_csv(
                predictions,
                [{"file": image_path.as_posix(), "rating": "80", "max_similarity": "", "mean_similarity": "", "nearest": "", "error": ""}],
                ["file", "rating", "max_similarity", "mean_similarity", "nearest", "error"],
            )

            bumble_train.evaluate_labels(
                Namespace(
                    labels=str(labels),
                    manifest="unused.csv",
                    predictions=str(predictions),
                    split="validation",
                    threshold=62.34,
                    output=str(report),
                )
            )

            rows = list(csv.DictReader(report.open(encoding="utf-8")))
            self.assertEqual(rows[0]["metric"], "prediction")
            self.assertEqual(rows[0]["mae"], "20.000000")

    def test_recrop_selected_preserves_selected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jpg"
            crop = root / "cropped" / "raw.jpg"
            selected = root / "selected" / "train" / "raw.jpg"
            image = Image.new("RGB", (100, 200), color=(20, 40, 60))
            image.save(raw)
            write_csv(
                root / "selection.csv",
                [manifest_row("train", selected) | {"raw_path": raw.as_posix(), "crop_path": crop.as_posix()}],
                bumble_train.SELECTION_FIELDS,
            )

            bumble_train.recrop_selected(
                Namespace(
                    manifest=str(root / "selection.csv"),
                    crop_left=0.0,
                    crop_top=0.0,
                    crop_right=1.0,
                    crop_bottom=0.7,
                    no_mask_share_icon=True,
                )
            )

            self.assertTrue(selected.exists())
            with Image.open(selected) as selected_image:
                self.assertEqual(selected_image.size, (100, 140))


def manifest_row(split: str, selected_path: Path) -> dict[str, str]:
    return {
        "split": split,
        "selection_reason": "test",
        "timestamp": "1",
        "screenshot": selected_path.name,
        "raw_path": selected_path.as_posix(),
        "crop_path": selected_path.as_posix(),
        "selected_path": selected_path.as_posix(),
        "method": "face_biased",
        "action": "right",
        "score": "75",
        "face_biased": "75",
        "multimodal": "75",
        "ridge": "75",
        "knn": "75",
        "component_spread": "0",
        "score_band": "high",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
