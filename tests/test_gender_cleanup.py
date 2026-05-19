from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gender_cleanup import discover_images, gender_label, move_to_quarantine, quarantine_action, unused_path


class GenderCleanupTests(unittest.TestCase):
    def test_gender_label_matches_insightface_mapping(self) -> None:
        self.assertEqual(gender_label(0), "female")
        self.assertEqual(gender_label(1), "male")
        self.assertEqual(gender_label(None), "unknown")

    def test_quarantine_action_separates_male_and_unknown(self) -> None:
        self.assertEqual(quarantine_action({"gender": "male", "action": "unknown"}, move_unknown=False, dry_run=False), "moved_male")
        self.assertEqual(quarantine_action({"gender": "unknown", "action": "unknown"}, move_unknown=True, dry_run=False), "moved_unknown")
        self.assertEqual(quarantine_action({"gender": "female", "action": "keep"}, move_unknown=True, dry_run=False), None)
        self.assertEqual(quarantine_action({"gender": "male", "action": "unknown"}, move_unknown=False, dry_run=True), "would_move_male")

    def test_discover_images_finds_supported_extensions_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (root / "Selfie10.jpg").write_bytes(b"x")
            (root / "Selfie2.webp").write_bytes(b"x")
            (nested / "Selfie1.png").write_bytes(b"x")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            paths = discover_images(root)

            self.assertEqual([path.name for path in paths], ["Selfie1.png", "Selfie2.webp", "Selfie10.jpg"])

    def test_unused_path_adds_suffix_when_target_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "image.jpg"
            target.write_bytes(b"x")

            self.assertEqual(unused_path(target).name, "image_2.jpg")

    def test_move_to_quarantine_preserves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            nested = source_root / "nested"
            nested.mkdir(parents=True)
            image = nested / "image.jpg"
            image.write_bytes(b"x")
            output_root = root / "removed"

            moved = move_to_quarantine(image, source_root, output_root)

            self.assertEqual(moved, output_root / "nested" / "image.jpg")
            self.assertTrue(moved.exists())
            self.assertFalse(image.exists())


if __name__ == "__main__":
    unittest.main()
