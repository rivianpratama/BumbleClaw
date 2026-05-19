from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kpop_profile_curate import (
    Candidate,
    dedupe_paths,
    identity_key,
    round_robin_take,
    select_balanced,
)


def candidate(path: str, identity: str, predicted_rating: float, embedding_value: float | None = None) -> Candidate:
    value = predicted_rating if embedding_value is None else embedding_value
    return Candidate(
        path=Path(path),
        identity=identity,
        embedding=np.asarray([value, 1.0], dtype=np.float32),
        gender="female",
        age="",
        face_count=1,
        face_area_ratio=0.5,
        bbox="",
        predicted_rating=predicted_rating,
        max_similarity=0.8,
    )


class KpopProfileCurateTests(unittest.TestCase):
    def test_identity_key_uses_numeric_identity_folder_across_top_level_splits(self) -> None:
        root = Path("dataset")

        self.assertEqual(identity_key(root / "1" / "0000" / "a.jpg", root), "0000")
        self.assertEqual(identity_key(root / "2" / "0000" / "a.jpg", root), "0000")

    def test_dedupe_paths_marks_later_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "1" / "0000" / "a.jpg"
            duplicate = root / "2" / "0000" / "a.jpg"
            unique = root / "1" / "0001" / "b.jpg"
            first.parent.mkdir(parents=True)
            duplicate.parent.mkdir(parents=True)
            unique.parent.mkdir(parents=True)
            first.write_bytes(b"same")
            duplicate.write_bytes(b"same")
            unique.write_bytes(b"different")

            unique_paths, duplicate_paths = dedupe_paths([first, duplicate, unique])

            self.assertEqual(unique_paths, [first, unique])
            self.assertEqual(duplicate_paths, {duplicate})

    def test_round_robin_take_balances_before_filling_extra_slots(self) -> None:
        per_identity = {
            "0000": [candidate("a1.jpg", "0000", 90), candidate("a2.jpg", "0000", 80)],
            "0001": [candidate("b1.jpg", "0001", 70)],
            "0002": [candidate("c1.jpg", "0002", 60), candidate("c2.jpg", "0002", 50)],
        }

        selected = round_robin_take(per_identity, 4)

        self.assertEqual([item.path.name for item in selected], ["a1.jpg", "b1.jpg", "c1.jpg", "a2.jpg"])

    def test_select_balanced_respects_max_per_identity(self) -> None:
        items = [
            candidate("a1.jpg", "0000", 90),
            candidate("a2.jpg", "0000", 80),
            candidate("a3.jpg", "0000", 70),
            candidate("b1.jpg", "0001", 60),
            candidate("b2.jpg", "0001", 50),
        ]

        selected = select_balanced(items, target_count=10, max_per_identity=2, top_per_identity_pool=35)

        counts = {}
        for item in selected:
            counts[item.identity] = counts.get(item.identity, 0) + 1
        self.assertEqual(counts, {"0000": 2, "0001": 2})

    def test_select_balanced_prefers_higher_predicted_scores_for_candidate_pool(self) -> None:
        items = [
            candidate("low.jpg", "0000", 30, embedding_value=1),
            candidate("middle.jpg", "0000", 60, embedding_value=2),
            candidate("high.jpg", "0000", 90, embedding_value=3),
        ]

        selected = select_balanced(items, target_count=2, max_per_identity=2, top_per_identity_pool=2)

        self.assertEqual({item.path.name for item in selected}, {"high.jpg", "middle.jpg"})


if __name__ == "__main__":
    unittest.main()
