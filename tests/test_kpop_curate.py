from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from kpop_curate import Candidate, identity_key, round_robin_take, select_balanced


def candidate(path: str, identity: str, value: float) -> Candidate:
    return Candidate(
        path=Path(path),
        identity=identity,
        embedding=np.asarray([value, 1.0], dtype=np.float32),
        gender="female",
        age="",
        face_count=1,
        face_area_ratio=value,
        bbox="",
    )


class KpopCurateTests(unittest.TestCase):
    def test_identity_key_uses_person_subfolder_when_present(self) -> None:
        root = Path("archive")

        self.assertEqual(identity_key(root / "aespa" / "karina" / "a.jpg", root), "aespa/karina")
        self.assertEqual(identity_key(root / "aespa" / "loose.jpg", root), "aespa/_loose")

    def test_round_robin_take_balances_identity_buckets(self) -> None:
        per_identity = {
            "a": [candidate("a1.jpg", "a", 1), candidate("a2.jpg", "a", 2)],
            "b": [candidate("b1.jpg", "b", 1)],
            "c": [candidate("c1.jpg", "c", 1), candidate("c2.jpg", "c", 2)],
        }

        selected = round_robin_take(per_identity, 4)

        self.assertEqual([item.path.name for item in selected], ["a1.jpg", "b1.jpg", "c1.jpg", "a2.jpg"])

    def test_select_balanced_respects_max_per_identity(self) -> None:
        items = [
            candidate("a1.jpg", "a", 1),
            candidate("a2.jpg", "a", 2),
            candidate("a3.jpg", "a", 3),
            candidate("b1.jpg", "b", 1),
            candidate("b2.jpg", "b", 2),
        ]

        selected = select_balanced(items, target_count=10, max_per_identity=2)

        counts = {}
        for item in selected:
            counts[item.identity] = counts.get(item.identity, 0) + 1
        self.assertEqual(counts, {"a": 2, "b": 2})


if __name__ == "__main__":
    unittest.main()
