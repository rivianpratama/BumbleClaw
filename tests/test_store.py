from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_similarity.store import load_store, save_store


class ReferenceStoreTests(unittest.TestCase):
    def test_saves_and_loads_insightface_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.npz"

            save_store(
                path,
                embeddings=[np.asarray([1.0, 0.0], dtype=np.float32)],
                paths=["references/1.jpg"],
                ratings=[95],
                backend="insightface",
                model_name="buffalo_l",
                provider="cuda",
                det_size=640,
                det_thresh=0.25,
            )
            store = load_store(path)

            self.assertEqual(store.backend, "insightface")
            self.assertEqual(store.model_name, "buffalo_l")
            self.assertEqual(store.provider, "cuda")
            self.assertEqual(store.det_size, 640)
            self.assertAlmostEqual(store.det_thresh, 0.25)
            self.assertEqual(store.paths, ["references/1.jpg"])


if __name__ == "__main__":
    unittest.main()
