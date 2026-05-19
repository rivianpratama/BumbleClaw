from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_similarity.clip_store import load_clip_store, normalize_path_key, save_clip_store


class ClipStoreTests(unittest.TestCase):
    def test_saves_and_loads_clip_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            save_clip_store(
                path,
                embeddings=[np.asarray([1.0, 0.0], dtype=np.float32)],
                paths=["references/1.jpg"],
                model_name="openai/clip-vit-base-patch32",
                device="cuda",
            )

            store = load_clip_store(path)

        self.assertEqual(store.paths, ["references/1.jpg"])
        self.assertEqual(store.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(store.device, "cuda")
        self.assertEqual(store.embeddings.shape, (1, 2))

    def test_normalize_path_key_matches_case_and_slashes(self) -> None:
        self.assertEqual(normalize_path_key(r"References\A\B.jpg"), "references/a/b.jpg")


if __name__ == "__main__":
    unittest.main()
