from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from face_similarity import clip_runtime


class ClipRuntimeTests(unittest.TestCase):
    def test_configure_clip_cache_sets_d_drive_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clip_runtime.configure_clip_cache()

            self.assertEqual(os.environ["HF_HOME"], str(clip_runtime.HF_CACHE_DIR))
            self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"], str(clip_runtime.HF_CACHE_DIR / "hub"))
            self.assertEqual(os.environ["TRANSFORMERS_CACHE"], str(clip_runtime.HF_CACHE_DIR / "transformers"))
            self.assertEqual(os.environ["TORCH_HOME"], str(clip_runtime.HF_CACHE_DIR / "torch"))

    def test_knn_method_does_not_reexec(self) -> None:
        with patch("os.execv") as execv:
            clip_runtime.ensure_clip_runtime("knn")

        execv.assert_not_called()

    def test_multimodal_method_reexecs_into_clip_venv(self) -> None:
        with patch.object(sys, "executable", r"C:\repo\.venv\Scripts\python.exe"):
            with patch.object(sys, "argv", ["bumble_phone_auto.py", "--loop", "--delay", "3"]):
                with patch("face_similarity.clip_runtime._same_path", return_value=False):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("os.execv") as execv:
                            clip_runtime.ensure_clip_runtime("face_biased")

        execv.assert_called_once_with(
            str(clip_runtime.CLIP_VENV_PYTHON),
            [str(clip_runtime.CLIP_VENV_PYTHON), "bumble_phone_auto.py", "--loop", "--delay", "3"],
        )


if __name__ == "__main__":
    unittest.main()
