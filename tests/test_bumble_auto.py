from __future__ import annotations

import io
import csv
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile

import bumble_auto
from face_similarity.cli_pause import pause_if_requested, read_key
from face_similarity.prediction import RatingPrediction


def config(
    *,
    loop: bool = False,
    delay: float | None = None,
    mode_247: bool = False,
) -> bumble_auto.AutomationConfig:
    return bumble_auto.AutomationConfig(
        setup_name="",
        store_path=Path("store.npz"),
        regressor_path=Path("model.joblib"),
        multimodal_regressor_path=Path("multimodal.joblib"),
        method="face_biased",
        face_weight=0.22,
        k=11,
        provider="auto",
        threshold=54.0,
        decision_mode="threshold",
        preference_model_path=Path("models/bumble_preference_classifier.joblib"),
        blend_preference_model_path=None,
        preference_threshold=None,
        dynamic_threshold=False,
        dynamic_mode="rolling",
        dynamic_window=200,
        dynamic_target_right_rate=0.25,
        dynamic_min_history=50,
        dynamic_min_threshold=48.0,
        dynamic_max_threshold=62.0,
        dynamic_preference_threshold=False,
        dynamic_preference_mode="rolling",
        dynamic_preference_window=200,
        dynamic_preference_target_right_rate=0.20,
        dynamic_preference_min_history=50,
        dynamic_preference_min_threshold=0.45,
        dynamic_preference_max_threshold=0.75,
        loop=loop,
        delay=delay,
        mode_247=mode_247,
        profile_dir=Path(".bumble_browser"),
        screenshot_path=Path("results/bumble_current.png"),
        log_dir=Path(r"D:\BumbleLog"),
        log_quality=45,
        log_max_width=720,
        log_format="webp",
    )


class FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    def press(self, key: str) -> None:
        self.pressed.append(key)


class FakeLocator:
    def __init__(self, text: str) -> None:
        self.text = text

    def inner_text(self, timeout: int) -> str:
        return self.text


class FakePage:
    def __init__(self, *, text: str = "", scrolled: bool = False) -> None:
        self.text = text
        self.keyboard = FakeKeyboard()
        self.evaluated: list[str] = []
        self.screenshots: list[dict[str, object]] = []
        self.waits: list[int] = []
        self._scrolled = scrolled

    def locator(self, selector: str) -> FakeLocator:
        self.selector = selector
        return FakeLocator(self.text)

    def evaluate(self, script: str) -> dict[str, object] | None:
        self.evaluated.append(script)
        if script == bumble_auto.SCROLL_STATE_SCRIPT:
            return {"scrolled": self._scrolled, "count": 1}
        if script == bumble_auto.RESET_SCROLL_SCRIPT:
            self._scrolled = False
            return None
        raise AssertionError(f"unexpected script: {script}")

    def wait_for_timeout(self, timeout: int) -> None:
        self.waits.append(timeout)

    def screenshot(self, **kwargs: object) -> None:
        self.screenshots.append(kwargs)


class BumbleAutoTests(unittest.TestCase):
    def test_decision_key_uses_left_below_threshold_otherwise_right(self) -> None:
        self.assertEqual(bumble_auto.decision_key(54.99), "ArrowLeft")
        self.assertEqual(bumble_auto.decision_key(55.0), "ArrowRight")
        self.assertEqual(bumble_auto.decision_key(100.0), "ArrowRight")

    def test_parse_args_requires_delay_for_loop(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_auto.parse_args(["--loop"])

    def test_parse_args_allows_247_loop_without_delay(self) -> None:
        cfg = bumble_auto.parse_args(["--loop", "--247"])

        self.assertTrue(cfg.loop)
        self.assertTrue(cfg.mode_247)
        self.assertIsNone(cfg.delay)

    def test_parse_args_supports_preference_decision_mode(self) -> None:
        cfg = bumble_auto.parse_args(
            [
                "--decision-mode",
                "preference",
                "--preference-model",
                "models/custom.joblib",
                "--preference-threshold",
                "0.42",
            ]
        )

        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/custom.joblib"))
        self.assertEqual(cfg.preference_threshold, 0.42)

    def test_parse_args_supports_base_generation_setups(self) -> None:
        setups = {
            "original": (
                "Original",
                "embeddings/reference_store.npz",
                "models/rating_regressor.joblib",
                "models/rating_regressor_multimodal.joblib",
                0.50,
            ),
            "round1": (
                "Round1",
                "embeddings/reference_store_bumble_combined.npz",
                "models/rating_regressor_bumble_combined.joblib",
                "models/rating_regressor_multimodal_bumble_combined.joblib",
                0.50,
            ),
            "round2": (
                "Round2",
                "embeddings/reference_store_bumble_combined_round2.npz",
                "models/rating_regressor_bumble_combined_round2.joblib",
                "models/rating_regressor_multimodal_bumble_combined_round2.joblib",
                0.22,
            ),
            "round3": (
                "Round3",
                "embeddings/reference_store_bumble_combined_round3.npz",
                "models/rating_regressor_bumble_combined_round3.joblib",
                "models/rating_regressor_multimodal_bumble_combined_round3.joblib",
                0.44,
            ),
        }

        for setup, expected in setups.items():
            with self.subTest(setup=setup):
                cfg = bumble_auto.parse_args(["--setup", setup, "--loop", "--delay", "0"])

                self.assertEqual(cfg.setup_name, expected[0])
                self.assertEqual(cfg.store_path, Path(expected[1]))
                self.assertEqual(cfg.regressor_path, Path(expected[2]))
                self.assertEqual(cfg.multimodal_regressor_path, Path(expected[3]))
                self.assertAlmostEqual(cfg.face_weight, expected[4])
                self.assertEqual(cfg.method, "face_biased")
                self.assertEqual(cfg.decision_mode, "threshold")
                self.assertAlmostEqual(cfg.threshold, 55.0)
                self.assertFalse(cfg.dynamic_threshold)
                self.assertFalse(cfg.dynamic_preference_threshold)

    def test_parse_args_supports_experimental1_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "experimental1", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "Experimental1")
        self.assertEqual(cfg.store_path, Path("embeddings/reference_store_bumble_combined_round2.npz"))
        self.assertEqual(cfg.regressor_path, Path("models/rating_regressor_bumble_combined_round2.joblib"))
        self.assertEqual(
            cfg.multimodal_regressor_path,
            Path("models/rating_regressor_multimodal_bumble_combined_round2.joblib"),
        )
        self.assertEqual(cfg.method, "face_biased")
        self.assertAlmostEqual(cfg.face_weight, 0.30)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_experimental1.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.556059)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertEqual(cfg.dynamic_preference_mode, "from_logs")
        self.assertEqual(cfg.dynamic_preference_window, 200)
        self.assertEqual(cfg.dynamic_preference_min_history, 50)
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.20)
        self.assertAlmostEqual(cfg.dynamic_preference_min_threshold, 0.45)
        self.assertAlmostEqual(cfg.dynamic_preference_max_threshold, 0.75)

    def test_parse_args_supports_experimental2_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "experimental2", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "Experimental2")
        self.assertEqual(cfg.store_path, Path("embeddings/reference_store_bumble_combined_round3.npz"))
        self.assertEqual(cfg.regressor_path, Path("models/rating_regressor_bumble_combined_round3.joblib"))
        self.assertEqual(
            cfg.multimodal_regressor_path,
            Path("models/rating_regressor_multimodal_bumble_combined_round3.joblib"),
        )
        self.assertEqual(cfg.method, "face_biased")
        self.assertAlmostEqual(cfg.face_weight, 0.30)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_experimental2.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.593991)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertEqual(cfg.dynamic_preference_mode, "from_logs")
        self.assertEqual(cfg.dynamic_preference_window, 200)
        self.assertEqual(cfg.dynamic_preference_min_history, 50)
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.20)
        self.assertAlmostEqual(cfg.dynamic_preference_min_threshold, 0.45)
        self.assertAlmostEqual(cfg.dynamic_preference_max_threshold, 0.75)

    def test_parse_args_supports_experimental3_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "experimental3", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "Experimental3")
        self.assertEqual(cfg.store_path, Path("embeddings/reference_store.npz"))
        self.assertEqual(cfg.regressor_path, Path("models/rating_regressor.joblib"))
        self.assertEqual(cfg.multimodal_regressor_path, Path("models/rating_regressor_multimodal.joblib"))
        self.assertEqual(cfg.method, "multimodalx_original")
        self.assertAlmostEqual(cfg.face_weight, 0.50)
        self.assertAlmostEqual(cfg.threshold, 67.342307)
        self.assertEqual(cfg.decision_mode, "threshold")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_classifier.joblib"))
        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)
        self.assertFalse(cfg.dynamic_preference_threshold)

    def test_parse_args_supports_multimodalx_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX")
        self.assertEqual(cfg.method, "multimodalx")
        self.assertEqual(cfg.decision_mode, "threshold")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_classifier.joblib"))
        self.assertAlmostEqual(cfg.threshold, 56.863525)
        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "from_logs")
        self.assertEqual(cfg.dynamic_window, 200)
        self.assertEqual(cfg.dynamic_min_history, 50)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)

    def test_parse_args_supports_multimodalx2_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx2", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX2")
        self.assertEqual(cfg.method, "multimodalx2")
        self.assertEqual(cfg.k, 9)
        self.assertEqual(cfg.decision_mode, "threshold")
        self.assertAlmostEqual(cfg.threshold, 55.835886)
        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)

    def test_parse_args_supports_multimodalx3_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx3", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX3")
        self.assertEqual(cfg.method, "face_biased")
        self.assertAlmostEqual(cfg.face_weight, 0.44)
        self.assertEqual(cfg.k, 11)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_multimodalx3.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.493593)
        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.20)

    def test_parse_args_supports_multimodalx4_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx4", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX4")
        self.assertEqual(cfg.method, "multimodalx2")
        self.assertEqual(cfg.k, 9)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_multimodalx4.joblib"))
        self.assertEqual(cfg.blend_preference_model_path, Path("models/bumble_preference_classifier.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.519971)
        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.15)

    def test_parse_args_supports_multimodalx5_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx5", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX5")
        self.assertEqual(cfg.method, "multimodalx5")
        self.assertAlmostEqual(cfg.face_weight, 0.60 / 0.65)
        self.assertEqual(cfg.k, 11)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_multimodalx5.joblib"))
        self.assertEqual(cfg.blend_preference_model_path, Path("models/bumble_preference_classifier.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.489530)
        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.20)
        self.assertAlmostEqual(cfg.dynamic_preference_min_threshold, 0.0)

    def test_parse_args_supports_multimodalx6_setup(self) -> None:
        cfg = bumble_auto.parse_args(["--setup", "multimodalx6", "--loop", "--delay", "0"])

        self.assertEqual(cfg.setup_name, "MultimodalX6")
        self.assertEqual(cfg.store_path, Path("embeddings/reference_store_bumble_combined_round2.npz"))
        self.assertEqual(cfg.method, "face_biased")
        self.assertAlmostEqual(cfg.face_weight, 0.30)
        self.assertAlmostEqual(cfg.threshold, 55.0)
        self.assertFalse(cfg.dynamic_threshold)
        self.assertEqual(cfg.decision_mode, "preference")
        self.assertEqual(cfg.preference_model_path, Path("models/bumble_preference_round2_veto.joblib"))
        self.assertAlmostEqual(cfg.preference_threshold or 0.0, 0.527431)
        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertEqual(cfg.dynamic_preference_mode, "from_logs")
        self.assertAlmostEqual(cfg.dynamic_preference_target_right_rate, 0.20)

    def test_parse_args_supports_dynamic_from_logs(self) -> None:
        cfg = bumble_auto.parse_args(["--dynamic-from-logs"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "from_logs")

    def test_parse_args_supports_dynamic_rolling(self) -> None:
        cfg = bumble_auto.parse_args(["--dynamic-rolling"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "rolling")

    def test_parse_args_supports_adaptive_dynamic_rolling(self) -> None:
        cfg = bumble_auto.parse_args(["--adaptive-dynamic-rolling"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "adaptive_rolling")

    def test_parse_args_supports_adaptive_dynamic_preference_rolling(self) -> None:
        cfg = bumble_auto.parse_args(["--adaptive-dynamic-preference-rolling"])

        self.assertTrue(cfg.dynamic_preference_threshold)
        self.assertEqual(cfg.dynamic_preference_mode, "adaptive_rolling")

    def test_parse_args_supports_dynamic_percentile(self) -> None:
        cfg = bumble_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "70"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.30)

    def test_parse_args_supports_fractional_dynamic_percentile(self) -> None:
        cfg = bumble_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "0.8"])

        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)

    def test_parse_args_rejects_invalid_dynamic_percentile(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "100"])

    def test_parse_args_rejects_multiple_dynamic_modes(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_auto.parse_args(["--dynamic-from-logs", "--dynamic-rolling"])

    def test_dynamic_threshold_uses_recent_matching_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with (log_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "screenshot",
                        "method",
                        "action",
                        "score",
                        "face_biased",
                        "multimodal",
                        "ridge",
                        "knn",
                        "store_path",
                        "regressor_path",
                        "multimodal_regressor_path",
                        "threshold",
                        "face_weight",
                        "k",
                        "provider",
                        "delay",
                        "mode_247",
                    ],
                )
                writer.writeheader()
                for index, score in enumerate(range(40, 80)):
                    writer.writerow(
                        {
                            "timestamp": str(index),
                            "method": "face_biased",
                            "score": str(score),
                            "regressor_path": "model.joblib",
                            "multimodal_regressor_path": "multimodal.joblib",
                            "face_weight": "0.22",
                        }
                    )
            cfg = bumble_auto.AutomationConfig(
                **{
                    **config().__dict__,
                    "log_dir": log_dir,
                    "dynamic_threshold": True,
                    "dynamic_mode": "from_logs",
                    "dynamic_window": 20,
                    "dynamic_min_history": 10,
                    "dynamic_min_threshold": 50.0,
                    "dynamic_max_threshold": 60.0,
                }
            )

            self.assertAlmostEqual(bumble_auto.decision_threshold(cfg), 60.0)

    def test_dynamic_from_logs_caps_min_history_to_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with (log_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "method", "score", "regressor_path", "multimodal_regressor_path", "face_weight"],
                )
                writer.writeheader()
                for index, score in enumerate(range(40, 65)):
                    writer.writerow(
                        {
                            "timestamp": str(index),
                            "method": "face_biased",
                            "score": str(score),
                            "regressor_path": "model.joblib",
                            "multimodal_regressor_path": "multimodal.joblib",
                            "face_weight": "0.22",
                        }
                    )
            cfg = bumble_auto.AutomationConfig(
                **{
                    **config().__dict__,
                    "log_dir": log_dir,
                    "dynamic_threshold": True,
                    "dynamic_mode": "from_logs",
                    "dynamic_window": 25,
                    "dynamic_min_history": 50,
                    "dynamic_min_threshold": 48.0,
                    "dynamic_max_threshold": 62.0,
                }
            )

            self.assertNotEqual(bumble_auto.decision_threshold(cfg), 54.0)

    def test_dynamic_rolling_uses_current_session_scores_only(self) -> None:
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "dynamic_threshold": True,
                "dynamic_mode": "rolling",
                "dynamic_window": 20,
                "dynamic_min_history": 10,
                "dynamic_min_threshold": 50.0,
                "dynamic_max_threshold": 60.0,
            }
        )

        self.assertEqual(bumble_auto.decision_threshold(cfg), 54.0)
        self.assertEqual(
            bumble_auto.decision_threshold(cfg, session_scores=list(range(40, 59))),
            54.0,
        )
        self.assertAlmostEqual(
            bumble_auto.decision_threshold(cfg, session_scores=list(range(40, 60))),
            54.25,
        )

    def test_adaptive_dynamic_rolling_uses_fixed_threshold_until_twenty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with (log_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "method", "score", "regressor_path", "multimodal_regressor_path", "face_weight"],
                )
                writer.writeheader()
                for index, score in enumerate(range(40, 59)):
                    writer.writerow(
                        {
                            "timestamp": str(index),
                            "method": "face_biased",
                            "score": str(score),
                            "regressor_path": "model.joblib",
                            "multimodal_regressor_path": "multimodal.joblib",
                            "face_weight": "0.22",
                        }
                    )
            cfg = bumble_auto.AutomationConfig(
                **{
                    **config().__dict__,
                    "log_dir": log_dir,
                    "dynamic_threshold": True,
                    "dynamic_mode": "adaptive_rolling",
                    "dynamic_window": 200,
                    "dynamic_min_history": 50,
                    "dynamic_min_threshold": 0.0,
                    "dynamic_max_threshold": 100.0,
                }
            )

            self.assertEqual(bumble_auto.decision_threshold(cfg), 54.0)
            with (log_dir / "scores.csv").open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "method", "score", "regressor_path", "multimodal_regressor_path", "face_weight"],
                )
                writer.writerow(
                    {
                        "timestamp": "19",
                        "method": "face_biased",
                        "score": "59",
                        "regressor_path": "model.joblib",
                        "multimodal_regressor_path": "multimodal.joblib",
                        "face_weight": "0.22",
                    }
                )

            self.assertAlmostEqual(bumble_auto.decision_threshold(cfg), 54.25)

    def test_dynamic_preference_threshold_uses_recent_matching_probability_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with (log_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "setup_name",
                        "method",
                        "preference_probability",
                        "regressor_path",
                        "multimodal_regressor_path",
                        "face_weight",
                        "decision_mode",
                        "preference_model_path",
                    ],
                )
                writer.writeheader()
                for index in range(60):
                    writer.writerow(
                        {
                            "timestamp": str(index),
                            "setup_name": "Experimental1",
                            "method": "face_biased",
                            "preference_probability": str(index / 100),
                            "regressor_path": "model.joblib",
                            "multimodal_regressor_path": "multimodal.joblib",
                            "face_weight": "0.22",
                            "decision_mode": "preference",
                            "preference_model_path": "models/bumble_preference_classifier.joblib",
                        }
                    )
            cfg = bumble_auto.AutomationConfig(
                **{
                    **config().__dict__,
                    "setup_name": "Experimental1",
                    "log_dir": log_dir,
                    "decision_mode": "preference",
                    "dynamic_preference_threshold": True,
                    "dynamic_preference_mode": "from_logs",
                    "dynamic_preference_window": 50,
                    "dynamic_preference_min_history": 50,
                    "dynamic_preference_target_right_rate": 0.20,
                    "dynamic_preference_min_threshold": 0.45,
                    "dynamic_preference_max_threshold": 0.75,
                }
            )
            model = SimpleNamespace(threshold=0.556059)

            self.assertAlmostEqual(
                bumble_auto.decision_preference_threshold(cfg, preference_model=model),
                0.492,
            )

    def test_dynamic_preference_rolling_waits_for_full_window(self) -> None:
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "decision_mode": "preference",
                "dynamic_preference_threshold": True,
                "dynamic_preference_mode": "rolling",
                "dynamic_preference_window": 5,
                "dynamic_preference_min_history": 2,
                "dynamic_preference_target_right_rate": 0.20,
                "dynamic_preference_min_threshold": 0.45,
                "dynamic_preference_max_threshold": 0.75,
            }
        )
        model = SimpleNamespace(threshold=0.556059)

        self.assertEqual(
            bumble_auto.decision_preference_threshold(
                cfg,
                preference_model=model,
                session_preference_probabilities=[0.1, 0.2, 0.3, 0.4],
            ),
            0.556059,
        )
        self.assertAlmostEqual(
            bumble_auto.decision_preference_threshold(
                cfg,
                preference_model=model,
                session_preference_probabilities=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            0.45,
        )

    def test_adaptive_dynamic_preference_rolling_uses_matching_logs_after_ten_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with (log_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "method",
                        "preference_probability",
                        "regressor_path",
                        "multimodal_regressor_path",
                        "face_weight",
                        "decision_mode",
                        "preference_model_path",
                    ],
                )
                writer.writeheader()
                for index in range(10):
                    writer.writerow(
                        {
                            "timestamp": str(index),
                            "method": "face_biased",
                            "preference_probability": str(0.10 + 0.05 * index),
                            "regressor_path": "model.joblib",
                            "multimodal_regressor_path": "multimodal.joblib",
                            "face_weight": "0.22",
                            "decision_mode": "preference",
                            "preference_model_path": "models/bumble_preference_classifier.joblib",
                        }
                    )
            cfg = bumble_auto.AutomationConfig(
                **{
                    **config().__dict__,
                    "log_dir": log_dir,
                    "decision_mode": "preference",
                    "dynamic_preference_threshold": True,
                    "dynamic_preference_mode": "adaptive_rolling",
                    "dynamic_preference_window": 200,
                    "dynamic_preference_min_history": 50,
                    "dynamic_preference_min_threshold": 0.0,
                    "dynamic_preference_max_threshold": 1.0,
                }
            )

            self.assertAlmostEqual(
                bumble_auto.decision_preference_threshold(
                    cfg,
                    preference_model=SimpleNamespace(threshold=0.556059),
                ),
                0.46,
            )

    def test_scroll_to_top_detects_and_resets_scrolled_page(self) -> None:
        page = FakePage(scrolled=True)

        scrolled = bumble_auto.scroll_to_top_if_needed(page)

        self.assertTrue(scrolled)
        self.assertFalse(page._scrolled)
        self.assertEqual(page.waits, [250])

    def test_perform_iteration_scores_and_presses_right(self) -> None:
        page = FakePage()
        store = SimpleNamespace()

        prediction = RatingPrediction(
            rating=54.0,
            method="face_biased",
            face_rating=45.0,
            multimodal_rating=55.0,
            knn_rating=60.0,
        )
        with patch("bumble_auto.score_screenshot", return_value=prediction):
            with patch("bumble_auto.save_profile_log", return_value=Path("logged.jpg")):
                result = bumble_auto.perform_iteration(
                    page,
                    store,
                    config(),
                    1,
                    face_regressor=SimpleNamespace(),
                    multimodal_regressor=SimpleNamespace(),
                )

        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.rating, 54.0)
        self.assertEqual(result.screenshot, "logged.jpg")
        self.assertEqual(result.key, "ArrowRight")
        self.assertEqual(result.threshold, 54.0)
        self.assertEqual(page.keyboard.pressed, ["ArrowRight"])
        self.assertEqual(len(page.screenshots), 1)

    def test_preference_decision_uses_probability_threshold(self) -> None:
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "decision_mode": "preference",
                "preference_threshold": 0.7,
            }
        )
        prediction = RatingPrediction(
            rating=40.0,
            method="face_biased",
            face_rating=45.0,
            multimodal_rating=55.0,
            knn_rating=60.0,
        )
        model = SimpleNamespace(threshold=0.5)

        with patch("bumble_auto.preference_probability", return_value=0.75):
            key, probability, threshold = bumble_auto.preference_decision(
                prediction,
                cfg,
                score_threshold=54.0,
                preference_model=model,
            )

        self.assertEqual(key, "ArrowRight")
        self.assertEqual(probability, 0.75)
        self.assertEqual(threshold, 0.7)

    def test_preference_decision_passes_old_p_like_feature(self) -> None:
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "decision_mode": "preference",
                "preference_threshold": 0.7,
            }
        )
        prediction = RatingPrediction(
            rating=40.0,
            method="multimodalx5",
            face_rating=45.0,
            multimodal_rating=55.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.preference_probability", return_value=0.75) as probability:
            bumble_auto.preference_decision(
                prediction,
                cfg,
                score_threshold=54.0,
                preference_model=SimpleNamespace(threshold=0.5),
                old_p_like=0.42,
            )

        self.assertEqual(probability.call_args.args[1]["old_p_like"], 0.42)

    def test_score_multimodalx_returns_1_to_100_blend(self) -> None:
        prediction = RatingPrediction(
            rating=68.0,
            method="face_biased",
            face_rating=40.0,
            multimodal_rating=80.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.preference_probability", return_value=0.60):
            blended, probability = bumble_auto.score_multimodalx(prediction, config(), SimpleNamespace())

        self.assertEqual(blended.method, "multimodalx")
        self.assertAlmostEqual(blended.rating, 49.4)
        self.assertAlmostEqual(probability, 0.60)

    def test_experimental3_scores_log_domain_proxy(self) -> None:
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "setup_name": "Experimental3",
                "log_quality": 50,
                "log_max_width": 720,
            }
        )
        prediction = RatingPrediction(34.0, "face_biased", 30.0, 38.0, 42.0)

        with patch("bumble_auto.save_compressed_image") as compress:
            with patch("bumble_auto.score_screenshot", return_value=prediction) as score:
                result = bumble_auto.score_iteration_screenshot(
                    cfg,
                    SimpleNamespace(),
                    face_regressor=SimpleNamespace(),
                    multimodal_regressor=SimpleNamespace(),
                )

        self.assertEqual(result, prediction)
        compress.assert_called_once()
        self.assertEqual(compress.call_args.args[0], cfg.screenshot_path)
        self.assertEqual(compress.call_args.kwargs["quality"], 50)
        self.assertEqual(compress.call_args.kwargs["max_width"], 720)
        self.assertNotEqual(score.call_args.args[0], cfg.screenshot_path)

    def test_score_multimodalx2_includes_knn_blend(self) -> None:
        cfg = bumble_auto.AutomationConfig(**{**config().__dict__, "method": "multimodalx2"})
        prediction = RatingPrediction(
            rating=68.0,
            method="face_biased",
            face_rating=40.0,
            multimodal_rating=80.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.preference_probability", return_value=0.60):
            blended, probability = bumble_auto.score_multimodalx(prediction, cfg, SimpleNamespace())

        self.assertEqual(blended.method, "multimodalx2")
        self.assertAlmostEqual(blended.rating, 58.6)
        self.assertAlmostEqual(probability, 0.60)

    def test_score_original_multimodalx_keeps_equal_original_core(self) -> None:
        cfg = bumble_auto.AutomationConfig(**{**config().__dict__, "method": "multimodalx_original", "face_weight": 0.50})
        prediction = RatingPrediction(
            rating=60.0,
            method="face_biased",
            face_rating=40.0,
            multimodal_rating=80.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.preference_probability", return_value=0.60):
            blended, probability = bumble_auto.score_multimodalx(prediction, cfg, SimpleNamespace())

        self.assertEqual(blended.method, "multimodalx_original")
        self.assertAlmostEqual(blended.rating, 60.0)
        self.assertAlmostEqual(probability, 0.60)

    def test_score_multimodalx5_uses_p80_tuned_blend(self) -> None:
        cfg = bumble_auto.AutomationConfig(**{**config().__dict__, "method": "multimodalx5"})
        prediction = RatingPrediction(
            rating=68.0,
            method="face_biased",
            face_rating=40.0,
            multimodal_rating=80.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.preference_probability", return_value=0.60) as probability:
            blended, p_like = bumble_auto.score_multimodalx(prediction, cfg, SimpleNamespace())

        self.assertEqual(blended.method, "multimodalx5")
        self.assertAlmostEqual(blended.rating, 49.0)
        self.assertAlmostEqual(p_like, 0.60)
        self.assertAlmostEqual(probability.call_args.args[1]["face_weight"], 0.30)

    def test_multimodalx_preference_iteration_uses_blend_then_veto_model(self) -> None:
        page = FakePage()
        cfg = bumble_auto.AutomationConfig(
            **{
                **config().__dict__,
                "method": "multimodalx2",
                "decision_mode": "preference",
                "preference_threshold": 0.5,
            }
        )
        prediction = RatingPrediction(
            rating=68.0,
            method="face_biased",
            face_rating=40.0,
            multimodal_rating=80.0,
            knn_rating=60.0,
        )
        blend_model = SimpleNamespace(model_name="blend")
        veto_model = SimpleNamespace(model_name="veto", threshold=0.5)

        with patch("bumble_auto.score_screenshot", return_value=prediction):
            with patch("bumble_auto.score_multimodalx", return_value=(RatingPrediction(
                rating=58.6,
                method="multimodalx2",
                face_rating=40.0,
                multimodal_rating=80.0,
                knn_rating=60.0,
            ), 0.6)) as blend:
                with patch("bumble_auto.preference_probability", return_value=0.75):
                    with patch("bumble_auto.save_profile_log", return_value=Path("logged.jpg")):
                        result = bumble_auto.perform_iteration(
                            page,
                            SimpleNamespace(),
                            cfg,
                            1,
                            face_regressor=SimpleNamespace(),
                            multimodal_regressor=SimpleNamespace(),
                            preference_model=veto_model,
                            blend_preference_model=blend_model,
                        )

        blend.assert_called_once()
        self.assertIs(blend.call_args.args[2], blend_model)
        self.assertEqual(result.key, "ArrowRight")
        self.assertAlmostEqual(result.preference_probability or 0.0, 0.75)

    def test_perform_iteration_retries_then_rejects_left_when_no_face_detected(self) -> None:
        page = FakePage()
        store = SimpleNamespace()

        with patch("bumble_auto.score_screenshot", side_effect=ValueError("No face detected")):
            with patch("bumble_auto.time.sleep"):
                with patch("bumble_auto.save_profile_log", return_value=Path("logged.webp")):
                    result = bumble_auto.perform_iteration(
                        page,
                        store,
                        config(),
                        1,
                        face_regressor=SimpleNamespace(),
                        multimodal_regressor=SimpleNamespace(),
                    )

        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.rating, 0.0)
        self.assertEqual(result.key, "ArrowLeft")
        self.assertEqual(page.keyboard.pressed, ["ArrowLeft"])

    def test_perform_iteration_stops_on_repeated_no_face_zero_score(self) -> None:
        page = FakePage()
        store = SimpleNamespace()

        with patch("bumble_auto.score_screenshot", side_effect=ValueError("No face detected")):
            with patch("bumble_auto.time.sleep"):
                with patch("bumble_auto.save_profile_log", return_value=Path("logged.webp")):
                    result = bumble_auto.perform_iteration(
                        page,
                        store,
                        config(),
                        2,
                        face_regressor=SimpleNamespace(),
                        multimodal_regressor=SimpleNamespace(),
                        previous_score_signature=("0.0000", "", "", "0.0000"),
                    )

        self.assertEqual(result.stop_reason, "Score is identical to previous screenshot")
        self.assertEqual(result.rating, 0.0)
        self.assertEqual(page.keyboard.pressed, [])

    def test_perform_iteration_stops_without_key_on_duplicate_score(self) -> None:
        page = FakePage()
        store = SimpleNamespace()
        prediction = RatingPrediction(
            rating=50.0,
            method="face_biased",
            face_rating=45.0,
            multimodal_rating=55.0,
            knn_rating=60.0,
        )

        with patch("bumble_auto.score_screenshot", return_value=prediction):
            with patch("bumble_auto.save_profile_log", return_value=Path("logged.jpg")):
                result = bumble_auto.perform_iteration(
                    page,
                    store,
                    config(),
                    2,
                    face_regressor=SimpleNamespace(),
                    multimodal_regressor=SimpleNamespace(),
                    previous_score_signature=("50.0000", "45.0000", "55.0000", "60.0000"),
                )

        self.assertEqual(result.stop_reason, "Score is identical to previous screenshot")
        self.assertEqual(result.screenshot, "logged.jpg")
        self.assertEqual(page.keyboard.pressed, [])

    def test_perform_iteration_stops_without_key_on_quota_prompt(self) -> None:
        page = FakePage(text="You are out of likes. Come back later.")
        store = SimpleNamespace()

        result = bumble_auto.perform_iteration(
            page,
            store,
            config(),
            1,
            face_regressor=SimpleNamespace(),
            multimodal_regressor=SimpleNamespace(),
        )

        self.assertEqual(result.stop_reason, "Bumble quota or empty prompt detected")
        self.assertEqual(page.keyboard.pressed, [])
        self.assertEqual(page.screenshots, [])

    def test_loop_repeats_after_success_and_stops_on_failure(self) -> None:
        page = FakePage()
        store = SimpleNamespace()
        first = bumble_auto.IterationResult(iteration=1, scrolled=False, rating=75.0, key="ArrowRight")
        second = bumble_auto.IterationResult(iteration=2, scrolled=False, stop_reason="Screenshot could not be scored")
        sleeper = Mock()

        with patch("bumble_auto.perform_iteration", side_effect=[first, second]):
            with patch("builtins.print"):
                pause_handler = Mock(return_value=False)
                results = bumble_auto.run_decision_loop(
                    page,
                    store,
                    config(loop=True, delay=0.1),
                    sleeper=sleeper,
                    pause_handler=pause_handler,
                )

        self.assertEqual(results, [first, second])
        sleeper.assert_called_once_with(0.1)
        self.assertEqual(pause_handler.call_count, 2)

    def test_pause_if_requested_waits_for_enter_after_p(self) -> None:
        resume_reader = Mock(return_value="")

        paused = pause_if_requested(key_reader=lambda: "P", resume_reader=resume_reader)

        self.assertTrue(paused)
        resume_reader.assert_called_once_with("Paused. Press Enter to resume.")

    def test_windows_pause_reader_polls_even_when_stdin_is_not_tty(self) -> None:
        with patch("face_similarity.cli_pause.sys.stdin.isatty", return_value=False):
            with patch("msvcrt.kbhit", return_value=True):
                with patch("msvcrt.getwch", return_value="p"):
                    self.assertEqual(read_key(), "p")

    def test_247_loop_delay_adds_biased_short_delay(self) -> None:
        rng = Mock()
        rng.triangular.return_value = 0.5

        delay, checkpoint = bumble_auto.loop_delay_seconds(
            config(loop=True, mode_247=True),
            now=10.0,
            last_hourly_pause_at=0.0,
            rng=rng,
        )

        self.assertEqual(delay, 0.5)
        self.assertEqual(checkpoint, 0.0)
        rng.triangular.assert_called_once_with(0.0, 3.0, 0.5)

    def test_247_loop_delay_adds_hourly_pause(self) -> None:
        rng = Mock()
        rng.triangular.return_value = 0.5
        rng.uniform.return_value = 150.0

        delay, checkpoint = bumble_auto.loop_delay_seconds(
            config(loop=True, delay=1.0, mode_247=True),
            now=3601.0,
            last_hourly_pause_at=0.0,
            rng=rng,
        )

        self.assertEqual(delay, 151.5)
        self.assertEqual(checkpoint, 3601.0)
        rng.uniform.assert_called_once_with(120, 180)

    def test_print_iteration_rings_bell_on_stop(self) -> None:
        with patch("bumble_auto.ring_bell") as bell:
            bumble_auto.print_iteration(bumble_auto.IterationResult(iteration=1, scrolled=False, stop_reason="stopped"))

        bell.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
