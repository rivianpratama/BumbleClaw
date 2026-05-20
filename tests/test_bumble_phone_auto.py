from __future__ import annotations

import io
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bumble_phone_auto
from face_similarity.prediction import RatingPrediction


def config(
    *,
    loop: bool = False,
    delay: float | None = None,
    mode_247: bool = False,
) -> bumble_phone_auto.PhoneAutomationConfig:
    return bumble_phone_auto.PhoneAutomationConfig(
        store_path=Path("store.npz"),
        regressor_path=Path("model.joblib"),
        multimodal_regressor_path=Path("multimodal.joblib"),
        method="face_biased",
        face_weight=0.22,
        k=11,
        provider="auto",
        threshold=54.0,
        dynamic_threshold=False,
        dynamic_mode="rolling",
        dynamic_window=200,
        dynamic_target_right_rate=0.25,
        dynamic_min_history=50,
        dynamic_min_threshold=48.0,
        dynamic_max_threshold=62.0,
        loop=loop,
        delay=delay,
        mode_247=mode_247,
        screenshot_path=Path("results/phone_current.png"),
        log_dir=Path(r"D:\BumbleLog"),
        log_quality=45,
        log_max_width=720,
        log_format="webp",
        adb_path="adb",
        serial=None,
        left_swipe=(850, 1400, 150, 1400, 250),
        right_swipe=(150, 1400, 850, 1400, 250),
    )


class BumblePhoneAutoTests(unittest.TestCase):
    def test_decision_action_uses_left_below_threshold_otherwise_right(self) -> None:
        self.assertEqual(bumble_phone_auto.decision_action(53.99), "left")
        self.assertEqual(bumble_phone_auto.decision_action(54.0), "right")
        self.assertEqual(bumble_phone_auto.decision_action(100.0), "right")

    def test_parse_args_requires_delay_for_loop(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_phone_auto.parse_args(["--loop"])

    def test_parse_args_allows_247_loop_without_delay(self) -> None:
        cfg = bumble_phone_auto.parse_args(["--loop", "--247"])

        self.assertTrue(cfg.loop)
        self.assertTrue(cfg.mode_247)
        self.assertIsNone(cfg.delay)

    def test_parse_args_supports_dynamic_from_logs(self) -> None:
        cfg = bumble_phone_auto.parse_args(["--dynamic-from-logs"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "from_logs")

    def test_parse_args_supports_dynamic_rolling(self) -> None:
        cfg = bumble_phone_auto.parse_args(["--dynamic-rolling"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertEqual(cfg.dynamic_mode, "rolling")

    def test_parse_args_supports_dynamic_percentile(self) -> None:
        cfg = bumble_phone_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "70"])

        self.assertTrue(cfg.dynamic_threshold)
        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.30)

    def test_parse_args_supports_fractional_dynamic_percentile(self) -> None:
        cfg = bumble_phone_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "0.8"])

        self.assertAlmostEqual(cfg.dynamic_target_right_rate, 0.20)

    def test_parse_args_rejects_invalid_dynamic_percentile(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_phone_auto.parse_args(["--dynamic-from-logs", "--dynamic-percentile", "100"])

    def test_parse_args_rejects_multiple_dynamic_modes(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_phone_auto.parse_args(["--dynamic-from-logs", "--dynamic-rolling"])

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
            cfg = bumble_phone_auto.PhoneAutomationConfig(
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

            self.assertAlmostEqual(bumble_phone_auto.decision_threshold(cfg), 60.0)

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
            cfg = bumble_phone_auto.PhoneAutomationConfig(
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

            self.assertNotEqual(bumble_phone_auto.decision_threshold(cfg), 54.0)

    def test_dynamic_rolling_uses_current_session_scores_only(self) -> None:
        cfg = bumble_phone_auto.PhoneAutomationConfig(
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

        self.assertEqual(bumble_phone_auto.decision_threshold(cfg), 54.0)
        self.assertEqual(
            bumble_phone_auto.decision_threshold(cfg, session_scores=list(range(40, 59))),
            54.0,
        )
        self.assertAlmostEqual(
            bumble_phone_auto.decision_threshold(cfg, session_scores=list(range(40, 60))),
            54.25,
        )

    def test_parse_swipe_requires_five_integers(self) -> None:
        self.assertEqual(bumble_phone_auto.parse_swipe("1,2,3,4,5"), (1, 2, 3, 4, 5))
        with self.assertRaises(Exception):
            bumble_phone_auto.parse_swipe("1,2,3")

    def test_capture_screen_writes_png_bytes_from_adb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            cfg = bumble_phone_auto.PhoneAutomationConfig(
                **{**cfg.__dict__, "screenshot_path": Path(directory) / "screen.png"}
            )
            runner = Mock(
                return_value=subprocess.CompletedProcess(
                    ["adb"],
                    returncode=0,
                    stdout=b"\x89PNG\r\n",
                    stderr=b"",
                )
            )

            bumble_phone_auto.capture_screen(cfg, runner=runner)

            self.assertEqual(cfg.screenshot_path.read_bytes(), b"\x89PNG\r\n")
            runner.assert_called_once_with(["adb", "exec-out", "screencap", "-p"], capture_output=True, check=False)

    def test_swipe_phone_uses_right_coordinates(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess(["adb"], returncode=0, stdout=b"", stderr=b""))

        bumble_phone_auto.swipe_phone(config(), "right", runner=runner)

        runner.assert_called_once_with(
            ["adb", "shell", "input", "swipe", "150", "1400", "850", "1400", "250"],
            capture_output=True,
            check=False,
        )

    def test_perform_iteration_scores_and_swipes(self) -> None:
        cfg = config()
        store = SimpleNamespace()

        with patch("bumble_phone_auto.capture_screen") as capture:
            prediction = RatingPrediction(
                rating=75.0,
                method="face_biased",
                face_rating=70.0,
                multimodal_rating=80.0,
                knn_rating=65.0,
            )
            with patch("bumble_phone_auto.score_screenshot", return_value=prediction):
                with patch("bumble_phone_auto.save_profile_log", return_value=Path("logged.jpg")):
                    with patch("bumble_phone_auto.swipe_phone") as swipe:
                        result = bumble_phone_auto.perform_iteration(
                            cfg,
                            store,
                            1,
                            face_regressor=SimpleNamespace(),
                            multimodal_regressor=SimpleNamespace(),
                        )

        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.rating, 75.0)
        self.assertEqual(result.screenshot, "logged.jpg")
        self.assertEqual(result.action, "right")
        self.assertEqual(result.threshold, 54.0)
        capture.assert_called_once_with(cfg)
        swipe.assert_called_once_with(cfg, "right")

    def test_perform_iteration_stops_without_swiping_on_duplicate_score(self) -> None:
        cfg = config()
        store = SimpleNamespace()
        prediction = RatingPrediction(
            rating=75.0,
            method="face_biased",
            face_rating=70.0,
            multimodal_rating=80.0,
            knn_rating=65.0,
        )

        with patch("bumble_phone_auto.capture_screen"):
            with patch("bumble_phone_auto.score_screenshot", return_value=prediction):
                with patch("bumble_phone_auto.save_profile_log", return_value=Path("logged.jpg")):
                    with patch("bumble_phone_auto.swipe_phone") as swipe:
                        result = bumble_phone_auto.perform_iteration(
                            cfg,
                            store,
                            2,
                            face_regressor=SimpleNamespace(),
                            multimodal_regressor=SimpleNamespace(),
                            previous_score_signature=("75.0000", "70.0000", "80.0000", "65.0000"),
                        )

        self.assertEqual(result.stop_reason, "Score is identical to previous screenshot")
        self.assertEqual(result.screenshot, "logged.jpg")
        swipe.assert_not_called()

    def test_loop_repeats_after_success_and_stops_on_failure(self) -> None:
        first = bumble_phone_auto.IterationResult(iteration=1, rating=75.0, action="right")
        second = bumble_phone_auto.IterationResult(iteration=2, stop_reason="failed")
        sleeper = Mock()

        with patch("bumble_phone_auto.perform_iteration", side_effect=[first, second]):
            with patch("builtins.print"):
                results = bumble_phone_auto.run_loop(
                    config(loop=True, delay=0.1),
                    SimpleNamespace(),
                    face_regressor=SimpleNamespace(),
                    multimodal_regressor=SimpleNamespace(),
                    sleeper=sleeper,
                )

        self.assertEqual(results, [first, second])
        sleeper.assert_called_once_with(0.1)

    def test_247_loop_delay_adds_biased_short_delay(self) -> None:
        rng = Mock()
        rng.triangular.return_value = 0.5

        delay, checkpoint = bumble_phone_auto.loop_delay_seconds(
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

        delay, checkpoint = bumble_phone_auto.loop_delay_seconds(
            config(loop=True, delay=1.0, mode_247=True),
            now=3601.0,
            last_hourly_pause_at=0.0,
            rng=rng,
        )

        self.assertEqual(delay, 151.5)
        self.assertEqual(checkpoint, 3601.0)
        rng.uniform.assert_called_once_with(120, 180)

    def test_print_iteration_rings_bell_on_stop(self) -> None:
        with patch("bumble_phone_auto.ring_bell") as bell:
            bumble_phone_auto.print_iteration(bumble_phone_auto.IterationResult(iteration=1, stop_reason="stopped"))

        bell.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
