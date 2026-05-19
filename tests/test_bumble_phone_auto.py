from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bumble_phone_auto
from face_similarity.prediction import RatingPrediction


def config(*, loop: bool = False, delay: float | None = None) -> bumble_phone_auto.PhoneAutomationConfig:
    return bumble_phone_auto.PhoneAutomationConfig(
        store_path=Path("store.npz"),
        regressor_path=Path("model.joblib"),
        multimodal_regressor_path=Path("multimodal.joblib"),
        method="face_biased",
        face_weight=0.5,
        k=11,
        provider="auto",
        threshold=60.0,
        loop=loop,
        delay=delay,
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
        self.assertEqual(bumble_phone_auto.decision_action(63.2), "left")
        self.assertEqual(bumble_phone_auto.decision_action(63.3), "right")
        self.assertEqual(bumble_phone_auto.decision_action(100.0), "right")

    def test_parse_args_requires_delay_for_loop(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_phone_auto.parse_args(["--loop"])

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

    def test_print_iteration_rings_bell_on_stop(self) -> None:
        with patch("bumble_phone_auto.ring_bell") as bell:
            bumble_phone_auto.print_iteration(bumble_phone_auto.IterationResult(iteration=1, stop_reason="stopped"))

        bell.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
