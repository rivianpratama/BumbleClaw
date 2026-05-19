from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bumble_auto
from face_similarity.prediction import RatingPrediction


def config(*, loop: bool = False, delay: float | None = None) -> bumble_auto.AutomationConfig:
    return bumble_auto.AutomationConfig(
        store_path=Path("store.npz"),
        regressor_path=Path("model.joblib"),
        multimodal_regressor_path=Path("multimodal.joblib"),
        method="face_biased",
        face_weight=0.22,
        k=11,
        provider="auto",
        threshold=62.34,
        loop=loop,
        delay=delay,
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
        self.assertEqual(bumble_auto.decision_key(62.33), "ArrowLeft")
        self.assertEqual(bumble_auto.decision_key(62.34), "ArrowRight")
        self.assertEqual(bumble_auto.decision_key(100.0), "ArrowRight")

    def test_parse_args_requires_delay_for_loop(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                bumble_auto.parse_args(["--loop"])

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
            rating=62.34,
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
        self.assertEqual(result.rating, 62.34)
        self.assertEqual(result.screenshot, "logged.jpg")
        self.assertEqual(result.key, "ArrowRight")
        self.assertEqual(page.keyboard.pressed, ["ArrowRight"])
        self.assertEqual(len(page.screenshots), 1)

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
                results = bumble_auto.run_decision_loop(page, store, config(loop=True, delay=0.1), sleeper=sleeper)

        self.assertEqual(results, [first, second])
        sleeper.assert_called_once_with(0.1)

    def test_print_iteration_rings_bell_on_stop(self) -> None:
        with patch("bumble_auto.ring_bell") as bell:
            bumble_auto.print_iteration(bumble_auto.IterationResult(iteration=1, scrolled=False, stop_reason="stopped"))

        bell.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
