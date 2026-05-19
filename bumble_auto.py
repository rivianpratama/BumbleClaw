from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from face_similarity.prediction import (
    DEFAULT_FACE_BIAS_WEIGHT,
    DEFAULT_MULTIMODAL_REGRESSOR_PATH,
    PREDICTION_METHODS,
    RatingPrediction,
    predict_image_rating,
)
from face_similarity.clip_runtime import ensure_clip_runtime
from face_similarity.automation_log import DEFAULT_BUMBLE_LOG_DIR, DEFAULT_LOG_FORMAT, LOG_FORMATS, save_profile_log
from face_similarity.regressor import DEFAULT_REGRESSOR_PATH, RatingRegressor, load_regressor
from face_similarity.store import ReferenceStore, load_store
from face_similarity.warnings import suppress_known_third_party_warnings

BUMBLE_URL = "https://bumble.com/app"
DEFAULT_STORE = "embeddings/reference_store.npz"
DEFAULT_PROFILE_DIR = ".bumble_browser"
DEFAULT_SCREENSHOT = "results/bumble_current.png"
DEFAULT_THRESHOLD = 63.3
BELL = "\a"

QUOTA_PROMPTS = (
    "out of likes",
    "no likes left",
    "daily limit",
    "limit reached",
    "come back later",
    "check back later",
    "no more profiles",
    "no one new",
    "that's everyone",
    "you've seen everyone",
    "you have seen everyone",
)

SCROLL_STATE_SCRIPT = """
() => {
  const elements = [
    document.scrollingElement,
    document.documentElement,
    document.body,
    ...Array.from(document.querySelectorAll("*")).filter((element) => {
      const style = window.getComputedStyle(element);
      const scrollable = /(auto|scroll)/.test(style.overflowY);
      return scrollable && element.scrollHeight > element.clientHeight + 1;
    }),
  ].filter(Boolean);
  const unique = Array.from(new Set(elements));
  return {
    scrolled: unique.some((element) => element.scrollTop > 1),
    count: unique.length,
  };
}
"""

RESET_SCROLL_SCRIPT = """
() => {
  const elements = [
    document.scrollingElement,
    document.documentElement,
    document.body,
    ...Array.from(document.querySelectorAll("*")).filter((element) => element.scrollTop > 1),
  ].filter(Boolean);
  Array.from(new Set(elements)).forEach((element) => {
    element.scrollTop = 0;
  });
  window.scrollTo(0, 0);
}
"""


@dataclass(frozen=True)
class AutomationConfig:
    store_path: Path
    regressor_path: Path
    multimodal_regressor_path: Path
    method: str
    face_weight: float
    k: int
    provider: str
    threshold: float
    loop: bool
    delay: float | None
    profile_dir: Path
    screenshot_path: Path
    log_dir: Path
    log_quality: int
    log_max_width: int
    log_format: str


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    scrolled: bool
    rating: float | None = None
    screenshot: str | None = None
    key: str | None = None
    score_signature: tuple[str, str, str, str] | None = None
    stop_reason: str | None = None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> AutomationConfig:
    parser = argparse.ArgumentParser(description="Score the visible Bumble Web profile and press left or right.")
    parser.add_argument("--store", default=DEFAULT_STORE, help="Reference store path")
    parser.add_argument("--regressor", default=DEFAULT_REGRESSOR_PATH, help="Face-only regressor path")
    parser.add_argument("--multimodal-regressor", default=DEFAULT_MULTIMODAL_REGRESSOR_PATH, help="Multimodal regressor path")
    parser.add_argument("--method", choices=PREDICTION_METHODS, default="face_biased", help="Scoring method")
    parser.add_argument("--face-weight", type=float, default=DEFAULT_FACE_BIAS_WEIGHT, help="Face-only weight for face_biased method")
    parser.add_argument("--k", type=int, default=11, help="Number of nearest references to use")
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Embedding provider for Bumble screenshots",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Swipe right when score is at least this value")
    parser.add_argument("--loop", action="store_true", help="Keep scoring and acting until a stop condition is hit")
    parser.add_argument("--delay", type=float, help="Seconds to wait after each action in loop mode")
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR, help="Persistent Playwright browser profile folder")
    parser.add_argument("--screenshot", default=DEFAULT_SCREENSHOT, help="Path to save the current Bumble screenshot")
    parser.add_argument("--log-dir", default=str(DEFAULT_BUMBLE_LOG_DIR), help="Directory for compressed profile logs")
    parser.add_argument("--log-quality", type=int, default=45, help="JPEG quality for profile logs")
    parser.add_argument("--log-max-width", type=int, default=720, help="Maximum logged image width")
    parser.add_argument("--log-format", choices=sorted(LOG_FORMATS), default=DEFAULT_LOG_FORMAT, help="Compressed log image format")
    args = parser.parse_args(argv)

    if args.loop and args.delay is None:
        parser.error("--delay is required with --loop")
    if args.delay is not None and args.delay < 0:
        parser.error("--delay must be greater than or equal to 0")
    if args.k < 1:
        parser.error("--k must be at least 1")

    return AutomationConfig(
        store_path=Path(args.store),
        regressor_path=Path(args.regressor),
        multimodal_regressor_path=Path(args.multimodal_regressor),
        method=args.method,
        face_weight=args.face_weight,
        k=args.k,
        provider=args.provider,
        threshold=args.threshold,
        loop=args.loop,
        delay=args.delay,
        profile_dir=Path(args.profile_dir),
        screenshot_path=Path(args.screenshot),
        log_dir=Path(args.log_dir),
        log_quality=args.log_quality,
        log_max_width=args.log_max_width,
        log_format=args.log_format,
    )


def decision_key(rating: float, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    return "ArrowLeft" if rating < threshold else "ArrowRight"


def has_quota_prompt(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(prompt in normalized for prompt in QUOTA_PROMPTS)


def page_has_quota_prompt(page: Any) -> bool:
    try:
        text = page.locator("body").inner_text(timeout=1000)
    except Exception:
        return False
    return has_quota_prompt(text)


def scroll_to_top_if_needed(page: Any) -> bool:
    state = page.evaluate(SCROLL_STATE_SCRIPT)
    scrolled = bool(state.get("scrolled")) if isinstance(state, dict) else bool(state)
    if scrolled:
        page.evaluate(RESET_SCROLL_SCRIPT)
        page.wait_for_timeout(250)
    return scrolled


def score_screenshot(
    path: Path,
    store: ReferenceStore,
    *,
    k: int,
    provider: str,
    method: str,
    face_weight: float,
    face_regressor: RatingRegressor | None,
    multimodal_regressor: RatingRegressor | None,
) -> RatingPrediction:
    return predict_image_rating(
        path,
        store=store,
        method=method,
        k=k,
        provider=provider,
        face_regressor=face_regressor,
        multimodal_regressor=multimodal_regressor,
        face_weight=face_weight,
        include_components=True,
    )


def perform_iteration(
    page: Any,
    store: ReferenceStore,
    config: AutomationConfig,
    iteration: int,
    *,
    face_regressor: RatingRegressor | None,
    multimodal_regressor: RatingRegressor | None,
    previous_score_signature: tuple[str, str, str, str] | None = None,
    is_retry: bool = False,
) -> IterationResult:
    if page_has_quota_prompt(page):
        return IterationResult(iteration=iteration, scrolled=False, stop_reason="Bumble quota or empty prompt detected")

    scrolled = scroll_to_top_if_needed(page)
    if page_has_quota_prompt(page):
        return IterationResult(iteration=iteration, scrolled=scrolled, stop_reason="Bumble quota or empty prompt detected")

    config.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(config.screenshot_path), full_page=False)
        prediction = score_screenshot(
            config.screenshot_path,
            store,
            k=config.k,
            provider=config.provider,
            method=config.method,
            face_weight=config.face_weight,
            face_regressor=face_regressor,
            multimodal_regressor=multimodal_regressor,
        )
    except Exception as exc:
        if "No face detected" in str(exc):
            if not is_retry:
                print(f"[{iteration}] No face detected. Waiting 5 seconds to retry...")
                time.sleep(5.0)
                return perform_iteration(
                    page,
                    store,
                    config,
                    iteration,
                    face_regressor=face_regressor,
                    multimodal_regressor=multimodal_regressor,
                    previous_score_signature=previous_score_signature,
                    is_retry=True,
                )
            
            print(f"[{iteration}] No face detected on retry. Swiping left.")
            prediction = RatingPrediction(
                rating=0.0,
                method=config.method,
                face_rating=None,
                multimodal_rating=None,
                knn_rating=0.0,
            )
            key = "ArrowLeft"
            logged_path = save_profile_log(
                config.screenshot_path,
                prediction=prediction,
                action=key,
                log_dir=config.log_dir,
                quality=config.log_quality,
                max_width=config.log_max_width,
                image_format=config.log_format,
            )
            page.keyboard.press(key)
            return IterationResult(
                iteration=iteration,
                scrolled=scrolled,
                rating=prediction.rating,
                screenshot=str(logged_path),
                key=key,
                score_signature=prediction_signature(prediction),
            )

        return IterationResult(
            iteration=iteration,
            scrolled=scrolled,
            stop_reason="Screenshot could not be scored",
            error=str(exc),
        )

    score_signature = prediction_signature(prediction)
    key = decision_key(prediction.rating, threshold=config.threshold)
    logged_path = save_profile_log(
        config.screenshot_path,
        prediction=prediction,
        action=key,
        log_dir=config.log_dir,
        quality=config.log_quality,
        max_width=config.log_max_width,
        image_format=config.log_format,
    )
    if score_signature == previous_score_signature:
        return IterationResult(
            iteration=iteration,
            scrolled=scrolled,
            rating=prediction.rating,
            screenshot=str(logged_path),
            key=key,
            score_signature=score_signature,
            stop_reason="Score is identical to previous screenshot",
        )
    page.keyboard.press(key)
    return IterationResult(
        iteration=iteration,
        scrolled=scrolled,
        rating=prediction.rating,
        screenshot=str(logged_path),
        key=key,
        score_signature=score_signature,
    )


def print_iteration(result: IterationResult) -> None:
    prefix = f"[{result.iteration}]"
    if result.stop_reason:
        ring_bell()
        print(f"{prefix} STOP scrolled={result.scrolled} reason={result.stop_reason}")
        if result.error:
            print(f"{prefix} ERROR {result.error}")
        return

    assert result.rating is not None
    assert result.key is not None
    print(f"{prefix} score={result.rating:.4f} key={result.key} scrolled={result.scrolled} screenshot={result.screenshot}")


def run_decision_loop(
    page: Any,
    store: ReferenceStore,
    config: AutomationConfig,
    *,
    face_regressor: RatingRegressor | None = None,
    multimodal_regressor: RatingRegressor | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[IterationResult]:
    results: list[IterationResult] = []
    iteration = 1
    previous_score_signature: tuple[str, str, str, str] | None = None

    while True:
        result = perform_iteration(
            page,
            store,
            config,
            iteration,
            face_regressor=face_regressor,
            multimodal_regressor=multimodal_regressor,
            previous_score_signature=previous_score_signature,
        )
        print_iteration(result)
        results.append(result)

        if result.stop_reason or not config.loop:
            return results

        previous_score_signature = result.score_signature
        assert config.delay is not None
        sleeper(config.delay)
        iteration += 1


def prediction_signature(prediction: RatingPrediction) -> tuple[str, str, str, str]:
    return (
        format_component(prediction.rating),
        format_component(prediction.face_rating),
        format_component(prediction.multimodal_rating),
        format_component(prediction.knn_rating),
    )


def format_component(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def ring_bell() -> None:
    print(BELL, end="", flush=True)
    if os.name != "nt":
        return
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        winsound.Beep(1200, 500)
    except Exception:
        pass


def open_bumble_page(config: AutomationConfig) -> tuple[Any, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "playwright is required. Run: pip install -r requirements.txt, then python -m playwright install chromium"
        ) from exc

    playwright = sync_playwright().start()
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(config.profile_dir),
        headless=False,
        viewport={"width": 1280, "height": 900},
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(BUMBLE_URL, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    return playwright, context


def main(argv: list[str] | None = None) -> int:
    suppress_known_third_party_warnings()
    config = parse_args(argv)
    ensure_clip_runtime(config.method)
    store = load_store(config.store_path)
    face_regressor = load_regressor(config.regressor_path) if config.method in {"regressor", "face_biased"} else None
    multimodal_regressor = (
        load_regressor(config.multimodal_regressor_path) if config.method in {"multimodal", "face_biased"} else None
    )
    playwright, context = open_bumble_page(config)

    try:
        page = context.pages[0] if context.pages else context.new_page()
        print("Use the opened browser to log in if needed. Press Enter when the Bumble profile is visible.")
        print(f"Using method={config.method}")
        if face_regressor is not None:
            print(f"Using face regressor={face_regressor.model_name} from {config.regressor_path}")
        if multimodal_regressor is not None:
            print(f"Using multimodal regressor={multimodal_regressor.model_name} from {config.multimodal_regressor_path}")
        if face_regressor is None and multimodal_regressor is None:
            print(f"Using KNN scorer with k={config.k}")
        input()
        run_decision_loop(
            page,
            store,
            config,
            face_regressor=face_regressor,
            multimodal_regressor=multimodal_regressor,
        )
    except KeyboardInterrupt:
        ring_bell()
        print("Stopped by user.")
    finally:
        context.close()
        playwright.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
