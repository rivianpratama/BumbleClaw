from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from face_similarity.prediction import (
    PREDICTION_METHODS,
    RatingPrediction,
    predict_image_rating,
)
from face_similarity.clip_runtime import ensure_clip_runtime
from face_similarity.automation_log import DEFAULT_BUMBLE_LOG_DIR, DEFAULT_LOG_FORMAT, LOG_FORMATS, save_profile_log
from face_similarity.dynamic_threshold import (
    LOG_MODE,
    ROLLING_MODE,
    DynamicThresholdConfig,
    effective_threshold,
    percentile_to_target_right_rate,
)
from face_similarity.regressor import RatingRegressor, load_regressor
from face_similarity.store import ReferenceStore, load_store
from face_similarity.warnings import suppress_known_third_party_warnings

BUMBLE_URL = "https://bumble.com/app"
DEFAULT_STORE = "embeddings/reference_store_bumble_combined_round2.npz"
DEFAULT_REGRESSOR_PATH = "models/rating_regressor_bumble_combined_round2.joblib"
DEFAULT_MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal_bumble_combined_round2.joblib"
DEFAULT_FACE_BIAS_WEIGHT = 0.22
DEFAULT_PROFILE_DIR = ".bumble_browser"
DEFAULT_SCREENSHOT = "results/bumble_current.png"
DEFAULT_THRESHOLD = 54.0
DEFAULT_DYNAMIC_WINDOW = 200
DEFAULT_DYNAMIC_TARGET_RIGHT_RATE = 0.25
DEFAULT_DYNAMIC_MIN_HISTORY = 50
DEFAULT_DYNAMIC_MIN_THRESHOLD = 48.0
DEFAULT_DYNAMIC_MAX_THRESHOLD = 62.0
HOURLY_PAUSE_INTERVAL_SECONDS = 60 * 60
HOURLY_PAUSE_MIN_SECONDS = 2 * 60
HOURLY_PAUSE_MAX_SECONDS = 3 * 60
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
    dynamic_threshold: bool
    dynamic_mode: str
    dynamic_window: int
    dynamic_target_right_rate: float
    dynamic_min_history: int
    dynamic_min_threshold: float
    dynamic_max_threshold: float
    loop: bool
    delay: float | None
    mode_247: bool
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
    threshold: float | None = None
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
    parser.add_argument("--dynamic-threshold", action="store_true", help="Alias for --dynamic-rolling")
    parser.add_argument("--dynamic-from-logs", action="store_true", help="Use scores.csv history immediately for dynamic threshold")
    parser.add_argument("--dynamic-rolling", action="store_true", help="Recalculate dynamic threshold from the latest current-session scores")
    parser.add_argument("--dynamic-window", type=int, default=DEFAULT_DYNAMIC_WINDOW, help="Logged profile count for dynamic threshold")
    parser.add_argument(
        "--dynamic-target-right-rate",
        type=float,
        default=DEFAULT_DYNAMIC_TARGET_RIGHT_RATE,
        help="Target right-swipe rate for dynamic threshold",
    )
    parser.add_argument("--dynamic-percentile", type=float, help="Score percentile for dynamic threshold, for example 75 or 0.75")
    parser.add_argument("--dynamic-min-history", type=int, default=DEFAULT_DYNAMIC_MIN_HISTORY, help="Minimum matching logs before --dynamic-from-logs activates, capped by --dynamic-window")
    parser.add_argument("--dynamic-min-threshold", type=float, default=DEFAULT_DYNAMIC_MIN_THRESHOLD, help="Lower clamp for dynamic threshold")
    parser.add_argument("--dynamic-max-threshold", type=float, default=DEFAULT_DYNAMIC_MAX_THRESHOLD, help="Upper clamp for dynamic threshold")
    parser.add_argument("--loop", action="store_true", help="Keep scoring and acting until a stop condition is hit")
    parser.add_argument("--delay", type=float, help="Seconds to wait after each action in loop mode")
    parser.add_argument("--247", action="store_true", dest="mode_247", help="Add random human-like loop delays and hourly pauses")
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR, help="Persistent Playwright browser profile folder")
    parser.add_argument("--screenshot", default=DEFAULT_SCREENSHOT, help="Path to save the current Bumble screenshot")
    parser.add_argument("--log-dir", default=str(DEFAULT_BUMBLE_LOG_DIR), help="Directory for compressed profile logs")
    parser.add_argument("--log-quality", type=int, default=45, help="JPEG quality for profile logs")
    parser.add_argument("--log-max-width", type=int, default=720, help="Maximum logged image width")
    parser.add_argument("--log-format", choices=sorted(LOG_FORMATS), default=DEFAULT_LOG_FORMAT, help="Compressed log image format")
    args = parser.parse_args(argv)

    if args.loop and args.delay is None and not args.mode_247:
        parser.error("--delay is required with --loop")
    if args.delay is not None and args.delay < 0:
        parser.error("--delay must be greater than or equal to 0")
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.dynamic_window < 1:
        parser.error("--dynamic-window must be at least 1")
    if args.dynamic_min_history < 1:
        parser.error("--dynamic-min-history must be at least 1")
    if not 0 < args.dynamic_target_right_rate < 1:
        parser.error("--dynamic-target-right-rate must be between 0 and 1")
    dynamic_target_right_rate = args.dynamic_target_right_rate
    if args.dynamic_percentile is not None:
        try:
            dynamic_target_right_rate = percentile_to_target_right_rate(args.dynamic_percentile)
        except ValueError as exc:
            parser.error(f"--dynamic-percentile {exc}")
    if args.dynamic_min_threshold > args.dynamic_max_threshold:
        parser.error("--dynamic-min-threshold must be less than or equal to --dynamic-max-threshold")
    dynamic_flag_count = sum(bool(flag) for flag in (args.dynamic_threshold, args.dynamic_from_logs, args.dynamic_rolling))
    if dynamic_flag_count > 1:
        parser.error("choose only one of --dynamic-threshold, --dynamic-from-logs, or --dynamic-rolling")
    dynamic_mode = LOG_MODE if args.dynamic_from_logs else ROLLING_MODE

    return AutomationConfig(
        store_path=Path(args.store),
        regressor_path=Path(args.regressor),
        multimodal_regressor_path=Path(args.multimodal_regressor),
        method=args.method,
        face_weight=args.face_weight,
        k=args.k,
        provider=args.provider,
        threshold=args.threshold,
        dynamic_threshold=dynamic_flag_count > 0,
        dynamic_mode=dynamic_mode,
        dynamic_window=args.dynamic_window,
        dynamic_target_right_rate=dynamic_target_right_rate,
        dynamic_min_history=args.dynamic_min_history,
        dynamic_min_threshold=args.dynamic_min_threshold,
        dynamic_max_threshold=args.dynamic_max_threshold,
        loop=args.loop,
        delay=args.delay,
        mode_247=args.mode_247,
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


def log_config(config: AutomationConfig, *, threshold: float | None = None) -> dict[str, object]:
    return {
        "method": config.method,
        "store_path": config.store_path,
        "regressor_path": config.regressor_path,
        "multimodal_regressor_path": config.multimodal_regressor_path,
        "threshold": config.threshold if threshold is None else threshold,
        "dynamic_enabled": config.dynamic_threshold,
        "dynamic_mode": config.dynamic_mode if config.dynamic_threshold else "",
        "dynamic_window": config.dynamic_window if config.dynamic_threshold else "",
        "dynamic_target_right_rate": config.dynamic_target_right_rate if config.dynamic_threshold else "",
        "dynamic_percentile": (1.0 - config.dynamic_target_right_rate) * 100.0 if config.dynamic_threshold else "",
        "dynamic_min_history": config.dynamic_min_history if config.dynamic_threshold else "",
        "dynamic_min_threshold": config.dynamic_min_threshold if config.dynamic_threshold else "",
        "dynamic_max_threshold": config.dynamic_max_threshold if config.dynamic_threshold else "",
        "face_weight": config.face_weight,
        "k": config.k,
        "provider": config.provider,
        "delay": config.delay,
        "mode_247": config.mode_247,
    }


def decision_threshold(config: AutomationConfig, *, session_scores: Sequence[float] | None = None) -> float:
    return effective_threshold(
        fixed_threshold=config.threshold,
        dynamic=DynamicThresholdConfig(
            enabled=config.dynamic_threshold,
            mode=config.dynamic_mode,
            window=config.dynamic_window,
            target_right_rate=config.dynamic_target_right_rate,
            min_history=config.dynamic_min_history,
            min_threshold=config.dynamic_min_threshold,
            max_threshold=config.dynamic_max_threshold,
        ),
        log_dir=config.log_dir,
        current_config=log_config(config),
        session_scores=session_scores,
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
    session_scores: Sequence[float] | None = None,
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
                    session_scores=session_scores,
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
            threshold = decision_threshold(config, session_scores=session_scores)
            key = "ArrowLeft"
            logged_path = save_profile_log(
                config.screenshot_path,
                prediction=prediction,
                action=key,
                log_dir=config.log_dir,
                quality=config.log_quality,
                max_width=config.log_max_width,
                image_format=config.log_format,
                config=log_config(config, threshold=threshold),
            )
            page.keyboard.press(key)
            return IterationResult(
                iteration=iteration,
                scrolled=scrolled,
                rating=prediction.rating,
                screenshot=str(logged_path),
                key=key,
                threshold=threshold,
                score_signature=prediction_signature(prediction),
            )

        return IterationResult(
            iteration=iteration,
            scrolled=scrolled,
            stop_reason="Screenshot could not be scored",
            error=str(exc),
        )

    score_signature = prediction_signature(prediction)
    threshold = decision_threshold(config, session_scores=session_scores)
    key = decision_key(prediction.rating, threshold=threshold)
    logged_path = save_profile_log(
        config.screenshot_path,
        prediction=prediction,
        action=key,
        log_dir=config.log_dir,
        quality=config.log_quality,
        max_width=config.log_max_width,
        image_format=config.log_format,
        config=log_config(config, threshold=threshold),
    )
    if score_signature == previous_score_signature:
        return IterationResult(
            iteration=iteration,
            scrolled=scrolled,
            rating=prediction.rating,
            screenshot=str(logged_path),
            key=key,
            threshold=threshold,
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
        threshold=threshold,
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
    threshold = "" if result.threshold is None else f" threshold={result.threshold:.4f}"
    print(
        f"{prefix} score={result.rating:.4f} key={result.key}{threshold} "
        f"scrolled={result.scrolled} screenshot={result.screenshot}"
    )


def run_decision_loop(
    page: Any,
    store: ReferenceStore,
    config: AutomationConfig,
    *,
    face_regressor: RatingRegressor | None = None,
    multimodal_regressor: RatingRegressor | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> list[IterationResult]:
    results: list[IterationResult] = []
    iteration = 1
    previous_score_signature: tuple[str, str, str, str] | None = None
    session_scores: list[float] = []
    last_hourly_pause_at = clock()
    rng = rng or random.Random()

    while True:
        result = perform_iteration(
            page,
            store,
            config,
            iteration,
            face_regressor=face_regressor,
            multimodal_regressor=multimodal_regressor,
            previous_score_signature=previous_score_signature,
            session_scores=session_scores,
        )
        print_iteration(result)
        results.append(result)

        if result.stop_reason or not config.loop:
            return results

        previous_score_signature = result.score_signature
        if result.rating is not None:
            session_scores.append(result.rating)
            del session_scores[:-config.dynamic_window]
        delay, last_hourly_pause_at = loop_delay_seconds(
            config,
            now=clock(),
            last_hourly_pause_at=last_hourly_pause_at,
            rng=rng,
        )
        sleeper(delay)
        iteration += 1


def loop_delay_seconds(
    config: AutomationConfig,
    *,
    now: float,
    last_hourly_pause_at: float,
    rng: random.Random,
) -> tuple[float, float]:
    delay = config.delay or 0.0
    if not config.mode_247:
        return delay, last_hourly_pause_at

    delay += random_247_delay_seconds(rng)
    if now - last_hourly_pause_at >= HOURLY_PAUSE_INTERVAL_SECONDS:
        delay += rng.uniform(HOURLY_PAUSE_MIN_SECONDS, HOURLY_PAUSE_MAX_SECONDS)
        last_hourly_pause_at = now
    return delay, last_hourly_pause_at


def random_247_delay_seconds(rng: random.Random) -> float:
    return min(rng.triangular(0.0, 3.0, 0.5), 2.999)


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
