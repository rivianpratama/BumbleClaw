from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from face_similarity.prediction import (
    PREDICTION_METHODS,
    RatingPrediction,
    predict_image_rating,
)
from face_similarity.automation_log import DEFAULT_BUMBLE_LOG_DIR, DEFAULT_LOG_FORMAT, LOG_FORMATS, save_profile_log
from face_similarity.clip_runtime import ensure_clip_runtime
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

DEFAULT_STORE = "embeddings/reference_store_bumble_combined_round2.npz"
DEFAULT_REGRESSOR_PATH = "models/rating_regressor_bumble_combined_round2.joblib"
DEFAULT_MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal_bumble_combined_round2.joblib"
DEFAULT_FACE_BIAS_WEIGHT = 0.22
DEFAULT_SCREENSHOT = "results/phone_current.png"
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


@dataclass(frozen=True)
class PhoneAutomationConfig:
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
    screenshot_path: Path
    log_dir: Path
    log_quality: int
    log_max_width: int
    log_format: str
    adb_path: str
    serial: str | None
    left_swipe: tuple[int, int, int, int, int]
    right_swipe: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    rating: float | None = None
    screenshot: str | None = None
    action: str | None = None
    threshold: float | None = None
    score_signature: tuple[str, str, str, str] | None = None
    stop_reason: str | None = None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> PhoneAutomationConfig:
    parser = argparse.ArgumentParser(description="Score the connected Android phone screen and swipe Bumble left or right.")
    parser.add_argument("--store", default=DEFAULT_STORE, help="Reference store path")
    parser.add_argument("--regressor", default=DEFAULT_REGRESSOR_PATH, help="Face-only regressor path")
    parser.add_argument("--multimodal-regressor", default=DEFAULT_MULTIMODAL_REGRESSOR_PATH, help="Multimodal regressor path")
    parser.add_argument("--method", choices=PREDICTION_METHODS, default="face_biased", help="Scoring method")
    parser.add_argument("--face-weight", type=float, default=DEFAULT_FACE_BIAS_WEIGHT, help="Face-only weight for face_biased method")
    parser.add_argument("--k", type=int, default=11, help="Number of nearest references to use for KNN mode")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto", help="Embedding provider")
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
    parser.add_argument("--loop", action="store_true", help="Keep scoring and swiping until stopped")
    parser.add_argument("--delay", type=float, help="Seconds to wait after each swipe in loop mode")
    parser.add_argument("--247", action="store_true", dest="mode_247", help="Add random human-like loop delays and hourly pauses")
    parser.add_argument("--screenshot", default=DEFAULT_SCREENSHOT, help="Path to save the phone screenshot")
    parser.add_argument("--log-dir", default=str(DEFAULT_BUMBLE_LOG_DIR), help="Directory for compressed profile logs")
    parser.add_argument("--log-quality", type=int, default=45, help="JPEG quality for profile logs")
    parser.add_argument("--log-max-width", type=int, default=720, help="Maximum logged image width")
    parser.add_argument("--log-format", choices=sorted(LOG_FORMATS), default=DEFAULT_LOG_FORMAT, help="Compressed log image format")
    parser.add_argument("--adb", default="adb", help="ADB executable path")
    parser.add_argument("--serial", help="ADB device serial when multiple devices are connected")
    parser.add_argument(
        "--left-swipe",
        default="850,1400,150,1400,250",
        help="Reject swipe as x1,y1,x2,y2,duration_ms",
    )
    parser.add_argument(
        "--right-swipe",
        default="150,1400,850,1400,250",
        help="Accept swipe as x1,y1,x2,y2,duration_ms",
    )
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

    return PhoneAutomationConfig(
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
        screenshot_path=Path(args.screenshot),
        log_dir=Path(args.log_dir),
        log_quality=args.log_quality,
        log_max_width=args.log_max_width,
        log_format=args.log_format,
        adb_path=args.adb,
        serial=args.serial,
        left_swipe=parse_swipe(args.left_swipe),
        right_swipe=parse_swipe(args.right_swipe),
    )


def parse_swipe(value: str) -> tuple[int, int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("swipe must be x1,y1,x2,y2,duration_ms")
    try:
        x1, y1, x2, y2, duration = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("swipe values must be integers") from exc
    if duration < 0:
        raise argparse.ArgumentTypeError("swipe duration must be non-negative")
    return x1, y1, x2, y2, duration


def decision_action(rating: float, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    return "left" if rating < threshold else "right"


def adb_base(config: PhoneAutomationConfig) -> list[str]:
    command = [config.adb_path]
    if config.serial:
        command.extend(["-s", config.serial])
    return command


def ensure_device(config: PhoneAutomationConfig) -> None:
    result = subprocess.run([*adb_base(config), "get-state"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or result.stdout.strip() != "device":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"No authorized ADB device found. {detail}".strip())


def capture_screen(config: PhoneAutomationConfig, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    config.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    result = runner([*adb_base(config), "exec-out", "screencap", "-p"], capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore") if isinstance(result.stderr, bytes) else str(result.stderr)
        raise RuntimeError(f"ADB screencap failed: {error.strip()}")
    if not result.stdout:
        raise RuntimeError("ADB screencap returned no image data")
    config.screenshot_path.write_bytes(result.stdout)


def swipe_phone(
    config: PhoneAutomationConfig,
    action: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    swipe = config.left_swipe if action == "left" else config.right_swipe
    x1, y1, x2, y2, duration = swipe
    result = runner(
        [*adb_base(config), "shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore") if isinstance(result.stderr, bytes) else str(result.stderr)
        raise RuntimeError(f"ADB swipe failed: {error.strip()}")


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


def log_config(config: PhoneAutomationConfig, *, threshold: float | None = None) -> dict[str, object]:
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


def decision_threshold(config: PhoneAutomationConfig, *, session_scores: Sequence[float] | None = None) -> float:
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
    config: PhoneAutomationConfig,
    store: ReferenceStore,
    iteration: int,
    *,
    face_regressor: RatingRegressor | None,
    multimodal_regressor: RatingRegressor | None,
    previous_score_signature: tuple[str, str, str, str] | None = None,
    session_scores: Sequence[float] | None = None,
    is_retry: bool = False,
) -> IterationResult:
    try:
        capture_screen(config)
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
        score_signature = prediction_signature(prediction)
        threshold = decision_threshold(config, session_scores=session_scores)
        action = decision_action(prediction.rating, threshold=threshold)
        logged_path = save_profile_log(
            config.screenshot_path,
            prediction=prediction,
            action=action,
            log_dir=config.log_dir,
            quality=config.log_quality,
            max_width=config.log_max_width,
            image_format=config.log_format,
            config=log_config(config, threshold=threshold),
        )
        if score_signature == previous_score_signature:
            return IterationResult(
                iteration=iteration,
                rating=prediction.rating,
                screenshot=str(logged_path),
                action=action,
                threshold=threshold,
                score_signature=score_signature,
                stop_reason="Score is identical to previous screenshot",
            )
        swipe_phone(config, action)
    except Exception as exc:
        if "No face detected" in str(exc):
            if not is_retry:
                print(f"[{iteration}] No face detected. Waiting 5 seconds to retry...")
                time.sleep(5.0)
                return perform_iteration(
                    config,
                    store,
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
            action = "left"
            logged_path = save_profile_log(
                config.screenshot_path,
                prediction=prediction,
                action=action,
                log_dir=config.log_dir,
                quality=config.log_quality,
                max_width=config.log_max_width,
                image_format=config.log_format,
                config=log_config(config, threshold=threshold),
            )
            swipe_phone(config, action)
            return IterationResult(
                iteration=iteration,
                rating=prediction.rating,
                screenshot=str(logged_path),
                action=action,
                threshold=threshold,
                score_signature=prediction_signature(prediction),
            )

        return IterationResult(iteration=iteration, stop_reason="Phone screen could not be scored or swiped", error=str(exc))
    return IterationResult(
        iteration=iteration,
        rating=prediction.rating,
        screenshot=str(logged_path),
        action=action,
        threshold=threshold,
        score_signature=score_signature,
    )


def print_iteration(result: IterationResult) -> None:
    prefix = f"[{result.iteration}]"
    if result.stop_reason:
        ring_bell()
        print(f"{prefix} STOP reason={result.stop_reason}")
        if result.error:
            print(f"{prefix} ERROR {result.error}")
        return

    assert result.rating is not None
    assert result.action is not None
    threshold = "" if result.threshold is None else f" threshold={result.threshold:.4f}"
    print(f"{prefix} score={result.rating:.4f} swipe={result.action}{threshold} screenshot={result.screenshot}")


def run_loop(
    config: PhoneAutomationConfig,
    store: ReferenceStore,
    *,
    face_regressor: RatingRegressor | None,
    multimodal_regressor: RatingRegressor | None,
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
            config,
            store,
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
    config: PhoneAutomationConfig,
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


def main(argv: list[str] | None = None) -> int:
    suppress_known_third_party_warnings()
    config = parse_args(argv)
    ensure_clip_runtime(config.method)
    ensure_device(config)
    store = load_store(config.store_path)
    face_regressor = load_regressor(config.regressor_path) if config.method in {"regressor", "face_biased"} else None
    multimodal_regressor = (
        load_regressor(config.multimodal_regressor_path) if config.method in {"multimodal", "face_biased"} else None
    )

    print(f"Using method={config.method}")
    if face_regressor is not None:
        print(f"Using face regressor={face_regressor.model_name} from {config.regressor_path}")
    if multimodal_regressor is not None:
        print(f"Using multimodal regressor={multimodal_regressor.model_name} from {config.multimodal_regressor_path}")
    if face_regressor is None and multimodal_regressor is None:
        print(f"Using KNN scorer with k={config.k}")
    print("Open Bumble on the phone and show a profile, then press Enter.")
    input()

    try:
        run_loop(config, store, face_regressor=face_regressor, multimodal_regressor=multimodal_regressor)
    except KeyboardInterrupt:
        ring_bell()
        print("Stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
