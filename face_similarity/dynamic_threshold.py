from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

LOG_MODE = "from_logs"
ROLLING_MODE = "rolling"
ADAPTIVE_ROLLING_MODE = "adaptive_rolling"
ADAPTIVE_SCORE_ROLLING_MIN_HISTORY = 20
ADAPTIVE_VALUE_ROLLING_MIN_HISTORY = 10


@dataclass(frozen=True)
class DynamicThresholdConfig:
    enabled: bool
    mode: str
    window: int
    target_right_rate: float
    min_history: int
    min_threshold: float
    max_threshold: float


def effective_threshold(
    *,
    fixed_threshold: float,
    dynamic: DynamicThresholdConfig,
    log_dir: Path,
    current_config: Mapping[str, object],
    session_scores: Sequence[float] | None = None,
) -> float:
    if not dynamic.enabled:
        return fixed_threshold
    if dynamic.mode not in {LOG_MODE, ROLLING_MODE, ADAPTIVE_ROLLING_MODE}:
        raise ValueError(f"unknown dynamic threshold mode: {dynamic.mode}")
    if dynamic.mode == ROLLING_MODE:
        scores = list(session_scores or [])[-dynamic.window:]
        min_history = dynamic.window
    else:
        scores = recent_values(
            log_dir / "scores.csv",
            value_field="score",
            limit=dynamic.window,
            current_config=current_config,
        )
        minimum = ADAPTIVE_SCORE_ROLLING_MIN_HISTORY if dynamic.mode == ADAPTIVE_ROLLING_MODE else dynamic.min_history
        min_history = min(minimum, dynamic.window)
    return threshold_from_scores(
        scores,
        fixed_threshold=fixed_threshold,
        target_right_rate=dynamic.target_right_rate,
        min_history=min_history,
        min_threshold=dynamic.min_threshold,
        max_threshold=dynamic.max_threshold,
    )


def effective_value_threshold(
    *,
    fixed_threshold: float,
    dynamic: DynamicThresholdConfig,
    log_dir: Path,
    current_config: Mapping[str, object],
    value_field: str,
    session_values: Sequence[float] | None = None,
) -> float:
    if not dynamic.enabled:
        return fixed_threshold
    if dynamic.mode not in {LOG_MODE, ROLLING_MODE, ADAPTIVE_ROLLING_MODE}:
        raise ValueError(f"unknown dynamic threshold mode: {dynamic.mode}")
    if dynamic.mode == ROLLING_MODE:
        values = list(session_values or [])[-dynamic.window:]
        min_history = dynamic.window
    else:
        values = recent_values(
            log_dir / "scores.csv",
            value_field=value_field,
            limit=dynamic.window,
            current_config=current_config,
        )
        minimum = ADAPTIVE_VALUE_ROLLING_MIN_HISTORY if dynamic.mode == ADAPTIVE_ROLLING_MODE else dynamic.min_history
        min_history = min(minimum, dynamic.window)
    return threshold_from_scores(
        values,
        fixed_threshold=fixed_threshold,
        target_right_rate=dynamic.target_right_rate,
        min_history=min_history,
        min_threshold=dynamic.min_threshold,
        max_threshold=dynamic.max_threshold,
    )


def threshold_from_scores(
    scores: Sequence[float],
    *,
    fixed_threshold: float,
    target_right_rate: float,
    min_history: int,
    min_threshold: float,
    max_threshold: float,
) -> float:
    if len(scores) < min_history:
        return fixed_threshold
    threshold = quantile(scores, 1.0 - target_right_rate)
    return clamp(threshold, min_threshold, max_threshold)


def percentile_to_target_right_rate(percentile: float) -> float:
    probability = percentile / 100.0 if percentile > 1.0 else percentile
    if not 0.0 < probability < 1.0:
        raise ValueError("percentile must be between 0 and 1 or 0 and 100")
    return 1.0 - probability


def recent_scores(
    csv_path: Path,
    *,
    limit: int,
    current_config: Mapping[str, object],
) -> list[float]:
    return recent_values(csv_path, value_field="score", limit=limit, current_config=current_config)


def recent_values(
    csv_path: Path,
    *,
    value_field: str,
    limit: int,
    current_config: Mapping[str, object],
) -> list[float]:
    if limit < 1 or not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row_matches_config(row, current_config):
                value = parse_float(row.get(value_field))
                if value is not None:
                    rows.append(value)
    return rows[-limit:]


def row_matches_config(row: Mapping[str, str], current_config: Mapping[str, object]) -> bool:
    expected = {
        "setup_name": current_config.get("setup_name"),
        "method": current_config.get("method"),
        "face_weight": current_config.get("face_weight"),
        "regressor_path": current_config.get("regressor_path"),
        "multimodal_regressor_path": current_config.get("multimodal_regressor_path"),
        "decision_mode": current_config.get("decision_mode"),
        "preference_model_path": current_config.get("preference_model_path"),
    }
    for field, expected_value in expected.items():
        actual = row.get(field, "")
        if expected_value is None or actual == "":
            continue
        if normalized_config_value(actual) != normalized_config_value(expected_value):
            return False
    return True


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if probability <= 0:
        return ordered[0]
    if probability >= 1:
        return ordered[-1]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalized_config_value(value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "\\").casefold()
